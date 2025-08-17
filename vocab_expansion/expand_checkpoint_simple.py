#!/usr/bin/env python3
"""
Robust checkpoint expansion for audio tokens (Gemma3→MaxText layout).

- Restores using MaxText's Orbax CheckpointManager (same as the converter).
- Unwraps TrainState(params={"params": ...}) to reach the unified embedding.
- Re-saves the result as a CheckpointManager step-0 checkpoint (manager layout).

Assumes original vocab=262_144 and adds 2_048 new tokens => 264_192 total.
"""

import json
from pathlib import Path
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

# Use MaxText's helper, same as in the converter.
from MaxText import checkpointing
from flax.training import train_state  # only to wrap for saving as TrainState again

# -------------------
# Constants
# -------------------
OLD_VOCAB_SIZE = 262_144
NEW_VOCAB_SIZE = 264_192  # = 262_144 + 2_048
NUM_NEW_TOKENS = 2_048
EMBED_DIM_HINT = 2_560  # Gemma3-4b embedding dim (informational)

# Unused token IDs (same definition you used earlier)
UNUSED_TOKEN_IDS = []
UNUSED_TOKEN_IDS.extend(range(6, 105))            # 99 tokens
UNUSED_TOKEN_IDS.extend(range(256001, 262144))    # 6143 tokens
assert len(UNUSED_TOKEN_IDS) == 6242

# -------------------
# Helpers
# -------------------

def _resolve_root_and_step(path: Path) -> Tuple[Path, int]:
    """
    Accept either:
      - the top-level manager dir (with step subdirs like /0, /1, ...)
      - a concrete step dir (e.g., .../0)
    Return (root_dir, step_number).
    """
    p = path
    if p.is_dir() and p.name.isdigit():
        return p.parent, int(p.name)

    # Otherwise assume top-level manager dir; pick latest numeric step.
    step_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.isdigit()]
    if not step_dirs:
        raise FileNotFoundError(
            f"No step subdirectories found under: {p}. "
            "Expected something like .../<root>/0/"
        )
    step = max(int(d.name) for d in step_dirs)
    return p, step


def _unwrap_params_tree(restored: Any) -> Tuple[dict, str]:
    """
    Return (params_tree, style) where:
      - params_tree is the inner dict containing model params (token_embedder, decoder, etc.)
      - style describes wrapping so we can re-wrap before saving:
          "trainstate_double"  -> TrainState(params={"params": ...})
          "trainstate_single"  -> TrainState(params={...})
          "leaf_double"        -> {"params": {"params": ...}}
          "leaf_single"        -> {"params": {...}}
    """
    # TrainState-like (Flax): has attribute 'params'
    if hasattr(restored, "params"):
        outer = restored.params
        if isinstance(outer, dict) and "params" in outer:  # double-nested
            return outer["params"], "trainstate_double"
        return outer, "trainstate_single"

    # Dict leaves
    if isinstance(restored, dict) and "params" in restored:
        if isinstance(restored["params"], dict) and "params" in restored["params"]:
            return restored["params"]["params"], "leaf_double"
        return restored["params"], "leaf_single"

    # Fallback: assume 'restored' itself is the params tree
    if isinstance(restored, dict):
        return restored, "leaf_single"

    raise ValueError("Unsupported checkpoint structure; cannot find parameters tree.")


def _rewrap_for_trainstate(params_tree: dict, style_in: str) -> train_state.TrainState:
    """
    Re-wrap updated params in a TrainState object so we can save with the Manager.
    We always produce TrainState(step=0, params={...}) and mirror 'double' nesting if needed.
    """
    if style_in in ("trainstate_double", "leaf_double"):
        params_wrapped = {"params": params_tree}
    else:
        params_wrapped = params_tree

    return train_state.TrainState(
        step=0,
        apply_fn=None,
        params=params_wrapped,
        tx=None,
        opt_state={},
    )

# -------------------
# Main expansion
# -------------------

def expand_checkpoint(checkpoint_path: str, output_path: str, seed: int = 42):
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    print(f"Resolving checkpoint path: {checkpoint_path}")
    root_dir, step_num = _resolve_root_and_step(checkpoint_path)
    print(f" -> root: {root_dir}")
    print(f" -> step: {step_num}")

    # Create a manager pointing at the root and restore that step
    enable_checkpointing = True
    async_checkpointing = False
    save_interval_steps = 1
    manager = checkpointing.create_orbax_checkpoint_manager(
        str(root_dir), enable_checkpointing, async_checkpointing, save_interval_steps
    )
    restored = manager.restore(step_num)
    # If the MaxText helper returns a dict of items, unwrap the single item.
    if isinstance(restored, dict) and len(restored) == 1:
        restored = list(restored.values())[0]

    params_tree, style_in = _unwrap_params_tree(restored)

    # Reach the unified embedding (the same tensor you wrote during conversion)
    # In your converter this lived at jax_weights["token_embedder"]["embedding"]【6:convert_gemma3_chkpt.py†turn1file6†L1-L3】
    if "token_embedder" not in params_tree or "embedding" not in params_tree["token_embedder"]:
        raise KeyError("Could not find token_embedder/embedding in checkpoint params.")

    old_embeddings = params_tree["token_embedder"]["embedding"]
    actual_vocab_size, actual_embed_dim = old_embeddings.shape
    print(f"Original embedding shape: {old_embeddings.shape}")  # 【2:expand_checkpoint_simple.py†turn1file2†L66-L71】

    if actual_vocab_size != OLD_VOCAB_SIZE:
        print(f"WARNING: expected vocab size {OLD_VOCAB_SIZE}, got {actual_vocab_size}")
    if actual_embed_dim != EMBED_DIM_HINT:
        print(f"Note: embedding dim is {actual_embed_dim} (hint {EMBED_DIM_HINT} for Gemma3-4b)")

    # Compute initialization from the average of the unused-token embeddings
    print("\nExtracting unused token embeddings & computing average…")
    idx = jnp.array(UNUSED_TOKEN_IDS)
    unused_embeddings = jnp.take(old_embeddings, idx, axis=0)
    avg_unused = unused_embeddings.mean(axis=0, keepdims=True)
    print(f"Average-unused L2 norm: {jnp.linalg.norm(avg_unused):.4f}")

    print(f"\nInitializing {NUM_NEW_TOKENS} new embeddings.")
    key = jax.random.PRNGKey(seed)
    noise_scale = 0.02
    noise = jax.random.normal(key, (NUM_NEW_TOKENS, actual_embed_dim), dtype=old_embeddings.dtype) * noise_scale
    new_embeddings = jnp.tile(avg_unused.astype(old_embeddings.dtype), (NUM_NEW_TOKENS, 1)) + noise

    # Concatenate to form the expanded table
    expanded_embeddings = jnp.concatenate([old_embeddings, new_embeddings], axis=0)
    print(f"Expanded embedding shape: {expanded_embeddings.shape}")
    if expanded_embeddings.shape[0] != NEW_VOCAB_SIZE:
        raise ValueError(f"Expected {NEW_VOCAB_SIZE} rows, got {expanded_embeddings.shape[0]}")

    # Mutate params and re-wrap in TrainState for manager-style saving
    params_tree = dict(params_tree)  # shallow copy
    params_tree["token_embedder"] = dict(params_tree["token_embedder"])
    params_tree["token_embedder"]["embedding"] = expanded_embeddings

    state_to_save = _rewrap_for_trainstate(params_tree, style_in)

    # Save as a manager checkpoint to the requested output path at step 0
    print(f"\nSaving expanded checkpoint (manager format) to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    out_manager = checkpointing.create_orbax_checkpoint_manager(
        str(output_path), enable_checkpointing, async_checkpointing, save_interval_steps
    )
    _ = checkpointing.save_checkpoint(out_manager, 0, state_to_save)
    print("Checkpoint saved at step 0.")

    # Write a small metadata sidecar
    meta = {
        "source_root": str(root_dir),
        "source_step": int(step_num),
        "original_vocab_size": int(actual_vocab_size),
        "new_vocab_size": NEW_VOCAB_SIZE,
        "num_new_tokens": NUM_NEW_TOKENS,
        "embed_dim": int(actual_embed_dim),
        "unused_token_ids_used": len(UNUSED_TOKEN_IDS),
        "init": {"method": "avg_unused + noise", "noise_scale": float(noise_scale), "seed": int(seed)},
    }
    with open(output_path / "expansion_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {output_path / 'expansion_metadata.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Expand checkpoint for audio tokens (manager-safe).")
    parser.add_argument("--checkpoint-path", required=True,
                        help="Path to converted checkpoint root (same dir you passed to the converter), or to a step dir like .../0")
    parser.add_argument("--output-path", required=True,
                        help="Directory to write the expanded manager checkpoint (will contain a new step 0).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    expand_checkpoint(args.checkpoint_path, args.output_path, args.seed)
