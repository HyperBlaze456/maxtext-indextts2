#!/usr/bin/env python3
"""
Expand Gemma 3 (MaxText-converted) token embeddings to match a repurposed tokenizer
that appended new tokens (e.g., +2048 <AUDIO_*>), and initialize the new rows.

Usage:
  python expand_gemma3_embedder.py \
      --checkpoint-path /path/to/converted_gemma3_ckpt \
      --tokenizer-dir ./gemma3-audio-tokenizer \
      --in-step 0 --out-step 1 \
      --std 0.02 --seed 0

Notes:
- Requires the MaxText package (same import used in test_load.py) and JAX.
- Reads new vocab size from tokenizer_dir/audio_token_metadata.json (written by repurpose_tokenizer.py).
- Expands:
    * params/params/token_embedder/embedding (rows = vocab size)
    * any output projection matrices whose shape matches (hidden_size, vocab_size)
      or (vocab_size, hidden_size), and any bias vectors of length vocab_size.
- New rows/cols are initialized from truncated normal N(0, std^2) with std=--std.
- Aliased <unused*> -> <AUDIO_i> entries reuse existing rows; only truly new IDs are added.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, List

import jax
import jax.numpy as jnp

from MaxText import checkpointing


def _trunc_normal(key, shape, std, dtype):
    """Sample truncated normal ([-2, 2] standard deviations), then scale by std."""
    x = jax.random.truncated_normal(key, lower=-2.0, upper=2.0, shape=shape, dtype=dtype)
    return x * std


def _expand_rows(old: jnp.ndarray, new_rows: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([old, new_rows], axis=0)


def _expand_cols(old: jnp.ndarray, new_cols: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([old, new_cols], axis=1)


def _walk_params(tree: Dict[str, Any], path: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], Any, Dict[str, Any], str]]:
    """
    Yield (path, value, parent_dict, leaf_key) for all leaves in a nested dict.
    """
    out = []
    for k, v in tree.items():
        if isinstance(v, dict):
            out.extend(_walk_params(v, path + (k,)))
        else:
            out.append((path + (k,), v, tree, k))
    return out


ess = "params"


def expand_checkpoint_embeddings(
    checkpoint_path: str,
    tokenizer_dir: str,
    in_step: int = 0,
    out_step: int = 1,
    std: float = 0.02,
    seed: int = 0,
):
    # --- Load tokenizer metadata to get target vocab size ---
    tok_dir = Path(tokenizer_dir)
    meta_path = tok_dir / "audio_token_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find {meta_path}. Did you run repurpose_tokenizer.py?")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_vocab_size = int(meta["new_vocab_size"])  # final tokenizer length
    original_vocab_size = int(meta["original_vocab_size"])
    appended_count = int(meta["appended_audio_count"])  # purely new tokens
    print(f"[Tokenizer] original_vocab_size={original_vocab_size}, new_vocab_size={target_vocab_size}, appended={appended_count}")

    # --- Restore checkpoint ---
    print(f"[Checkpoint] Restoring from {checkpoint_path}, step={in_step}")
    manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_path, True, False, 1
    )
    restored = manager.restore(in_step)

    # Navigate to params
    items = restored["items"]
    params_root = items["params"]["params"]

    # --- Token embedder ---
    if "token_embedder" not in params_root or "embedding" not in params_root["token_embedder"]:
        raise KeyError("Could not find params['params']['token_embedder']['embedding'] in checkpoint.")

    emb = params_root["token_embedder"]["embedding"]
    if emb.ndim != 2:
        raise ValueError(f"Expected embedding matrix of rank-2, got shape {emb.shape}")

    old_vocab_size, hidden_size = int(emb.shape[0]), int(emb.shape[1])
    emb_dtype = emb.dtype
    print(f"[Embedder] current shape={emb.shape} (vocab={old_vocab_size}, hidden={hidden_size}), dtype={emb_dtype}")

    if target_vocab_size <= old_vocab_size:
        print("[Embedder] No expansion needed: target_vocab_size <= current_vocab_size")
    else:
        delta = target_vocab_size - old_vocab_size
        print(f"[Embedder] Expanding by {delta} rows to reach {target_vocab_size}.")

        key = jax.random.PRNGKey(seed)
        new_rows = _trunc_normal(key, shape=(delta, hidden_size), std=std, dtype=emb_dtype)

        # Concatenate
        new_emb = _expand_rows(emb, new_rows)
        params_root["token_embedder"]["embedding"] = new_emb
        print(f"[Embedder] New shape={new_emb.shape}")

    # --- Expand any output projection heads that depend on vocab size ---
    # Heuristics:
    #  - kernel shaped (hidden_size, old_vocab_size) -> expand columns
    #  - kernel shaped (old_vocab_size, hidden_size) -> expand rows
    #  - bias shaped (old_vocab_size,) -> expand length
    expanded_heads = 0
    expanded_bias = 0
    for path, leaf, parent, key_name in _walk_params(params_root):
        if isinstance(leaf, jnp.ndarray):
            shape = tuple(int(s) for s in leaf.shape)
            # Skip the token embedder (already handled)
            if path[-2:] == ("token_embedder", "embedding"):
                continue

            # Expand kernels
            if len(shape) == 2:
                if shape == (hidden_size, old_vocab_size):
                    delta = target_vocab_size - old_vocab_size
                    if delta > 0:
                        key = jax.random.PRNGKey(seed + expanded_heads + 1)
                        new_cols = _trunc_normal(key, (hidden_size, delta), std, leaf.dtype)
                        parent[key_name] = _expand_cols(leaf, new_cols)
                        expanded_heads += 1
                        print(f"[Head] Expanded cols at {'/'.join(path)}: {shape} -> {parent[key_name].shape}")
                elif shape == (old_vocab_size, hidden_size):
                    delta = target_vocab_size - old_vocab_size
                    if delta > 0:
                        key = jax.random.PRNGKey(seed + expanded_heads + 1)
                        new_rows = _trunc_normal(key, (delta, hidden_size), std, leaf.dtype)
                        parent[key_name] = _expand_rows(leaf, new_rows)
                        expanded_heads += 1
                        print(f"[Head] Expanded rows at {'/'.join(path)}: {shape} -> {parent[key_name].shape}")

            # Expand biases
            if len(shape) == 1 and shape[0] == old_vocab_size:
                delta = target_vocab_size - old_vocab_size
                if delta > 0:
                    zeros = jnp.zeros((delta,), dtype=leaf.dtype)
                    parent[key_name] = jnp.concatenate([leaf, zeros], axis=0)
                    expanded_bias += 1
                    print(f"[Bias] Expanded at {'/'.join(path)}: {shape} -> {parent[key_name].shape}")

    print(f"[Summary] Expanded {expanded_heads} projection matrices and {expanded_bias} bias vectors that matched vocab size.")

    # --- Save to new step ---
    print(f"[Checkpoint] Saving to step={out_step}")
    manager.save(out_step, restored)
    print("[Done] Saved expanded checkpoint.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-path", required=True, help="Path to converted checkpoint root (directory).")
    ap.add_argument("--tokenizer-dir", required=True, help="Path to tokenizer dir created by repurpose_tokenizer.py")
    ap.add_argument("--in-step", type=int, default=0, help="Step number to load from (default: 0)")
    ap.add_argument("--out-step", type=int, default=1, help="Step number to save to (default: 1)")
    ap.add_argument("--std", type=float, default=0.02, help="Initializer std for new rows/cols (default: 0.02)")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed for initializer")
    args = ap.parse_args()

    expand_checkpoint_embeddings(
        checkpoint_path=args.checkpoint_path,
        tokenizer_dir=args.tokenizer_dir,
        in_step=args.in_step,
        out_step=args.out_step,
        std=args.std,
        seed=args.seed,
    )
