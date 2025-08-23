#!/usr/bin/env python3
"""
Expand Gemma 3 (MaxText-converted) token embeddings under three modes and initialize
new entries with Gemma's default (truncated normal, std=0.02 unless overridden).

Modes (choose ONE):
  1) --append-count N          : append exactly N new token rows (and heads/biases)
  2) --tokenizer-dir DIR       : match vocab size found in audio_token_metadata.json
  3) --pad-to-pow2             : pad to next power-of-two greater than current vocab

Examples:
  # Your requested fix: append exactly 2048 rows, keep std=0.02
  python expand_gemma3_embedder.py \
    --checkpoint-path /path/to/converted_gemma3_ckpt \
    --append-count 2048 --in-step 0 --out-step 1 --std 0.02 --seed 0

  # Match a repurposed tokenizer directory (from repurpose_tokenizer.py)
  python expand_gemma3_embedder.py --checkpoint-path CKPT --tokenizer-dir TOK --in-step 0 --out-step 1

  # Pad to next power-of-two (if you want to re-align sizes)
  python expand_gemma3_embedder.py --checkpoint-path CKPT --pad-to-pow2

Notes:
- Requires the MaxText package (same import used in test_load.py) and JAX.
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
from typing import Any, Dict, Tuple, List, Optional

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


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _resolve_target_vocab(
    old_vocab_size: int,
    tokenizer_dir: Optional[str],
    append_count: Optional[int],
    pad_to_pow2: bool,
) -> int:
    # Priority: explicit append > tokenizer dir > pad-to-pow2 flag (relative to current)
    if append_count is not None:
        if append_count < 0:
            raise ValueError("--append-count must be non-negative")
        return old_vocab_size + append_count

    if tokenizer_dir is not None:
        meta_path = Path(tokenizer_dir) / "audio_token_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Could not find {meta_path}. Did you run repurpose_tokenizer.py?")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta["new_vocab_size"])  # final length produced by tokenizer rewrite

    if pad_to_pow2:
        return _next_pow2(old_vocab_size if (old_vocab_size & (old_vocab_size - 1)) else old_vocab_size)

    raise ValueError("You must specify one of: --append-count, --tokenizer-dir, or --pad-to-pow2")


def expand_checkpoint_embeddings(
    checkpoint_path: str,
    tokenizer_dir: Optional[str],
    append_count: Optional[int],
    pad_to_pow2: bool,
    in_step: int = 0,
    out_step: int = 1,
    std: float = 0.02,
    seed: int = 0,
):
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

    # Decide target size
    target_vocab_size = _resolve_target_vocab(old_vocab_size, tokenizer_dir, append_count, pad_to_pow2)
    print(f"[Target] target_vocab_size={target_vocab_size}")

    if target_vocab_size <= old_vocab_size:
        print("[Embedder] No expansion needed: target_vocab_size <= current_vocab_size")
    else:
        delta = target_vocab_size - old_vocab_size
        print(f"[Embedder] Expanding by {delta} rows to reach {target_vocab_size}.")
        key = jax.random.PRNGKey(seed)
        new_rows = _trunc_normal(key, shape=(delta, hidden_size), std=std, dtype=emb_dtype)
        params_root["token_embedder"]["embedding"] = _expand_rows(emb, new_rows)
        print(f"[Embedder] New shape={params_root['token_embedder']['embedding'].shape}")

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
                # (hidden, vocab)
                if shape == (hidden_size, old_vocab_size):
                    if target_vocab_size > old_vocab_size:
                        key = jax.random.PRNGKey(seed + expanded_heads + 1)
                        new_cols = _trunc_normal(key, (hidden_size, target_vocab_size - old_vocab_size), std, leaf.dtype)
                        parent[key_name] = _expand_cols(leaf, new_cols)
                        expanded_heads += 1
                        print(f"[Head] Expanded cols at {'/'.join(path)}: {shape} -> {parent[key_name].shape}")
                # (vocab, hidden)
                elif shape == (old_vocab_size, hidden_size):
                    if target_vocab_size > old_vocab_size:
                        key = jax.random.PRNGKey(seed + expanded_heads + 1)
                        new_rows = _trunc_normal(key, (target_vocab_size - old_vocab_size, hidden_size), std, leaf.dtype)
                        parent[key_name] = _expand_rows(leaf, new_rows)
                        expanded_heads += 1
                        print(f"[Head] Expanded rows at {'/'.join(path)}: {shape} -> {parent[key_name].shape}")

            # Expand biases
            if len(shape) == 1 and shape[0] == old_vocab_size:
                if target_vocab_size > old_vocab_size:
                    zeros = jnp.zeros((target_vocab_size - old_vocab_size,), dtype=leaf.dtype)
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
    # Modes
    ap.add_argument("--append-count", type=int, default=None, help="Append exactly this many new tokens to the current vocab size.")
    ap.add_argument("--tokenizer-dir", default=None, help="Path to tokenizer dir created by repurpose_tokenizer.py (reads audio_token_metadata.json).")
    ap.add_argument("--pad-to-pow2", action="store_true", help="Pad to next power-of-two >= current vocab size.")
    # IO and init
    ap.add_argument("--in-step", type=int, default=0, help="Step number to load from (default: 0)")
    ap.add_argument("--out-step", type=int, default=1, help="Step number to save to (default: 1)")
    ap.add_argument("--std", type=float, default=0.02, help="Initializer std for new rows/cols (default: 0.02)")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed for initializer")
    args = ap.parse_args()

    expand_checkpoint_embeddings(
        checkpoint_path=args.checkpoint_path,
        tokenizer_dir=args.tokenizer_dir,
        append_count=args.append_count,
        pad_to_pow2=args.pad_to_pow2,
        in_step=args.in_step,
        out_step=args.out_step,
        std=args.std,
        seed=args.seed,
    )
#!/usr/bin/env python3
"""
Expand Gemma 3 (MaxText-converted) token embeddings under three modes and initialize
new entries with Gemma's default (truncated normal, std=0.02 unless overridden).

Modes (choose ONE):
  1) --append-count N          : append exactly N new token rows (and heads/biases)
  2) --tokenizer-dir DIR       : match vocab size found in audio_token_metadata.json
  3) --pad-to-pow2             : pad to next power-of-two greater than current vocab

Examples:
  # Your requested fix: append exactly 2048 rows, keep std=0.02
  python expand_gemma3_embedder.py \
    --checkpoint-path /path/to/converted_gemma3_ckpt \
    --append-count 2048 --in-step 0 --out-step 1 --std 0.02 --seed 0

  # Match a repurposed tokenizer directory (from repurpose_tokenizer.py)
  python expand_gemma3_embedder.py --checkpoint-path CKPT --tokenizer-dir TOK --in-step 0 --out-step 1

  # Pad to next power-of-two (if you want to re-align sizes)
  python expand_gemma3_embedder.py --checkpoint-path CKPT --pad-to-pow2

Notes:
- Requires the MaxText package (same import used in test_load.py) and JAX.
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
from typing import Any, Dict, Tuple, List, Optional

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


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _resolve_target_vocab(
    old_vocab_size: int,
    tokenizer_dir: Optional[str],
    append_count: Optional[int],
    pad_to_pow2: bool,
) -> int:
    # Priority: explicit append > tokenizer dir > pad-to-pow2 flag (relative to current)
    if append_count is not None:
        if append_count < 0:
            raise ValueError("--append-count must be non-negative")
        return old_vocab_size + append_count

    if tokenizer_dir is not None:
        meta_path = Path(tokenizer_dir) / "audio_token_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Could not find {meta_path}. Did you run repurpose_tokenizer.py?")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta["new_vocab_size"])  # final length produced by tokenizer rewrite

    if pad_to_pow2:
        return _next_pow2(old_vocab_size if (old_vocab_size & (old_vocab_size - 1)) else old_vocab_size)

    raise ValueError("You must specify one of: --append-count, --tokenizer-dir, or --pad-to-pow2")


def expand_checkpoint_embeddings(
    checkpoint_path: str,
    tokenizer_dir: Optional[str],
    append_count: Optional[int],
    pad_to_pow2: bool,
    in_step: int = 0,
    out_step: int = 1,
    std: float = 0.02,
    seed: int = 0,
):
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

    # Decide target size
    target_vocab_size = _resolve_target_vocab(old_vocab_size, tokenizer_dir, append_count, pad_to_pow2)
    print(f"[Target] target_vocab_size={target_vocab_size}")

    if target_vocab_size <= old_vocab_size:
        print("[Embedder] No expansion needed: target_vocab_size <= current_vocab_size")
    else:
        delta = target_vocab_size - old_vocab_size
        print(f"[Embedder] Expanding by {delta} rows to reach {target_vocab_size}.")
        key = jax.random.PRNGKey(seed)
        new_rows = _trunc_normal(key, shape=(delta, hidden_size), std=std, dtype=emb_dtype)
        params_root["token_embedder"]["embedding"] = _expand_rows(emb, new_rows)
        print(f"[Embedder] New shape={params_root['token_embedder']['embedding'].shape}")

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
                # (hidden, vocab)
                if shape == (hidden_size, old_vocab_size):
                    if target_vocab_size > old_vocab_size:
                        key = jax.random.PRNGKey(seed + expanded_heads + 1)
                        new_cols = _trunc_normal(key, (hidden_size, target_vocab_size - old_vocab_size), std, leaf.dtype)
                        parent[key_name] = _expand_cols(leaf, new_cols)
                        expanded_heads += 1
                        print(f"[Head] Expanded cols at {'/'.join(path)}: {shape} -> {parent[key_name].shape}")
                # (vocab, hidden)
                elif shape == (old_vocab_size, hidden_size):
                    if target_vocab_size > old_vocab_size:
                        key = jax.random.PRNGKey(seed + expanded_heads + 1)
                        new_rows = _trunc_normal(key, (target_vocab_size - old_vocab_size, hidden_size), std, leaf.dtype)
                        parent[key_name] = _expand_rows(leaf, new_rows)
                        expanded_heads += 1
                        print(f"[Head] Expanded rows at {'/'.join(path)}: {shape} -> {parent[key_name].shape}")

            # Expand biases
            if len(shape) == 1 and shape[0] == old_vocab_size:
                if target_vocab_size > old_vocab_size:
                    zeros = jnp.zeros((target_vocab_size - old_vocab_size,), dtype=leaf.dtype)
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
    # Modes
    ap.add_argument("--append-count", type=int, default=None, help="Append exactly this many new tokens to the current vocab size.")
    ap.add_argument("--tokenizer-dir", default=None, help="Path to tokenizer dir created by repurpose_tokenizer.py (reads audio_token_metadata.json).")
    ap.add_argument("--pad-to-pow2", action="store_true", help="Pad to next power-of-two >= current vocab size.")
    # IO and init
    ap.add_argument("--in-step", type=int, default=0, help="Step number to load from (default: 0)")
    ap.add_argument("--out-step", type=int, default=1, help="Step number to save to (default: 1)")
    ap.add_argument("--std", type=float, default=0.02, help="Initializer std for new rows/cols (default: 0.02)")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed for initializer")
    args = ap.parse_args()

    expand_checkpoint_embeddings(
        checkpoint_path=args.checkpoint_path,
        tokenizer_dir=args.tokenizer_dir,
        append_count=args.append_count,
        pad_to_pow2=args.pad_to_pow2,
        in_step=args.in_step,
        out_step=args.out_step,
        std=args.std,
        seed=args.seed,
    )
