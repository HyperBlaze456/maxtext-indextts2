#!/usr/bin/env python3
"""
Simple checkpoint expansion for audio tokens.

Expands embeddings from 262,144 to 264,192 tokens:
- Uses existing embeddings for 6242 repurposed tokens
- Adds 2048 new embeddings initialized from unused token average
"""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


# Fixed constants
OLD_VOCAB_SIZE = 262_144
NEW_VOCAB_SIZE = 264_192  # 262_144 + 2_048
NUM_NEW_TOKENS = 2_048
EMBED_DIM = 2_560  # Gemma3-4b embedding dimension

# Hardcoded unused token IDs (same as in repurpose_tokenizer.py)
UNUSED_TOKEN_IDS = []

# First range: IDs 6 to 104 (99 tokens)
UNUSED_TOKEN_IDS.extend(range(6, 105))

# Second range: IDs 256001 to 262143 (6143 tokens)  
UNUSED_TOKEN_IDS.extend(range(256001, 262144))

# Total: 6242 unused token IDs
assert len(UNUSED_TOKEN_IDS) == 6242, f"Expected 6242 unused tokens, got {len(UNUSED_TOKEN_IDS)}"


def expand_checkpoint(
    checkpoint_path: str,
    output_path: str,
    seed: int = 42
):
    """Expand checkpoint embeddings for audio tokens.
    
    Args:
        checkpoint_path: Path to original MaxText checkpoint
        output_path: Path to save expanded checkpoint
        seed: Random seed for initialization
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(checkpoint_path)
    
    # Get embeddings from standard location
    if "params" not in ckpt:
        raise ValueError("Invalid checkpoint: missing 'params'")
    
    params = ckpt["params"]
    
    if "token_embedder" not in params:
        raise ValueError("Invalid checkpoint: missing 'token_embedder'")
    
    if "embedding" not in params["token_embedder"]:
        raise ValueError("Invalid checkpoint: missing embeddings")
    
    # Get original embeddings
    old_embeddings = params["token_embedder"]["embedding"]
    actual_vocab_size, actual_embed_dim = old_embeddings.shape
    
    print(f"Original embedding shape: {old_embeddings.shape}")
    
    # Verify dimensions
    if actual_vocab_size != OLD_VOCAB_SIZE:
        print(f"WARNING: Expected vocab size {OLD_VOCAB_SIZE}, got {actual_vocab_size}")
    
    if actual_embed_dim != EMBED_DIM:
        print(f"Note: Embedding dimension is {actual_embed_dim} (expected {EMBED_DIM} for Gemma3-4b)")
    
    # Extract embeddings of unused tokens
    print("\nExtracting unused token embeddings...")
    unused_embeddings = old_embeddings[UNUSED_TOKEN_IDS]
    print(f"Unused embeddings shape: {unused_embeddings.shape}")
    
    # Calculate initialization for new tokens
    # Use average of unused embeddings as base
    avg_unused = unused_embeddings.mean(axis=0, keepdims=True)
    print(f"Average unused embedding norm: {jnp.linalg.norm(avg_unused):.4f}")
    
    # Initialize new embeddings
    print(f"\nInitializing {NUM_NEW_TOKENS} new embeddings...")
    key = jax.random.PRNGKey(seed)
    
    # Add small noise to break symmetry
    noise_scale = 0.02
    noise = jax.random.normal(key, (NUM_NEW_TOKENS, actual_embed_dim)) * noise_scale
    
    # New embeddings = average of unused + noise
    new_embeddings = jnp.tile(avg_unused, (NUM_NEW_TOKENS, 1)) + noise
    
    # Verify new embeddings
    new_norms = jnp.linalg.norm(new_embeddings, axis=1)
    print(f"New embeddings norm - mean: {new_norms.mean():.4f}, std: {new_norms.std():.4f}")
    
    # Concatenate old and new embeddings
    expanded_embeddings = jnp.concatenate([old_embeddings, new_embeddings], axis=0)
    print(f"\nExpanded embeddings shape: {expanded_embeddings.shape}")
    
    # Verify final shape
    if expanded_embeddings.shape[0] != NEW_VOCAB_SIZE:
        raise ValueError(f"Expected {NEW_VOCAB_SIZE} embeddings, got {expanded_embeddings.shape[0]}")
    
    # Update checkpoint
    params["token_embedder"]["embedding"] = expanded_embeddings
    ckpt["params"] = params
    
    # Save expanded checkpoint
    print(f"\nSaving expanded checkpoint to {output_path}...")
    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    checkpointer.save(output_path, ckpt)
    print("Checkpoint saved successfully!")
    
    # Save metadata
    metadata = {
        "original_checkpoint": str(checkpoint_path),
        "original_vocab_size": OLD_VOCAB_SIZE,
        "new_vocab_size": NEW_VOCAB_SIZE,
        "num_new_tokens": NUM_NEW_TOKENS,
        "embed_dim": actual_embed_dim,
        "unused_token_ids_used": len(UNUSED_TOKEN_IDS),
        "initialization": {
            "method": "average_unused_plus_noise",
            "noise_scale": noise_scale,
            "seed": seed,
            "base_norm": float(jnp.linalg.norm(avg_unused)),
            "new_embeddings_norm_mean": float(new_norms.mean()),
            "new_embeddings_norm_std": float(new_norms.std())
        },
        "token_allocation": {
            "repurposed_audio": "6242 tokens (IDs from unused)",
            "new_audio": "1950 tokens (IDs 262144-264093)",
            "new_unused": "98 tokens (IDs 264094-264191)"
        }
    }
    
    metadata_path = output_dir.parent / f"{output_dir.name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Expansion Summary:")
    print("="*60)
    print(f"  Original: {OLD_VOCAB_SIZE} tokens x {actual_embed_dim} dims")
    print(f"  Expanded: {NEW_VOCAB_SIZE} tokens x {actual_embed_dim} dims")
    print(f"  Added: {NUM_NEW_TOKENS} new embeddings")
    print(f"  Initialization: Average of {len(UNUSED_TOKEN_IDS)} unused tokens + noise")
    
    return expanded_embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Expand checkpoint for audio tokens")
    parser.add_argument("--checkpoint-path", required=True, help="Path to original checkpoint")
    parser.add_argument("--output-path", required=True, help="Path for expanded checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    expand_checkpoint(args.checkpoint_path, args.output_path, args.seed)