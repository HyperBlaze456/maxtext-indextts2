#!/usr/bin/env python3
"""
Verify the audio token conversion worked correctly.
"""

import json
from pathlib import Path
import jax.numpy as jnp
import orbax.checkpoint as ocp
from transformers import AutoTokenizer


def verify_conversion(
    tokenizer_dir: str = "./gemma3-audio-tokenizer",
    checkpoint_path: str = None
):
    """Verify tokenizer and checkpoint conversion.
    
    Args:
        tokenizer_dir: Path to modified tokenizer
        checkpoint_path: Path to expanded checkpoint (optional)
    """
    print("="*60)
    print("Verifying Audio Token Conversion")
    print("="*60)
    
    # Load metadata
    metadata_path = Path(tokenizer_dir) / "audio_token_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("\nTokenizer metadata:")
        print(f"  Original vocab: {metadata['original_vocab_size']}")
        print(f"  New vocab: {metadata['new_vocab_size']}")
        print(f"  Audio tokens: {metadata['num_audio_tokens']}")
        print(f"  Unused tokens: {metadata['num_unused_tokens']}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Verify audio tokens
    print("\nVerifying audio tokens...")
    audio_test_cases = [
        (0, "<AUDIO_0>"),
        (100, "<AUDIO_100>"),
        (6241, "<AUDIO_6241>"),  # Last repurposed
        (6242, "<AUDIO_6242>"),  # First new
        (8191, "<AUDIO_8191>"),  # Last audio
    ]
    
    for idx, token in audio_test_cases:
        token_id = tokenizer.convert_tokens_to_ids([token])[0]
        if token_id != tokenizer.unk_token_id:
            print(f"  ✓ {token:15} -> ID {token_id}")
        else:
            print(f"  ✗ {token:15} NOT FOUND")
    
    # Verify unused tokens
    print("\nVerifying replacement unused tokens...")
    unused_test_cases = [
        (0, "<unused0>"),
        (50, "<unused50>"),
        (97, "<unused97>"),
    ]
    
    for idx, token in unused_test_cases:
        token_id = tokenizer.convert_tokens_to_ids([token])[0]
        if token_id != tokenizer.unk_token_id:
            print(f"  ✓ {token:15} -> ID {token_id}")
        else:
            print(f"  ✗ {token:15} NOT FOUND")
    
    # Test encoding/decoding
    print("\nTesting encoding/decoding...")
    test_texts = [
        "Hello world",
        "Audio: <AUDIO_0> <AUDIO_100> <AUDIO_8191>",
        "Mixed: text <AUDIO_5000> more text <unused50>",
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        print(f"  Original: '{text}'")
        print(f"  Tokens:   {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  Decoded:  '{decoded}'")
        print()
    
    # Verify checkpoint if provided
    if checkpoint_path:
        print("\n" + "="*60)
        print("Verifying Checkpoint")
        print("="*60)
        
        # Load checkpoint metadata if exists
        ckpt_metadata_path = Path(checkpoint_path).parent / f"{Path(checkpoint_path).name}_metadata.json"
        if ckpt_metadata_path.exists():
            with open(ckpt_metadata_path, 'r') as f:
                ckpt_metadata = json.load(f)
            print("\nCheckpoint metadata:")
            print(f"  Original vocab: {ckpt_metadata['original_vocab_size']}")
            print(f"  New vocab: {ckpt_metadata['new_vocab_size']}")
            print(f"  Embed dim: {ckpt_metadata['embed_dim']}")
            print(f"  Initialization: {ckpt_metadata['initialization']['method']}")
        
        # Load checkpoint
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpointer = ocp.PyTreeCheckpointer()
        ckpt = checkpointer.restore(checkpoint_path)
        
        if "params" in ckpt and "token_embedder" in ckpt["params"]:
            embeddings = ckpt["params"]["token_embedder"]["embedding"]
            vocab_size, embed_dim = embeddings.shape
            
            print(f"Embedding matrix shape: {embeddings.shape}")
            
            # Check if sizes match tokenizer
            if vocab_size == len(tokenizer):
                print(f"  ✓ Checkpoint vocab matches tokenizer ({vocab_size})")
            else:
                print(f"  ✗ Size mismatch! Checkpoint: {vocab_size}, Tokenizer: {len(tokenizer)}")
            
            # Analyze embeddings
            print("\nEmbedding statistics:")
            
            # Original tokens (first 262,144)
            orig_embeddings = embeddings[:262144]
            orig_norms = jnp.linalg.norm(orig_embeddings, axis=1)
            print(f"  Original tokens (0-262143):")
            print(f"    Norm - mean: {orig_norms.mean():.4f}, std: {orig_norms.std():.4f}")
            
            # New tokens (last 2,048)
            if vocab_size > 262144:
                new_embeddings = embeddings[262144:]
                new_norms = jnp.linalg.norm(new_embeddings, axis=1)
                print(f"  New tokens (262144-{vocab_size-1}):")
                print(f"    Norm - mean: {new_norms.mean():.4f}, std: {new_norms.std():.4f}")
            
            # Check for any NaN/Inf
            has_nan = jnp.any(jnp.isnan(embeddings))
            has_inf = jnp.any(jnp.isinf(embeddings))
            
            if has_nan or has_inf:
                print("\n  ✗ WARNING: Embeddings contain NaN or Inf values!")
            else:
                print("\n  ✓ No NaN or Inf values found")
        else:
            print("  ✗ Could not find embeddings in checkpoint")
    
    print("\n" + "="*60)
    print("Verification Complete")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify audio token conversion")
    parser.add_argument("--tokenizer-dir", default="./gemma3-audio-tokenizer", 
                       help="Path to modified tokenizer")
    parser.add_argument("--checkpoint-path", help="Path to expanded checkpoint (optional)")
    
    args = parser.parse_args()
    verify_conversion(args.tokenizer_dir, args.checkpoint_path)