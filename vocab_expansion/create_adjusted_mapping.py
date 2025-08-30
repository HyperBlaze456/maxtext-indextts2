#!/usr/bin/env python3
"""
Create an adjusted audio token mapping that properly handles the skipped index 262144.

The soft token at index 262144 should not be mapped to any embeddings. 
This means all token indices > 262144 need to be adjusted by subtracting 1
to get the correct embedding index.
"""

import json
import argparse
from pathlib import Path


def create_adjusted_mapping(original_mapping_path):
    """
    Create an adjusted mapping that handles the index 262144 skip.
    
    For token remapping during inference:
    - Tokens with index < 262144: use as-is
    - Token at index 262144: should never appear (soft token)
    - Tokens with index > 262144: subtract 1 to get embedding index
    """
    
    # Load original mapping
    with open(original_mapping_path, 'r') as f:
        original_data = json.load(f)
    
    original_mappings = original_data['audio_mappings']
    
    # Create adjusted mapping
    adjusted_mappings = {}
    embedding_to_audio = {}  # Reverse mapping for verification
    
    for token_idx_str, audio_id in original_mappings.items():
        token_idx = int(token_idx_str)
        
        # Skip the soft token at 262144
        if token_idx == 262144:
            print(f"Skipping soft token at index 262144")
            continue
        
        # Calculate the embedding index
        if token_idx < 262144:
            embedding_idx = token_idx
        else:
            # 262145 -> 262144, or else embedding gets overflowed. Rest of it gets shifted -1 too.
            embedding_idx = token_idx - 1
        
        # Store both mappings
        adjusted_mappings[str(embedding_idx)] = {
            'audio_id': audio_id,
            'original_token_idx': token_idx
        }
        
        # Store reverse mapping for audio_id -> embedding_idx
        if audio_id >= 0:  # Skip padding tokens (-1)
            embedding_to_audio[str(audio_id)] = embedding_idx
    
    # Calculate statistics
    total_audio_tokens = sum(1 for v in adjusted_mappings.values() if v['audio_id'] >= 0)
    padding_tokens = sum(1 for v in adjusted_mappings.values() if v['audio_id'] == -1)
    max_embedding_idx = max(int(k) for k in adjusted_mappings.keys())
    
    # Create the adjusted mapping file
    adjusted_data = {
        'embedding_to_audio': {k: v['audio_id'] for k, v in adjusted_mappings.items()},
        'audio_to_embedding': embedding_to_audio,
        'detailed_mappings': adjusted_mappings,
        'stats': {
            'total_mappings': len(adjusted_mappings),
            'total_audio_tokens': total_audio_tokens,
            'padding_tokens': padding_tokens,
            'max_embedding_index': max_embedding_idx,
            'original_vocab_size': original_data.get('vocab_size', 264192),
            'adjusted_vocab_size': original_data.get('vocab_size', 264192) - 1,  # One less due to skipped 262144
            'note': 'Soft token at original index 262144 is excluded. Indices > 262144 are shifted down by 1.'
        }
    }
    
    # Save adjusted mapping
    output_path = 'audio_token_mapping_adjusted.json'
    with open(output_path, 'w') as f:
        json.dump(adjusted_data, f, indent=2)
    
    print(f"\nAdjusted mapping created successfully!")
    print(f"Statistics:")
    print(f"- Total mappings: {adjusted_data['stats']['total_mappings']}")
    print(f"- Audio tokens: {adjusted_data['stats']['total_audio_tokens']}")
    print(f"- Padding tokens: {adjusted_data['stats']['padding_tokens']}")
    print(f"- Max embedding index: {adjusted_data['stats']['max_embedding_index']}")
    print(f"- Adjusted vocab size: {adjusted_data['stats']['adjusted_vocab_size']}")
    print(f"\nSaved to: {output_path}")
    
    return adjusted_data


def verify_mapping(adjusted_data):
    """Verify the adjusted mapping is correct."""
    
    print("\nVerifying mapping...")
    
    # Check that no embedding index equals or exceeds the adjusted vocab size
    max_idx = adjusted_data['stats']['max_embedding_index']
    vocab_size = adjusted_data['stats']['adjusted_vocab_size']
    
    if max_idx >= vocab_size:
        print(f"WARNING: Max embedding index {max_idx} >= vocab size {vocab_size}")
    else:
        print(f"✓ Max embedding index {max_idx} < vocab size {vocab_size}")
    
    # Check audio token coverage
    audio_to_emb = adjusted_data['audio_to_embedding']
    audio_ids = sorted([int(k) for k in audio_to_emb.keys()])
    
    # Check for gaps in audio IDs
    expected_audio = set(range(8192))  # 0-8191 for audio tokens
    mapped_audio = set(audio_ids)
    missing_audio = expected_audio - mapped_audio
    
    if missing_audio:
        print(f"WARNING: Missing audio IDs: {len(missing_audio)} tokens not mapped")
        print(f"  First few missing: {sorted(missing_audio)[:10]}")
    else:
        print(f"✓ All 8192 audio tokens are mapped")
    
    # Verify no embedding index is used twice
    embedding_indices = list(adjusted_data['embedding_to_audio'].keys())
    if len(embedding_indices) == len(set(embedding_indices)):
        print(f"✓ No duplicate embedding indices")
    else:
        print(f"WARNING: Duplicate embedding indices found!")
    
    print("\nVerification complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create adjusted audio token mapping")
    parser.add_argument("input_path", help="Path to original audio_token_mapping.json")
    parser.add_argument("--output", default="vocab_expansion/audio_token_mapping_adjusted.json", 
                        help="Output path for adjusted mapping")
    args = parser.parse_args()
    
    adjusted_data = create_adjusted_mapping(args.input_path)
    verify_mapping(adjusted_data)