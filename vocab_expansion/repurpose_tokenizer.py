#!/usr/bin/env python3
"""
Clean script to repurpose Gemma's unused tokens as audio tokens.

Token allocation:
- 6242 existing unused tokens -> <AUDIO_0> to <AUDIO_6241>
- 1950 new tokens -> <AUDIO_6242> to <AUDIO_8191>
- 98 new tokens -> <unused0> to <unused97>

Total: 8192 audio tokens + 98 replacement unused tokens
"""

import json
from pathlib import Path
from transformers import AutoTokenizer


# Fixed mapping of unused token IDs in Gemma tokenizer
# These are hardcoded based on Gemma's tokenizer structure
UNUSED_TOKEN_IDS = {}

# First range: <unused0> to <unused98> have IDs 6 to 104
for i in range(99):
    UNUSED_TOKEN_IDS[f"<unused{i}>"] = 6 + i

# Second range: <unused99> to <unused6241> have IDs 256001 to 262143  
for i in range(99, 6242):
    UNUSED_TOKEN_IDS[f"<unused{i}>"] = 256001 + (i - 99)


def repurpose_tokenizer(
    base_model: str = "google/gemma-3-4b",
    output_dir: str = "./gemma3-audio-tokenizer"
):
    """Repurpose unused tokens and add new tokens for audio.
    
    Args:
        base_model: Base Gemma model name
        output_dir: Directory to save modified tokenizer
    """
    print("Loading Gemma tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")
    
    # Step 1: Verify unused tokens exist and get their IDs
    print("\nVerifying unused tokens...")
    vocab = tokenizer.get_vocab()
    verified_unused = {}
    
    for token_name, expected_id in UNUSED_TOKEN_IDS.items():
        if token_name in vocab:
            actual_id = vocab[token_name]
            if actual_id != expected_id:
                print(f"  WARNING: {token_name} has ID {actual_id}, expected {expected_id}")
            verified_unused[token_name] = actual_id
        else:
            print(f"  ERROR: {token_name} not found in vocabulary")
    
    print(f"Found {len(verified_unused)} unused tokens to repurpose")
    
    # Step 2: Create mappings
    # Map old unused tokens to audio tokens
    token_remapping = {}
    for i, (unused_token, token_id) in enumerate(verified_unused.items()):
        audio_token = f"<AUDIO_{i}>"
        token_remapping[unused_token] = audio_token
    
    # Step 3: Modify tokenizer vocabulary
    # This is done by manipulating the tokenizer's internal structures
    # Note: This approach works for SentencePiece tokenizers (like Gemma)
    
    # Get the tokenizer's vocabulary as a dict
    vocab = tokenizer.get_vocab()
    
    # Create new vocabulary with renamed tokens
    new_vocab = {}
    for token, token_id in vocab.items():
        if token in token_remapping:
            # Rename unused token to audio token
            new_vocab[token_remapping[token]] = token_id
        else:
            new_vocab[token] = token_id
    
    # Step 4: Add new tokens
    # Add 1950 new audio tokens
    new_audio_tokens = []
    for i in range(6242, 8192):  # 6242 to 8191
        new_audio_tokens.append(f"<AUDIO_{i}>")
    
    # Add 98 new unused tokens
    new_unused_tokens = []
    for i in range(98):
        new_unused_tokens.append(f"<unused{i}>")
    
    # Add all new tokens
    all_new_tokens = new_audio_tokens + new_unused_tokens
    print(f"\nAdding {len(all_new_tokens)} new tokens...")
    
    # For Gemma/SentencePiece, we need to add these as special tokens
    special_tokens_dict = {"additional_special_tokens": all_new_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    
    print(f"Added {num_added} new tokens")
    print(f"New vocabulary size: {len(tokenizer)}")
    
    # Step 5: Save the modified tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved modified tokenizer to {output_dir}")
    
    # Step 6: Save metadata
    metadata = {
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": len(tokenizer),
        "num_audio_tokens": 8192,
        "num_unused_tokens": 98,
        "repurposed_tokens": len(verified_unused),
        "added_tokens": num_added,
        "audio_token_ranges": [
            {"start": 0, "end": 6241, "type": "repurposed", "count": 6242},
            {"start": 6242, "end": 8191, "type": "new", "count": 1950}
        ],
        "unused_token_ids": list(UNUSED_TOKEN_IDS.values()),
        "first_new_token_id": original_vocab_size,
        "last_new_token_id": len(tokenizer) - 1
    }
    
    metadata_path = output_path / "audio_token_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Step 7: Verify the conversion
    print("\n" + "="*60)
    print("Verification:")
    print("="*60)
    
    # Check some audio tokens
    test_audio_tokens = ["<AUDIO_0>", "<AUDIO_100>", "<AUDIO_6241>", "<AUDIO_6242>", "<AUDIO_8191>"]
    for token in test_audio_tokens:
        if tokenizer.convert_tokens_to_ids([token])[0] != tokenizer.unk_token_id:
            token_id = tokenizer.convert_tokens_to_ids([token])[0]
            print(f"  ✓ {token:15} -> ID {token_id}")
        else:
            print(f"  ✗ {token:15} not found")
    
    # Check new unused tokens
    test_unused = ["<unused0>", "<unused50>", "<unused97>"]
    for token in test_unused:
        if tokenizer.convert_tokens_to_ids([token])[0] != tokenizer.unk_token_id:
            token_id = tokenizer.convert_tokens_to_ids([token])[0]
            print(f"  ✓ {token:15} -> ID {token_id}")
        else:
            print(f"  ✗ {token:15} not found")
    
    print("\nConversion complete!")
    return tokenizer, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Repurpose Gemma unused tokens for audio")
    parser.add_argument("--base-model", default="google/gemma-2-2b", help="Base model name")
    parser.add_argument("--output-dir", default="./gemma3-audio-tokenizer", help="Output directory")
    
    args = parser.parse_args()
    repurpose_tokenizer(args.base_model, args.output_dir)