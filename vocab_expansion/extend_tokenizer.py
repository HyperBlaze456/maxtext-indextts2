#!/usr/bin/env python3
"""
Extend tokenizer vocabulary for audio tokens and create adjusted embedding mappings.
This combines tokenizer extension and index adjustment to handle the soft token at 262144.
"""

import re
import json
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer
from pathlib import Path


def find_unused_tokens_parallel(vocab_batch):
    """Find unused tokens in a batch of vocabulary items."""
    unused_pattern = re.compile(r'^<unused(\d+)>$')
    unused_tokens = {}

    for token, idx in vocab_batch:
        match = unused_pattern.match(token)
        if match:
            unused_num = int(match.group(1))
            if unused_num < 6242:  # Only existing unused tokens
                unused_tokens[idx] = unused_num

    return unused_tokens


def create_adjusted_embedding_index(token_idx, soft_token_idx=262144):
    """
    Calculate the embedding index for a token, accounting for the skipped soft token.
    
    Args:
        token_idx: The tokenizer vocabulary index
        soft_token_idx: The index of the soft token to skip (default: 262144)
    
    Returns:
        The adjusted embedding index
    """
    if token_idx == soft_token_idx:
        return None  # Soft token should not have an embedding
    elif token_idx < soft_token_idx:
        return token_idx
    else:
        # Shift down by 1 to account for skipped soft token
        return token_idx - 1


def extend_tokenizer_with_audio_tokens(
    tokenizer_name="google/gemma-3-4b-pt",
    num_workers=8,
    save_tokenizer=False,
    tokenizer_output_path="extended_tokenizer",
    mapping_output_path="audio_token_mapping_complete.json"
):
    """
    Extend tokenizer with audio tokens and create adjusted embedding mappings.
    
    This function:
    1. Finds existing unused tokens in the vocabulary
    2. Adds new audio and padding tokens to reach 8192 audio tokens
    3. Creates mappings between tokenizer indices and audio IDs
    4. Adjusts indices to account for the soft token at 262144
    5. Returns both raw and adjusted mappings for embeddings
    """
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Original vocabulary size: {len(tokenizer.vocab)}")

    # Check if soft token exists at 262144
    soft_token_idx = 262144
    if soft_token_idx in tokenizer.get_vocab().values():
        print(f"Found soft token at index {soft_token_idx}, will be excluded from embeddings")

    # Convert vocab to list for parallel processing
    vocab_items = list(tokenizer.vocab.items())

    # Split vocab for parallel processing
    batch_size = len(vocab_items) // num_workers
    batches = []
    for i in range(num_workers):
        start = i * batch_size
        end = start + batch_size if i < num_workers - 1 else len(vocab_items)
        batches.append(vocab_items[start:end])

    # Find existing unused tokens using parallel processing
    print("Finding existing unused tokens...")
    unused_token_map = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(find_unused_tokens_parallel, batches)
        for result in results:
            unused_token_map.update(result)

    print(f"Found {len(unused_token_map)} existing unused tokens")

    # Sort unused tokens by their unused number for clean ordering
    sorted_unused = sorted(unused_token_map.items(), key=lambda x: x[1])

    # Create mappings
    tokenizer_to_audio = {}  # token_idx -> audio_id
    embedding_to_audio = {}  # embedding_idx -> audio_id
    audio_to_embedding = {}  # audio_id -> embedding_idx
    detailed_mappings = {}   # embedding_idx -> {audio_id, original_token_idx}

    # Map existing unused tokens to audio tokens
    for token_idx, unused_num in sorted_unused:
        if token_idx != soft_token_idx:  # Skip soft token
            audio_id = unused_num
            tokenizer_to_audio[token_idx] = audio_id
            
            # Calculate adjusted embedding index
            embedding_idx = create_adjusted_embedding_index(token_idx, soft_token_idx)
            if embedding_idx is not None:
                embedding_to_audio[embedding_idx] = audio_id
                audio_to_embedding[audio_id] = embedding_idx
                detailed_mappings[embedding_idx] = {
                    'audio_id': audio_id,
                    'original_token_idx': token_idx,
                    'token': tokenizer.convert_ids_to_tokens(token_idx)
                }

    # Add new tokens for remaining audio tokens
    new_tokens = []
    
    # Add 1950 audio tokens
    for i in range(1950):
        new_tokens.append(f"<unused{6242 + i}>")
    
    # Add 2 special tokens for LM-TTS
    new_tokens.append("e_<BT>")  # Begin Text marker
    new_tokens.append("e_<BA>")  # Begin Audio marker
    
    # Add 96 padding tokens to reach 2048 (multiple of 256)
    for i in range(96):
        new_tokens.append(f"<pad_audio_{i}>")

    # Add new tokens to tokenizer
    print(f"Adding {len(new_tokens)} new tokens...")
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Successfully added {num_added} tokens")

    # Map new audio token indices
    for i in range(1950):
        token = f"<unused{6242 + i}>"
        token_idx = tokenizer.convert_tokens_to_ids(token)
        audio_id = 6242 + i
        tokenizer_to_audio[token_idx] = audio_id
        
        # Calculate adjusted embedding index
        embedding_idx = create_adjusted_embedding_index(token_idx, soft_token_idx)
        if embedding_idx is not None:
            embedding_to_audio[embedding_idx] = audio_id
            audio_to_embedding[audio_id] = embedding_idx
            detailed_mappings[embedding_idx] = {
                'audio_id': audio_id,
                'original_token_idx': token_idx,
                'token': token
            }

    # Map special LM-TTS tokens - these get their own unique audio IDs
    # Using audio IDs 8192 and 8193 for special tokens
    special_tokens = [
        ("e_<BT>", 8192),  # Begin Text marker
        ("e_<BA>", 8193)   # Begin Audio marker
    ]
    
    for token, audio_id in special_tokens:
        token_idx = tokenizer.convert_tokens_to_ids(token)
        tokenizer_to_audio[token_idx] = audio_id
        
        # Calculate adjusted embedding index
        embedding_idx = create_adjusted_embedding_index(token_idx, soft_token_idx)
        if embedding_idx is not None:
            embedding_to_audio[embedding_idx] = audio_id
            audio_to_embedding[audio_id] = embedding_idx
            detailed_mappings[embedding_idx] = {
                'audio_id': audio_id,
                'original_token_idx': token_idx,
                'token': token,
                'special_token': True
            }

    # Map padding tokens
    for i in range(96):
        token = f"<pad_audio_{i}>"
        token_idx = tokenizer.convert_tokens_to_ids(token)
        tokenizer_to_audio[token_idx] = -1  # Padding marker
        
        # Calculate adjusted embedding index
        embedding_idx = create_adjusted_embedding_index(token_idx, soft_token_idx)
        if embedding_idx is not None:
            embedding_to_audio[embedding_idx] = -1
            detailed_mappings[embedding_idx] = {
                'audio_id': -1,
                'original_token_idx': token_idx,
                'token': token
            }

    # Calculate statistics
    total_audio_tokens = sum(1 for v in embedding_to_audio.values() if v >= 0)
    padding_tokens = sum(1 for v in embedding_to_audio.values() if v == -1)
    max_embedding_idx = max(embedding_to_audio.keys())
    max_token_idx = max(tokenizer_to_audio.keys())
    
    # Create comprehensive output data
    output_data = {
        # Primary mappings (using embedding indices)
        'embedding_to_audio': {str(k): v for k, v in embedding_to_audio.items()},
        'audio_to_embedding': {str(k): v for k, v in audio_to_embedding.items()},
        
        # Raw tokenizer mappings (for reference)
        'tokenizer_to_audio': {str(k): v for k, v in tokenizer_to_audio.items()},
        
        # Detailed mappings with all information
        'detailed_mappings': {str(k): v for k, v in detailed_mappings.items()},
        
        # Statistics and metadata
        'stats': {
            'original_vocab_size': len(tokenizer.vocab) - num_added,
            'extended_vocab_size': len(tokenizer.vocab),
            'adjusted_embedding_size': len(tokenizer.vocab) - 1,  # -1 for soft token
            'total_audio_tokens': total_audio_tokens,
            'padding_tokens': padding_tokens,
            'total_mappings': len(embedding_to_audio),
            'max_token_index': max_token_idx,
            'max_embedding_index': max_embedding_idx,
            'soft_token_index': soft_token_idx,
            'existing_unused_mapped': sum(1 for v in tokenizer_to_audio.values() if 0 <= v < 6242),
            'new_audio_added': 1950,
            'special_tokens_added': 2,
            'padding_added': 96
        },
        
        'notes': {
            'soft_token': f'Soft token at index {soft_token_idx} is excluded from embeddings',
            'index_adjustment': f'Token indices > {soft_token_idx} are shifted down by 1 for embedding indices',
            'audio_range': 'Audio IDs range from 0 to 8191',
            'padding_marker': 'Padding tokens are marked with audio_id = -1'
        }
    }
    
    # Save mapping
    with open(mapping_output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nMapping saved to '{mapping_output_path}'")
    
    # Optionally save the extended tokenizer
    if save_tokenizer:
        tokenizer.save_pretrained(tokenizer_output_path)
        print(f"Extended tokenizer saved to '{tokenizer_output_path}'")
    
    return {
        'tokenizer': tokenizer,
        'mapping_data': output_data
    }


def verify_extended_tokenizer(result):
    """Comprehensive verification of the extended tokenizer and mappings."""
    
    tokenizer = result['tokenizer']
    mapping_data = result['mapping_data']
    stats = mapping_data['stats']
    
    print("\n" + "="*60)
    print("VERIFICATION REPORT")
    print("="*60)
    
    # 1. Verify tokenizer extension
    print("\n1. Tokenizer Extension:")
    print(f"   - Original vocab size: {stats['original_vocab_size']}")
    print(f"   - Extended vocab size: {stats['extended_vocab_size']}")
    print(f"   - Tokens added: {stats['extended_vocab_size'] - stats['original_vocab_size']}")
    
    # 2. Verify audio token coverage
    print("\n2. Audio Token Coverage:")
    audio_to_emb = mapping_data['audio_to_embedding']
    audio_ids = sorted([int(k) for k in audio_to_emb.keys()])
    
    expected_audio = set(range(8192))
    mapped_audio = set(audio_ids)
    missing_audio = expected_audio - mapped_audio
    
    if missing_audio:
        print(f"   ⚠ Missing {len(missing_audio)} audio IDs")
        print(f"     First 10 missing: {sorted(missing_audio)[:10]}")
    else:
        print(f"   ✓ All 8192 audio tokens are mapped")
    
    # 3. Verify embedding index adjustment
    print("\n3. Embedding Index Adjustment:")
    max_emb_idx = stats['max_embedding_index']
    adjusted_size = stats['adjusted_embedding_size']
    
    if max_emb_idx < adjusted_size:
        print(f"   ✓ Max embedding index ({max_emb_idx}) < adjusted size ({adjusted_size})")
    else:
        print(f"   ⚠ Max embedding index ({max_emb_idx}) >= adjusted size ({adjusted_size})")
    
    # 4. Verify no duplicate embeddings
    print("\n4. Embedding Uniqueness:")
    emb_indices = list(mapping_data['embedding_to_audio'].keys())
    if len(emb_indices) == len(set(emb_indices)):
        print(f"   ✓ No duplicate embedding indices")
    else:
        print(f"   ⚠ Duplicate embedding indices found!")
    
    # 5. Summary statistics
    print("\n5. Summary Statistics:")
    print(f"   - Existing unused tokens mapped: {stats['existing_unused_mapped']}")
    print(f"   - New audio tokens added: {stats['new_audio_added']}")
    print(f"   - Padding tokens added: {stats['padding_added']}")
    print(f"   - Total audio tokens: {stats['total_audio_tokens']}")
    print(f"   - Total padding tokens: {stats['padding_tokens']}")
    
    # 6. Check specific important tokens
    print("\n6. Important Token Checks:")
    
    # Check soft token is excluded
    soft_idx = str(stats['soft_token_index'])
    if soft_idx not in mapping_data['tokenizer_to_audio']:
        print(f"   ✓ Soft token ({soft_idx}) correctly excluded from tokenizer mapping")
    
    # Adjusted embedding for soft token should not exist
    adjusted_soft = create_adjusted_embedding_index(int(soft_idx))
    if adjusted_soft is None or str(adjusted_soft) not in mapping_data['embedding_to_audio']:
        print(f"   ✓ Soft token has no embedding mapping")
    
    # Check a sample of token -> embedding conversions
    print("\n7. Sample Token Conversions:")
    sample_tokens = [
        ('First audio token', 0),
        ('Last existing unused', 6241),
        ('First new audio', 6242),
        ('Last audio token', 8191),
    ]
    
    for desc, audio_id in sample_tokens:
        if str(audio_id) in audio_to_emb:
            emb_idx = audio_to_emb[str(audio_id)]
            detail = mapping_data['detailed_mappings'][str(emb_idx)]
            print(f"   - {desc} (audio {audio_id}):")
            print(f"     Token idx: {detail['original_token_idx']}, Embedding idx: {emb_idx}")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    
    # Return overall success
    return len(missing_audio) == 0 and max_emb_idx < adjusted_size


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extend tokenizer with audio tokens and create adjusted mappings"
    )
    parser.add_argument(
        "--tokenizer", 
        default="google/gemma-3-4b-pt",
        help="Base tokenizer to extend"
    )
    parser.add_argument(
        "--save-tokenizer",
        action="store_true",
        help="Save the extended tokenizer"
    )
    parser.add_argument(
        "--tokenizer-output",
        default="extended_tokenizer",
        help="Path to save extended tokenizer"
    )
    parser.add_argument(
        "--mapping-output",
        default="audio_token_mapping_complete.json",
        help="Path to save mapping file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers"
    )
    
    args = parser.parse_args()
    
    # Extend tokenizer and create mappings
    result = extend_tokenizer_with_audio_tokens(
        tokenizer_name=args.tokenizer,
        num_workers=args.workers,
        save_tokenizer=args.save_tokenizer,
        tokenizer_output_path=args.tokenizer_output,
        mapping_output_path=args.mapping_output
    )
    
    # Verify everything
    success = verify_extended_tokenizer(result)
    
    if success:
        print("\n✅ Successfully extended tokenizer and created adjusted mappings!")
    else:
        print("\n⚠️ Extension completed with warnings - please review the verification report")