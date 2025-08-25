import re
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer
import json


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


def map_audio_tokens(tokenizer_name="google/gemma-3-4b-pt", num_workers=8):
    """
    Map unused tokens to audio tokens and add new tokens.
    Ignores the soft token at index 262144.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Original vocabulary size: {len(tokenizer.vocab)}")

    # Check if soft token exists at 262144
    if 262144 in tokenizer.get_vocab().values():
        print(f"Found soft token at index 262144, will be ignored in mapping")

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

    # Create audio token mapping (ignoring soft token at 262144)
    audio_token_mapping = {}

    # Map existing unused tokens to audio tokens
    for token_idx, unused_num in unused_token_map.items():
        if token_idx != 262144:  # Skip soft token
            audio_token_mapping[token_idx] = unused_num

    # Add new tokens for remaining audio tokens
    new_tokens = []

    # Add 1950 audio tokens
    for i in range(1950):
        new_tokens.append(f"<unused{6242 + i}>")

    # Add 98 padding tokens to reach 2048 (multiple of 256)
    for i in range(98):
        new_tokens.append(f"<pad_audio_{i}>")

    # Add new tokens to tokenizer
    print(f"Adding {len(new_tokens)} new tokens...")
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Successfully added {num_added} tokens")

    # Map new token indices to audio IDs
    for i in range(1950):
        token_id = tokenizer.convert_tokens_to_ids(f"<unused{6242 + i}>")
        audio_token_mapping[token_id] = 6242 + i

    # Padding tokens get -1 as marker
    for i in range(98):
        token_id = tokenizer.convert_tokens_to_ids(f"<pad_audio_{i}>")
        audio_token_mapping[token_id] = -1

    print("\nSummary:")
    print(f"- Mapped {sum(1 for v in audio_token_mapping.values() if 0 <= v < 6242)} existing unused tokens")
    print(f"- Added {1950} new audio tokens")
    print(f"- Added {98} padding tokens")
    print(f"- Final vocabulary size: {len(tokenizer.vocab)}")
    print(f"- Total audio tokens mapped: {sum(1 for v in audio_token_mapping.values() if v >= 0)}")

    return {
        'tokenizer': tokenizer,
        'audio_token_mapping': audio_token_mapping,
        'vocab_size': len(tokenizer.vocab)
    }


def verify_mapping(result):
    """Verify the audio token mapping is correct."""
    mapping = result['audio_token_mapping']

    # Count tokens by type
    existing = sum(1 for v in mapping.values() if 0 <= v < 6242)
    new_audio = sum(1 for v in mapping.values() if 6242 <= v < 8192)
    padding = sum(1 for v in mapping.values() if v == -1)

    print("\nVerification:")
    print(f"- Existing unused tokens: {existing}")
    print(f"- New audio tokens: {new_audio}")
    print(f"- Padding tokens: {padding}")
    print(f"- Total audio coverage: {existing + new_audio}/8192")

    # Check if soft token is excluded
    if 262144 not in mapping:
        print("- Soft token (262144) correctly excluded from mapping")

    return existing + new_audio == 8192


if __name__ == "__main__":
    # Map tokens
    result = map_audio_tokens("google/gemma-3-4b-pt", num_workers=8)

    # Verify
    is_valid = verify_mapping(result)

    if is_valid:
        print("\n✓ Mapping completed successfully!")

        # Save mapping
        with open('audio_token_mapping.json', 'w') as f:
            mapping_data = {
                'audio_mappings': {str(k): int(v) for k, v in result['audio_token_mapping'].items()},
                'vocab_size': result['vocab_size'],
                'note': 'Soft token at index 262144 is excluded from audio mappings'
            }
            json.dump(mapping_data, f, indent=2)
        print("Saved to 'audio_token_mapping.json'")
    else:
        print("\n✗ Verification failed!")