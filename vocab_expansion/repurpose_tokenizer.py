from transformers import AutoTokenizer
import json
import os

def create_audio_token_mapping(name="google/gemma-3-4b-pt"):
    tokenizer = AutoTokenizer.from_pretrained(name)

    original_vocab_size = len(tokenizer)
    NUM_UNUSED_TOKENS = 6242  # <unused0> to <unused6241>
    NUM_ADDITIONAL_AUDIO_TOKENS = 1950
    NUM_PADDING_TOKENS = 98  # Reduce this by 5, later on modified as
    TOTAL_NEW_TOKENS = NUM_ADDITIONAL_AUDIO_TOKENS + NUM_PADDING_TOKENS  # 2048
    TOTAL_AUDIO_TOKENS = 8192

    assert NUM_UNUSED_TOKENS + NUM_ADDITIONAL_AUDIO_TOKENS == TOTAL_AUDIO_TOKENS, \
        f"Token count mismatch: {NUM_UNUSED_TOKENS} + {NUM_ADDITIONAL_AUDIO_TOKENS} != {TOTAL_AUDIO_TOKENS}"

    unused_to_audio_mapping = {}
    audio_token_list = []

    # Map existing unused tokens to audio tokens
    print(f"\nMapping {NUM_UNUSED_TOKENS} unused tokens to audio tokens...")
    for i in range(NUM_UNUSED_TOKENS):
        unused_token = f"<unused{i}>"
        audio_token = f"<audio_{i:04d}>"

        # Check if unused token exists in vocabulary
        if unused_token in tokenizer.get_vocab():
            token_id = tokenizer.get_vocab()[unused_token]
            unused_to_audio_mapping[unused_token] = {
                "audio_token": audio_token,
                "original_id": token_id,
                "audio_index": i
            }
            audio_token_list.append(audio_token)
        else:
            print(f"Warning: {unused_token} not found in vocabulary")

    # Create additional audio tokens
    additional_audio_tokens = []
    print(f"\nCreating {NUM_ADDITIONAL_AUDIO_TOKENS} additional audio tokens...")
    for i in range(NUM_ADDITIONAL_AUDIO_TOKENS):
        audio_index = NUM_UNUSED_TOKENS + i
        audio_token = f"<audio_{audio_index:04d}>"
        additional_audio_tokens.append(audio_token)
        audio_token_list.append(audio_token)

    # Create padding tokens (reserved for future use)
    padding_tokens = []
    print(f"\nCreating {NUM_PADDING_TOKENS} padding tokens for future use...")
    for i in range(NUM_PADDING_TOKENS):
        padding_token = f"<pad_reserved_{i:02d}>"
        padding_tokens.append(padding_token)

    # Prepare tokens to add to tokenizer
    all_new_tokens = additional_audio_tokens + padding_tokens

    # Add new tokens to tokenizer
    print(f"\nAdding {len(all_new_tokens)} new tokens to tokenizer...")
    num_added = tokenizer.add_tokens(all_new_tokens)
    print(f"Successfully added {num_added} tokens")

    # Get new vocabulary size
    new_vocab_size = len(tokenizer)
    print(f"New vocabulary size: {new_vocab_size}")
    print(f"Total vocabulary increase: {new_vocab_size - original_vocab_size}")

    # Create comprehensive mapping
    audio_token_mapping = {
        "config": {
            "num_unused_tokens": NUM_UNUSED_TOKENS,
            "num_additional_audio_tokens": NUM_ADDITIONAL_AUDIO_TOKENS,
            "num_padding_tokens": NUM_PADDING_TOKENS,
            "total_audio_tokens": TOTAL_AUDIO_TOKENS,
            "total_new_tokens_added": TOTAL_NEW_TOKENS,
            "original_vocab_size": original_vocab_size,
            "new_vocab_size": new_vocab_size
        },
        "unused_to_audio": unused_to_audio_mapping,
        "additional_audio_tokens": additional_audio_tokens,
        "padding_tokens": padding_tokens,
        "all_audio_tokens": audio_token_list
    }

    audio_to_id = {}
    id_to_audio = {}

    # Map all audio tokens to their IDs
    for audio_token in audio_token_list:
        if audio_token in tokenizer.get_vocab():
            token_id = tokenizer.get_vocab()[audio_token]
        else:
            # For new tokens, get their ID after adding
            token_id = tokenizer.convert_tokens_to_ids(audio_token)

        audio_to_id[audio_token] = token_id
        id_to_audio[token_id] = audio_token

    audio_token_mapping["audio_to_id"] = audio_to_id
    audio_token_mapping["id_to_audio"] = id_to_audio

    return tokenizer, audio_token_mapping


def save_mapping(mapping, output_dir="./audio_token_mapping"):
    """
    Save the mapping to JSON files for later use.

    Args:
        mapping: The audio token mapping dictionary
        output_dir: Directory to save the mapping files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full mapping
    with open(os.path.join(output_dir, "full_mapping.json"), "w") as f:
        # Convert integer keys to strings for JSON serialization
        json_safe_mapping = mapping.copy()
        if "id_to_audio" in json_safe_mapping:
            json_safe_mapping["id_to_audio"] = {
                str(k): v for k, v in json_safe_mapping["id_to_audio"].items()
            }
        json.dump(json_safe_mapping, f, indent=2)

    # Save just the audio token list for quick reference
    with open(os.path.join(output_dir, "audio_tokens.json"), "w") as f:
        json.dump(mapping["all_audio_tokens"], f, indent=2)

    # Save configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(mapping["config"], f, indent=2)

    print(f"\nMapping saved to {output_dir}/")


def verify_mapping(tokenizer, mapping):
    """
    Verify the mapping is correct and all tokens are accessible.

    Args:
        tokenizer: The modified tokenizer
        mapping: The audio token mapping
    """
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)

    # Verify unused token mapping
    print(f"\nVerifying unused token mappings...")
    num_verified = 0
    for unused_token, info in list(mapping["unused_to_audio"].items())[:5]:  # Check first 5
        if unused_token in tokenizer.get_vocab():
            num_verified += 1
            print(f"✓ {unused_token} -> {info['audio_token']}")

    # Verify new audio tokens
    print(f"\nVerifying new audio tokens...")
    for token in mapping["additional_audio_tokens"][:5]:  # Check first 5
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            print(f"✓ {token} -> ID: {token_id}")
        else:
            print(f"✗ {token} not found!")

    # Summary statistics
    print(f"\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total audio tokens available: {len(mapping['all_audio_tokens'])}")
    print(f"Reused unused tokens: {len(mapping['unused_to_audio'])}")
    print(f"New audio tokens added: {len(mapping['additional_audio_tokens'])}")
    print(f"Padding tokens reserved: {len(mapping['padding_tokens'])}")
    print(f"Total vocabulary size: {len(tokenizer)}")


def main():
    tokenizer_name = "google/gemma-3-4b-pt"

    try:
        tokenizer, mapping = create_audio_token_mapping(tokenizer_name)

        save_mapping(mapping)

        verify_mapping(tokenizer, mapping)

        # Example usage
        print(f"\n" + "=" * 50)
        print("EXAMPLE USAGE")
        print("=" * 50)

        # Encode some text with audio tokens
        sample_text = "Hello <audio_0000> world <audio_1000>"
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)

        print(f"Original text: {sample_text}")
        print(f"Encoded IDs: {encoded}")
        print(f"Decoded text: {decoded}")

        # Save the modified tokenizer
        output_tokenizer_dir = "./gemma_audio_tokenizer"
        tokenizer.save_pretrained(output_tokenizer_dir)
        print(f"\nModified tokenizer saved to {output_tokenizer_dir}/")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()