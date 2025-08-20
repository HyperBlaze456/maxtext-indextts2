#!/usr/bin/env python3
"""Debug script to check token counts"""

import json
from pathlib import Path
import sys

def check_tokenizer_metadata(tokenizer_dir):
    """Check the token count in the metadata"""
    tok_dir = Path(tokenizer_dir)
    
    # Check metadata
    meta_path = tok_dir / "audio_token_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print("=== audio_token_metadata.json ===")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        
        print(f"\nToken count calculation:")
        print(f"  Original vocab size: {meta['original_vocab_size']}")
        print(f"  New vocab size: {meta['new_vocab_size']}")
        print(f"  Difference: {meta['new_vocab_size'] - meta['original_vocab_size']}")
        print(f"  Appended audio count: {meta['appended_audio_count']}")
        
        if meta['new_vocab_size'] - meta['original_vocab_size'] != meta['appended_audio_count']:
            print(f"  WARNING: Mismatch! Expected diff = appended_audio_count")
    
    # Check alias map
    alias_path = tok_dir / "audio_alias_map.json"
    if alias_path.exists():
        alias = json.loads(alias_path.read_text())
        print("\n=== audio_alias_map.json ===")
        print(f"  alias_count: {alias.get('alias_count', 'N/A')}")
        print(f"  append_count_requested: {alias.get('append_count_requested', 'N/A')}")
        print(f"  append_count_added: {alias.get('append_count_added', 'N/A')}")
        
        # Check the actual new tokens
        new_tokens = alias.get('new_audio_token_ids', {})
        if new_tokens:
            print(f"  Number of new_audio_token_ids: {len(new_tokens)}")
            token_ids = sorted(new_tokens.values())
            print(f"  Token ID range: {min(token_ids)} to {max(token_ids)}")
            print(f"  Total span: {max(token_ids) - min(token_ids) + 1}")

    # Try to load actual tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        print(f"\n=== Actual tokenizer ===")
        print(f"  Vocab size: {len(tok)}")
        print(f"  Vocab size (from vocab): {len(tok.get_vocab())}")
        
        # Check if there's a mismatch
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if len(tok) != meta['new_vocab_size']:
                print(f"  WARNING: Tokenizer size {len(tok)} != metadata new_vocab_size {meta['new_vocab_size']}")
    except Exception as e:
        print(f"\n  Could not load tokenizer: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_token_count.py <tokenizer_dir>")
        sys.exit(1)
    
    check_tokenizer_metadata(sys.argv[1])