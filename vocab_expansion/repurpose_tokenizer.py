#!/usr/bin/env python3
"""
Repurpose Gemma's <unused*> tokens by aliasing them to <AUDIO_i> (no SPM edits),
and append additional <AUDIO_*> tokens at the end of the vocab.

- We DO NOT rename existing tokens inside the SentencePiece model.
- We DO create `audio_alias_map.json` that maps <AUDIO_i> <-> <unusedi>.
- We DO append `--append-count` new <AUDIO_*> tokens using add_special_tokens,
  starting from the first index AFTER the aliased range.
- We ensure no name collisions with existing tokens.

Usage:
  python repurpose_audio_tokens.py \
      --base-model google/gemma-3-4b \
      --output-dir ./gemma3-audio-tokenizer \
      --append-count 2048
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


UNUSED_RE = re.compile(r"^<unused(\d+)>$")


def find_contiguous_unused(tokenizer) -> Tuple[List[str], Dict[str, int]]:
    """
    Return the contiguous sequence <unused0>, <unused1>, ... as they exist in the vocab.
    Stops at the first missing index. Also returns their ids.
    """
    vocab = tokenizer.get_vocab()
    names, ids = [], {}
    i = 0
    while True:
        name = f"<unused{i}>"
        if name in vocab:
            names.append(name)
            ids[name] = vocab[name]
            i += 1
        else:
            break
    return names, ids


def collect_all_unused(tokenizer) -> List[Tuple[int, str]]:
    """
    Collect all <unusedN> present (not necessarily contiguous) as (N, token_str), sorted by N.
    Useful for diagnostics; not used for aliasing.
    """
    out = []
    for tok in tokenizer.get_vocab().keys():
        m = UNUSED_RE.match(tok)
        if m:
            out.append((int(m.group(1)), tok))
    return sorted(out, key=lambda x: x[0])


def build_alias_map(unused_names: List[str]) -> Dict[str, str]:
    """
    Build <AUDIO_i> -> <unusedi> alias for the contiguous block.
    """
    return {f"<AUDIO_{i}>": f"<unused{i}>" for i in range(len(unused_names))}


def append_new_audio_tokens(tokenizer, start_index: int, count: int) -> List[str]:
    """
    Append <AUDIO_start_index> ... <AUDIO_{start_index+count-1}> as additional special tokens,
    skipping any that already exist in the vocab to avoid collisions.
    """
    candidates = [f"<AUDIO_{i}>" for i in range(start_index, start_index + count)]
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in candidates if t not in existing]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return new_tokens


def repurpose_tokenizer(
    base_model: str,
    output_dir: str,
    append_count: int = 2048,
):
    print(f"Loading tokenizer: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model)
    original_vocab_size = len(tok)
    print(f"Original vocab size: {original_vocab_size}")

    # 1) Discover contiguous <unused0..K-1>
    contiguous_unused, contiguous_ids = find_contiguous_unused(tok)
    if not contiguous_unused:
        raise RuntimeError("No <unused*> tokens found; cannot build alias map.")

    # (Diagnostics) how many unused in total (any indices)
    all_unused = collect_all_unused(tok)
    print(f"Contiguous <unused0..{len(contiguous_unused)-1}> count: {len(contiguous_unused)}")
    if len(all_unused) != len(contiguous_unused):
        print(f"Note: Found {len(all_unused)} total <unusedN> entries (non-contiguous as well).")

    # 2) Build alias map <AUDIO_i> -> <unusedi> for contiguous block only
    alias_map = build_alias_map(contiguous_unused)  # e.g., {"<AUDIO_0>":"<unused0>", ...}

    # 3) Append new <AUDIO_*> tokens AFTER the aliased range
    start_new = len(contiguous_unused)
    print(f"Appending {append_count} new audio tokens starting from index {start_new}...")
    newly_added = append_new_audio_tokens(tok, start_new, append_count)
    print(f"Actually added {len(newly_added)} new tokens (skipped any that already existed).")
    new_vocab_size = len(tok)
    print(f"New vocab size: {new_vocab_size}")

    # 4) Save tokenizer
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(outdir)
    print(f"Saved tokenizer to: {outdir}")

    # 5) Save alias map JSON (easy to load elsewhere)
    # Also include reverse map and useful ID lookups.
    reverse_alias = {v: k for k, v in alias_map.items()}  # <unusedi> -> <AUDIO_i>

    # Map known <unusedi> piece IDs for convenience
    audio_to_existing_id = {
        audio: contiguous_ids[unused] for audio, unused in alias_map.items()
    }

    # IDs for newly added <AUDIO_*> tokens
    new_audio_token_ids = {
        t: tok.convert_tokens_to_ids(t) for t in newly_added
    }

    alias_payload = {
        "base_model": base_model,
        "alias_count": len(alias_map),
        "append_count_requested": append_count,
        "append_count_added": len(newly_added),
        "audio_to_unused": alias_map,
        "unused_to_audio": reverse_alias,
        "audio_to_existing_id": audio_to_existing_id,
        "new_audio_token_ids": new_audio_token_ids,
    }
    (outdir / "audio_alias_map.json").write_text(json.dumps(alias_payload, indent=2), encoding="utf-8")
    print(f"Wrote alias map: {outdir / 'audio_alias_map.json'}")

    # 6) Save compact metadata
    metadata = {
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "repurposed_audio_count": len(alias_map),
        "appended_audio_count": len(newly_added),
        "first_appended_audio": newly_added[0] if newly_added else None,
        "last_appended_audio": newly_added[-1] if newly_added else None,
    }
    (outdir / "audio_token_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {outdir / 'audio_token_metadata.json'}")

    # 7) Quick verification prints
    probe = [
        "<AUDIO_0>",
        f"<AUDIO_{min(100, len(alias_map)-1)}>",
        f"<AUDIO_{len(alias_map)-1}>",     # last aliased
        f"<AUDIO_{len(alias_map)}>",       # first appended
        f"<AUDIO_{len(alias_map)+append_count-1}>",  # last intended appended
    ]
    print("\nVerification:")
    for t in probe:
        tid = tok.convert_tokens_to_ids(t)
        if tid == tok.unk_token_id:
            print(f"  ✗ {t:>15}: not in tokenizer (expected for aliased entries; use alias map).")
        else:
            print(f"  ✓ {t:>15}: id={tid}")

    print("\nDone.")
    return tok, alias_payload, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/gemma-3-4b")
    parser.add_argument("--output-dir", default="./gemma3-audio-tokenizer")
    parser.add_argument("--append-count", type=int, default=2048,
                        help="How many new <AUDIO_*> tokens to append after the aliased range.")
    args = parser.parse_args()
    repurpose_tokenizer(args.base_model, args.output_dir, append_count=args.append_count)
