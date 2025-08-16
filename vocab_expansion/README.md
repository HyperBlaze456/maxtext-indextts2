# Vocabulary Expansion Tools for MaxText

This directory contains tools for expanding the vocabulary of pre-trained models to support additional tokens (e.g., audio tokens for multimodal training).

## Overview

When extending a language model to handle new modalities like audio, you need to:
1. Extend the tokenizer to recognize new special tokens
2. Expand the model's embedding matrix to accommodate these tokens
3. Initialize the new embeddings appropriately

## Scripts

### 1. `extend_tokenizer.py`
Extends a HuggingFace tokenizer with additional special tokens.

**Basic usage:**
```bash
# Extend Gemma3 tokenizer with 8192 audio tokens
python extend_tokenizer.py \
    --base-model "google/gemma-2-2b" \
    --output-dir "./gemma3-audio-tokenizer" \
    --num-audio-tokens 8192

# Verify the extended tokenizer
python extend_tokenizer.py \
    --output-dir "./gemma3-audio-tokenizer" \
    --verify-only
```

### 2. `expand_checkpoint.py`
Expands the embedding matrix in a MaxText checkpoint to match the new vocabulary size.

**Basic usage:**
```bash
# Expand checkpoint vocabulary from 262,144 to 270,336 tokens
python expand_checkpoint.py \
    --checkpoint-path /path/to/original/checkpoint \
    --output-path /path/to/expanded/checkpoint \
    --old-vocab-size 262144 \
    --new-vocab-size 270336 \
    --init-method average

# Verify the expanded checkpoint
python expand_checkpoint.py \
    --output-path /path/to/expanded/checkpoint \
    --verify-only
```

## Initialization Methods

The `expand_checkpoint.py` script supports three initialization methods for new embeddings:

1. **`average`** (Recommended for audio tokens)
   - Initializes new tokens as the average of all existing embeddings
   - Adds small random noise to break symmetry
   - Places new tokens in a reasonable location in embedding space

2. **`special_tokens`**
   - Uses existing special tokens (like `<pad>`, `<eos>`) as initialization basis
   - Good when new tokens are functionally similar to special tokens

3. **`random`**
   - Random initialization matching the distribution of existing embeddings
   - May require more training to converge

## Complete Workflow

### Step 1: Extend the Tokenizer
```bash
python extend_tokenizer.py \
    --base-model "google/gemma-2-2b" \
    --output-dir "./gemma3-audio-tokenizer" \
    --num-audio-tokens 8192
```

### Step 2: Expand the Checkpoint
```bash
python expand_checkpoint.py \
    --checkpoint-path /path/to/gemma3/checkpoint \
    --output-path /path/to/gemma3-audio/checkpoint \
    --old-vocab-size 262144 \
    --new-vocab-size 270336 \
    --init-method average
```

### Step 3: Update MaxText Config
Create or modify your config file (e.g., `MaxText/configs/models/gemma3-4b-audio.yml`):
```yaml
# Updated vocabulary size
vocab_size: 270_336  # 262,144 + 8,192

# Optional: Training strategies for new tokens
freeze_existing_embeddings_steps: 1000  # Freeze old embeddings initially
new_token_lr_multiplier: 10.0          # Higher learning rate for new tokens
```

### Step 4: Use in Training
```python
# Load the extended tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./gemma3-audio-tokenizer")

# Use audio tokens in your text
text = "The sound <AUDIO_0> is followed by <AUDIO_1>"
tokens = tokenizer.encode(text)

# Train with MaxText using the expanded checkpoint
# python -m MaxText.train MaxText/configs/models/gemma3-4b-audio.yml \
#     load_parameters_path=/path/to/gemma3-audio/checkpoint
```

## Best Practices

1. **Initialize thoughtfully**: The `average` method works well for most cases, placing new tokens in a reasonable embedding space location.

2. **Train gradually**: Consider freezing existing embeddings initially while new tokens adapt.

3. **Monitor carefully**: Track both original token performance (should not degrade) and new token usage.

4. **Adjust learning rates**: New tokens may benefit from higher initial learning rates.

5. **Validate alignment**: Always verify that tokenizer IDs match the model's expectations.

## Community Practices for Vocabulary Extension

Based on successful vocabulary extensions in models like:
- **mT5**: Extended for multilingual support
- **BLOOM**: Added tokens for new languages
- **LLaMA**: Community extensions for specialized domains

Key insights:
- Average initialization + small noise is most robust
- Freezing existing embeddings for 500-1000 steps helps stability
- New tokens typically need 5-10x more gradient updates initially
- Weight tying (shared embeddings/LM head) naturally helps new tokens learn faster

## Troubleshooting

**Issue: Tokenizer and model vocab size mismatch**
- Ensure the config file `vocab_size` matches the extended tokenizer
- Verify checkpoint was properly expanded

**Issue: New tokens not learning**
- Check initialization isn't all zeros
- Increase learning rate multiplier for new tokens
- Ensure tokens are actually being used in training data

**Issue: Original model performance degrades**
- Use embedding freezing for initial training
- Reduce learning rate
- Check for gradient explosion in new embeddings

## References

- [How to add tokens to a tokenizer](https://huggingface.co/docs/transformers/main/en/tasks/token_classification#add-tokens-to-the-tokenizer)
- [Extending T5 for new languages](https://arxiv.org/abs/2010.11934)
- [Vocabulary adaptation techniques](https://arxiv.org/abs/2007.15779)