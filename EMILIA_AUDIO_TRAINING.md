Emilia Audio Training Quickstart (with absolute paths)

This memo shows the exact commands to prepare the tokenizer/mapping and run training with the Emilia dataset using the new `emilia_audio` input pipeline.

Assumptions
- Repo root: `D:/리서치/JAXLearn/maxtext-indextts2`
- You have Python and pip available in your environment.
- If the Emilia dataset on Hugging Face is private, you have a valid token in `$env:HF_TOKEN` (PowerShell) or you will pass it explicitly.

1) Install dependencies

PowerShell
```
cd D:/리서치/JAXLearn/maxtext-indextts2
pip install -r requirements.txt
pip install torch librosa soundfile
```

2) Extend tokenizer and create audio mapping

This adds 8,192 audio tokens plus the special separators `e_<BT>` and `e_<BA>`, and produces an embedding-to-audio mapping JSON.

PowerShell
```
$REPO = "D:/리서치/JAXLearn/maxtext-indextts2"
$TOK_OUT = "$REPO/assets/gemma3-audio-tokenizer"

python "$REPO/vocab_expansion/extend_tokenizer.py" `
  --tokenizer google/gemma-3-4b-pt `
  --save-tokenizer `
  --tokenizer-output "$TOK_OUT" `
  --mapping-output "$REPO/vocab_expansion/audio_token_mapping_complete.json"
```

Notes
- The training pipeline expects the mapping JSON to have a top-level key `embedding_to_audio`. The command above writes the correct format to `vocab_expansion/audio_token_mapping_complete.json`.

3) (Optional) Expand an existing checkpoint embedding table

If you are fine-tuning from a Gemma3 checkpoint, expand its token embedding table by +8192 rows to match the extended tokenizer.

PowerShell
```
$REPO = "D:/리서치/JAXLearn/maxtext-indextts2"
python "$REPO/vocab_expansion/expand_embedder.py" `
  --checkpoint-path "D:/checkpoints/gemma3_4b_converted" `
  --num-tokens 8192 `
  --out-step 0
```

4) Run training with the `emilia_audio` pipeline

Use the provided config `MaxText/configs/experimental/emilia_audio_gemma3_4b.yml` and override critical fields on the CLI (absolute paths, vocab size, dataset id, etc.). Passing `vocab_size` on the CLI ensures it is not overwritten by the model config.

PowerShell
```
$REPO = "D:/리서치/JAXLearn/maxtext-indextts2"
$CFG  = "$REPO/MaxText/configs/experimental/emilia_audio_gemma3_4b.yml"
$TOK  = "$REPO/assets/gemma3-audio-tokenizer"
$MAP  = "$REPO/vocab_expansion/audio_token_mapping_complete.json"

python -m MaxText.train $CFG `
  vocab_size=270336 `
  tokenizer_path="$TOK" `
  audio_token_mapping_path="$MAP" `
  dataset_type=emilia_audio `
  hf_path="YOUR_ORG/YOUR_EMILIA_DATASET" `
  train_split=train `
  hf_data_dir='' `
  hf_train_files='' `
  hf_access_token="$env:HF_TOKEN" `
  base_output_directory="gs://YOUR_BUCKET/maxtext" `
  run_name="emilia_audio_run" `
  eval_interval=0
```

Replace
- `YOUR_ORG/YOUR_EMILIA_DATASET` with your actual HF dataset id.
- `gs://YOUR_BUCKET/maxtext` with your GCS output path (or a local path if desired).
- If public, you can drop `hf_access_token=...`.

5) Using precomputed semantic tokens (optional)

If your Emilia dataset already contains semantic token lists (arrays of ints 0..8191), skip on-the-fly audio tokenization by adding:

```
use_precomputed_semantic=true precomputed_semantic_path=json.semantic_tokens
```

Example
```
python -m MaxText.train $CFG `
  vocab_size=270336 `
  tokenizer_path="$TOK" `
  audio_token_mapping_path="$MAP" `
  dataset_type=emilia_audio `
  hf_path="YOUR_ORG/YOUR_EMILIA_DATASET" `
  use_precomputed_semantic=true `
  precomputed_semantic_path="json.semantic_tokens" `
  base_output_directory="gs://YOUR_BUCKET/maxtext" `
  run_name="emilia_audio_run_precomp" `
  eval_interval=0
```

Reference details
- Input pipeline: `MaxText/input_pipeline/_emilia_audio_processing.py`
  - Builds text via HF tokenizer; audio via Wav2Vec2-BERT + RepCodec or reads precomputed tokens.
  - Sequence format: `[e_<BT>] + text_ids + [e_<BA>] + mapped_audio_ids`.
  - Masks all target positions before `e_<BA>`, so loss applies only to audio tokens.
- Special tokens: added by the extend script (stored in the tokenizer dir) and must exist for training to succeed.
- `vocab_size` must match the extended tokenizer and any checkpoint expansion.

