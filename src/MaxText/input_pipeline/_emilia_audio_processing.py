# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

import numpy as np
import torch

import datasets
import grain.python as grain
import jax
import transformers

from MaxText.input_pipeline import _input_pipeline_utils
from MaxText import multihost_dataloading

from .maskgct.semantic_utils import (
    build_semantic_model,
    build_semantic_codec,
    SemanticTokenizer,
)
from .maskgct.config import SemanticCodecConfig

from MaxText.input_pipeline import _hf_data_processing


class _AddSegPos(grain.MapTransform):
    """Add segmentation and position tensors for non-packed sequences.

    - inputs_segmentation/targets_segmentation: 1 where token != pad_id, else 0
    - inputs_position/targets_position: 0..L-1 masked by segmentation
    """

    def __init__(self, pad_id: int):
        self.pad_id = int(pad_id)

    def map(self, element):
        def _one(arr: np.ndarray):
            seg = (arr != self.pad_id).astype(np.int32)
            pos = (np.arange(arr.shape[-1], dtype=np.int32)[None, :] if arr.ndim == 2 else np.arange(arr.shape[-1], dtype=np.int32))
            return seg, pos * (seg != 0)

        inputs = element["inputs"]
        targets = element["targets"]
        if inputs.ndim == 1:
            in_seg, in_pos = _one(inputs)
            tg_seg, tg_pos = _one(targets)
        else:
            in_segs, in_poss, tg_segs, tg_poss = [], [], [], []
            for i in range(inputs.shape[0]):
                iseg, ipos = _one(inputs[i])
                tseg, tpos = _one(targets[i])
                in_segs.append(iseg)
                in_poss.append(ipos)
                tg_segs.append(tseg)
                tg_poss.append(tpos)
            in_seg = np.stack(in_segs, axis=0)
            in_pos = np.stack(in_poss, axis=0)
            tg_seg = np.stack(tg_segs, axis=0)
            tg_pos = np.stack(tg_poss, axis=0)

        element["inputs_segmentation"] = in_seg
        element["targets_segmentation"] = tg_seg
        element["inputs_position"] = in_pos
        element["targets_position"] = tg_pos
        return element


def get_semantic_tokenizer(device: str | None = None, cfg: SemanticCodecConfig | None = None) -> SemanticTokenizer:
    """
    Stay aware of inefficiency on TPU device as it is forced to run on CPU.
    Which device would do the heavy lifting? That's the question.
    """

    if device is None:
        if torch.cuda.is_available():
            device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda"
        else:
            device = "cpu"
    if cfg is None:
        cfg = SemanticCodecConfig()

    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg, device)

    tokenizer = SemanticTokenizer(
        semantic_model=semantic_model,
        semantic_codec=semantic_codec,
        semantic_mean=semantic_mean,
        semantic_std=semantic_std,
        device=device,
    )
    return tokenizer

def _audio_to_embedding_map_from_embedding_to_audio(embedding_to_audio: dict[str, int]) -> dict[int, int]:
    """Invert mapping: embedding_idx -> audio_id  to  audio_id -> embedding_idx"""
    a2e_map: dict[int, int] = {}
    for emb_idx_str, audio_id in embedding_to_audio.items():
        emb_idx = int(emb_idx_str)
        a2e_map[int(audio_id)] = emb_idx
    return a2e_map

def _load_audio_mapping(path: str) -> dict[int, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "embedding_to_audio" in data:
        return _audio_to_embedding_map_from_embedding_to_audio(data["embedding_to_audio"])
    raise ValueError(
        "Unsupported audio mapping format. Expect a dict with key 'embedding_to_audio'."
    )

def _map_semantic_tokens_to_model_ids(semantic_tokens: np.ndarray, audio_to_embedding: dict[int, int]) -> np.ndarray:
    # Build a vectorized LUT for speed
    max_audio_id = max(audio_to_embedding.keys()) if audio_to_embedding else -1
    lut_size = max(8192, max_audio_id + 1)
    lut = np.full((lut_size,), -1, dtype=np.int32)
    for a, e in audio_to_embedding.items():
        if 0 <= a < lut_size:
            lut[a] = int(e)
    # Clip tokens into LUT range then map
    clipped = np.clip(semantic_tokens, 0, lut_size - 1)
    return lut[clipped]

def process_batch(batch: dict[str, any],
                  hf_tokenizer: transformers.PreTrainedTokenizer,
                  semantic_tokenizer: SemanticTokenizer,
                  audio_to_embedding: dict[int, int],
                  sep_cond_text_id: int,
                  sep_text_audio_id: int,
                  mask_id: int,
                  sample_rate: int = 16000,
                  precomputed_semantic_path: str | None = None,
                  ):
    """
    Build per-example token sequences from an Emilia-style HF batch.

    Returns a Python list of np.int32 arrays (ragged), one per example:
      [sep_cond_text_id] + text_bpe_ids + [sep_text_audio_id] + semantic_audio_ids_mapped

    Notes on expected batch structure (Emilia dataset):
    - Audio waveforms:
        Prefer batch['mp3'] as a list of numpy arrays. Fallback keys: 'audio', 'speech', 'waveform'.
    - Texts:
        Prefer batch['text'] as a list of strings. If not present, batch['json'] may be
        a dict of lists with key 'text' or a list of dicts containing 'text'.
    """

    # 1) Extract audio list (list of 1D numpy arrays) unless using precomputed semantics
    audio_list = None
    if not precomputed_semantic_path:
        for k in ("mp3", "audio", "speech", "waveform"):
            if k in batch:
                audio_list = batch[k]
                break
        if audio_list is None:
            raise KeyError(
                "Audio array list not found in batch. Expected one of keys: 'mp3', 'audio', 'speech', 'waveform'."
            )
        if not isinstance(audio_list, list):
            raise ValueError("Audio batch field must be a list of waveforms")

    # 2) Extract texts (list of strings)
    texts: list[str] = []
    if "text" in batch and isinstance(batch["text"], list):
        texts = batch["text"]
    elif "json" in batch:
        meta = batch["json"]
        if isinstance(meta, dict) and "text" in meta and isinstance(meta["text"], list):
            texts = meta["text"]
        elif isinstance(meta, list):
            # list of dicts
            texts = [m.get("text", "") if isinstance(m, dict) else "" for m in meta]
    if not texts:
        raise KeyError("Text list not found in batch. Expected 'text' list or 'json' containing texts.")

    if audio_list is not None and len(texts) != len(audio_list):
        raise ValueError(f"Mismatched batch sizes: {len(texts)} texts vs {len(audio_list)} audio samples")

    # 3) Tokenize texts to BPE ids (no special tokens; no truncation here)
    text_tok = hf_tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
    )
    # Ensure list-of-lists
    text_ids_list: list[list[int]] = text_tok["input_ids"]

    # 4) Semantic tokens: precomputed path or tokenize on-the-fly
    sem_tokens_list: list[np.ndarray] = []
    if precomputed_semantic_path:
        # Support dot path, e.g. 'json.semantic_tokens' or 'semantic_tokens'
        parts = precomputed_semantic_path.split('.')
        if parts[0] == 'json':
            json_val = batch.get('json')
            if isinstance(json_val, list):
                key = parts[1] if len(parts) > 1 else 'semantic_tokens'
                for item in json_val:
                    vals = item.get(key, []) if isinstance(item, dict) else []
                    sem_tokens_list.append(np.asarray(vals, dtype=np.int32))
            elif isinstance(json_val, dict):
                key = parts[1] if len(parts) > 1 else 'semantic_tokens'
                vals = json_val.get(key)
                if not isinstance(vals, list):
                    raise KeyError(f"json.{key} not found as list in batch")
                for v in vals:
                    sem_tokens_list.append(np.asarray(v, dtype=np.int32))
            else:
                raise KeyError("'json' not found or unsupported structure when accessing precomputed semantics")
        else:
            vals = batch.get(precomputed_semantic_path)
            if not isinstance(vals, list):
                raise KeyError(f"Precomputed semantic path '{precomputed_semantic_path}' not found as list in batch")
            for v in vals:
                sem_tokens_list.append(np.asarray(v, dtype=np.int32))
    else:
        # Tokenize audio waveforms
        # Support both batch and per-item tokenize depending on tokenizer implementation
        if hasattr(semantic_tokenizer, 'tokenize_batch'):
            sem_tokens_list = semantic_tokenizer.tokenize_batch(
                audio_list,
                sampling_rate=sample_rate,
                return_ragged=True,
                return_lengths=False,
                to_numpy=True,
            )
        else:
            for audio in audio_list:
                sem = semantic_tokenizer.tokenize(audio, sampling_rate=sample_rate)
                if isinstance(sem, torch.Tensor):
                    sem = sem.detach().cpu().numpy()
                sem_tokens_list.append(np.asarray(sem, dtype=np.int32))

    # 5) Map semantic tokens (0..8191) to model embedding-space ids
    mapped_sem_ids_list: list[np.ndarray] = []
    for sem in sem_tokens_list:
        sem = np.asarray(sem, dtype=np.int32)
        mapped = _map_semantic_tokens_to_model_ids(sem, audio_to_embedding)
        mapped_sem_ids_list.append(mapped)

    # 6) Build final per-example sequences
    seqs: list[np.ndarray] = []
    for text_ids, sem_ids in zip(text_ids_list, mapped_sem_ids_list):
        # [sep_cond_text] + text + [sep_text_audio] + semantic_ids
        seq = np.asarray([sep_cond_text_id] + list(map(int, text_ids)) + [sep_text_audio_id] + sem_ids.tolist(), dtype=np.int32)
        seqs.append(seq)

    # Return as a dict suitable for HF batched map
    return {"tokens": [s.tolist() for s in seqs]}


def make_emilia_audio_train_iterator(
    config,
    global_mesh,
    process_indices_train,
):
    """
    Build a Grain iterator for TTS (Emilia) training data.

    Steps:
      - Load HF dataset
      - Map to per-example ragged token sequences via process_batch
      - Convert to Grain datasource and apply padding/batching
      - Mask prefill region before `sep_text_audio_id` so loss applies only to audio tokens
    """

    # 1) Load HF dataset (streaming like other pipelines)
    train_ds = datasets.load_dataset(
        config.hf_path,
        data_dir=config.hf_data_dir,
        data_files=config.hf_train_files,
        split=config.train_split,
        streaming=True,
        token=config.hf_access_token,
    )

    # 2) Build tokenizers and mapping
    hf_tok = transformers.AutoTokenizer.from_pretrained(
        config.tokenizer_path,
        add_bos_token=config.add_bos,
        add_eos_token=config.add_eos,
        legacy=False,
        token=config.hf_access_token,
    )
    if hf_tok.pad_token_id is not None:
        pad_id = hf_tok.pad_token_id
    elif hf_tok.unk_token_id is not None:
        pad_id = hf_tok.unk_token_id
    else:
        pad_id = -1

    # Resolve separator tokens from the vocab extension utilities.
    sep_cond_text_id = hf_tok.convert_tokens_to_ids("e_<BT>")
    sep_text_audio_id = hf_tok.convert_tokens_to_ids("e_<BA>")
    if sep_cond_text_id is None or sep_cond_text_id < 0 or sep_text_audio_id is None or sep_text_audio_id < 0:
        raise ValueError(
            "Required special tokens e_<BT> and e_<BA> not found in tokenizer. "
            "Ensure your tokenizer has been extended per vocab_expansion/extend_tokenizer.py."
        )
    # Use pad_id as mask value to keep ignored positions simple.
    mask_id = pad_id

    use_pre = getattr(config, "use_precomputed_semantic", False)
    pre_path = getattr(config, "precomputed_semantic_path", "") or None

    semantic_tok = None if use_pre else get_semantic_tokenizer()
    audio_to_embedding = _load_audio_mapping(config.audio_token_mapping_path)

    # 3) Batched map to produce 'tokens' column (ragged per example)
    train_ds = train_ds.map(
        process_batch,
        batched=True,
        fn_kwargs=dict(
            hf_tokenizer=hf_tok,
            semantic_tokenizer=semantic_tok,
            audio_to_embedding=audio_to_embedding,
            sep_cond_text_id=sep_cond_text_id,
            sep_text_audio_id=sep_text_audio_id,
            mask_id=mask_id,
            sample_rate=16000,
            precomputed_semantic_path=pre_path,
        ),
        remove_columns=None,
    )

    # 4) Wrap as Grain datasource
    dataset = _input_pipeline_utils.HFDataSource(
        dataset=train_ds,
        dataloading_host_index=process_indices_train.index(jax.process_index()),
        dataloading_host_count=len(process_indices_train),
        num_threads=1,
        generate_padding_example=False,
        max_target_length=config.max_target_length,
        data_column_names=["tokens"],
    )

    # 5) Define Grain operations: inputs/targets -> pad/batch -> mask prefill -> shift
    operations: list = []
    operations.append(_input_pipeline_utils.HFNormalizeFeatures("tokens"))
    if getattr(config, "packing", True):
        length_struct = {col: config.max_target_length for col in ("inputs", "targets")}
        operations.append(
            grain.experimental.PackAndBatchOperation(
                batch_size=config.global_batch_size_to_load // jax.process_count(),
                length_struct=length_struct,
            )
        )
        operations.append(_input_pipeline_utils.ReformatPacking(("inputs", "targets")))
    else:
        operations.append(_input_pipeline_utils.PadToMaxLength(config.max_target_length, pad_id))
        operations.append(
            grain.Batch(batch_size=config.global_batch_size_to_load // jax.process_count(), drop_remainder=True)
        )
        operations.append(_AddSegPos(pad_id))
    operations.append(_input_pipeline_utils.MaskPrefillWithSeparator(sep_token_id=sep_text_audio_id, mask_id=mask_id))
    # Ignore pad, optional BOS, and mask_id during loss
    ignore_ids = [pad_id]
    if getattr(hf_tok, "bos_token_id", None) is not None:
        ignore_ids.append(hf_tok.bos_token_id)
    ignore_ids.append(mask_id)
    operations.append(_input_pipeline_utils.ShiftData(ignored_ids=ignore_ids, axis=1))

    # Dummy sampler (IterableDataset has no random access)
    dummy_index_sampler = grain.IndexSampler(
        num_records=len(dataset),
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=process_indices_train.index(jax.process_index()),
            shard_count=len(process_indices_train),
            drop_remainder=False,
        ),
        shuffle=False,
        seed=0,
    )

    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=dummy_index_sampler,
        worker_count=1,
        worker_buffer_size=1,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=128),
    )

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)
    return multihost_gen


def make_emilia_audio_eval_iterator(
    config,
    global_mesh,
    process_indices_eval,
):
    raise NotImplementedError(
        "Emilia audio dataset provides only a 'train' split. Set eval_interval=0 or provide a custom eval pipeline."
    )

