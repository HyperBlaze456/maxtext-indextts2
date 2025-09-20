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
"""
MaskGCT: Masked Generative Codec Transformer

This module provides complete implementation of MaskGCT components:
- Semantic tokenization (Wav2Vec2-BERT based)
- Acoustic tokenization (Multi-layer VQ codec)
- Semantic-to-Acoustic generation (S2A model)
"""

# Semantic components
from .semantic_utils import (
    build_semantic_model,
    build_semantic_codec,
    SemanticTokenizer
)
from .acoustic.repcodec_model import RepCodec

# Acoustic components
from .acoustic.codec import CodecEncoder, CodecDecoder
from .acoustic.quantize_extended import VectorQuantize, LookupFreeQuantize

# S2A components
from .s2a.maskgct_s2a import MaskGCT_S2A
from .s2a.llama_nar import DiffLlama, DiffLlamaPrefix
from .s2a.masking_utils import (
    get_mask_layer_schedule,
    create_random_mask,
    create_causal_mask,
    create_padding_mask
)

# Configuration
from .config import (
    MaskGCTConfig,
    SemanticCodecConfig,
    AcousticEncoderConfig,
    AcousticDecoderConfig,
    S2AModelConfig,
    get_default_config,
    get_small_config,
    get_large_config
)

__all__ = [
    # Semantic
    "build_semantic_model",
    "build_semantic_codec",
    "SemanticTokenizer",
    "RepCodec",
    # Acoustic
    "CodecEncoder",
    "CodecDecoder",
    "VectorQuantize",
    "LookupFreeQuantize",
    # S2A
    "MaskGCT_S2A",
    "DiffLlama",
    "DiffLlamaPrefix",
    # Utils
    "get_mask_layer_schedule",
    "create_random_mask",
    "create_causal_mask",
    "create_padding_mask",
    # Config
    "MaskGCTConfig",
    "SemanticCodecConfig",
    "AcousticEncoderConfig",
    "AcousticDecoderConfig",
    "S2AModelConfig",
    "get_default_config",
    "get_small_config",
    "get_large_config"
]

