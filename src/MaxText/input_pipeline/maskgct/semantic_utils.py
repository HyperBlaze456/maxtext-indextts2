# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
from .acoustic.repcodec_model import RepCodec


def build_semantic_model(device):
    """Build Wav2Vec2-BERT model for semantic feature extraction"""
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    semantic_model.to(device)
    
    # Load pre-computed statistics
    stat_mean_var = torch.load("./MaxText/input_pipeline/maskgct/wav2vec2bert_stats.pt")
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)
    
    return semantic_model, semantic_mean, semantic_std


def build_semantic_codec(cfg, device):
    """Build RepCodec for semantic tokenization"""
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    semantic_codec.to(device)
    return semantic_codec


class SemanticTokenizer:
    """
    Semantic tokenizer that converts Wav2Vec2-BERT's 17th layer representation 
    into discrete tokens with 8192 vocabulary size.
    """
    
    def __init__(self, semantic_model, semantic_codec, semantic_mean, semantic_std, device):
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model = semantic_model
        self.semantic_codec = semantic_codec
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std
        self.device = device
    
    @torch.no_grad()
    def extract_features(self, speech, sampling_rate=16000):
        """
        Extract features from raw speech
        
        Args:
            speech: Single audio array or list of audio arrays for batch processing
            sampling_rate: Sample rate of the audio (default 16kHz)
            
        Returns:
            input_features: Processed audio features (B, T, C)
            attention_mask: Attention mask for the features (B, T)
        """
        # Handle both single and batch inputs
        if not isinstance(speech, list):
            speech = [speech]
        
        inputs = self.processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        return input_features, attention_mask
    
    @torch.no_grad()
    def extract_semantic_code(self, input_features, attention_mask):
        """
        Extract semantic codes from Wav2Vec2-BERT features
        
        Args:
            input_features: Processed audio features
            attention_mask: Attention mask for the features
            
        Returns:
            semantic_code: Discrete token indices (B, T)
            rec_feat: Reconstructed features
        """
        # Get hidden states from Wav2Vec2-BERT
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Extract 17th layer features
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        
        # Normalize features using pre-computed statistics
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)
        
        # Quantize features to get discrete tokens
        semantic_code, rec_feat = self.semantic_codec.quantize(feat)  # (B, T)
        
        return semantic_code, rec_feat
    
    @torch.no_grad()
    def tokenize(self, speech, sampling_rate=16000):
        """
        Complete tokenization pipeline from raw speech to semantic tokens
        
        Args:
            speech: Raw audio waveform (single array) or list of waveforms (batch)
            sampling_rate: Sample rate of the audio (default 16kHz)
            
        Returns:
            semantic_tokens: Discrete token indices with 8192 vocabulary size
                            Shape: (T,) for single input or (B, T) for batch input
        """
        # Track if input was single sample
        is_single = not isinstance(speech, list)
        
        # Extract features (handles both single and batch)
        input_features, attention_mask = self.extract_features(speech, sampling_rate)
        input_features = input_features.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Get semantic codes
        semantic_tokens, _ = self.extract_semantic_code(input_features, attention_mask)
        
        # Return single sample without batch dimension if input was single
        if is_single and semantic_tokens.shape[0] == 1:
            return semantic_tokens.squeeze(0)
        
        return semantic_tokens
    
    @torch.no_grad()
    def tokenize_batch(
        self,
        speech_list,
        sampling_rate=16000,
        return_lengths: bool = False,
        return_ragged: bool = False,
        to_numpy: bool = True,
    ):
        """
        Batch tokenization pipeline for multiple audio samples

        Args:
            speech_list: List of raw audio waveforms
            sampling_rate: Sample rate of the audio (default 16kHz)
            return_lengths: If True, also return per-sample valid lengths.
            return_ragged: If True, return a Python list of per-sample sequences
                trimmed to their valid lengths (no padding).
            to_numpy: If True and return_ragged is True, convert tensors to
                numpy int32 arrays on CPU for downstream processing.

        Returns:
            If return_ragged=False:
              - semantic_tokens: torch.LongTensor [B, T] (padded)
              - optionally lengths: List[int] if return_lengths=True
            If return_ragged=True:
              - sequences: List[ArrayLike[int]] trimmed per example (no padding)
              - optionally lengths: List[int] if return_lengths=True
        """
        # Ensure input is a list
        if not isinstance(speech_list, list):
            raise ValueError("tokenize_batch expects a list of audio arrays")

        # Extract features for batch
        input_features, attention_mask = self.extract_features(speech_list, sampling_rate)
        input_features = input_features.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Get semantic codes
        semantic_tokens, _ = self.extract_semantic_code(input_features, attention_mask)

        # Derive per-sample valid lengths from attention mask
        lengths = attention_mask.sum(dim=1).tolist()

        if return_ragged:
            sequences = []
            for i, L in enumerate(lengths):
                seq = semantic_tokens[i, :L]
                if to_numpy:
                    seq = seq.detach().cpu().numpy().astype(np.int32)
                sequences.append(seq)
            if return_lengths:
                return sequences, lengths
            return sequences

        # Default: return padded batch (and lengths optionally)
        if return_lengths:
            return semantic_tokens, lengths
        return semantic_tokens
