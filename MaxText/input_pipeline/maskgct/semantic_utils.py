# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
from .repcodec_model import RepCodec


def build_semantic_model(device):
    """Build Wav2Vec2-BERT model for semantic feature extraction"""
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    semantic_model.to(device)
    
    # Load pre-computed statistics
    stat_mean_var = torch.load("./tokenizer/maskgct/wav2vec2bert_stats.pt")
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
    def tokenize_batch(self, speech_list, sampling_rate=16000):
        """
        Batch tokenization pipeline for multiple audio samples
        
        Args:
            speech_list: List of raw audio waveforms
            sampling_rate: Sample rate of the audio (default 16kHz)
            
        Returns:
            semantic_tokens: Batch of discrete token indices (B, T)
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
        
        return semantic_tokens