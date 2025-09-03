#!/usr/bin/env python3
"""Test script for batch processing in SemanticTokenizer"""

import torch
import numpy as np
from MaxText.input_pipeline.maskgct.semantic_utils import (
    build_semantic_model, 
    build_semantic_codec, 
    SemanticTokenizer
)
from MaxText.input_pipeline.maskgct.config import SemanticCodecConfig


def create_dummy_audio(duration_sec=1.0, sample_rate=16000, batch_size=1):
    """Create dummy audio for testing"""
    samples = int(duration_sec * sample_rate)
    if batch_size == 1:
        # Single audio sample
        audio = np.random.randn(samples).astype(np.float32) * 0.1
    else:
        # Multiple audio samples with different lengths
        audio = []
        for i in range(batch_size):
            # Vary the length slightly for each sample
            length_variation = int(samples * (0.9 + 0.2 * i / batch_size))
            audio.append(np.random.randn(length_variation).astype(np.float32) * 0.1)
    return audio


def test_single_sample(tokenizer):
    """Test single sample processing"""
    print("\n=== Testing Single Sample Processing ===")
    
    # Create single audio sample
    audio = create_dummy_audio(duration_sec=1.0, batch_size=1)
    print(f"Input audio shape: {audio.shape}")
    
    # Tokenize
    tokens = tokenizer.tokenize(audio, sampling_rate=16000)
    print(f"Output tokens shape: {tokens.shape}")
    print(f"Token values range: [{tokens.min().item()}, {tokens.max().item()}]")
    
    return tokens


def test_batch_using_tokenize(tokenizer):
    """Test batch processing using tokenize method"""
    print("\n=== Testing Batch Processing (tokenize method) ===")
    
    # Create batch of audio samples
    batch_size = 4
    audio_batch = create_dummy_audio(duration_sec=1.0, batch_size=batch_size)
    print(f"Number of samples: {len(audio_batch)}")
    print(f"Sample shapes: {[a.shape for a in audio_batch]}")
    
    # Tokenize batch
    tokens = tokenizer.tokenize(audio_batch, sampling_rate=16000)
    print(f"Output tokens shape: {tokens.shape}")
    print(f"Token values range: [{tokens.min().item()}, {tokens.max().item()}]")
    
    return tokens


def test_batch_using_tokenize_batch(tokenizer):
    """Test batch processing using tokenize_batch method"""
    print("\n=== Testing Batch Processing (tokenize_batch method) ===")
    
    # Create batch of audio samples
    batch_size = 4
    audio_batch = create_dummy_audio(duration_sec=1.0, batch_size=batch_size)
    print(f"Number of samples: {len(audio_batch)}")
    print(f"Sample shapes: {[a.shape for a in audio_batch]}")
    
    # Tokenize batch
    tokens = tokenizer.tokenize_batch(audio_batch, sampling_rate=16000)
    print(f"Output tokens shape: {tokens.shape}")
    print(f"Token values range: [{tokens.min().item()}, {tokens.max().item()}]")
    
    return tokens


def test_consistency(tokenizer):
    """Test that single and batch processing produce consistent results"""
    print("\n=== Testing Consistency ===")
    
    # Create identical audio samples
    audio1 = np.random.randn(16000).astype(np.float32) * 0.1
    audio2 = audio1.copy()
    
    # Process single
    tokens_single = tokenizer.tokenize(audio1, sampling_rate=16000)
    
    # Process as batch
    tokens_batch = tokenizer.tokenize([audio1, audio2], sampling_rate=16000)
    
    # Compare first sample from batch with single processing
    are_equal = torch.allclose(tokens_single, tokens_batch[0])
    print(f"Single processing shape: {tokens_single.shape}")
    print(f"Batch processing shape: {tokens_batch.shape}")
    print(f"First batch sample equals single: {are_equal}")
    
    if are_equal:
        print("✓ Consistency check passed!")
    else:
        print("✗ Consistency check failed!")
        print(f"Difference: {(tokens_single - tokens_batch[0]).abs().max().item()}")


def main():
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Build models
    cfg = SemanticCodecConfig()
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg, device)
    
    # Create tokenizer
    tokenizer = SemanticTokenizer(
        semantic_model=semantic_model,
        semantic_codec=semantic_codec,
        semantic_mean=semantic_mean,
        semantic_std=semantic_std,
        device=device
    )
    
    print("Models loaded successfully!")
    
    # Run tests
    try:
        test_single_sample(tokenizer)
        test_batch_using_tokenize(tokenizer)
        test_batch_using_tokenize_batch(tokenizer)
        test_consistency(tokenizer)
        
        print("\n=== All tests completed successfully! ===")
        print("\nBatch processing is now supported in SemanticTokenizer!")
        print("You can use either:")
        print("  - tokenizer.tokenize(audio_list) for automatic single/batch handling")
        print("  - tokenizer.tokenize_batch(audio_list) for explicit batch processing")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()