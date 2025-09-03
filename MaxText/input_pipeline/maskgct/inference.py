import torch
import librosa
import argparse
from .semantic_utils import build_semantic_model, build_semantic_codec, SemanticTokenizer
from .config import SemanticCodecConfig


def main():
    parser = argparse.ArgumentParser(description="Convert audio to semantic tokens")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Sample rate for processing (default: 16000)")
    
    args = parser.parse_args()
    
    print(f"Loading models on {args.device}...")
    
    # Build models
    cfg = SemanticCodecConfig()
    semantic_model, semantic_mean, semantic_std = build_semantic_model(args.device)
    semantic_codec = build_semantic_codec(cfg, args.device)
    
    # Create tokenizer
    tokenizer = SemanticTokenizer(
        semantic_model=semantic_model,
        semantic_codec=semantic_codec,
        semantic_mean=semantic_mean,
        semantic_std=semantic_std,
        device=args.device
    )
    
    # Load audio
    print(f"Loading audio from {args.audio_path}...")
    speech, sr = librosa.load(args.audio_path, sr=args.sample_rate)
    
    # Tokenize
    print("Extracting semantic tokens...")
    semantic_tokens = tokenizer.tokenize(speech, sampling_rate=args.sample_rate)
    
    # Print results
    print(f"\nResults:")
    print(f"- Input audio shape: {speech.shape}")
    print(f"- Semantic tokens shape: {semantic_tokens.shape}")
    print(f"- Vocabulary size: {cfg.codebook_size}")
    print(f"- Token sequence length: {semantic_tokens.shape[-1]}")
    print(f"\nFirst 50 tokens:")
    print(semantic_tokens[0, :50].cpu().numpy())
    
    # Save tokens
    output_path = args.audio_path.replace(".wav", "_tokens.pt")
    torch.save(semantic_tokens.cpu(), output_path)
    print(f"\nSaved tokens to: {output_path}")


if __name__ == "__main__":
    main()