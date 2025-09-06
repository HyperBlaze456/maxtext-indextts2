from .maskgct.semantic_utils import (
    build_semantic_model,
    build_semantic_codec,
    SemanticTokenizer,
)
from .maskgct.config import SemanticCodecConfig
import argparse
import torch


def get_semantic_tokenizer(device: str | None = None, cfg: SemanticCodecConfig | None = None) -> SemanticTokenizer:
    """Initialize and return a SemanticTokenizer ready for batch processing.

    This uses MaskGCT semantic utilities as the single source of truth.

    Args:
        device: Target device (e.g., "cuda" or "cpu"). Auto-detects if None.
        cfg: Optional SemanticCodecConfig. Uses default if None.

    Returns:
        An initialized SemanticTokenizer instance.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
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


# Placeholders to satisfy top-level imports in input_pipeline_interface.
# These can be implemented later when wiring the full Emilia audio pipeline.
def make_emilia_audio_train_iterator(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Emilia audio train iterator not yet implemented.")


def make_emilia_audio_eval_iterator(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Emilia audio eval iterator not yet implemented.")


def _create_dummy_audio_list(batch_size: int, duration_sec: float, sample_rate: int):
    """Create a list of random 1D float32 arrays emulating audio waveforms."""
    import numpy as np

    base_len = int(duration_sec * sample_rate)
    audio_list = []
    for i in range(batch_size):
        # Vary lengths slightly across the batch to exercise padding logic
        scale = 0.9 + 0.2 * (i / max(1, batch_size - 1)) if batch_size > 1 else 1.0
        length = max(1, int(base_len * scale))
        audio_list.append((np.random.randn(length).astype("float32") * 0.05))
    return audio_list


def main():  # pragma: no cover
    """Minimal batch demo using SemanticTokenizer with random audio-like arrays."""
    parser = argparse.ArgumentParser(description="Batch demo for SemanticTokenizer")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of samples in the batch")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration (seconds) per sample before variation")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for processing")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to use ("cuda"/"cpu"); auto-detects if omitted',
    )
    args = parser.parse_args()

    # Build tokenizer using semantic utils (original implementation)
    tokenizer = get_semantic_tokenizer(device=args.device)

    # Create random batch of 1D arrays
    audio_list = _create_dummy_audio_list(args.batch_size, args.duration, args.sample_rate)

    # Run batch tokenization (works for single or multiple)
    tokens = tokenizer.tokenize(audio_list if args.batch_size > 1 else audio_list[0], sampling_rate=args.sample_rate)

    # Summarize results
    if isinstance(tokens, torch.Tensor):
        shape = tuple(tokens.shape)
        tmin = tokens.min().item()
        tmax = tokens.max().item()
    else:
        shape = ("unknown",)
        tmin = tmax = "n/a"

    print("Semantic tokenization complete.")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Sample rate: {args.sample_rate}")
    print(f"- Tokens shape: {shape}")
    print(f"- Token value range: [{tmin}, {tmax}]")


if __name__ == "__main__":  # pragma: no cover
    main()


