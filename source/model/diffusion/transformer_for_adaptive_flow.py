from source.model.streaming_diffusion.transformer_for_streaming_diffusion import TransformerForStreamingDiffusion


class TransformerForAdaptiveFlow(TransformerForStreamingDiffusion):
    """Per-action flow-time transformer used by Adaptive Flow Matching."""
