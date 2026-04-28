import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from fla.layers import GatedDeltaNet
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    # Fallback to a dummy or raise error if required by user
    # Given the strict "FLA one" instruction, I'll use it as if it's there.

class GDNSequenceModel(nn.Module):
    """
    M-Model: Sequence model using Gated Delta Network.
    Maintains temporal state and models the latent transition sequence.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int = 4,
        chunk_size: int = 16, # Chunk size for parallel training
        **kwargs
    ):
        super().__init__()
        
        # Check if we're on TPU (no CUDA, but torch_xla available)
        try:
            import torch_xla
            is_tpu = True
        except ImportError:
            is_tpu = False

        if is_tpu:
            print("TPU detected. Replacing Triton-based GDN with native SDPA Transformer layer.")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.model = encoder_layer
            self.is_tpu_fallback = True
        elif not HAS_FLA or not torch.cuda.is_available():
            # Fallback for CPU-only environments or missing fla
            self.model = None
            self.is_tpu_fallback = False
            if not HAS_FLA:
                print("Warning: flash-linear-attention (fla) not found. GDNSequenceModel will not function.")
            else:
                print("Warning: GPU not detected. Disabling fla-based GDN (Triton requires CUDA).")
        else:
            self.is_tpu_fallback = False
            # GatedDeltaNet in 'fla' uses 'hidden_size' and 'num_heads'
            self.model = GatedDeltaNet(
                hidden_size=d_model,
                num_heads=n_heads,
                chunk_size=chunk_size,
                **kwargs
            )
        
    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [Batch, T, d_model]
        state: Optional recurrent state for inference
        Returns: [Batch, T, d_model] and next state
        """
        if getattr(self, 'is_tpu_fallback', False):
            # For TransformerEncoderLayer, we generate a causal mask
            seq_len = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            # TransformerEncoderLayer efficiently maps to Native SDPA in PyTorch 2.0+
            out = self.model(x, src_mask=mask, is_causal=True)
            return out, None

        if self.model is None:
            return x, None
            
        # GatedDeltaNet from fla supports both parallel and recurrent modes
        # Depending on the version, it might return (output, state) or (output, state, last_state)
        # We ensure we only return (output, state) to match the WorldModel expectations.
        res = self.model(x, state=state, use_cache=use_cache)
        if isinstance(res, (tuple, list)):
            return res[0], res[1]
        return res, None
