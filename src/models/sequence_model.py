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
        if not HAS_FLA:
            # We will use a mock if FLA is not found, but the user requested FLA.
            # In a real environment, we would ensure FLA is installed.
            self.model = None
            print("Warning: flash-linear-attention (fla) not found. GDNSequenceModel will not function.")
        else:
            self.model = GatedDeltaNet(
                d_model=d_model,
                n_heads=n_heads,
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
        if self.model is None:
            return x, None
            
        # GatedDeltaNet from fla supports both parallel and recurrent modes
        # depending on the presence of state/cache.
        return self.model(x, state=state, use_cache=use_cache)
