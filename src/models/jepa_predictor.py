import torch
import torch.nn as nn
from typing import Optional

class JEPAPredictor(nn.Module):
    """
    Memory-efficient JEPA Predictor using GDN (Gated DeltaNet).

    Predicts next state s_{t+1} from current state s_t and action z_a.

    Key design:
    - Uses GDN for O(N) complexity (vs O(N²) Transformer)
    - 6-layer depth for capacity
    - Bottleneck width (384) as per I-JEPA paper Table 14
    - NO residual connection to force learning
    - Concatenate state+action to preserve information

    Memory: ~100MB (vs ~1GB for Transformer predictor)
    Complexity: O(N * d²) linear (vs O(N² * d) quadratic)
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int = 6,
        num_heads: int = 8,
        bottleneck_dim: int = 384,  # I-JEPA paper: bottleneck improves performance
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # Input projection: concatenate state + action, then bottleneck
        self.input_proj = nn.Sequential(
            nn.Linear(d_model * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU()
        )

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, bottleneck_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # GDN layers for O(N) sequence modeling
        from .sequence_model import GDNSequenceModel
        self.gdn_layers = nn.ModuleList([
            GDNSequenceModel(
                d_model=bottleneck_dim,
                n_heads=num_heads,
                chunk_size=16
            )
            for _ in range(num_layers)
        ])

        # Layer norms between GDN layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(bottleneck_dim)
            for _ in range(num_layers)
        ])

        # Output projection back to d_model (NO RESIDUAL CONNECTION)
        self.output_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(
        self,
        s_t: torch.Tensor,
        z_a: torch.Tensor,
        step_idx: int = 0
    ) -> torch.Tensor:
        """
        Predict s_{t+1} from s_t and action z_a.

        Args:
            s_t: Current state latent [B, T, d_model]
            z_a: Action embedding [B, T, d_model]
            step_idx: Position in sequence for positional encoding

        Returns:
            s_next: Predicted next state [B, T, d_model]
        """
        B, T, D = s_t.shape

        # Concatenate state and action (preserve both)
        x = torch.cat([s_t, z_a], dim=-1)  # [B, T, 2*d_model]
        x = self.input_proj(x)              # [B, T, bottleneck_dim]

        # Add positional encoding
        pos = self.pos_embed[:, step_idx:step_idx+T, :]
        x = x + pos

        # Process through GDN layers (O(N) complexity)
        for gdn_layer, layer_norm in zip(self.gdn_layers, self.layer_norms):
            residual = x
            x, _ = gdn_layer(x, use_cache=False)
            x = layer_norm(x + residual)  # Residual within GDN stack only

        # Output projection (NO residual from input - force learning)
        s_next = self.output_proj(x)

        return s_next

    def forward_multistep(
        self,
        s_t: torch.Tensor,          # [B, d_model]
        action_embeds: torch.Tensor, # [B, k, d_model]
        k: int
    ) -> torch.Tensor:
        """
        Multi-step rollout for auxiliary loss.
        Predicts k steps into future without intermediate supervision.

        Args:
            s_t: Initial state [B, d_model]
            action_embeds: Action embeddings for k steps [B, k, d_model]
            k: Number of steps to predict

        Returns:
            predictions: [B, k, d_model] predicted latents at [t+1, t+2, ..., t+k]
        """
        predictions = []
        s_curr = s_t  # [B, d_model]

        for i in range(k):
            z_a = action_embeds[:, i, :]  # [B, d_model]

            # Single-step prediction
            s_curr_exp = s_curr.unsqueeze(1)  # [B, 1, d_model]
            z_a_exp = z_a.unsqueeze(1)        # [B, 1, d_model]

            s_next = self.forward(s_curr_exp, z_a_exp, step_idx=i)
            s_next = s_next.squeeze(1)  # [B, d_model]

            predictions.append(s_next)
            s_curr = s_next  # Autoregressive

        return torch.stack(predictions, dim=1)  # [B, k, d_model]
