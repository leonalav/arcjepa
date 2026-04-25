import torch
import torch.nn as nn
from typing import Optional

class JEPAPredictor(nn.Module):
    """
    Memory-efficient JEPA Predictor with bottleneck architecture.

    Key design from I-JEPA paper (Table 14):
    - Bottleneck width (384) improves performance vs full width (1024)
    - Lightweight MLP with depth for capacity
    - NO residual connection to force learning
    - Concatenate state+action to preserve information

    Memory: ~5M params (vs ~30M Transformer, ~40M multi-GDN)
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int = 6,
        num_heads: int = 8,
        bottleneck_dim: int = 384,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # Input projection: concatenate state + action, then bottleneck
        self.input_proj = nn.Sequential(
            nn.Linear(d_model * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, bottleneck_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Deep MLP with bottleneck (paper-faithful)
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(bottleneck_dim, bottleneck_dim * 2),
                nn.LayerNorm(bottleneck_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim * 2, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim)
            ])
        self.mlp_layers = nn.Sequential(*layers)

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

        # Deep MLP with bottleneck (paper-faithful)
        x = self.mlp_layers(x)

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
