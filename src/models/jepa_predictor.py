import torch
import torch.nn as nn
from typing import Optional

class JEPAPredictor(nn.Module):
    """
    Paper-faithful JEPA Predictor with increased capacity.

    Predicts next state s_{t+1} from current state s_t and action z_a.

    Key improvements over original:
    - 6 layers (vs 2) for increased capacity
    - Transformer architecture (vs MLP) for sequence modeling
    - NO residual connection to force learning
    - Concatenate state+action (vs add) to preserve information
    - Positional encoding for sequence awareness

    Complexity: O(T^2 * d) per forward pass, but T is small (1-7 steps with teacher forcing)
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        self.d_model = d_model

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # Input projection: concatenate state + action (preserve information)
        self.input_proj = nn.Linear(d_model * 2, d_model)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder with causal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection (NO RESIDUAL CONNECTION - force learning)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
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
        x = self.input_proj(x)              # [B, T, d_model]

        # Add positional encoding
        pos = self.pos_embed[:, step_idx:step_idx+T, :]
        x = x + pos

        # Transformer processing with causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, mask=causal_mask)

        # Output projection (NO residual - force learning)
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
