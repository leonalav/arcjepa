import math
import torch
import torch.nn as nn
from typing import Optional


class DiscreteViT(nn.Module):
    """
    V-Model: Spatial Encoder for ARC grids.

    Processes 4×4 patches over a 64×64 grid (paper spec: "1×1 or 3×3 localized
    grid patches with categorical embeddings", start_plan.md §3).

    4×4 patches chosen because:
      - 4 divides 64 exactly → 16×16 = 256 patch tokens per frame
      - Attention matrix [B, 257, 257] = ~0.13 MB per frame in bf16
        vs. 1×1 patches [B, 4097, 4097] = 32 MB → 254× smaller
      - Full B×T frames can pass through in one shot (no micro-batching)

    Input:  [Batch, 16, 16, d_model] — patch embeddings from GridEmbedding
    Output: [Batch, d_model] — [CLS] token as the global latent state s_t
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        init_std: float = 0.02
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])

        # [CLS] token for global aggregation (I-JEPA / V-JEPA style)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.norm = nn.LayerNorm(d_model)

        # ── Weight initialization ──────────────────────────────────────────
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        """CITATION: V-JEPA vision_transformer.py lines 128-143."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        """Scale down residual branch weights by 1/sqrt(2*layer_id).

        CITATION: V-JEPA vision_transformer.py lines 145-151. This prevents
        gradient explosion at initialization in deep Transformer stacks by
        ensuring that the contribution of deeper layers starts small.
        """
        for layer_id, layer in enumerate(self.layers):
            # nn.TransformerEncoderLayer: self_attn.out_proj and linear2
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'out_proj'):
                layer.self_attn.out_proj.weight.data.div_(
                    math.sqrt(2.0 * (layer_id + 1))
                )
            if hasattr(layer, 'linear2'):
                layer.linear2.weight.data.div_(
                    math.sqrt(2.0 * (layer_id + 1))
                )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    [Batch, Ph, Pw, d_model] — patch token grid (Ph=Pw=16 for 4×4 patches)
        mask: Optional [Batch, Ph*Pw] key-padding mask (True = ignore)
        Returns: [Batch, d_model] — CLS token latent state s_t
        """
        b, ph, pw, d = x.shape
        # Flatten patch grid into sequence: [Batch, Ph*Pw, d_model]
        x = x.view(b, ph * pw, d)

        # Prepend [CLS] token: [Batch, Ph*Pw+1, d_model]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if mask is not None:
            # Extend mask for CLS token (always attended to)
            cls_mask = torch.zeros((b, 1), device=mask.device, dtype=torch.bool)
            mask = torch.cat((cls_mask, mask), dim=1)

        # Process through Transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)

        # Return normalised CLS token as the frame latent state
        return self.norm(x[:, 0])
