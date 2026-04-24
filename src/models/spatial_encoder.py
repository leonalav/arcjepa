import torch
import torch.nn as nn
from typing import Optional

class DiscreteViT(nn.Module):
    """
    V-Model: Spatial Encoder for ARC grids.
    Processes 1x1 patches (single cells) using a Transformer architecture.
    """
    def __init__(
        self, 
        d_model: int, 
        nhead: int = 8, 
        num_layers: int = 4, 
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # [CLS] token for global aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [Batch, H, W, d_model] - grid of embeddings
        mask: Optional mask for variable grid sizes
        Returns: [Batch, d_model] latent state s_t
        """
        b, h, w, d = x.shape
        # Flatten HxW into sequence length N
        x = x.view(b, h * w, d)
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [Batch, HW+1, d_model]
        
        # If mask is provided, it should be [Batch, HW]
        # We need to add one 'False' at the beginning for the CLS token
        if mask is not None:
            cls_mask = torch.zeros((b, 1), device=mask.device, dtype=torch.bool)
            mask = torch.cat((cls_mask, mask), dim=1)
            
        # Transformer forward pass
        # TransformerEncoder expects src_key_padding_mask as [Batch, SeqLen]
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Return [CLS] token representation as the latent state s_t
        s_t = output[:, 0]
        return self.norm(s_t)
