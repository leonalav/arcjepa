import torch
import torch.nn as nn
import copy
from typing import Optional, Dict, Any

from .embeddings import GridEmbedding, ActionEmbedding, PositionalEncoding2D
from .spatial_encoder import DiscreteViT
from .sequence_model import GDNSequenceModel
from .jepa_predictor import JEPAPredictor
from .decoder import GridDecoder

class ARCJEPAWorldModel(nn.Module):
    """
    ARC-JEPA World Model: Combines spatial encoding, temporal sequence modeling,
    and latent prediction with EMA target encoders.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        num_vit_layers: int = 4,
        num_gdn_heads: int = 4,
        tau: float = 0.999
    ):
        super().__init__()
        self.d_model = d_model
        
        # Shared Embeddings
        self.grid_embed = GridEmbedding(d_model)
        self.pos_embed = PositionalEncoding2D(d_model)
        self.action_embed = ActionEmbedding(d_model)
        
        # Encoders
        self.online_encoder = DiscreteViT(d_model, nhead=n_heads, num_layers=num_vit_layers)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # Temporal / Sequence Model
        self.gdn = GDNSequenceModel(d_model, n_heads=num_gdn_heads)
        
        # Predictor
        self.predictor = JEPAPredictor(d_model)
        
        # Final State Decoder
        self.decoder = GridDecoder(d_model)

    def encode(self, grids: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """
        grids: [Batch, T, 64, 64]
        encoder: online_encoder or target_encoder
        Returns: [Batch, T, d_model] latent states
        """
        b, t, h, w = grids.shape
        # Flatten time for spatial encoding
        grids = grids.reshape(b * t, h, w)
        
        # Embed and add pos
        x = self.grid_embed(grids) # [BT, H, W, d_model]
        p = self.pos_embed(h, w)   # [H, W, d_model]
        x = x + p.unsqueeze(0)
        
        # Spatial encoding
        latents = encoder(x) # [BT, d_model]
        return latents.reshape(b, t, self.d_model)

    def forward(self, batch: Dict[str, torch.Tensor], context_ratio: float = 0.3) -> Dict[str, torch.Tensor]:
        """
        context_ratio: The percentage of the trajectory provided as ground-truth context.
                       The model must auto-regressively predict the rest.
        """
        states = batch['states']              # [B, T, 64, 64]
        actions = batch['actions']            # [B, T]
        cx, cy = batch['coords_x'], batch['coords_y']
        target_grids = batch['target_states'] # [B, T, 64, 64]
        
        B, T = states.shape[:2]
        
        # 1. Determine Context Window (K) vs Prediction Window
        K = max(1, int(T * context_ratio))
        
        # 2. Encode Context States (Online)
        # We only encode the first K states. The model is BLIND to states K through T.
        s_context = self.encode(states[:, :K], self.online_encoder) # [B, K, d_model]
        
        # 3. Build initial temporal state using GDN
        # We get the context features and the RNN hidden state at time K
        s_context_features, rnn_state = self.gdn(s_context, use_cache=True) 
        
        # 4. Auto-regressive Latent Rollout
        pred_latents = []
        
        # The predictor starts from the last valid context feature
        curr_state = s_context_features[:, -1] # [B, d_model]
        
        for t in range(K, T):
            # Embed the action for step t
            z_a_t = self.action_embed(actions[:, t], cx[:, t], cy[:, t]) # [B, d_model]
            
            # Predict next latent state
            next_latent = self.predictor(curr_state, z_a_t) # [B, d_model]
            pred_latents.append(next_latent)
            
            # To predict the step AFTER next, we feed our prediction back into the GDN
            # to update the temporal RNN state
            if t < T - 1:
                # GDN expects sequence dimension [B, 1, d_model]
                next_latent_seq = next_latent.unsqueeze(1)
                gdn_out, rnn_state = self.gdn(next_latent_seq, state=rnn_state, use_cache=True)
                curr_state = gdn_out.squeeze(1) # [B, d_model]
                
        pred_latents = torch.stack(pred_latents, dim=1) # [B, T-K, d_model]
        
        # 5. Encode target states (Target EMA) - No gradient
        with torch.no_grad():
            # We only calculate target latents for the steps we predicted
            s_next_target = self.encode(target_grids[:, K:T], self.target_encoder) # [B, T-K, d_model]
            
        # 6. Decode final predicted state for reconstruction loss
        # Decode the very last hallucinated state to see if it matches the true final grid
        final_state_logits = self.decoder(pred_latents[:, -1]) # [B, 16, 64, 64]
        
        return {
            'pred_latents': pred_latents,
            'target_latents': s_next_target,
            'decoder_logits': final_state_logits
        }
