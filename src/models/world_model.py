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
        tau: float = 0.999,
        multistep_k: int = 1
    ):
        super().__init__()
        self.d_model = d_model
        self.multistep_k = multistep_k

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

        # Policy Head: s_t -> (action_logits, x_logits, y_logits)
        # Enables AlphaZero-style pruning during latent search
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 9 + 64 + 64) # 9 actions + 64x + 64y
        )

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

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        context_ratio: float = 0.7,
        use_teacher_forcing: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing and no GDN feedback loop.

        Key changes from original:
        - Teacher forcing: use ground truth states as predictor input
        - GDN only for context processing, NOT in prediction loop
        - Store RAW predictor outputs for loss (no GDN processing)
        - Increased default context_ratio from 0.3 to 0.7

        Args:
            batch: Dictionary with states, actions, coords, target_states
            context_ratio: Fraction of sequence used as context (default 0.7)
            use_teacher_forcing: Use ground truth states during training (default True)

        Returns:
            Dictionary with pred_latents, target_latents, decoder_logits, policy_logits
        """
        states = batch['states']              # [B, T, 64, 64]
        actions = batch['actions']            # [B, T]
        cx, cy = batch['coords_x'], batch['coords_y']
        target_grids = batch['target_states'] # [B, T, 64, 64]

        B, T = states.shape[:2]
        K = max(1, int(T * context_ratio))

        # ═══════════════════════════════════════════════════════
        # PHASE 1: CONTEXT ENCODING (with GDN)
        # ═══════════════════════════════════════════════════════

        # Encode context states with online encoder
        s_context = self.encode(states[:, :K], self.online_encoder)  # [B, K, d_model]

        # Process context with GDN for temporal features
        s_context_features, _ = self.gdn(s_context, use_cache=True)

        # ═══════════════════════════════════════════════════════
        # PHASE 2: TEACHER-FORCED PREDICTION (NO GDN)
        # ═══════════════════════════════════════════════════════

        if use_teacher_forcing:
            # TEACHER FORCING: Use ground truth states as input
            # Encode states in smaller chunks to save memory
            all_states_encoded = []
            chunk_size = 4  # Process 4 frames at a time to reduce memory

            for i in range(0, T-1, chunk_size):
                end_idx = min(i + chunk_size, T-1)
                chunk_encoded = self.encode(states[:, i:end_idx], self.online_encoder)
                all_states_encoded.append(chunk_encoded)

            all_states_encoded = torch.cat(all_states_encoded, dim=1)  # [B, T-1, d_model]

            # Prepare action embeddings for prediction window
            action_embeds = []
            for t in range(K-1, T-1):
                z_a = self.action_embed(actions[:, t], cx[:, t], cy[:, t])
                action_embeds.append(z_a)
            action_embeds = torch.stack(action_embeds, dim=1)  # [B, T-K, d_model]

            # Batch prediction (efficient)
            s_input = all_states_encoded[:, K-1:T-1]  # [B, T-K, d_model]
            pred_latents = self.predictor(s_input, action_embeds, step_idx=K-1)  # [B, T-K, d_model]

            # Multi-step prediction (auxiliary)
            if self.multistep_k > 1 and K-1 + self.multistep_k <= T:
                s_init = all_states_encoded[:, K-1]  # [B, d_model]
                action_embeds_k = []
                for i in range(self.multistep_k):
                    z_a = self.action_embed(actions[:, K-1+i], cx[:, K-1+i], cy[:, K-1+i])
                    action_embeds_k.append(z_a)
                action_embeds_k = torch.stack(action_embeds_k, dim=1)

                multistep_pred_latents = self.predictor.forward_multistep(
                    s_init, action_embeds_k, self.multistep_k
                )
            else:
                multistep_pred_latents = None

        else:
            # AUTOREGRESSIVE: Use previous predictions (inference mode)
            pred_latents = []
            curr_state = s_context_features[:, -1]  # [B, d_model]

            for t in range(K-1, T-1):
                z_a = self.action_embed(actions[:, t], cx[:, t], cy[:, t])

                # Predict next state (NO GDN processing)
                s_next_pred = self.predictor(
                    curr_state.unsqueeze(1),
                    z_a.unsqueeze(1),
                    step_idx=t
                ).squeeze(1)

                pred_latents.append(s_next_pred)
                curr_state = s_next_pred  # Autoregressive

            pred_latents = torch.stack(pred_latents, dim=1)
            multistep_pred_latents = None

        # ═══════════════════════════════════════════════════════
        # PHASE 3: TARGET ENCODING (EMA)
        # ═══════════════════════════════════════════════════════

        with torch.no_grad():
            # Encode target states with target encoder (raw embeddings)
            s_next_target = self.encode(target_grids[:, K:T], self.target_encoder)  # [B, T-K, d_model]

            # Multi-step targets
            if self.multistep_k > 1 and K + self.multistep_k <= T:
                multistep_target_latents = self.encode(
                    target_grids[:, K:K+self.multistep_k],
                    self.target_encoder
                )
            else:
                multistep_target_latents = None

        # ═══════════════════════════════════════════════════════
        # PHASE 4: AUXILIARY OUTPUTS
        # ═══════════════════════════════════════════════════════

        # Decode final predicted state
        final_state_logits = self.decoder(pred_latents[:, -1])

        # Policy logits
        policy_logits = self.policy_head(pred_latents)

        output = {
            'pred_latents': pred_latents,
            'target_latents': s_next_target,
            'decoder_logits': final_state_logits,
            'policy_logits': policy_logits
        }

        if multistep_pred_latents is not None:
            output['multistep_pred_latents'] = multistep_pred_latents
            output['multistep_target_latents'] = multistep_target_latents

        return output
