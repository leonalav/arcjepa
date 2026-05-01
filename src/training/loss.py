import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .loss_components import FocalLoss, TemporalSpatialMask


class ARCJPELoss(nn.Module):
    """
    Combined Loss for ARC-JEPA — V-JEPA-faithful formulation.

    Loss formulation:
      L = L_jepa                        # JEPA MSE in encoder space (512D)
        + lambda_recon  * L_recon       # Grid reconstruction (cross-entropy or focal)
        + reg_coeff     * L_var_reg     # V-JEPA variance regularization
        + L_policy                      # AlphaZero-style policy
        + L_multistep                   # Auxiliary multi-step JEPA

    Key design decisions (with citations):

    1. JEPA loss uses MSE in the encoder's native latent space.
       CITATION: I-JEPA paper (Assran et al., 2023), Section 3:
         (1/M) * Σ Σ_{j∈B_i} || ŝ_{y_j} - s_{y_j} ||_2^2

    2. Target latents are layer-normalized.
       CITATION: V-JEPA official vjepa/train.py, forward_target(), line 426:
         h = F.layer_norm(h, (h.size(-1),))

    3. Variance regularization follows V-JEPA exactly.
       CITATION: V-JEPA official vjepa/train.py, lines 448-449:
         reg_fn(z) = sum([sqrt(z_i.var(dim=1) + 0.0001) for z_i in z]) / len(z)
         loss_reg  = mean(relu(1 - pstd_z))
       This replaces the VICReg covariance + variance terms which were
       causing 100x loss imbalance and catastrophic training instability.

    4. Predictor output is used directly for both JEPA loss and variance
       regularization, matching V-JEPA's approach. The predictor's internal
       LayerNorm + Linear projection handles scale matching to the
       layer-normed target, learned end-to-end.
    """
    def __init__(
        self,
        recon_weight: float = 0.01,
        reg_coeff: float = 1.0,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        temporal_weight_multiplier: float = 10.0
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.reg_coeff = reg_coeff
        self.use_focal = use_focal
        self.temporal_weight_multiplier = temporal_weight_multiplier

        if self.use_focal:
            self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.temporal_mask_fn = TemporalSpatialMask(
                changed_weight=temporal_weight_multiplier,
                unchanged_weight=1.0
            )

    def forward(self, outputs, targets):
        """
        outputs: Dict from WorldModel
        targets: Dict with 'final_state', 'states', 'target_states', 'actions', etc.
        """
        # Predictor output in encoder space (512D)
        pred_latents   = outputs['pred_latents']    # [B, T_pred, d_model]
        target_latents = outputs['target_latents']  # [B, T_pred, d_model] — layer_normed, no_grad

        decoder_logits  = outputs['decoder_logits']
        final_state_gt  = targets['final_state']

        seq_mask = targets.get('seq_mask', None)

        # ── 1. JEPA Loss — MSE in encoder latent space ──────────────────────
        # CITATION: I-JEPA paper (Assran et al., 2023), Section 3:
        #   (1/M) * Σ Σ_{j∈B_i} || ŝ_{y_j} - s_{y_j} ||_2^2
        # .detach() is belt-and-suspenders: target_latents produced under
        # no_grad in world_model.py, but this documents intent.
        if seq_mask is not None:
            T_pred = pred_latents.shape[1]
            mask = seq_mask[:, :T_pred].unsqueeze(-1)  # [B, T_pred, 1]

            mse_unreduced = (pred_latents - target_latents.detach()) ** 2
            valid_elements = mask.sum() * pred_latents.shape[-1]
            jepa_loss = (mse_unreduced * mask).sum() / (valid_elements + 1e-8)
        else:
            jepa_loss = F.mse_loss(pred_latents, target_latents.detach())

        # ── 2. Reconstruction Loss ──────────────────────────────────────────
        if self.use_focal:
            if 'states' in targets and 'target_states' in targets:
                temporal_mask = self.temporal_mask_fn.compute_for_final_state(
                    targets['states'], final_state_gt
                )
            else:
                temporal_mask = None
            recon_loss = self.focal_loss_fn(decoder_logits, final_state_gt, temporal_mask)
        else:
            recon_loss = F.cross_entropy(decoder_logits, final_state_gt)

        # ── 3. Variance Regularization (V-JEPA faithful) ────────────────────
        # CITATION: V-JEPA official vjepa/train.py, lines 448-449:
        #   def reg_fn(z):
        #       return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)
        #   loss_reg += torch.mean(F.relu(1.-pstd_z))
        #
        # V-JEPA computes variance of predictor output across the patch
        # dimension (dim=1) for each sample, then hinge-penalises dimensions
        # whose std < 1.0. We adapt this to our temporal setting: compute
        # variance across the batch-time dimension (flattened) for each feature.
        if seq_mask is not None:
            T_pred = pred_latents.size(1)
            flat_mask = seq_mask[:, :T_pred].reshape(-1, 1)  # [B*T_pred, 1]
            flat_pred = pred_latents.reshape(-1, pred_latents.size(-1))  # [B*T_pred, D]

            valid_count = flat_mask.sum() + 1e-8
            mean_val = (flat_pred * flat_mask).sum(dim=0, keepdim=True) / valid_count
            var_val = (((flat_pred - mean_val) ** 2) * flat_mask).sum(dim=0) / valid_count
            std_pred = torch.sqrt(var_val + 1e-4)
        else:
            flat_pred = pred_latents.reshape(-1, pred_latents.size(-1))
            std_pred = torch.sqrt(flat_pred.var(dim=0, unbiased=False) + 1e-4)

        var_reg = torch.mean(F.relu(1.0 - std_pred))

        # ── 4. Policy Loss ──────────────────────────────────────────────────
        policy_logits = outputs['policy_logits']  # [B, T_pred, 137]
        B, pred_T, _ = policy_logits.shape
        full_T = targets['actions'].shape[1]
        K = full_T - pred_T

        gt_actions = targets['actions'][:, K:]
        gt_x       = targets['coords_x'][:, K:]
        gt_y       = targets['coords_y'][:, K:]

        action_logits = policy_logits[:, :, :9]
        x_logits      = policy_logits[:, :, 9:73]
        y_logits      = policy_logits[:, :, 73:137]

        l_action = F.cross_entropy(action_logits.flatten(0, 1), gt_actions.flatten())
        l_x      = F.cross_entropy(x_logits.flatten(0, 1), gt_x.flatten())
        l_y      = F.cross_entropy(y_logits.flatten(0, 1), gt_y.flatten())

        policy_loss = (l_action + l_x + l_y) * 0.25

        # ── 5. Multi-step JEPA Loss (auxiliary) ─────────────────────────────
        if 'multistep_pred_latents' in outputs and 'multistep_target_latents' in outputs:
            multistep_jepa_loss = F.mse_loss(
                outputs['multistep_pred_latents'],
                outputs['multistep_target_latents']
            )
        else:
            multistep_jepa_loss = torch.tensor(0.0, device=pred_latents.device)

        # ── Total Loss ───────────────────────────────────────────────────────
        total_loss = (
            jepa_loss                          +
            self.recon_weight * recon_loss     +
            self.reg_coeff   * var_reg        +
            multistep_jepa_loss               +
            policy_loss
        )

        return {
            'loss':                 total_loss,
            'jepa_loss':            jepa_loss.detach(),
            'recon_loss':           recon_loss.detach(),
            'var_reg':              var_reg.detach(),
            'multistep_jepa_loss':  multistep_jepa_loss.detach(),
            'policy_loss':          policy_loss.detach()
        }
