import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .loss_components import VICRegCovarianceLoss, FocalLoss, TemporalSpatialMask

class ARCJPELoss(nn.Module):
    """
    Combined Loss for ARC-JEPA with modular research extensions.

    Loss formulation (collapse-resistant):
      L = L_jepa                        # JEPA L2 in RAW encoder space (512D)  ← KEY
        + lambda_recon  * L_recon       # Grid reconstruction
        + lambda_var    * L_std         # VICReg variance on RAW latents
        + lambda_cov    * L_cov_raw     # VICReg covariance on RAW latents
        + lambda_cov_p  * L_cov_proj    # VICReg covariance on projector outputs
        + L_policy                      # AlphaZero-style policy
        + L_multistep                   # Auxiliary multi-step JEPA

    Key design decisions (with citations):

    1. JEPA MSE in RAW latent space (512D), NOT projector space.
       CITATION: I-JEPA paper (Assran et al., 2023), Section 3, Loss Eq.:
         (1/M) * Σ Σ_{j∈B_i} || ŝ_{y_j} - s_{y_j} ||_2^2
       Both terms are in the encoder's native embedding space. There is no
       projector in the official I-JEPA/V-JEPA loss; adding one and computing
       loss there creates a collapse shortcut (projector collapses the 512D
       representation to satisfy MSE without needing rich latents).
       CITATION: V-JEPA official vjepa/train.py, loss_fn (lines 440-446):
         loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
       where z = predictor output, h = F.layer_norm(target_encoder(clips)).
       Both are in the encoder's native d_model space.

    2. VICReg variance + covariance apply to RAW pred latents (512D)
       so the encoder itself must produce decorrelated features.

    3. VICReg covariance ALSO applies to projector outputs (1024D) at a
       reduced weight so the projector is incentivised to span its full
       dimensionality rather than collapsing to a low-rank map.

    4. Covariance normalised by (1/D) per VICReg paper (Bardes et al., 2021),
       Equation 3: v(Z) = (1/d) * Σ_{i≠j} [C(Z)]_{ij}^2.
       Old normalisation (1/(D*(D-1))) was 512× too weak for D=512.
    """
    def __init__(
        self,
        recon_weight: float = 0.01,
        use_vicreg: bool = False,
        vicreg_weight: float = 25.0,    # cov weight on raw latents
        var_weight: float = 25.0,       # variance weight (new, separate knob)
        proj_cov_weight: float = 5.0,   # cov weight on projector outputs (reduced)
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        temporal_weight_multiplier: float = 10.0
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.use_vicreg = use_vicreg
        self.vicreg_weight = vicreg_weight       # cov on raw latents
        self.var_weight = var_weight             # variance on raw latents
        self.proj_cov_weight = proj_cov_weight   # cov on projector outputs
        self.use_focal = use_focal
        self.temporal_weight_multiplier = temporal_weight_multiplier

        if self.use_vicreg:
            self.vicreg_cov_loss = VICRegCovarianceLoss()

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
        # Raw 512-dim latents (encoder output)
        pred_latents   = outputs['pred_latents']    # [B, T_pred, 512]
        target_latents = outputs['target_latents']  # [B, T_pred, 512] — no_grad, EMA encoder

        # 1024-dim projector outputs.
        # projected_pred is used for cov_proj decorrelation (only).
        # projected_target is no longer used — JEPA loss is now in raw 512D space.
        projected_pred   = outputs['projected_pred_latents']    # [B, T_pred, 1024]  differentiable
        # projected_target kept in dict for potential monitoring but not consumed in loss.

        decoder_logits  = outputs['decoder_logits']
        final_state_gt  = targets['final_state']

        # ── 1. JEPA Loss — enforced in RAW encoder latent space (512D) ──────
        # CITATION: I-JEPA paper (Assran et al., 2023), Section 3, Loss Eq.:
        #   (1/M) * Σ Σ_{j∈B_i} || ŝ_{y_j} - s_{y_j} ||_2^2
        #   Both s_hat (predictor output) and s_y (target encoder output) are
        #   in the encoder's native embedding space — no projector involved.
        # CITATION: V-JEPA official vjepa/train.py, loss_fn (lines 440-446):
        #   loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
        #   zi = predictor output [B, N, D],  hi = F.layer_norm(target(clips))
        #   Both in encoder d_model space.
        #
        # WHY: Computing JEPA MSE in projector space was a collapse shortcut.
        # The projector could learn a degenerate 512→1024 map that minimises
        # MSE without the raw latents being rich or decorrelated.
        # Forcing loss in raw 512D space means the encoder itself must produce
        # representations that are predictable — no escape hatch.
        # .detach() is belt-and-suspenders: target_latents is already produced
        # under torch.no_grad() in world_model.py, but this documents intent
        # and guards against future refactoring that reattaches the target path.
        seq_mask = targets.get('seq_mask', None)
        
        if seq_mask is not None:
            # pred_latents shape: [B, T_pred, 512]
            T_pred = pred_latents.shape[1]
            
            # Align mask to prediction length and add feature dimension: [B, T_pred, 1]
            mask = seq_mask[:, :T_pred].unsqueeze(-1)
            
            # Compute unreduced L1 (as per V-JEPA paper): [B, T_pred, 512]
            l1_loss = F.l1_loss(pred_latents, target_latents.detach(), reduction='none')
            
            # Mask out the padded timesteps and compute mean over valid steps
            valid_elements = mask.sum() * l1_loss.shape[-1]
            jepa_loss = (l1_loss * mask).sum() / (valid_elements + 1e-8)
        else:
            jepa_loss = F.l1_loss(pred_latents, target_latents.detach())

        # ── 2. Reconstruction Loss ──────────────────────────────────────────
        if self.use_focal:
            if 'states' in targets and 'target_states' in targets:
                temporal_mask = self.temporal_mask_fn.compute_for_final_state(
                    targets['states'], final_state_gt
                )
            else:
                temporal_mask = None
            recon_loss = self.focal_loss_fn(decoder_logits, final_state_gt, temporal_mask)
            focal_loss = recon_loss.detach()
        else:
            recon_loss = F.cross_entropy(decoder_logits, final_state_gt)
            focal_loss = torch.tensor(0.0, device=target_latents.device)

        # ── 3. Variance Regularization on RAW latents (primary anti-collapse) ──
        flat_target_raw = target_latents.reshape(-1, target_latents.size(-1))
        flat_pred_raw   = pred_latents.reshape(-1, pred_latents.size(-1))

        seq_mask = targets.get('seq_mask', None)
        if seq_mask is not None:
            T_pred = pred_latents.size(1)
            # Expand mask to match latents shape: [B*T_pred, 1]
            flat_mask = seq_mask[:, :T_pred].reshape(-1, 1)
            
            # --- XLA-SAFE MASKED VARIANCE ---
            valid_count = flat_mask.sum() + 1e-8
            
            # 1. Masked Mean
            mean_pred = (flat_pred_raw * flat_mask).sum(dim=0, keepdim=True) / valid_count
            mean_target = (flat_target_raw * flat_mask).sum(dim=0, keepdim=True) / valid_count
            
            # 2. Masked Variance (mean of squared deviations)
            var_pred = (((flat_pred_raw - mean_pred) ** 2) * flat_mask).sum(dim=0) / valid_count
            var_target = (((flat_target_raw - mean_target) ** 2) * flat_mask).sum(dim=0) / valid_count
            
            std_pred = torch.sqrt(var_pred + 1e-4)
            with torch.no_grad():
                std_target = torch.sqrt(var_target + 1e-4)
        else:
            std_pred   = torch.sqrt(flat_pred_raw.var(dim=0, unbiased=False) + 1e-4)
            with torch.no_grad():
                std_target = torch.sqrt(flat_target_raw.var(dim=0, unbiased=False) + 1e-4)

        # VICReg-standard hinge: penalise dimensions whose std < 1.0
        std_loss = (
            torch.mean(F.relu(1.0 - std_target)) +
            torch.mean(F.relu(1.0 - std_pred))
        )

        # ── 4. Policy Loss ──────────────────────────────────────────────────
        policy_logits = outputs['policy_logits']  # [B, T-K, 137]
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

        # ── 5. Covariance Regularization ─────────────────────────────────────
        if self.use_vicreg:
            if seq_mask is not None:
                # --- XLA-SAFE COVARIANCE MASKING ---
                # We replace padded frames with the mean of the valid frames.
                # This ensures their deviation from the mean is exactly 0, so 
                # they contribute nothing to the covariance matrix, all without 
                # changing the tensor shape.
                T_pred = pred_latents.size(1)
                mask_3d = seq_mask[:, :T_pred].unsqueeze(-1)  # [B, T_pred, 1]
                
                valid_count = mask_3d.sum() + 1e-8
                
                pred_mean = (pred_latents * mask_3d).sum(dim=(0,1), keepdim=True) / valid_count
                safe_pred_latents = (pred_latents * mask_3d) + (pred_mean * (1.0 - mask_3d))
                
                proj_mean = (projected_pred * mask_3d).sum(dim=(0,1), keepdim=True) / valid_count
                safe_projected_pred = (projected_pred * mask_3d) + (proj_mean * (1.0 - mask_3d))
            else:
                safe_pred_latents = pred_latents
                safe_projected_pred = projected_pred

            cov_loss_raw = self.vicreg_cov_loss(safe_pred_latents)
            cov_loss_proj = self.vicreg_cov_loss(safe_projected_pred)

            cov_loss = (
                self.vicreg_weight   * cov_loss_raw +
                self.proj_cov_weight * cov_loss_proj
            )
        else:
            cov_loss_raw  = torch.tensor(0.0, device=target_latents.device)
            cov_loss_proj = torch.tensor(0.0, device=target_latents.device)
            cov_loss      = torch.tensor(0.0, device=target_latents.device)

        # ── 6. Multi-step JEPA Loss (auxiliary) ─────────────────────────────
        if 'multistep_pred_latents' in outputs and 'multistep_target_latents' in outputs:
            multistep_jepa_loss = F.l1_loss(
                outputs['multistep_pred_latents'],
                outputs['multistep_target_latents']
            )
        else:
            multistep_jepa_loss = torch.tensor(0.0, device=target_latents.device)

        # ── Total Loss ───────────────────────────────────────────────────────
        total_loss = (
            jepa_loss                          +
            self.recon_weight  * recon_loss    +
            self.var_weight    * std_loss      +   # ← now has weight 25 (was 1)
            cov_loss                           +   # ← vicreg_weight * raw + proj_cov_weight * proj
            multistep_jepa_loss                +
            policy_loss
        )

        return {
            'loss':                 total_loss,
            'jepa_loss':            jepa_loss.detach(),
            'recon_loss':           recon_loss.detach(),
            'std_loss':             std_loss.detach(),
            'cov_loss':             cov_loss.detach(),
            'cov_loss_raw':         cov_loss_raw.detach() if self.use_vicreg else cov_loss_raw,
            'cov_loss_proj':        cov_loss_proj.detach() if self.use_vicreg else cov_loss_proj,
            'focal_loss':           focal_loss,
            'multistep_jepa_loss':  multistep_jepa_loss.detach(),
            'policy_loss':          policy_loss.detach()
        }
