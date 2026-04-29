import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .loss_components import VICRegCovarianceLoss, FocalLoss, TemporalSpatialMask

class ARCJPELoss(nn.Module):
    """
    Combined Loss for ARC-JEPA with modular research extensions.

    Loss formulation (post-collapse fix):
      L = L_jepa_proj                   # JEPA MSE in projector space (1024D)
        + lambda_recon  * L_recon       # Grid reconstruction
        + lambda_var    * L_std         # VICReg variance on RAW latents  ← NEW
        + lambda_cov    * L_cov_raw     # VICReg covariance on RAW latents ← NEW
        + lambda_cov_p  * L_cov_proj    # VICReg covariance on projector outputs ← RETAINED
        + L_policy                      # AlphaZero-style policy
        + L_multistep                   # Auxiliary multi-step JEPA

    Key design decisions vs. prior version:
    1. JEPA MSE is now computed in projector space (1024D) so invariance is
       enforced in the higher-dimensional buffer, not the core 512D latent.
    2. VICReg variance + covariance apply to RAW target/pred latents (512D)
       so the encoder itself must produce decorrelated features.
    3. VICReg covariance also applies to projector outputs (1024D) at a
       reduced weight, so the projector is still incentivised to span its
       full dimensionality rather than collapsing to a low-rank mapping.
    4. Separate var_weight (default 25.0) matches the VICReg paper balance.
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
        target_latents = outputs['target_latents']  # [B, T_pred, 512]

        # 1024-dim projector outputs (for JEPA MSE and projector-level cov)
        projected_pred   = outputs['projected_pred_latents']    # [B, T_pred, 1024]
        projected_target = outputs['projected_target_latents']  # [B, T_pred, 1024]

        decoder_logits  = outputs['decoder_logits']
        final_state_gt  = targets['final_state']

        # ── 1. JEPA Loss — enforced in projector space (1024D) ─────────────
        # Operating in projector space keeps the invariance objective out of
        # the raw latent space so VICReg can freely decorrelate it there.
        # Paper: I-JEPA uses L2 distance; MSE provides stronger gradients.
        jepa_loss = F.mse_loss(projected_pred, projected_target)

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
        # Flatten [B, T, D] → [B*T, D] for sufficient sample count.
        flat_target_raw = target_latents.reshape(-1, target_latents.size(-1))
        flat_pred_raw   = pred_latents.reshape(-1, pred_latents.size(-1))

        # unbiased=False to prevent NaN on small effective batch sizes
        std_target = torch.sqrt(flat_target_raw.var(dim=0, unbiased=False) + 1e-4)
        std_pred   = torch.sqrt(flat_pred_raw.var(dim=0, unbiased=False) + 1e-4)

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
            # PRIMARY: Covariance on raw 512-dim target latents.
            # This forces the ENCODER to produce decorrelated features.
            # Gradients flow directly into the online encoder weights.
            cov_loss_raw = self.vicreg_cov_loss(target_latents)

            # SECONDARY: Covariance on 1024-dim projector outputs.
            # Keeps the projector incentivised to span its full dimensionality.
            # Reduced weight (proj_cov_weight < vicreg_weight) so it does not
            # dominate over the primary raw-latent decorrelation signal.
            cov_loss_proj = self.vicreg_cov_loss(projected_target)

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
            multistep_jepa_loss = F.mse_loss(
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
