import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .loss_components import VICRegCovarianceLoss, FocalLoss, TemporalSpatialMask

class ARCJPELoss(nn.Module):
    """
    Combined Loss for ARC-JEPA with modular research extensions.
    L = L_jepa + lambda * L_recon + L_std + [optional: L_cov, L_focal, L_multistep]
    """
    def __init__(
        self,
        recon_weight: float = 0.1,
        use_vicreg: bool = False,
        vicreg_weight: float = 1.0,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        temporal_weight_multiplier: float = 10.0
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.use_vicreg = use_vicreg
        self.vicreg_weight = vicreg_weight
        self.use_focal = use_focal
        self.temporal_weight_multiplier = temporal_weight_multiplier

        # Initialize loss components
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
        targets: Dict with 'final_state', 'states', 'target_states'
        """
        pred_latents = outputs['pred_latents']
        target_latents = outputs['target_latents']
        decoder_logits = outputs['decoder_logits']
        final_state_gt = targets['final_state']

        # 1. JEPA Loss (MSE)
        # We want predicted next latent to match target latent
        jepa_loss = F.mse_loss(pred_latents, target_latents)

        # 2. Reconstruction Loss (CrossEntropy or Focal)
        # final_state_gt is [B, 64, 64] with values 0-15
        # decoder_logits is [B, 16, 64, 64]
        if self.use_focal:
            # Compute temporal mask if states are provided
            if 'states' in targets and 'target_states' in targets:
                temporal_mask = self.temporal_mask_fn.compute_for_final_state(
                    targets['states'],
                    final_state_gt
                )
            else:
                temporal_mask = None

            recon_loss = self.focal_loss_fn(decoder_logits, final_state_gt, temporal_mask)
            focal_loss = recon_loss.item()
        else:
            recon_loss = F.cross_entropy(decoder_logits, final_state_gt)
            focal_loss = 0.0

        # 3. Variance Regularization (VICReg-style to prevent collapse)
        # Force the standard deviation of latents across the batch to be > 0.1
        std_target = torch.sqrt(target_latents.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(0.1 - std_target))

        # 4. Covariance Regularization (VICReg extension)
        if self.use_vicreg:
            cov_loss = self.vicreg_cov_loss(target_latents)
        else:
            cov_loss = torch.tensor(0.0, device=target_latents.device)

        # 5. Multi-step JEPA Loss (if provided)
        if 'multistep_pred_latents' in outputs and 'multistep_target_latents' in outputs:
            multistep_jepa_loss = F.mse_loss(
                outputs['multistep_pred_latents'],
                outputs['multistep_target_latents']
            )
        else:
            multistep_jepa_loss = torch.tensor(0.0, device=target_latents.device)

        # Total loss
        total_loss = (
            jepa_loss +
            self.recon_weight * recon_loss +
            std_loss +
            self.vicreg_weight * cov_loss +
            multistep_jepa_loss
        )

        return {
            'loss': total_loss,
            'jepa_loss': jepa_loss.item(),
            'recon_loss': recon_loss.item(),
            'std_loss': std_loss.item(),
            'cov_loss': cov_loss.item(),
            'focal_loss': focal_loss,
            'multistep_jepa_loss': multistep_jepa_loss.item()
        }
