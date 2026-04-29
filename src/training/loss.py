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
        recon_weight: float = 0.01,  # Reduced from 0.1
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

        # 1. JEPA Loss (MSE for stronger gradients)
        # MSE provides stronger gradients than L1 for small errors
        # Paper citation: I-JEPA uses L2 loss
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
            focal_loss = recon_loss.detach()
        else:
            recon_loss = F.cross_entropy(decoder_logits, final_state_gt)
            focal_loss = torch.tensor(0.0, device=target_latents.device)

        # 3. Variance Regularization (VICReg-style to prevent collapse)
        # Apply to PROJECTED latents to allow core latents to remain flexible
        projected_target = outputs.get('projected_target_latents', target_latents)
        projected_pred = outputs.get('projected_pred_latents', pred_latents)

        # Flatten [B, T, D] to [B*T, D] to ensure sufficient samples
        flat_target = projected_target.reshape(-1, projected_target.size(-1))
        flat_pred = projected_pred.reshape(-1, projected_pred.size(-1))
        
        # Calculate variance with unbiased=False (correction=0) to prevent NaN on small batches
        std_target = torch.sqrt(flat_target.var(dim=0, unbiased=False) + 1e-04)
        std_online = torch.sqrt(flat_pred.var(dim=0, unbiased=False) + 1e-04)
        
        # Increase threshold from 0.1 to 1.0 (VICReg standard)
        std_loss = torch.mean(F.relu(1.0 - std_target)) + torch.mean(F.relu(1.0 - std_online))

        # 4. Policy Loss (AlphaZero-style supervised policy training)
        policy_logits = outputs['policy_logits'] # [B, T-K, 137]
        B, pred_T, _ = policy_logits.shape
        full_T = targets['actions'].shape[1]
        K = full_T - pred_T
        
        gt_actions = targets['actions'][:, K:]
        gt_x = targets['coords_x'][:, K:]
        gt_y = targets['coords_y'][:, K:]
        
        # Split logits: 0:9 (actions), 9:73 (x), 73:137 (y)
        action_logits = policy_logits[:, :, :9]
        x_logits = policy_logits[:, :, 9:73]
        y_logits = policy_logits[:, :, 73:137]
        
        l_action = F.cross_entropy(action_logits.flatten(0, 1), gt_actions.flatten())
        l_x = F.cross_entropy(x_logits.flatten(0, 1), gt_x.flatten())
        l_y = F.cross_entropy(y_logits.flatten(0, 1), gt_y.flatten())
        
        # Scale policy loss so it provides meaningful gradient flow without overpowering JEPA
        policy_loss = (l_action + l_x + l_y) * 0.25

        # 5. Covariance Regularization (VICReg extension)
        if self.use_vicreg:
            # Covariance loss is explicitly applied to the projector output
            cov_loss = self.vicreg_cov_loss(projected_target)
        else:
            cov_loss = torch.tensor(0.0, device=target_latents.device)

        # 6. Multi-step JEPA Loss (if provided)
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
            multistep_jepa_loss +
            policy_loss
        )

        return {
            'loss': total_loss,
            'jepa_loss': jepa_loss.detach(),
            'recon_loss': recon_loss.detach(),
            'std_loss': std_loss.detach(),
            'cov_loss': cov_loss.detach(),
            'focal_loss': focal_loss.detach(),
            'multistep_jepa_loss': multistep_jepa_loss.detach(),
            'policy_loss': policy_loss.detach()
        }
