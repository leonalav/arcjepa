import torch
import torch.nn as nn
import torch.nn.functional as F

class ARCJPELoss(nn.Module):
    """
    Combined Loss for ARC-JEPA.
    L = L_jepa + lambda * L_recon
    """
    def __init__(self, recon_weight: float = 0.1):
        super().__init__()
        self.recon_weight = recon_weight

    def forward(self, outputs, targets):
        """
        outputs: Dict from WorldModel
        targets: Dict with 'final_state' (ground truth)
        """
        pred_latents = outputs['pred_latents']
        target_latents = outputs['target_latents']
        decoder_logits = outputs['decoder_logits']
        final_state_gt = targets['final_state']
        
        # 1. JEPA Loss (MSE)
        # We want predicted next latent to match target latent
        jepa_loss = F.mse_loss(pred_latents, target_latents)
        
        # 2. Reconstruction Loss (CrossEntropy)
        # final_state_gt is [B, 64, 64] with values 0-15
        # decoder_logits is [B, 16, 64, 64]
        recon_loss = F.cross_entropy(decoder_logits, final_state_gt)
        
        # 3. Variance Regularization (VICReg-style to prevent collapse)
        # Force the standard deviation of latents across the batch to be > 0.1
        std_target = torch.sqrt(target_latents.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(0.1 - std_target))
        
        total_loss = jepa_loss + self.recon_weight * recon_loss + std_loss
        
        return {
            'loss': total_loss,
            'jepa_loss': jepa_loss,
            'recon_loss': recon_loss,
            'std_loss': std_loss
        }
