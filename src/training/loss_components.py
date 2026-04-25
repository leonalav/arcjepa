import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VICRegCovarianceLoss(nn.Module):
    """
    VICReg Covariance Loss: Penalizes off-diagonal elements of the covariance matrix
    to force latent dimensions to be decorrelated and encode distinct information.
    """
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, target_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_latents: [B, T, D] or [B, D] tensor of latent representations

        Returns:
            Covariance loss (scalar)
        """
        # Flatten batch and time dimensions if needed
        if target_latents.dim() == 3:
            B, T, D = target_latents.shape
            latents = target_latents.reshape(B * T, D)
        else:
            latents = target_latents

        N = latents.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=latents.device)

        # Center the features
        latents_centered = latents - latents.mean(dim=0, keepdim=True)

        # Compute covariance matrix: [D, D]
        # Using N (biased) for stability with small batches
        cov_matrix = (latents_centered.T @ latents_centered) / N

        # Penalize off-diagonal elements (divide by D to match VICReg paper)
        # off_diag_loss = (sum(cov^2) - sum(diag(cov)^2)) / D
        off_diag_loss = ((cov_matrix ** 2).sum() - (torch.diag(cov_matrix) ** 2).sum()) / D

        return off_diag_loss


class FocalLoss(nn.Module):
    """
    Focal Loss: -α * (1-p_t)^γ * log(p_t)

    Focuses learning on hard examples by down-weighting easy examples.
    For ARC grids, this prevents the model from trivially predicting background pixels.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, num_classes: int = 16):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] predicted logits
            targets: [B, H, W] ground truth labels (0-15)
            weight_mask: [B, H, W] optional spatial weighting

        Returns:
            Focal loss (scalar)
        """
        # Compute cross-entropy per pixel
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B, H, W]

        # Compute probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Get probability of true class
        B, C, H, W = logits.shape
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        p_t = (probs * targets_one_hot).sum(dim=1)  # [B, H, W]

        # Focal modulation factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = self.alpha * focal_weight * ce_loss  # [B, H, W]

        # Apply spatial mask if provided
        if weight_mask is not None:
            focal_loss = focal_loss * weight_mask

        return focal_loss.mean()


class TemporalSpatialMask(nn.Module):
    """
    Temporal Spatial Mask: Creates a binary mask indicating which pixels changed
    between consecutive time steps, then applies differential weighting.

    Changed pixels receive higher weight to force the model to focus on
    semantically important regions rather than static background.
    """
    def __init__(self, changed_weight: float = 10.0, unchanged_weight: float = 1.0):
        super().__init__()
        self.changed_weight = changed_weight
        self.unchanged_weight = unchanged_weight

    def forward(
        self,
        states: torch.Tensor,
        target_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            states: [B, T, H, W] input states at time t
            target_states: [B, T, H, W] target states at time t+1

        Returns:
            weight_mask: [B, T, H, W] spatial weighting mask
        """
        # Compute binary mask: 1 where pixels changed, 0 where unchanged
        changed_mask = (states != target_states).float()  # [B, T, H, W]

        # Apply differential weighting
        weight_mask = (
            changed_mask * self.changed_weight +
            (1 - changed_mask) * self.unchanged_weight
        )

        return weight_mask

    def compute_for_final_state(
        self,
        states: torch.Tensor,
        final_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mask for final state only (used in reconstruction loss).

        Args:
            states: [B, T, H, W] input states
            final_state: [B, H, W] final target state

        Returns:
            weight_mask: [B, H, W] spatial weighting mask
        """
        # Compare final state with last input state
        last_state = states[:, -1, :, :]  # [B, H, W]

        # Compute binary mask
        changed_mask = (last_state != final_state).float()  # [B, H, W]

        # Apply differential weighting
        # Note: If grid is completely static (changed_mask all zeros),
        # this automatically evaluates to unchanged_weight (typically 1.0)
        weight_mask = (
            changed_mask * self.changed_weight +
            (1 - changed_mask) * self.unchanged_weight
        )

        return weight_mask
