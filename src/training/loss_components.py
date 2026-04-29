import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VICRegCovarianceLoss(nn.Module):
    """
    Feature decorrelation loss on the CORRELATION matrix.

    Operates on the correlation matrix (not raw covariance) so that the
    penalty is scale-invariant: a feature pair with corr=0.97 contributes
    0.97²=0.94 to the loss regardless of per-dimension variance magnitude.
    The variance term (std_loss) handles scale separately.

    Normalization: (1/D) * sum_off_diag( C² )
    ──────────────────────────────────────────────────────────────────────
    CITATION: VICReg paper (Bardes et al., 2021, arXiv:2105.04906),
    Section 2, Equation 3:

        v(Z) = (1/d) * Σ_{i≠j} [C(Z)]_{ij}²

    The paper normalises by D (the embedding dimension), NOT by D*(D-1).
    Our previous code divided by D*(D-1) = 261,632 for D=512, which is
    512× weaker than the paper's formula. With weight=25 this produced a
    covariance gradient contribution of ~0.5 vs std_loss ~15 — a 30×
    imbalance that left features correlated despite VICReg being active.
    """
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, latents_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents_input: [B, T, D] or [B, D] tensor of latent representations

        Returns:
            Scalar covariance loss: (1/D) * sum of squared off-diagonal
            entries of the per-batch correlation matrix.
        """
        # Flatten batch and time dimensions if needed
        if latents_input.dim() == 3:
            B, T, D = latents_input.shape
            latents = latents_input.reshape(B * T, D)
        else:
            latents = latents_input

        N, D = latents.shape
        if N <= 1:
            return torch.tensor(0.0, device=latents.device)

        # Standardize: zero mean, unit variance per dimension.
        # This converts the covariance matrix into the correlation matrix.
        latents_centered = latents - latents.mean(dim=0, keepdim=True)
        std = torch.sqrt(latents_centered.var(dim=0, unbiased=False) + self.eps)
        latents_normed = latents_centered / std.unsqueeze(0)

        # Compute correlation matrix [D, D].
        # Diagonal entries ≈ 1.0; off-diagonal entries ∈ [-1, 1].
        corr_matrix = (latents_normed.T @ latents_normed) / N

        # (1/D) * Σ_{i≠j} C_{ij}²
        # CITATION: VICReg paper Eq. 3 — normalise by D, not D*(D-1).
        # D*(D-1) is 512× too large for D=512, killing the gradient.
        off_diag_sq_sum = (corr_matrix ** 2).sum() - (torch.diagonal(corr_matrix) ** 2).sum()
        off_diag_loss = off_diag_sq_sum / D

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
