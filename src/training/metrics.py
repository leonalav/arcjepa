import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


def compute_latent_metrics(target_latents: torch.Tensor) -> Dict[str, float]:
    """
    Compute latent space health metrics.

    Args:
        target_latents: [B, T, D] or [B, D] tensor of latent representations

    Returns:
        Dict with std_mean, std_min, std_max, effective_rank, correlation_max, norm_mean
    """
    # Flatten batch and time dimensions if needed
    if target_latents.dim() == 3:
        B, T, D = target_latents.shape
        latents = target_latents.reshape(B * T, D)
    else:
        latents = target_latents

    # Standard deviation statistics
    # Use correction=0 (biased) to prevent NaN on single-sample shards
    std_per_dim = torch.std(latents, dim=0, correction=0)
    latent_std_mean = std_per_dim.mean().item()
    latent_std_min = std_per_dim.min().item()
    latent_std_max = std_per_dim.max().item()

    # L2 norm statistics
    latent_norm_mean = torch.norm(latents, dim=1).mean().item()

    # Effective rank via eigenvalue entropy
    latents_centered = latents - latents.mean(dim=0, keepdim=True)
    # Using N (biased) for stability
    N = latents.shape[0]
    cov_matrix = (latents_centered.T @ latents_centered) / N

    try:
        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-8]  # Filter numerical noise

        if len(eigenvalues) > 0:
            probs = eigenvalues / eigenvalues.sum()
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            effective_rank = torch.exp(entropy).item()
        else:
            effective_rank = 0.0
    except:
        effective_rank = 0.0

    # Maximum off-diagonal correlation
    try:
        # Compute correlation matrix
        std_matrix = std_per_dim.unsqueeze(0) * std_per_dim.unsqueeze(1)
        corr_matrix = cov_matrix / (std_matrix + 1e-8)

        # Extract off-diagonal elements
        mask = ~torch.eye(corr_matrix.shape[0], dtype=torch.bool, device=corr_matrix.device)
        off_diag = corr_matrix[mask]
        latent_correlation_max = torch.abs(off_diag).max().item() if len(off_diag) > 0 else 0.0
    except:
        latent_correlation_max = 0.0

    return {
        'latent_std_mean': latent_std_mean,
        'latent_std_min': latent_std_min,
        'latent_std_max': latent_std_max,
        'effective_rank': effective_rank,
        'latent_norm_mean': latent_norm_mean,
        'latent_correlation_max': latent_correlation_max
    }


def compute_prediction_metrics(
    decoder_logits: torch.Tensor,
    final_state: torch.Tensor,
    temporal_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute prediction quality metrics.

    Args:
        decoder_logits: [B, 16, H, W] logits from decoder
        final_state: [B, H, W] ground truth grid
        temporal_mask: [B, H, W] optional binary mask of changed pixels

    Returns:
        Dict with pixel_accuracy, foreground_accuracy, background_accuracy, etc.
    """
    # Get predictions
    predictions = torch.argmax(decoder_logits, dim=1)  # [B, H, W]

    # Overall pixel accuracy
    correct = (predictions == final_state).float()
    pixel_accuracy = correct.mean().item()

    # Foreground vs background accuracy
    foreground_mask = (final_state != 0)
    background_mask = (final_state == 0)

    if foreground_mask.any():
        foreground_accuracy = correct[foreground_mask].mean().item()
    else:
        foreground_accuracy = 0.0

    if background_mask.any():
        background_accuracy = correct[background_mask].mean().item()
    else:
        background_accuracy = 0.0

    # Changed vs unchanged pixel accuracy (if temporal mask provided)
    if temporal_mask is not None:
        # If temporal_mask is a trajectory [B, T, H, W], take the last step matching final_state
        if temporal_mask.dim() == 4:
            curr_mask = temporal_mask[:, -1]
        else:
            curr_mask = temporal_mask

        changed_mask = (curr_mask > 0.5)
        unchanged_mask = ~changed_mask

        if changed_mask.any():
            changed_pixel_accuracy = correct[changed_mask].mean().item()
        else:
            changed_pixel_accuracy = 0.0

        if unchanged_mask.any():
            unchanged_pixel_accuracy = correct[unchanged_mask].mean().item()
        else:
            unchanged_pixel_accuracy = 0.0
    else:
        changed_pixel_accuracy = 0.0
        unchanged_pixel_accuracy = 0.0

    # Per-color accuracy (for 16 ARC colors)
    per_color_accuracy = {}
    for color in range(16):
        color_mask = (final_state == color)
        if color_mask.any():
            per_color_accuracy[f'color_{color}_accuracy'] = correct[color_mask].mean().item()
        else:
            per_color_accuracy[f'color_{color}_accuracy'] = 0.0

    return {
        'pixel_accuracy': pixel_accuracy,
        'foreground_accuracy': foreground_accuracy,
        'background_accuracy': background_accuracy,
        'changed_pixel_accuracy': changed_pixel_accuracy,
        'unchanged_pixel_accuracy': unchanged_pixel_accuracy,
        **per_color_accuracy
    }


def compute_gradient_metrics(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient health metrics.

    Args:
        model: The world model (may be wrapped in DDP)

    Returns:
        Dict with grad_norm_encoder, grad_norm_predictor, grad_norm_decoder, grad_max, grad_min
    """
    # Unwrap DDP if needed
    if hasattr(model, 'module'):
        model = model.module

    # Compute gradient norms per component
    grad_norm_encoder = 0.0
    grad_norm_predictor = 0.0
    grad_norm_decoder = 0.0
    grad_max = 0.0
    grad_min = float('inf')

    # Online encoder gradients
    if hasattr(model, 'online_encoder'):
        for p in model.online_encoder.parameters():
            if p.grad is not None:
                grad_norm_encoder += p.grad.norm().item() ** 2
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_min = min(grad_min, p.grad.abs().min().item())
        grad_norm_encoder = grad_norm_encoder ** 0.5

    # Predictor gradients
    if hasattr(model, 'predictor'):
        for p in model.predictor.parameters():
            if p.grad is not None:
                grad_norm_predictor += p.grad.norm().item() ** 2
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_min = min(grad_min, p.grad.abs().min().item())
        grad_norm_predictor = grad_norm_predictor ** 0.5

    # Decoder gradients
    if hasattr(model, 'decoder'):
        for p in model.decoder.parameters():
            if p.grad is not None:
                grad_norm_decoder += p.grad.norm().item() ** 2
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_min = min(grad_min, p.grad.abs().min().item())
        grad_norm_decoder = grad_norm_decoder ** 0.5

    if grad_min == float('inf'):
        grad_min = 0.0

    return {
        'grad_norm_encoder': grad_norm_encoder,
        'grad_norm_predictor': grad_norm_predictor,
        'grad_norm_decoder': grad_norm_decoder,
        'grad_max': grad_max,
        'grad_min': grad_min
    }


def compute_data_statistics(
    states: torch.Tensor,
    target_states: torch.Tensor
) -> Dict[str, float]:
    """
    Compute data statistics.

    Args:
        states: [B, T, H, W] input states
        target_states: [B, T, H, W] target states

    Returns:
        Dict with noop_ratio, foreground_ratio, unique_colors_mean, grid_entropy_mean
    """
    B, T, H, W = states.shape

    # No-op ratio: fraction of transitions where s_t == s_{t+1}
    noop_count = 0
    total_transitions = 0

    for b in range(B):
        for t in range(T):
            if torch.equal(states[b, t], target_states[b, t]):
                noop_count += 1
            total_transitions += 1

    noop_ratio = noop_count / total_transitions if total_transitions > 0 else 0.0

    # Foreground ratio: fraction of non-zero pixels
    foreground_ratio = (target_states != 0).float().mean().item()

    # Unique colors per grid
    unique_colors_list = []
    for b in range(B):
        for t in range(T):
            unique_colors = torch.unique(target_states[b, t]).numel()
            unique_colors_list.append(unique_colors)
    unique_colors_mean = np.mean(unique_colors_list) if unique_colors_list else 0.0

    # Grid entropy (Shannon entropy of pixel distribution)
    entropy_list = []
    for b in range(B):
        for t in range(T):
            grid = target_states[b, t].flatten()
            # Compute histogram
            hist = torch.histc(grid.float(), bins=16, min=0, max=15)
            probs = hist / (hist.sum() + 1e-10)
            probs = probs[probs > 0]
            entropy = -(probs * torch.log(probs)).sum().item()
            entropy_list.append(entropy)
    grid_entropy_mean = np.mean(entropy_list) if entropy_list else 0.0

    return {
        'noop_ratio': noop_ratio,
        'foreground_ratio': foreground_ratio,
        'unique_colors_mean': unique_colors_mean,
        'grid_entropy_mean': grid_entropy_mean
    }
