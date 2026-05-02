import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


def compute_latent_metrics(target_latents: torch.Tensor, seq_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute latent space health metrics.

    Args:
        target_latents: [B, T, D] or [B, D] tensor of latent representations
        seq_mask: [B, T] optional boolean mask to filter padded frames

    Returns:
        Dict with std_mean, std_min, std_max, effective_rank, correlation_max, norm_mean
    """
    # Flatten batch and time dimensions if needed
    if target_latents.dim() == 3:
        B, T, D = target_latents.shape
        if seq_mask is not None:
            # Mask could be shorter than T (T_pred vs T)
            T_mask = min(T, seq_mask.shape[1])
            flat_mask = seq_mask[:, :T_mask].reshape(-1).bool()
            latents = target_latents[:, :T_mask, :].reshape(B * T_mask, D)[flat_mask]
        else:
            latents = target_latents.reshape(B * T, D)
    else:
        latents = target_latents

    if latents.shape[0] < 2:
        return {
            'latent_std_mean': 0.0,
            'latent_std_min': 0.0,
            'latent_std_max': 0.0,
            'effective_rank': 1.0,
            'latent_norm_mean': 0.0,
            'latent_correlation_max': 0.0
        }

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
        # CRITICAL: eigvalsh is not implemented for float16 in PyTorch.
        # During DeepSpeed mixed precision, this silently failed and returned 0.0.
        eigenvalues = torch.linalg.eigvalsh(cov_matrix.to(torch.float32))
        
        # Filter numerical noise. Since sum of variances is ~25, noise is very small
        eigenvalues = eigenvalues[eigenvalues > 1e-6]


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
    temporal_mask: Optional[torch.Tensor] = None,
    states: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute prediction quality metrics.

    Args:
        decoder_logits: [B, 16, H, W] logits from decoder
        final_state: [B, H, W] ground truth grid
        temporal_mask: [B, H, W] optional binary mask of changed pixels
        states: [B, T, H, W] optional full trajectory to find the active grid

    Returns:
        Dict with pixel_accuracy, foreground_accuracy, background_accuracy, etc.
    """
    # Get predictions
    predictions = torch.argmax(decoder_logits, dim=1)  # [B, H, W]

    # Find the active bounding box (fixes Deception 8: 1/16 accuracy baseline over padded canvas)
    active_mask = torch.zeros_like(final_state, dtype=torch.bool)
    for b in range(final_state.shape[0]):
        active_grid = final_state[b]
        if states is not None:
            active_grid = torch.max(active_grid, states[b].max(dim=0)[0])
            
        non_zero = torch.nonzero(active_grid)
        if len(non_zero) > 0:
            min_r, min_c = non_zero.min(dim=0)[0]
            max_r, max_c = non_zero.max(dim=0)[0]
            active_mask[b, min_r:max_r+1, min_c:max_c+1] = True
        else:
            active_mask[b] = True

    # Overall pixel accuracy restricted to active grid
    correct = (predictions == final_state).float()
    pixel_accuracy = correct[active_mask].mean().item() if active_mask.any() else correct.mean().item()

    # Foreground vs background accuracy restricted to active grid
    foreground_mask = (final_state != 0) & active_mask
    background_mask = (final_state == 0) & active_mask

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


def compute_action_metrics(
    action_logits: torch.Tensor,
    target_actions: torch.Tensor,
    available_actions_mask: Optional[torch.Tensor] = None,
    seq_mask: Optional[torch.Tensor] = None,
    top_k: int = 3,
) -> Dict[str, float]:
    pred_T = action_logits.shape[1]
    full_T = target_actions.shape[1]
    K = full_T - pred_T
    targets = target_actions[:, K:K + pred_T].to(action_logits.device)
    mask = torch.ones_like(targets, dtype=torch.bool) if seq_mask is None else seq_mask[:, K:K + pred_T].to(action_logits.device).bool()

    if available_actions_mask is not None:
        available = available_actions_mask[:, K:K + pred_T].to(action_logits.device).bool()
        masked_logits = action_logits.masked_fill(~available, torch.finfo(action_logits.dtype).min)
        target_valid = available.gather(-1, targets.clamp(0, available.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    else:
        available = None
        masked_logits = action_logits
        target_valid = torch.ones_like(mask, dtype=torch.bool)

    pred = masked_logits.argmax(dim=-1)
    valid_count = mask.float().sum().item()
    if valid_count == 0:
        return {'action_accuracy': 0.0, 'action_topk_accuracy': 0.0, 'invalid_action_rate': 0.0, 'target_validity_rate': 0.0}

    top_k = min(top_k, masked_logits.shape[-1])
    topk = masked_logits.topk(top_k, dim=-1).indices
    topk_hit = (topk == targets.unsqueeze(-1)).any(dim=-1)
    if available is not None:
        invalid_pred = ~available.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
    else:
        invalid_pred = torch.zeros_like(mask, dtype=torch.bool)

    return {
        'action_accuracy': ((pred == targets) & mask).float().sum().item() / valid_count,
        'action_topk_accuracy': (topk_hit & mask).float().sum().item() / valid_count,
        'invalid_action_rate': (invalid_pred & mask).float().sum().item() / valid_count,
        'target_validity_rate': (target_valid & mask).float().sum().item() / valid_count,
    }


def compute_terminal_metrics(
    terminal_logits: torch.Tensor,
    terminal_targets: torch.Tensor,
    seq_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    pred_T = terminal_logits.shape[1]
    full_T = terminal_targets.shape[1]
    K = full_T - pred_T
    targets = terminal_targets[:, K:K + pred_T].to(terminal_logits.device).float()
    mask = torch.ones_like(targets, dtype=torch.bool) if seq_mask is None else seq_mask[:, K:K + pred_T].to(terminal_logits.device).bool()
    preds = torch.sigmoid(terminal_logits) >= 0.5
    valid_count = mask.float().sum().item()
    if valid_count == 0:
        return {'terminal_accuracy': 0.0, 'terminal_positive_rate': 0.0}
    return {
        'terminal_accuracy': ((preds == targets.bool()) & mask).float().sum().item() / valid_count,
        'terminal_positive_rate': (preds & mask).float().sum().item() / valid_count,
    }


def compute_efficiency_metrics(
    efficiency_pred: torch.Tensor,
    efficiency_target: torch.Tensor,
    seq_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    pred_T = efficiency_pred.shape[1]
    full_T = efficiency_target.shape[1]
    K = full_T - pred_T
    target = efficiency_target[:, K:K + pred_T].to(efficiency_pred.device).float()
    mask = torch.ones_like(target) if seq_mask is None else seq_mask[:, K:K + pred_T].to(efficiency_pred.device).float()
    pred = torch.sigmoid(efficiency_pred)
    mae = (torch.abs(pred - target) * mask).sum() / (mask.sum() + 1e-8)
    return {
        'efficiency_mae': mae.item(),
        'efficiency_pred_mean': ((pred * mask).sum() / (mask.sum() + 1e-8)).item(),
        'efficiency_target_mean': ((target * mask).sum() / (mask.sum() + 1e-8)).item(),
    }


def compute_value_metrics(
    value_pred: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    seq_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    pred_T = value_pred.shape[1]
    full_T = targets['actions'].shape[1]
    K = full_T - pred_T
    if 'return_to_go' in targets:
        value_target = targets['return_to_go'][:, K:K + pred_T].to(value_pred.device).float().clamp(0.0, 1.0)
    elif 'episode_success' in targets:
        value_target = targets['episode_success'][:, K:K + pred_T].to(value_pred.device).float()
    else:
        value_target = targets.get('success', torch.zeros_like(targets['actions'], dtype=torch.float32))[:, K:K + pred_T].to(value_pred.device).float()
    mask = torch.ones_like(value_target) if seq_mask is None else seq_mask[:, K:K + pred_T].to(value_pred.device).float()
    pred = torch.sigmoid(value_pred)
    diff = torch.abs(pred - value_target)
    mae = (diff * mask).sum() / (mask.sum() + 1e-8)
    brier = (((pred - value_target) ** 2) * mask).sum() / (mask.sum() + 1e-8)
    positives = value_target > 0.5
    negatives = ~positives
    pos_mean = pred[positives & mask.bool()].mean().item() if (positives & mask.bool()).any() else 0.0
    neg_mean = pred[negatives & mask.bool()].mean().item() if (negatives & mask.bool()).any() else 0.0
    return {
        'value_mae': mae.item(),
        'value_brier': brier.item(),
        'value_positive_mean': pos_mean,
        'value_negative_mean': neg_mean,
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
    target_states: torch.Tensor,
    seq_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute data statistics. Expects CPU tensors to avoid XLA graph breaks.

    Args:
        states: [B, T, H, W] input states
        target_states: [B, T, H, W] target states
        seq_mask: [B, T] optional mask of valid unpadded frames

    Returns:
        Dict with noop_ratio, foreground_ratio, unique_colors_mean, grid_entropy_mean
    """
    B, T, H, W = states.shape

    # Deception 2 Fix: noop_ratio was measuring padding. Rename conceptually to padding_ratio (kept as noop_ratio for compatibility)
    if seq_mask is not None:
        padding_ratio = 1.0 - seq_mask.float().mean().item()
        
        # Filter valid states for metrics
        T_mask = min(T, seq_mask.shape[1])
        flat_mask = seq_mask[:, :T_mask].reshape(-1).bool()
        valid_target_states = target_states[:, :T_mask, :, :].reshape(B * T_mask, H, W)[flat_mask]
    else:
        padding_ratio = (states == target_states).reshape(B, T, -1).all(dim=-1).float().mean().item()
        valid_target_states = target_states.reshape(B * T, H, W)

    if len(valid_target_states) == 0:
        return {'noop_ratio': padding_ratio, 'padding_ratio': padding_ratio, 'foreground_ratio': 0.0, 'unique_colors_mean': 0.0, 'grid_entropy_mean': 0.0}

    # Deception 1 Fix: Calculate foreground ratio ONLY within the active grid of valid states
    active_mask = torch.zeros_like(valid_target_states, dtype=torch.bool)
    for i in range(valid_target_states.shape[0]):
        non_zero = torch.nonzero(valid_target_states[i])
        if len(non_zero) > 0:
            min_r, min_c = non_zero.min(dim=0)[0]
            max_r, max_c = non_zero.max(dim=0)[0]
            active_mask[i, min_r:max_r+1, min_c:max_c+1] = True
        else:
            active_mask[i] = True

    if active_mask.any():
        foreground_ratio = (valid_target_states[active_mask] != 0).float().mean().item()
    else:
        foreground_ratio = 0.0

    # Unique colors per valid grid
    N_valid = valid_target_states.shape[0]
    flat_grids = valid_target_states.reshape(N_valid, H * W).float()  # [N_valid, H*W]
    
    unique_counts = []
    for i in range(N_valid):
        hist = torch.histc(flat_grids[i], bins=16, min=0, max=15)
        unique_counts.append((hist > 0).sum().item())
    unique_colors_mean = np.mean(unique_counts) if unique_counts else 0.0

    # Grid entropy per valid grid
    entropy_list = []
    for i in range(N_valid):
        hist = torch.histc(flat_grids[i], bins=16, min=0, max=15)
        probs = hist / (hist.sum() + 1e-10)
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum().item()
        entropy_list.append(entropy)
    grid_entropy_mean = np.mean(entropy_list) if entropy_list else 0.0

    true_noop_ratio = (states == target_states).reshape(B, T, -1).all(dim=-1).float()
    if seq_mask is not None:
        true_noop_ratio = (true_noop_ratio * seq_mask[:, :T].float()).sum() / (seq_mask[:, :T].float().sum() + 1e-8)
        true_noop_ratio = true_noop_ratio.item()
    else:
        true_noop_ratio = true_noop_ratio.mean().item()

    return {
        'noop_ratio': true_noop_ratio,
        'padding_ratio': padding_ratio,
        'foreground_ratio': foreground_ratio,
        'unique_colors_mean': unique_colors_mean,
        'grid_entropy_mean': grid_entropy_mean
    }
