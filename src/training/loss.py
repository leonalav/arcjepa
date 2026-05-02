import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_components import FocalLoss, TemporalSpatialMask
from src.data.arc_schema import DEFAULT_NUM_ACTIONS, make_coord_mask, masked_action_logits


class ARCJPELoss(nn.Module):
    def __init__(
        self,
        recon_weight: float = 0.01,
        reg_coeff: float = 1.0,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        temporal_weight_multiplier: float = 10.0,
        policy_weight: float = 0.25,
        coord_weight: float = 0.25,
        terminal_weight: float = 0.1,
        value_weight: float = 0.1,
        efficiency_weight: float = 0.1,
        visit_policy_weight: float = 0.0,
        num_actions: int = DEFAULT_NUM_ACTIONS,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.reg_coeff = reg_coeff
        self.use_focal = use_focal
        self.temporal_weight_multiplier = temporal_weight_multiplier
        self.policy_weight = policy_weight
        self.coord_weight = coord_weight
        self.terminal_weight = terminal_weight
        self.value_weight = value_weight
        self.efficiency_weight = efficiency_weight
        self.visit_policy_weight = visit_policy_weight
        self.num_actions = num_actions

        if self.use_focal:
            self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.temporal_mask_fn = TemporalSpatialMask(
                changed_weight=temporal_weight_multiplier,
                unchanged_weight=1.0
            )

    def _prediction_slice(self, targets, pred_T: int):
        full_T = targets['actions'].shape[1]
        K = full_T - pred_T
        return slice(K, K + pred_T)

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        while mask.dim() < values.dim():
            mask = mask.unsqueeze(-1)
        return (values * mask).sum() / (mask.sum() * values.shape[-1] + 1e-8 if values.dim() > mask.dim() - 1 else mask.sum() + 1e-8)

    def _masked_ce(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').reshape(targets.shape)
        return (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)

    def forward(self, outputs, targets):
        pred_latents = outputs['pred_latents']
        target_latents = outputs['target_latents']
        decoder_logits = outputs['decoder_logits']
        final_state_gt = targets['final_state']
        B, pred_T, D = pred_latents.shape
        pred_slice = self._prediction_slice(targets, pred_T)
        seq_mask = targets.get('seq_mask', torch.ones(B, targets['actions'].shape[1], device=pred_latents.device))[:, pred_slice].to(pred_latents.device).float()

        mse_unreduced = (pred_latents - target_latents.detach()) ** 2
        jepa_loss = (mse_unreduced * seq_mask.unsqueeze(-1)).sum() / (seq_mask.sum() * D + 1e-8)

        if self.use_focal:
            if 'state_changed_mask' in targets:
                temporal_mask = targets['state_changed_mask'][:, -1].to(decoder_logits.device)
            elif 'states' in targets and 'target_states' in targets:
                temporal_mask = self.temporal_mask_fn.compute_for_final_state(targets['states'], final_state_gt)
            else:
                temporal_mask = None
            recon_loss = self.focal_loss_fn(decoder_logits, final_state_gt, temporal_mask)
        else:
            recon_loss = F.cross_entropy(decoder_logits, final_state_gt)

        flat_mask = seq_mask.reshape(-1, 1)
        flat_pred = pred_latents.reshape(-1, D)
        valid_count = flat_mask.sum() + 1e-8
        mean_val = (flat_pred * flat_mask).sum(dim=0, keepdim=True) / valid_count
        var_val = (((flat_pred - mean_val) ** 2) * flat_mask).sum(dim=0) / valid_count
        std_pred = torch.sqrt(var_val + 1e-4)
        var_reg = torch.mean(F.relu(1.0 - std_pred))

        gt_actions = targets['actions'][:, pred_slice].to(pred_latents.device).clamp(0, self.num_actions - 1)
        gt_x = targets['coords_x'][:, pred_slice].to(pred_latents.device).clamp(0, 63)
        gt_y = targets['coords_y'][:, pred_slice].to(pred_latents.device).clamp(0, 63)
        available_mask = targets.get('available_actions_mask', None)
        if available_mask is not None:
            available_mask = available_mask[:, pred_slice].to(pred_latents.device).bool()
            invalid_targets = (gt_actions > 0) & ~available_mask.gather(-1, gt_actions.unsqueeze(-1)).squeeze(-1)
            if bool((invalid_targets & (seq_mask > 0)).any().detach().cpu()):
                raise ValueError("Target action is not available under available_actions_mask")
        else:
            available_mask = torch.ones(B, pred_T, self.num_actions, dtype=torch.bool, device=pred_latents.device)
            available_mask[:, :, 0] = False

        action_logits = outputs.get('raw_action_logits', outputs.get('action_logits'))
        action_logits = masked_action_logits(action_logits, available_mask)
        masked_policy_loss = self._masked_ce(action_logits, gt_actions, seq_mask)

        coord_mask = targets.get('coord_mask', make_coord_mask(targets['actions']))[:, pred_slice].to(pred_latents.device).bool()
        coord_train_mask = seq_mask * coord_mask.float()
        if coord_train_mask.sum() > 0:
            coord_x_loss = self._masked_ce(outputs['x_logits'], gt_x, coord_train_mask)
            coord_y_loss = self._masked_ce(outputs['y_logits'], gt_y, coord_train_mask)
            coord_loss = coord_x_loss + coord_y_loss
        else:
            coord_loss = torch.tensor(0.0, device=pred_latents.device)

        terminal_target = targets.get('terminal', torch.zeros(B, targets['actions'].shape[1], device=pred_latents.device))[:, pred_slice].to(pred_latents.device).float()
        terminal_loss = F.binary_cross_entropy_with_logits(outputs['terminal_logits'], terminal_target, reduction='none')
        terminal_loss = (terminal_loss * seq_mask).sum() / (seq_mask.sum() + 1e-8)

        success_target = targets.get('success', torch.zeros(B, targets['actions'].shape[1], device=pred_latents.device))[:, pred_slice].to(pred_latents.device).float()
        episode_success = targets.get('episode_success', success_target)[:, pred_slice].to(pred_latents.device).float() if 'episode_success' in targets else success_target
        if 'return_to_go' in targets:
            value_target = targets['return_to_go'][:, pred_slice].to(pred_latents.device).float().clamp(0.0, 1.0)
            value_loss_unreduced = F.mse_loss(torch.sigmoid(outputs['value_pred']), value_target, reduction='none')
        else:
            value_target = episode_success
            value_loss_unreduced = F.binary_cross_entropy_with_logits(outputs['value_pred'], value_target, reduction='none')
        value_loss = (value_loss_unreduced * seq_mask).sum() / (seq_mask.sum() + 1e-8)

        if 'efficiency_target' in targets:
            efficiency_target = targets['efficiency_target'][:, pred_slice].to(pred_latents.device).float().clamp(0.0, 1.0)
        elif 'steps_to_win' in targets:
            steps = targets['steps_to_win'][:, pred_slice].to(pred_latents.device).float().clamp_min(1.0)
            efficiency_target = episode_success / steps
        else:
            efficiency_target = success_target
        efficiency_loss_unreduced = F.mse_loss(torch.sigmoid(outputs['efficiency_pred']), efficiency_target, reduction='none')
        efficiency_loss = (efficiency_loss_unreduced * seq_mask).sum() / (seq_mask.sum() + 1e-8)

        if self.visit_policy_weight > 0 and 'mcts_visit_policy' in targets:
            visit_target = targets['mcts_visit_policy'][:, pred_slice].to(pred_latents.device).float()
            visit_sum = visit_target.sum(dim=-1, keepdim=True)
            has_visit = (visit_sum.squeeze(-1) > 0) & (seq_mask > 0)
            visit_target = visit_target / visit_sum.clamp_min(1e-8)
            visit_log_probs = F.log_softmax(action_logits, dim=-1)
            visit_kl_unreduced = F.kl_div(visit_log_probs, visit_target, reduction='none').sum(dim=-1)
            visit_policy_loss = (visit_kl_unreduced * has_visit.float()).sum() / (has_visit.float().sum() + 1e-8)
        else:
            visit_policy_loss = torch.tensor(0.0, device=pred_latents.device)

        if 'multistep_pred_latents' in outputs and 'multistep_target_latents' in outputs:
            multistep_jepa_loss = F.mse_loss(outputs['multistep_pred_latents'], outputs['multistep_target_latents'])
        else:
            multistep_jepa_loss = torch.tensor(0.0, device=pred_latents.device)

        policy_loss = self.policy_weight * masked_policy_loss + self.coord_weight * coord_loss
        total_loss = (
            jepa_loss +
            self.recon_weight * recon_loss +
            self.reg_coeff * var_reg +
            multistep_jepa_loss +
            policy_loss +
            self.terminal_weight * terminal_loss +
            self.value_weight * value_loss +
            self.efficiency_weight * efficiency_loss +
            self.visit_policy_weight * visit_policy_loss
        )

        return {
            'loss': total_loss,
            'jepa_loss': jepa_loss.detach(),
            'recon_loss': recon_loss.detach(),
            'var_reg': var_reg.detach(),
            'multistep_jepa_loss': multistep_jepa_loss.detach(),
            'policy_loss': policy_loss.detach(),
            'masked_policy_loss': masked_policy_loss.detach(),
            'coord_loss': coord_loss.detach(),
            'terminal_loss': terminal_loss.detach(),
            'value_loss': value_loss.detach(),
            'efficiency_loss': efficiency_loss.detach(),
            'visit_policy_loss': visit_policy_loss.detach(),
        }
