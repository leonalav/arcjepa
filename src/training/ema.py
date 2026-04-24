import torch
import torch.nn as nn
import copy

class EMAUpdater:
    """
    Handles Exponential Moving Average updates for the Target Encoder.
    theta_target = tau * theta_target + (1 - tau) * theta_online
    """
    def __init__(self, online_model: nn.Module, target_model: nn.Module, tau: float = 0.999):
        self.online_model = online_model
        self.target_model = target_model
        self.tau = tau
        
        # Initialize target weights with online weights
        self.target_model.load_state_dict(self.online_model.state_dict())
        
        # Target model should not have gradients
        for param in self.target_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self):
        for online_params, target_params in zip(self.online_model.parameters(), self.target_model.parameters()):
            target_params.data.mul_(self.tau).add_(online_params.data, alpha=1 - self.tau)
