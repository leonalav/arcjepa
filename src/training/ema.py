import torch
import torch.nn as nn
import copy

class EMAUpdater:
    """
    Handles Exponential Moving Average updates for the Target Encoder.
    theta_target = tau * theta_target + (1 - tau) * theta_online

    I-JEPA paper (Assran et al., 2023, Appendix A.1):
      "We use a momentum value of 0.996, and linearly increase this value
       to 1.0 throughout pretraining."

    This annealing schedule creates a stronger learning signal early in
    training (larger divergence between online and target encoders) and
    gradually freezes the target encoder as training converges.
    """
    def __init__(
        self,
        online_model: nn.Module,
        target_model: nn.Module,
        tau_start: float = 0.996,
        tau_end: float = 1.0
    ):
        self.online_model = online_model
        self.target_model = target_model
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau = tau_start
        
        # Initialize target weights with online weights
        self.target_model.load_state_dict(self.online_model.state_dict())
        
        # Target model should not have gradients
        for param in self.target_model.parameters():
            param.requires_grad = False

    def set_progress(self, progress: float):
        """
        Update EMA momentum based on training progress.

        Args:
            progress: float in [0.0, 1.0] representing fraction of
                      training completed (e.g., current_epoch / total_epochs).
        """
        progress = max(0.0, min(1.0, progress))
        self.tau = self.tau_start + progress * (self.tau_end - self.tau_start)

    @torch.no_grad()
    def update(self):
        for online_params, target_params in zip(self.online_model.parameters(), self.target_model.parameters()):
            target_params.data.mul_(self.tau).add_(online_params.data, alpha=1 - self.tau)
