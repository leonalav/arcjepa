import os
import sys
import torch
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Discover project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports from the codebase
from src.models.world_model import ARCJEPAWorldModel
from src.training.ema import EMAUpdater
from src.models.embeddings import ActionEmbedding
from src.training.loss_components import VICRegCovarianceLoss, FocalLoss, TemporalSpatialMask
from src.training.metrics import (
    compute_latent_metrics,
    compute_prediction_metrics
)
from src.training.loss import ARCJPELoss
from src.models.jepa_predictor import JEPAPredictor
from src.inference import LatentMCTS, MCTSConfig, MCTSNode
from src.inference.presets import UNSUPERVISED_FAST, SUPERVISED_SHAPED_FAST
from src.inference.utils import grid_accuracy, decode_action_sequence
from src.inference.grid_analysis import (
    check_symmetry, check_completion, check_consistency,
    detect_objects, find_edges, find_frontier,
    object_level_accuracy, structural_similarity, measure_progress
)

# --- Fixtures ---

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def small_model(device):
    """Create a small test model for efficiency."""
    model = ARCJEPAWorldModel(d_model=32, n_heads=4, num_vit_layers=2).to(device)
    model.eval()
    return model

@pytest.fixture
def fast_mcts_config():
    """Fast config for MCTS testing."""
    return MCTSConfig(
        num_simulations=5,
        max_depth=3,
        coord_sampling="sparse",
        coord_stride=16,
        valid_actions=[1, 2]
    )

# --- Architecture Tests ---

class TestArchitecture:
    def test_model_shapes(self, device, small_model):
        batch_size = 2
        T = 4
        
        batch = {
            'states': torch.randint(0, 16, (batch_size, T, 64, 64)).to(device),
            'actions': torch.randint(0, 8, (batch_size, T)).to(device),
            'coords_x': torch.randint(0, 64, (batch_size, T)).to(device),
            'coords_y': torch.randint(0, 64, (batch_size, T)).to(device),
            'target_states': torch.randint(0, 16, (batch_size, T, 64, 64)).to(device)
        }
        
        outputs = small_model(batch)
        
        K = max(1, int(T * 0.3))
        expected_pred_steps = T - K
        
        assert outputs['pred_latents'].shape == (batch_size, expected_pred_steps, 32)
        assert outputs['target_latents'].shape == (batch_size, expected_pred_steps, 32)
        assert outputs['decoder_logits'].shape == (batch_size, 16, 64, 64)
        assert outputs['policy_logits'].shape == (batch_size, expected_pred_steps, 137)

    def test_ema_update(self, device):
        model = ARCJEPAWorldModel(d_model=64).to(device)
        updater = EMAUpdater(model.online_encoder, model.target_encoder, tau=0.0)
        
        with torch.no_grad():
            for p in model.online_encoder.parameters():
                p.add_(1.0)
        updater.update()
        
        for p1, p2 in zip(model.online_encoder.parameters(), model.target_encoder.parameters()):
            assert torch.allclose(p1, p2)

    def test_action_embeddings(self, device):
        d_model = 32
        embed = ActionEmbedding(d_model).to(device)
        
        a1 = embed(torch.tensor([1], device=device), torch.tensor([10], device=device), torch.tensor([20], device=device))
        a2 = embed(torch.tensor([2], device=device), torch.tensor([10], device=device), torch.tensor([20], device=device))
        assert not torch.allclose(a1, a2)

# --- Extension / Loss Tests ---

class TestExtensions:
    def test_vicreg_loss(self):
        vicreg = VICRegCovarianceLoss()
        latents = torch.randn(4, 8, 128)
        loss = vicreg(latents)
        assert loss.item() >= 0

    def test_focal_loss(self):
        focal = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(2, 16, 64, 64)
        targets = torch.randint(0, 16, (2, 64, 64))
        loss = focal(logits, targets)
        assert loss.item() >= 0

    def test_temporal_mask(self):
        mask_fn = TemporalSpatialMask()
        states = torch.randint(0, 16, (2, 4, 64, 64))
        target_states = states.clone()
        target_states[:, :, 5:10, 5:10] = 5
        mask = mask_fn(states, target_states)
        assert mask.shape == states.shape

    def test_multistep_predictor(self):
        predictor = JEPAPredictor(d_model=128)
        s_t_init = torch.randn(2, 128)
        action_embeds = torch.randn(2, 3, 128)
        output = predictor.forward_multistep(s_t_init, action_embeds, k=3)
        assert output.shape == (2, 3, 128)

    def test_policy_loss(self, device):
        criterion = ARCJPELoss()
        
        # Mock outputs
        outputs = {
            'pred_latents': torch.randn(2, 2, 32),
            'target_latents': torch.randn(2, 2, 32),
            'decoder_logits': torch.randn(2, 16, 64, 64),
            'policy_logits': torch.randn(2, 2, 137)
        }
        
        # Mock targets
        targets = {
            'final_state': torch.randint(0, 16, (2, 64, 64)),
            'actions': torch.randint(0, 8, (2, 4)),
            'coords_x': torch.randint(0, 64, (2, 4)),
            'coords_y': torch.randint(0, 64, (2, 4))
        }
        
        loss_dict = criterion(outputs.copy(), targets)
        assert 'policy_loss' in loss_dict
        assert loss_dict['policy_loss'] > 0

# --- MCTS Tests ---

class TestMCTS:
    def test_node_operations(self):
        s_t = torch.randn(1, 32)
        root = MCTSNode(s_t=s_t, rnn_state=None)
        assert root.visits == 0
        
        # Test add_child with prior
        child = root.add_child(action=1, coords=(10, 20), s_next=s_t, rnn_state_next=None, prior_p=0.5)
        assert child.prior_p == 0.5
        
        child.backpropagate(0.8)
        assert root.visits == 1
        assert root.total_value == 0.8

    def test_node_expansion_top_k(self):
        s_t = torch.randn(1, 32)
        node = MCTSNode(s_t=s_t)
        assert not node.is_fully_expanded(top_k=5)
        
        for i in range(5):
            node.add_child(action=1, coords=(i, i), s_next=s_t, rnn_state_next=None)
            
        assert node.is_fully_expanded(top_k=5)

    def test_search_supervised(self, device, small_model, fast_mcts_config):
        mcts = LatentMCTS(small_model, fast_mcts_config, device)
        input_grid = torch.randint(0, 16, (64, 64), device=device)
        target_grid = torch.randint(0, 16, (64, 64), device=device)
        
        result = mcts.solve_puzzle(input_grid, target_grid, mode="supervised")
        assert result["success"] in [True, False]
        assert "action_sequence" in result

    def test_search_unsupervised(self, device, small_model):
        config = UNSUPERVISED_FAST
        config.num_simulations = 5
        mcts = LatentMCTS(small_model, config, device)
        input_grid = torch.randint(0, 16, (64, 64), device=device)
        
        result = mcts.solve_puzzle(input_grid, target_grid=None, mode="unsupervised")
        assert result["mode"] == "unsupervised"
        assert result["final_accuracy"] is None

# --- Grid Analysis Tests ---

class TestGridAnalysis:
    def test_heuristics(self):
        grid = torch.zeros(64, 64, dtype=torch.long)
        grid[10:20, 10:20] = 1 # Square
        
        assert check_symmetry(grid) > 0.9
        assert check_completion(grid) > 0.5
        assert len(detect_objects(grid)) == 1
        
    def test_shaped_rewards(self):
        pred = torch.zeros(64, 64, dtype=torch.long)
        target = torch.zeros(64, 64, dtype=torch.long)
        target[10:20, 10:20] = 1 # Object 1
        target[30:40, 30:40] = 2 # Object 2
        
        # Partial overlap on Object 1 (IoU=0.5, threshold is >0.5, so 0 matches)
        pred[10:15, 10:20] = 1
        # Perfect match on Object 2 (IoU=1.0, 1 match)
        pred[30:40, 30:40] = 2
        
        obj_acc = object_level_accuracy(pred, target)
        # Should be 0.5 (1 match / 2 targets)
        assert 0.0 < obj_acc < 1.0

if __name__ == "__main__":
    # Allow running directly with python
    pytest.main([__file__])
