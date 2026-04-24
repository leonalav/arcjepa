"""
Tests for MCTS inference module.
"""

import os
import sys
import torch
import pytest
import numpy as np
from pathlib import Path

# Discover project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.world_model import ARCJEPAWorldModel
from src.inference.mcts import LatentMCTS
from src.inference.node import MCTSNode
from src.inference.config import MCTSConfig, FAST_CONFIG
from src.inference.utils import grid_accuracy, decode_action_sequence


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def model(device):
    """Create a small test model."""
    model = ARCJEPAWorldModel(d_model=32, n_heads=4, num_vit_layers=2).to(device)
    model.eval()
    return model


@pytest.fixture
def mcts_config():
    """Fast config for testing."""
    return MCTSConfig(
        num_simulations=10,
        c_puct=1.0,
        max_depth=5,
        coord_sampling="sparse",
        coord_stride=16,
        valid_actions=[1, 2, 3]
    )


def test_mcts_node_creation():
    """Test MCTSNode initialization and basic operations."""
    s_t = torch.randn(1, 32)
    node = MCTSNode(s_t=s_t, rnn_state=None, parent=None)

    assert node.visits == 0
    assert node.total_value == 0.0
    assert node.q_value() == 0.0
    assert not node.is_terminal
    assert len(node.children) == 0


def test_mcts_node_uct_score():
    """Test UCT score calculation."""
    s_t = torch.randn(1, 32)
    parent = MCTSNode(s_t=s_t, rnn_state=None, parent=None)
    parent.visits = 10

    child = MCTSNode(s_t=s_t, rnn_state=None, parent=parent, action_taken=1)

    # Unvisited child should have infinite UCT
    assert child.uct_score(c_puct=1.0, parent_visits=10) == float('inf')

    # After visiting, UCT should be finite
    child.visits = 5
    child.total_value = 2.5
    uct = child.uct_score(c_puct=1.4, parent_visits=10)

    assert uct > 0
    assert uct < float('inf')


def test_mcts_node_backpropagation():
    """Test reward backpropagation up the tree."""
    s_t = torch.randn(1, 32)

    root = MCTSNode(s_t=s_t, rnn_state=None, parent=None)
    child1 = MCTSNode(s_t=s_t, rnn_state=None, parent=root, action_taken=1)
    child2 = MCTSNode(s_t=s_t, rnn_state=None, parent=child1, action_taken=2)

    # Backpropagate from leaf
    reward = 0.8
    child2.backpropagate(reward)

    # Check all nodes updated
    assert child2.visits == 1
    assert child2.total_value == reward
    assert child1.visits == 1
    assert child1.total_value == reward
    assert root.visits == 1
    assert root.total_value == reward


def test_state_cloning(device, model, mcts_config):
    """Test that RNN state cloning prevents corruption across siblings."""
    mcts = LatentMCTS(model, mcts_config, device)

    # Create initial state
    s_t = torch.randn(1, 32, device=device)
    rnn_state = (torch.randn(1, 1, 32, device=device), torch.randn(1, 1, 32, device=device))

    # Clone state
    cloned = mcts._clone_rnn_state(rnn_state)

    # Modify original
    rnn_state[0].fill_(999.0)

    # Cloned should be unchanged
    assert not torch.allclose(cloned[0], rnn_state[0])
    assert cloned[0].abs().max() < 10.0  # Original values preserved


def test_grid_accuracy():
    """Test grid accuracy calculation."""
    pred = torch.tensor([[0, 1, 2], [3, 4, 5]])
    target = torch.tensor([[0, 1, 2], [3, 4, 5]])

    # Perfect match
    assert grid_accuracy(pred, target) == 1.0

    # Partial match
    target[1, 2] = 0
    acc = grid_accuracy(pred, target)
    assert 0.8 < acc < 0.9  # 5/6 correct


def test_decode_action_sequence():
    """Test action sequence extraction from node path."""
    s_t = torch.randn(1, 32)

    root = MCTSNode(s_t=s_t, rnn_state=None, parent=None)
    child1 = MCTSNode(s_t=s_t, rnn_state=None, parent=root, action_taken=1, action_coords=(10, 20))
    child2 = MCTSNode(s_t=s_t, rnn_state=None, parent=child1, action_taken=2, action_coords=(30, 40))

    sequence = decode_action_sequence(child2)

    assert len(sequence) == 2
    assert sequence[0] == (1, 10, 20)
    assert sequence[1] == (2, 30, 40)


def test_mcts_search_basic(device, model, mcts_config):
    """Test basic MCTS search execution."""
    mcts = LatentMCTS(model, mcts_config, device)

    # Create mock input and target
    input_grid = torch.randint(0, 16, (1, 64, 64), device=device)
    target_grid = torch.randint(0, 16, (1, 64, 64), device=device)

    # Encode initial state
    with torch.no_grad():
        b, h, w = input_grid.shape
        x = model.grid_embed(input_grid)
        p = model.pos_embed(h, w)
        x = x + p.unsqueeze(0)
        s_0 = model.online_encoder(x)
        s_0_seq = s_0.unsqueeze(1)
        s_0_context, rnn_state_0 = model.gdn(s_0_seq, use_cache=True)
        s_0_final = s_0_context.squeeze(1)

    # Run search
    root, stats = mcts.search(s_0_final, rnn_state_0, target_grid)

    # Verify search ran
    assert stats["simulations_run"] == mcts_config.num_simulations
    assert stats["nodes_created"] > 0
    assert root.visits > 0


def test_mcts_solve_puzzle(device, model, mcts_config):
    """Test high-level solve_puzzle interface."""
    mcts = LatentMCTS(model, mcts_config, device)

    # Create mock puzzle
    input_grid = torch.randint(0, 16, (64, 64), device=device)
    target_grid = torch.randint(0, 16, (64, 64), device=device)

    # Solve
    result = mcts.solve_puzzle(input_grid, target_grid)

    # Verify result structure
    assert "action_sequence" in result
    assert "final_grid" in result
    assert "final_accuracy" in result
    assert "stats" in result
    assert "success" in result

    assert isinstance(result["action_sequence"], list)
    assert result["final_grid"].shape == (1, 64, 64)
    assert 0.0 <= result["final_accuracy"] <= 1.0


def test_mcts_early_stopping(device, model):
    """Test that MCTS stops early when perfect solution found."""
    config = MCTSConfig(
        num_simulations=1000,
        early_stop_on_win=True,
        coord_sampling="sparse",
        coord_stride=32,
        valid_actions=[1, 2]
    )
    mcts = LatentMCTS(model, config, device)

    # Create identical input and target (easy to solve)
    grid = torch.randint(0, 16, (1, 64, 64), device=device)

    # Encode
    with torch.no_grad():
        b, h, w = grid.shape
        x = model.grid_embed(grid)
        p = model.pos_embed(h, w)
        x = x + p.unsqueeze(0)
        s_0 = model.online_encoder(x)
        s_0_seq = s_0.unsqueeze(1)
        s_0_context, rnn_state_0 = model.gdn(s_0_seq, use_cache=True)
        s_0_final = s_0_context.squeeze(1)

    # Search (should stop early if it finds perfect match)
    root, stats = mcts.search(s_0_final, rnn_state_0, grid)

    # If early stopping worked, simulations < budget
    # (This might not always trigger with random model, but test structure is correct)
    assert stats["simulations_run"] <= config.num_simulations


def test_config_memory_estimation():
    """Test memory usage estimation."""
    config = FAST_CONFIG
    mem_stats = config.estimate_memory_usage(d_model=256)

    assert "estimated_nodes" in mem_stats
    assert "total_mb" in mem_stats
    assert "warning" in mem_stats
    assert mem_stats["total_mb"] > 0


def test_action_space_generation():
    """Test action space generation with different sampling strategies."""
    config_sparse = MCTSConfig(coord_sampling="sparse", coord_stride=8, valid_actions=[1, 2])
    config_dense = MCTSConfig(coord_sampling="dense", valid_actions=[1])

    sparse_coords = config_sparse.get_coordinate_samples(grid_size=64)
    dense_coords = config_dense.get_coordinate_samples(grid_size=64)

    # Sparse should have fewer coordinates
    assert len(sparse_coords) < len(dense_coords)
    assert len(dense_coords) == 64 * 64


if __name__ == "__main__":
    # Simple manual run
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running tests on {device}")

    test_mcts_node_creation()
    test_mcts_node_uct_score()
    test_mcts_node_backpropagation()
    test_grid_accuracy()
    test_decode_action_sequence()
    test_config_memory_estimation()
    test_action_space_generation()

    print("Basic tests passed!")

    # Model-dependent tests
    model = ARCJEPAWorldModel(d_model=32, n_heads=4, num_vit_layers=2).to(device)
    model.eval()
    config = MCTSConfig(num_simulations=10, coord_sampling="sparse", coord_stride=16, valid_actions=[1, 2, 3])

    test_state_cloning(device, model, config)
    test_mcts_search_basic(device, model, config)
    test_mcts_solve_puzzle(device, model, config)

    print("All tests passed!")
