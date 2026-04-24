"""
Tests for unsupervised MCTS and shaped rewards.
"""

import torch
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.world_model import ARCJEPAWorldModel
from src.inference import LatentMCTS, MCTSConfig
from src.inference.presets import (
    UNSUPERVISED_FAST,
    UNSUPERVISED_BALANCED,
    SUPERVISED_SHAPED_FAST,
)
from src.inference.grid_analysis import (
    check_symmetry,
    check_completion,
    check_consistency,
    detect_objects,
    find_edges,
    find_frontier,
)


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def model(device):
    model = ARCJEPAWorldModel(d_model=32, n_heads=4, num_vit_layers=2).to(device)
    model.eval()
    return model


def test_unsupervised_evaluation(device, model):
    """Test unsupervised evaluation without target grid."""
    config = UNSUPERVISED_FAST
    config.num_simulations = 5
    mcts = LatentMCTS(model, config, device)

    input_grid = torch.randint(0, 16, (64, 64), device=device)

    # Should work without target
    result = mcts.solve_puzzle(input_grid, target_grid=None, mode="unsupervised")

    assert result["mode"] == "unsupervised"
    assert result["final_accuracy"] is None  # No target to compare
    assert result["final_grid"] is not None
    assert len(result["action_sequence"]) >= 0


def test_supervised_shaped_rewards(device, model):
    """Test supervised mode with shaped rewards."""
    config = SUPERVISED_SHAPED_FAST
    config.num_simulations = 5
    mcts = LatentMCTS(model, config, device)

    input_grid = torch.randint(0, 16, (64, 64), device=device)
    target_grid = torch.randint(0, 16, (64, 64), device=device)

    result = mcts.solve_puzzle(input_grid, target_grid, mode="supervised")

    assert result["mode"] == "supervised"
    assert result["final_accuracy"] is not None
    assert 0.0 <= result["final_accuracy"] <= 1.0


def test_mode_validation():
    """Test that modes validate inputs correctly."""
    device = torch.device('cpu')
    model = ARCJEPAWorldModel(d_model=32).to(device)
    model.eval()

    config = MCTSConfig(evaluation_mode="supervised", num_simulations=5)
    mcts = LatentMCTS(model, config, device)

    input_grid = torch.randint(0, 16, (64, 64))

    # Supervised mode without target should fail
    with pytest.raises(ValueError, match="Supervised mode requires target_grid"):
        mcts.solve_puzzle(input_grid, target_grid=None)


def test_heuristic_coordinate_sampling(device):
    """Test heuristic sampling finds relevant coordinates."""
    # Create a grid with clear structure
    grid = torch.zeros(64, 64, dtype=torch.long, device=device)
    grid[10:20, 10:20] = 1  # Square object
    grid[30:40, 30:40] = 2  # Another square

    config = MCTSConfig(coord_sampling="heuristic")
    coords = config.get_coordinate_samples(grid_size=64, current_grid=grid)

    # Should find edges and frontiers
    assert len(coords) > 0
    assert len(coords) < 64 * 64  # Should be selective

    # Check that some coordinates are near objects
    coord_set = set(coords)
    assert (10, 10) in coord_set or (11, 11) in coord_set  # Near first object


def test_symmetry_detection():
    """Test symmetry detection."""
    # Perfectly symmetric grid
    grid = torch.zeros(64, 64, dtype=torch.long)
    grid[20:40, 20:40] = 1
    sym_score = check_symmetry(grid)
    assert sym_score > 0.9

    # Asymmetric grid
    grid[10, 10] = 2
    sym_score = check_symmetry(grid)
    assert sym_score < 1.0


def test_completion_detection():
    """Test completion heuristic."""
    # Clean grid with few colors
    grid = torch.zeros(64, 64, dtype=torch.long)
    grid[10:20, 10:20] = 1
    grid[30:40, 30:40] = 2
    completion = check_completion(grid)
    assert completion > 0.5

    # Noisy grid with many isolated pixels
    noisy_grid = torch.randint(0, 16, (64, 64))
    noisy_completion = check_completion(noisy_grid)
    assert noisy_completion < completion


def test_consistency_check():
    """Test consistency between input and prediction."""
    input_grid = torch.zeros(64, 64, dtype=torch.long)
    input_grid[10:20, 10:20] = 1

    # Prediction preserves input
    pred_grid = input_grid.clone()
    pred_grid[30:40, 30:40] = 2
    consistency = check_consistency(pred_grid, input_grid)
    assert consistency > 0.7

    # Prediction destroys input
    bad_pred = torch.zeros(64, 64, dtype=torch.long)
    bad_pred[50:60, 50:60] = 3
    bad_consistency = check_consistency(bad_pred, input_grid)
    assert bad_consistency < consistency


def test_object_detection():
    """Test connected component detection."""
    grid = torch.zeros(64, 64, dtype=torch.long)
    grid[10:20, 10:20] = 1  # Object 1
    grid[30:40, 30:40] = 2  # Object 2
    grid[50, 50] = 3  # Object 3 (single pixel)

    objects = detect_objects(grid)
    assert len(objects) == 3


def test_edge_detection():
    """Test edge pixel detection."""
    grid = torch.zeros(64, 64, dtype=torch.long)
    grid[10:20, 10:20] = 1

    edges = find_edges(grid)
    assert len(edges) > 0

    # Edges should be on the boundary
    edge_set = set(edges)
    assert (10, 10) in edge_set  # Corner
    assert (15, 15) not in edge_set  # Interior


def test_frontier_detection():
    """Test frontier cell detection."""
    grid = torch.zeros(64, 64, dtype=torch.long)
    grid[10:20, 10:20] = 1

    frontier = find_frontier(grid)
    assert len(frontier) > 0

    # Frontier should be adjacent to object
    frontier_set = set(frontier)
    assert (9, 10) in frontier_set or (10, 9) in frontier_set


def test_mode_switching():
    """Test switching between supervised and unsupervised modes."""
    device = torch.device('cpu')
    model = ARCJEPAWorldModel(d_model=32).to(device)
    model.eval()

    config = MCTSConfig(evaluation_mode="supervised", num_simulations=5)
    mcts = LatentMCTS(model, config, device)

    input_grid = torch.randint(0, 16, (64, 64))
    target_grid = torch.randint(0, 16, (64, 64))

    # Override to unsupervised
    result = mcts.solve_puzzle(input_grid, target_grid=None, mode="unsupervised")
    assert result["mode"] == "unsupervised"

    # Config should be restored
    assert mcts.config.evaluation_mode == "supervised"


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running tests on {device}")

    # Grid analysis tests
    test_symmetry_detection()
    test_completion_detection()
    test_consistency_check()
    test_object_detection()
    test_edge_detection()
    test_frontier_detection()
    print("✓ Grid analysis tests passed")

    # Model-dependent tests
    model = ARCJEPAWorldModel(d_model=32, n_heads=4, num_vit_layers=2).to(device)
    model.eval()

    test_unsupervised_evaluation(device, model)
    test_supervised_shaped_rewards(device, model)
    test_mode_validation()
    test_heuristic_coordinate_sampling(device)
    test_mode_switching()

    print("✓ All unsupervised MCTS tests passed!")
