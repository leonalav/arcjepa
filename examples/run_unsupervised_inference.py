"""
Example: Test-time inference with unsupervised MCTS.
Demonstrates solving ARC puzzles without target grid.
"""

import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.world_model import ARCJEPAWorldModel
from src.inference import LatentMCTS, UNSUPERVISED_BALANCED
from src.inference.utils import visualize_search_tree, compute_rollout_statistics


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained World Model."""
    model = ARCJEPAWorldModel(d_model=256, n_heads=8, num_vit_layers=4).to(device)

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found, using random initialization")

    model.eval()
    return model


def create_test_puzzle(device):
    """Create a simple test puzzle."""
    # Input: small square
    input_grid = torch.zeros(64, 64, dtype=torch.long, device=device)
    input_grid[20:30, 20:30] = 1

    return input_grid


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load model
    print("Loading model...")
    model = load_model("checkpoints/model_latest.pt", device)

    # Create MCTS solver (unsupervised mode)
    print("\nInitializing unsupervised MCTS...")
    config = UNSUPERVISED_BALANCED
    print(f"  Simulations: {config.num_simulations}")
    print(f"  Mode: {config.evaluation_mode}")
    print(f"  Sampling: {config.coord_sampling}")

    mcts = LatentMCTS(model, config, device)

    # Create test puzzle
    print("\nCreating test puzzle...")
    input_grid = create_test_puzzle(device)

    # Solve without target grid
    print("\nRunning MCTS search (no target grid)...")
    print("This demonstrates TRUE test-time inference.\n")

    with torch.no_grad():
        result = mcts.solve_puzzle(
            input_grid,
            target_grid=None,  # ← No target!
            mode="unsupervised"
        )

    # Display results
    print("="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nMode: {result['mode']}")
    print(f"Actions taken: {len(result['action_sequence'])}")

    print(f"\nSearch statistics:")
    stats = result['stats']
    print(f"  Simulations: {stats['simulations_run']}")
    print(f"  Nodes created: {stats['nodes_created']}")
    print(f"  Max depth: {stats['max_depth_reached']}")

    print(f"\nAction sequence:")
    for i, (action, x, y) in enumerate(result['action_sequence'][:10]):
        print(f"  {i+1}. ACTION{action} at ({x}, {y})")
    if len(result['action_sequence']) > 10:
        print(f"  ... and {len(result['action_sequence']) - 10} more")

    # Show predicted grid
    pred_grid = result['final_grid'].squeeze(0)
    print(f"\nPredicted grid (16x16 preview):")
    print_grid(pred_grid[:16, :16])

    # Tree statistics
    tree_stats = compute_rollout_statistics(result['root_node'])
    print(f"\nTree statistics:")
    print(f"  Total nodes: {tree_stats['total_nodes']}")
    print(f"  Tree depth: {tree_stats['max_depth']}")
    print(f"  Root Q-value: {tree_stats['root_q']:.4f}")

    print("\n" + "="*60)
    print("✓ Unsupervised inference complete!")
    print("\nNote: Without ground truth, we cannot verify correctness.")
    print("The solution is based on heuristic quality metrics:")
    print("  - Symmetry detection")
    print("  - Completion heuristic")
    print("  - Consistency with input")


def print_grid(grid, size=16):
    """Print ASCII representation of grid."""
    chars = '.123456789ABCDEF'
    g = grid[:size, :size].cpu().numpy()
    for row in g:
        print('  ' + ''.join(chars[min(val, 15)] for val in row))


if __name__ == "__main__":
    main()
