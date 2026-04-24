"""
Example script demonstrating MCTS inference for ARC puzzle solving.

This script shows how to:
1. Load a trained World Model checkpoint
2. Load an ARC test puzzle
3. Run MCTS to search for a solution
4. Visualize the search process and results
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.world_model import ARCJEPAWorldModel
from src.inference.mcts import LatentMCTS
from src.inference.config import MCTSConfig, FAST_CONFIG, BALANCED_CONFIG, THOROUGH_CONFIG
from src.inference.utils import (
    visualize_search_tree,
    compute_rollout_statistics,
    format_action_sequence
)


def load_model(checkpoint_path: str, device: torch.device) -> ARCJEPAWorldModel:
    """
    Load trained World Model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Create model with same architecture as training
    model = ARCJEPAWorldModel(
        d_model=256,
        n_heads=8,
        num_vit_layers=4,
        num_gdn_heads=4
    ).to(device)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model (for demonstration only)")

    model.eval()
    return model


def load_puzzle(puzzle_path: str, device: torch.device) -> tuple:
    """
    Load an ARC puzzle from file.

    Args:
        puzzle_path: Path to puzzle file (numpy .npy or text format)
        device: Device to load tensors on

    Returns:
        (input_grid, target_grid) as torch tensors
    """
    # For demonstration, create a simple mock puzzle
    # In practice, load from actual ARC dataset
    print(f"Loading puzzle from {puzzle_path}")

    # Mock puzzle: simple pattern completion
    input_grid = torch.zeros(64, 64, dtype=torch.long, device=device)
    target_grid = torch.zeros(64, 64, dtype=torch.long, device=device)

    # Create a simple pattern
    input_grid[10:20, 10:20] = 1
    target_grid[10:20, 10:20] = 1
    target_grid[30:40, 30:40] = 2

    return input_grid, target_grid


def visualize_grids(input_grid: torch.Tensor, target_grid: torch.Tensor, predicted_grid: torch.Tensor):
    """
    Print ASCII visualization of grids.

    Args:
        input_grid: Input grid [H, W]
        target_grid: Target grid [H, W]
        predicted_grid: Predicted grid [H, W]
    """
    def grid_to_str(grid, size=16):
        """Convert grid to ASCII representation."""
        g = grid[:size, :size].cpu().numpy()
        chars = '.123456789ABCDEF'
        lines = []
        for row in g:
            lines.append(''.join(chars[min(val, 15)] for val in row))
        return '\n'.join(lines)

    print("\n=== Input Grid (16x16 preview) ===")
    print(grid_to_str(input_grid))

    print("\n=== Target Grid (16x16 preview) ===")
    print(grid_to_str(target_grid))

    print("\n=== Predicted Grid (16x16 preview) ===")
    print(grid_to_str(predicted_grid))


def main():
    parser = argparse.ArgumentParser(description="Run MCTS inference on ARC puzzle")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_latest.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--puzzle",
        type=str,
        default="data/test_puzzle.npy",
        help="Path to puzzle file"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["fast", "balanced", "thorough"],
        default="balanced",
        help="MCTS configuration preset"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="Number of MCTS simulations (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--visualize-tree",
        action="store_true",
        help="Print search tree visualization"
    )

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Select config
    config_map = {
        "fast": FAST_CONFIG,
        "balanced": BALANCED_CONFIG,
        "thorough": THOROUGH_CONFIG
    }
    config = config_map[args.config]

    if args.simulations is not None:
        config.num_simulations = args.simulations

    print(f"\nMCTS Configuration:")
    print(f"  Simulations: {config.num_simulations}")
    print(f"  c_puct: {config.c_puct}")
    print(f"  Max depth: {config.max_depth}")
    print(f"  Coord sampling: {config.coord_sampling} (stride={config.coord_stride})")

    # Estimate memory usage
    mem_stats = config.estimate_memory_usage(d_model=256)
    print(f"\nEstimated VRAM usage: {mem_stats['total_mb']} MB")
    print(f"  Estimated nodes: {mem_stats['estimated_nodes']}")
    print(f"  Warning level: {mem_stats['warning']}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, device)

    # Load puzzle
    print("\nLoading puzzle...")
    input_grid, target_grid = load_puzzle(args.puzzle, device)

    # Create MCTS solver
    print("\nInitializing MCTS...")
    mcts = LatentMCTS(model, config, device)

    # Solve puzzle
    print(f"\nRunning MCTS search ({config.num_simulations} simulations)...")
    print("This may take a few minutes depending on configuration...")

    with torch.no_grad():
        result = mcts.solve_puzzle(input_grid, target_grid)

    # Display results
    print("\n" + "="*60)
    print("SEARCH RESULTS")
    print("="*60)

    print(f"\nSuccess: {result['success']}")
    print(f"Final Accuracy: {result['final_accuracy']:.4f}")

    print(f"\nSearch Statistics:")
    stats = result['stats']
    print(f"  Simulations run: {stats['simulations_run']}")
    print(f"  Nodes created: {stats['nodes_created']}")
    print(f"  Terminal nodes found: {stats['terminal_nodes_found']}")
    print(f"  Max depth reached: {stats['max_depth_reached']}")

    print(f"\nAction Sequence ({len(result['action_sequence'])} steps):")
    print(format_action_sequence(result['action_sequence']))

    # Visualize grids
    visualize_grids(input_grid, target_grid, result['final_grid'].squeeze(0))

    # Tree statistics
    tree_stats = compute_rollout_statistics(result['root_node'])
    print(f"\nTree Statistics:")
    print(f"  Total nodes: {tree_stats['total_nodes']}")
    print(f"  Max depth: {tree_stats['max_depth']}")
    print(f"  Terminal nodes: {tree_stats['terminal_nodes']}")
    print(f"  Root Q-value: {tree_stats['root_q']:.4f}")

    # Visualize tree
    if args.visualize_tree:
        print("\n" + "="*60)
        print("SEARCH TREE VISUALIZATION")
        print("="*60)
        print(visualize_search_tree(result['root_node'], max_depth=5, top_k=3))

    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()
