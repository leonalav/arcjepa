"""
Configuration for MCTS search algorithm.
"""

from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class MCTSConfig:
    """Configuration for Latent MCTS search."""

    # Search budget
    num_simulations: int = 5000

    # UCT exploration constant (higher = more exploration)
    c_puct: float = 1.4

    # Maximum rollout depth (prevents infinite loops)
    max_depth: int = 20

    # Valid action indices to consider (1-7 are game actions, 8 is SUBMIT)
    valid_actions: List[int] = None

    # Coordinate sampling strategy
    coord_sampling: str = "sparse"  # "sparse", "dense", or "heuristic"
    coord_stride: int = 4  # For sparse sampling: sample every Nth pixel

    # Memory management
    enable_pruning: bool = False  # Prune low-value branches
    pruning_threshold: float = 0.01  # Prune nodes with Q < threshold
    max_tree_nodes: int = 50000  # Maximum nodes before forced pruning

    # Early stopping
    early_stop_on_win: bool = True  # Stop search if perfect match found

    # Evaluation
    reward_shaping: str = "binary"  # "binary", "shaped", or "unsupervised"
    evaluation_mode: str = "supervised"  # "supervised" or "unsupervised"

    def __post_init__(self):
        if self.valid_actions is None:
            # Default: all 7 game actions (exclude SUBMIT=8 during search)
            self.valid_actions = [1, 2, 3, 4, 5, 6, 7]

    def get_coordinate_samples(self, grid_size: int = 64, current_grid: Optional[Any] = None) -> List[tuple]:
        """
        Generate coordinate samples based on sampling strategy.

        Args:
            grid_size: Size of the grid
            current_grid: Optional current grid state for heuristic sampling

        Returns:
            List of (x, y) tuples to consider for actions.
        """
        if self.coord_sampling == "dense":
            # All coordinates (expensive!)
            return [(x, y) for x in range(grid_size) for y in range(grid_size)]

        elif self.coord_sampling == "sparse":
            # Sample every Nth coordinate
            coords = []
            for x in range(0, grid_size, self.coord_stride):
                for y in range(0, grid_size, self.coord_stride):
                    coords.append((x, y))
            return coords

        elif self.coord_sampling == "heuristic":
            # Content-aware sampling (requires current_grid)
            if current_grid is None:
                # Safe fallback to sparse if no grid provided for heuristics
                coords = []
                for x in range(0, grid_size, self.coord_stride):
                    for y in range(0, grid_size, self.coord_stride):
                        coords.append((x, y))
                return coords

            # Import here to avoid circular dependency
            from .grid_analysis import find_edges, find_frontier, find_symmetry_points

            coords = set()

            # Object boundaries
            coords.update(find_edges(current_grid))

            # Frontier cells
            coords.update(find_frontier(current_grid))

            # Symmetry points
            coords.update(find_symmetry_points(current_grid))

            # Always include corners and center
            coords.update([
                (0, 0), (0, grid_size-1),
                (grid_size-1, 0), (grid_size-1, grid_size-1),
                (grid_size//2, grid_size//2)
            ])

            return list(coords)

        else:
            raise ValueError(f"Unknown coord_sampling: {self.coord_sampling}")

    def estimate_memory_usage(self, d_model: int = 256) -> dict:
        """
        Estimate VRAM usage for MCTS search.

        Args:
            d_model: Model dimension

        Returns:
            Dictionary with memory estimates in MB
        """
        # Per-node memory:
        # - s_t: [1, d_model] float32 = d_model * 4 bytes
        # - rnn_state: varies, assume ~2x d_model for GDN cache
        # - Python overhead: ~200 bytes per node

        bytes_per_node = (d_model * 4) + (d_model * 8) + 200

        # Estimate tree size based on branching factor
        num_coords = len(self.get_coordinate_samples())
        branching_factor = len(self.valid_actions) * num_coords

        # Approximate tree nodes (not all branches fully expanded)
        estimated_nodes = min(self.num_simulations, self.max_tree_nodes)

        total_bytes = estimated_nodes * bytes_per_node
        total_mb = total_bytes / (1024 * 1024)

        return {
            "estimated_nodes": estimated_nodes,
            "bytes_per_node": bytes_per_node,
            "total_mb": round(total_mb, 2),
            "branching_factor": branching_factor,
            "warning": "high" if total_mb > 1000 else "medium" if total_mb > 500 else "low"
        }


# Preset configurations for different use cases

FAST_CONFIG = MCTSConfig(
    num_simulations=1000,
    c_puct=1.0,
    max_depth=10,
    coord_sampling="sparse",
    coord_stride=8,
)

BALANCED_CONFIG = MCTSConfig(
    num_simulations=5000,
    c_puct=1.4,
    max_depth=20,
    coord_sampling="sparse",
    coord_stride=4,
)

THOROUGH_CONFIG = MCTSConfig(
    num_simulations=10000,
    c_puct=2.0,
    max_depth=30,
    coord_sampling="sparse",
    coord_stride=2,
    enable_pruning=True,
)
