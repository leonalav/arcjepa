from dataclasses import dataclass
from typing import Any, List, Optional

from src.data.arc_schema import DEFAULT_NUM_ACTIONS, action_uses_coordinates, default_available_actions


@dataclass
class MCTSConfig:
    num_simulations: int = 5000
    c_puct: float = 1.4
    max_depth: int = 20
    valid_actions: Optional[List[int]] = None
    allow_submit: bool = True
    num_actions: int = DEFAULT_NUM_ACTIONS
    coord_sampling: str = "sparse"
    coord_stride: int = 4
    enable_pruning: bool = False
    pruning_threshold: float = 0.01
    max_tree_nodes: int = 50000
    early_stop_on_win: bool = True
    reward_shaping: str = "binary"
    evaluation_mode: str = "supervised"
    use_value_head: bool = True

    def __post_init__(self):
        if self.valid_actions is None:
            mask = default_available_actions(self.num_actions)
            self.valid_actions = mask.nonzero(as_tuple=False).flatten().tolist()
            if not self.allow_submit and self.num_actions > 0:
                self.valid_actions = [a for a in self.valid_actions if a != self.num_actions - 1]

    def actions_from_mask(self, available_actions_mask: Optional[Any] = None) -> List[int]:
        if available_actions_mask is None:
            return list(self.valid_actions)
        if hasattr(available_actions_mask, 'nonzero'):
            valid = available_actions_mask.bool().nonzero(as_tuple=False).flatten().tolist()
        else:
            valid = [i for i, enabled in enumerate(available_actions_mask) if enabled]
        return [int(a) for a in valid if int(a) in self.valid_actions or self.valid_actions is None]

    def get_coordinate_samples(self, grid_size: int = 64, current_grid: Optional[Any] = None) -> List[tuple]:
        if self.coord_sampling == "dense":
            return [(x, y) for x in range(grid_size) for y in range(grid_size)]
        if self.coord_sampling == "sparse":
            return [(x, y) for x in range(0, grid_size, self.coord_stride) for y in range(0, grid_size, self.coord_stride)]
        if self.coord_sampling == "heuristic":
            if current_grid is None:
                return [(x, y) for x in range(0, grid_size, self.coord_stride) for y in range(0, grid_size, self.coord_stride)]
            from .grid_analysis import find_edges, find_frontier, find_symmetry_points
            coords = set()
            coords.update(find_edges(current_grid))
            coords.update(find_frontier(current_grid))
            coords.update(find_symmetry_points(current_grid))
            coords.update([(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1), (grid_size//2, grid_size//2)])
            return list(coords)
        raise ValueError(f"Unknown coord_sampling: {self.coord_sampling}")

    def action_space(self, available_actions_mask: Optional[Any] = None, current_grid: Optional[Any] = None) -> List[tuple]:
        coords = self.get_coordinate_samples(current_grid=current_grid)
        action_space = []
        for action in self.actions_from_mask(available_actions_mask):
            if action_uses_coordinates(action):
                action_space.extend((action, x, y) for x, y in coords)
            else:
                action_space.append((action, 0, 0))
        return action_space

    def estimate_memory_usage(self, d_model: int = 256) -> dict:
        bytes_per_node = (d_model * 4) + (d_model * 8) + 200
        branching_factor = len(self.action_space())
        estimated_nodes = min(self.num_simulations, self.max_tree_nodes)
        total_mb = estimated_nodes * bytes_per_node / (1024 * 1024)
        return {
            "estimated_nodes": estimated_nodes,
            "bytes_per_node": bytes_per_node,
            "total_mb": round(total_mb, 2),
            "branching_factor": branching_factor,
            "warning": "high" if total_mb > 1000 else "medium" if total_mb > 500 else "low"
        }


FAST_CONFIG = MCTSConfig(num_simulations=1000, c_puct=1.0, max_depth=10, coord_sampling="sparse", coord_stride=8)
BALANCED_CONFIG = MCTSConfig(num_simulations=5000, c_puct=1.4, max_depth=20, coord_sampling="sparse", coord_stride=4)
THOROUGH_CONFIG = MCTSConfig(num_simulations=10000, c_puct=2.0, max_depth=30, coord_sampling="sparse", coord_stride=2, enable_pruning=True)
