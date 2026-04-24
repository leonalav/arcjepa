"""
Unsupervised MCTS configuration presets for test-time inference.
"""

from .config import MCTSConfig

# Test-time inference configs (no target grid required)

UNSUPERVISED_FAST = MCTSConfig(
    num_simulations=1000,
    c_puct=1.0,
    max_depth=10,
    coord_sampling="heuristic",
    evaluation_mode="unsupervised",
    reward_shaping="unsupervised",
    early_stop_on_win=False,  # No ground truth to check
)

UNSUPERVISED_BALANCED = MCTSConfig(
    num_simulations=5000,
    c_puct=1.4,
    max_depth=20,
    coord_sampling="heuristic",
    evaluation_mode="unsupervised",
    reward_shaping="unsupervised",
    early_stop_on_win=False,
)

UNSUPERVISED_THOROUGH = MCTSConfig(
    num_simulations=10000,
    c_puct=2.0,
    max_depth=30,
    coord_sampling="heuristic",
    evaluation_mode="unsupervised",
    reward_shaping="unsupervised",
    enable_pruning=True,
    early_stop_on_win=False,
)

# Supervised configs with shaped rewards (for validation/training)

SUPERVISED_SHAPED_FAST = MCTSConfig(
    num_simulations=1000,
    c_puct=1.0,
    max_depth=10,
    coord_sampling="sparse",
    coord_stride=8,
    evaluation_mode="supervised",
    reward_shaping="shaped",
)

SUPERVISED_SHAPED_BALANCED = MCTSConfig(
    num_simulations=5000,
    c_puct=1.4,
    max_depth=20,
    coord_sampling="sparse",
    coord_stride=4,
    evaluation_mode="supervised",
    reward_shaping="shaped",
)
