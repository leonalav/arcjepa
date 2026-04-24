"""
Inference module for ARC-JEPA World Model.
Contains MCTS and other search algorithms for test-time puzzle solving.
"""

from .mcts import LatentMCTS
from .node import MCTSNode
from .config import MCTSConfig
from .utils import grid_accuracy, decode_action_sequence
from .presets import (
    UNSUPERVISED_FAST,
    UNSUPERVISED_BALANCED,
    UNSUPERVISED_THOROUGH,
    SUPERVISED_SHAPED_FAST,
    SUPERVISED_SHAPED_BALANCED,
)

__all__ = [
    'LatentMCTS',
    'MCTSNode',
    'MCTSConfig',
    'grid_accuracy',
    'decode_action_sequence',
    'UNSUPERVISED_FAST',
    'UNSUPERVISED_BALANCED',
    'UNSUPERVISED_THOROUGH',
    'SUPERVISED_SHAPED_FAST',
    'SUPERVISED_SHAPED_BALANCED',
]
