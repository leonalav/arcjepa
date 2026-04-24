"""
# written by leonalav. thanks to claude for the latent feature!
Latent MCTS: Monte Carlo Tree Search in the latent space of ARC-JEPA World Model.

This implementation operates entirely in latent space, using the trained World Model
to predict state transitions without decoding intermediate states. Only leaf nodes
are decoded for evaluation against the target grid.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any
import random

from .node import MCTSNode
from .config import MCTSConfig
from .utils import grid_accuracy


class LatentMCTS:
    """
    Monte Carlo Tree Search in latent space.

    Uses the World Model's predictor and GDN to explore action sequences,
    evaluating only leaf nodes by decoding to pixel space.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MCTSConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Latent MCTS.

        Args:
            model: ARCJEPAWorldModel instance
            config: MCTS configuration
            device: Device to run on (defaults to model's device)
        """
        self.model = model
        self.config = config or MCTSConfig()
        self.device = device or next(model.parameters()).device

        # Extract model components
        self.encoder = model.online_encoder
        self.predictor = model.predictor
        self.gdn = model.gdn
        self.decoder = model.decoder
        self.action_embed = model.action_embed
        self.grid_embed = model.grid_embed
        self.pos_embed = model.pos_embed

        # Statistics
        self.stats = {
            "simulations_run": 0,
            "nodes_created": 0,
            "terminal_nodes_found": 0,
            "max_depth_reached": 0,
        }

    def search(
        self,
        initial_state: torch.Tensor,
        initial_rnn_state: Optional[Any],
        target_grid: torch.Tensor,
        num_simulations: Optional[int] = None
    ) -> Tuple[MCTSNode, Dict[str, Any]]:
        """
        Run MCTS search from initial state.

        Args:
            initial_state: Initial latent state [1, d_model]
            initial_rnn_state: Initial GDN recurrent cache
            target_grid: Target grid to match [1, 64, 64]
            num_simulations: Number of simulations (overrides config)

        Returns:
            (root_node, search_stats)
        """
        num_sims = num_simulations or self.config.num_simulations

        # Create root node
        root = MCTSNode(
            s_t=initial_state,
            rnn_state=initial_rnn_state,
            parent=None,
            action_taken=None,
            action_coords=None
        )

        # Generate valid action-coordinate pairs
        valid_action_coords = self._generate_action_space()

        # Reset statistics
        self.stats = {
            "simulations_run": 0,
            "nodes_created": 1,
            "terminal_nodes_found": 0,
            "max_depth_reached": 0,
        }

        # Run simulations
        with torch.no_grad():
            for sim in range(num_sims):
                # Phase 1: Selection
                node, depth = self._select(root, valid_action_coords)

                # Update max depth
                self.stats["max_depth_reached"] = max(self.stats["max_depth_reached"], depth)

                # Check if we hit max depth
                if depth >= self.config.max_depth:
                    # Evaluate and backpropagate without expansion
                    reward = self._evaluate(node.s_t, target_grid)
                    node.backpropagate(reward)
                    self.stats["simulations_run"] += 1
                    continue

                # Phase 2: Expansion
                if not node.is_terminal:
                    child = self._expand(node, valid_action_coords)
                    if child is not None:
                        self.stats["nodes_created"] += 1
                        node = child  # Evaluate the new child

                # Phase 3: Evaluation
                reward = self._evaluate(node.s_t, target_grid)

                # Check for WIN condition
                if reward >= 1.0 and not node.is_terminal:
                    node.is_terminal = True
                    self.stats["terminal_nodes_found"] += 1

                    if self.config.early_stop_on_win:
                        # Backpropagate and stop search
                        node.backpropagate(reward)
                        self.stats["simulations_run"] = sim + 1
                        break

                # Phase 4: Backpropagation
                node.backpropagate(reward)

                self.stats["simulations_run"] += 1

                # Memory management: prune if needed
                if self.config.enable_pruning and self.stats["nodes_created"] > self.config.max_tree_nodes:
                    self._prune_tree(root)

        return root, self.stats

    def _select(self, root: MCTSNode, valid_action_coords: list) -> Tuple[MCTSNode, int]:
        """
        Phase 1: Selection using UCT.

        Traverse down the tree selecting children with highest UCT score
        until we reach a node that is not fully expanded or is terminal.

        Args:
            root: Root node to start selection from
            valid_action_coords: List of valid (action, x, y) tuples

        Returns:
            (selected_node, depth)
        """
        node = root
        depth = 0

        while not node.is_terminal and node.is_fully_expanded(valid_action_coords):
            if not node.children:
                break
            node = node.select_best_child(self.config.c_puct)
            depth += 1

        return node, depth

    def _expand(self, node: MCTSNode, valid_action_coords: list) -> Optional[MCTSNode]:
        """
        Phase 2: Expansion.

        Add one new child to the node by trying an unexplored action.

        Args:
            node: Node to expand from
            valid_action_coords: List of valid (action, x, y) tuples

        Returns:
            Newly created child node, or None if no actions available
        """
        untried = node.get_untried_actions(valid_action_coords)

        if not untried:
            return None

        # Pick a random untried action
        action, x, y = random.choice(untried)

        # Predict next state
        s_next, rnn_next = self._predict_next_state(
            node.s_t,
            node.rnn_state,
            action,
            x,
            y
        )

        # Create child node
        child = node.add_child(action, (x, y), s_next, rnn_next)

        return child

    def _evaluate(self, s_t: torch.Tensor, target_grid: torch.Tensor) -> float:
        """
        Phase 3: Evaluation.

        Decode latent state and compare against target grid.

        Args:
            s_t: Latent state [1, d_model]
            target_grid: Target grid [1, 64, 64]

        Returns:
            Reward in [0.0, 1.0]
        """
        # Decode latent state to grid
        logits = self.decoder(s_t)  # [1, 16, 64, 64]
        pred_grid = logits.argmax(dim=1)  # [1, 64, 64]

        # Calculate accuracy
        if self.config.reward_shaping == "binary":
            reward = grid_accuracy(pred_grid, target_grid)
        else:
            # Continuous reward shaping could be added here
            reward = grid_accuracy(pred_grid, target_grid)

        return reward

    def _predict_next_state(
        self,
        s_t: torch.Tensor,
        rnn_state: Optional[Any],
        action: int,
        x: int,
        y: int
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Predict next latent state given current state and action.

        Args:
            s_t: Current latent state [1, d_model]
            rnn_state: Current GDN recurrent cache
            action: Action index (1-7)
            x: X coordinate
            y: Y coordinate

        Returns:
            (s_next, rnn_state_next)
        """
        # CRITICAL: Clone rnn_state to prevent in-place corruption
        rnn_state_cloned = self._clone_rnn_state(rnn_state)

        # Embed action
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        x_tensor = torch.tensor([x], dtype=torch.long, device=self.device)
        y_tensor = torch.tensor([y], dtype=torch.long, device=self.device)

        z_a = self.action_embed(action_tensor, x_tensor, y_tensor)  # [1, d_model]

        # Predict next latent
        s_next_pred = self.predictor(s_t, z_a)  # [1, d_model]

        # Update GDN cache
        s_next_seq = s_next_pred.unsqueeze(1)  # [1, 1, d_model]
        s_next_context, rnn_state_next = self.gdn(
            s_next_seq,
            state=rnn_state_cloned,
            use_cache=True
        )

        s_next = s_next_context.squeeze(1)  # [1, d_model]

        return s_next, rnn_state_next

    def _clone_rnn_state(self, rnn_state: Optional[Any]) -> Optional[Any]:
        """
        Safely clone GDN recurrent state to prevent in-place corruption.

        Args:
            rnn_state: GDN recurrent cache (tuple, tensor, or None)

        Returns:
            Cloned state
        """
        if rnn_state is None:
            return None
        elif isinstance(rnn_state, tuple):
            return tuple(
                t.clone() if isinstance(t, torch.Tensor) else t
                for t in rnn_state
            )
        elif isinstance(rnn_state, torch.Tensor):
            return rnn_state.clone()
        else:
            # Unknown type, return as-is (risky but better than crashing)
            return rnn_state

    def _generate_action_space(self) -> List[Tuple[int, int, int]]:
        """
        Generate list of valid (action, x, y) tuples to consider.

        Returns:
            List of (action, x, y) tuples
        """
        coords = self.config.get_coordinate_samples()
        action_space = []

        for action in self.config.valid_actions:
            for x, y in coords:
                action_space.append((action, x, y))

        return action_space

    def _prune_tree(self, root: MCTSNode):
        """
        Prune low-value branches to save memory.

        Args:
            root: Root node of the tree
        """
        # Simple pruning: remove children with Q < threshold
        def _prune_recursive(node: MCTSNode):
            if not node.children:
                return

            # Prune children
            to_remove = []
            for key, child in node.children.items():
                if child.q_value() < self.config.pruning_threshold:
                    to_remove.append(key)

            for key in to_remove:
                del node.children[key]

            # Recurse on remaining children
            for child in node.children.values():
                _prune_recursive(child)

        _prune_recursive(root)

    def get_best_action_sequence(self, root: MCTSNode) -> List[Tuple[int, int, int]]:
        """
        Extract best action sequence from search tree.

        Follows the path with highest visit counts.

        Args:
            root: Root node of the search tree

        Returns:
            List of (action, x, y) tuples
        """
        sequence = []
        node = root

        while node.children:
            best_action = node.get_best_action()
            if best_action is None:
                break

            sequence.append(best_action)
            node = node.children[best_action]

        return sequence

    def solve_puzzle(
        self,
        input_grid: torch.Tensor,
        target_grid: torch.Tensor,
        num_simulations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        High-level interface to solve an ARC puzzle using MCTS.

        Args:
            input_grid: Input grid [1, 64, 64] or [64, 64]
            target_grid: Target grid [1, 64, 64] or [64, 64]
            num_simulations: Number of MCTS simulations

        Returns:
            Dictionary with solution and statistics
        """
        # Ensure correct shapes
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)
        if target_grid.dim() == 2:
            target_grid = target_grid.unsqueeze(0)

        input_grid = input_grid.to(self.device)
        target_grid = target_grid.to(self.device)

        # Encode initial state
        with torch.no_grad():
            # Embed grid
            b, h, w = input_grid.shape
            x = self.grid_embed(input_grid.view(b, h, w))  # [1, 64, 64, d_model]
            p = self.pos_embed(h, w)  # [64, 64, d_model]
            x = x + p.unsqueeze(0)

            # Encode to latent
            s_0 = self.encoder(x)  # [1, d_model]

            # Initialize GDN cache
            s_0_seq = s_0.unsqueeze(1)  # [1, 1, d_model]
            s_0_context, rnn_state_0 = self.gdn(s_0_seq, use_cache=True)
            s_0_final = s_0_context.squeeze(1)  # [1, d_model]

        # Run MCTS search
        root, stats = self.search(s_0_final, rnn_state_0, target_grid, num_simulations)

        # Extract best action sequence
        action_sequence = self.get_best_action_sequence(root)

        # Get final predicted grid
        best_node = root
        for action in action_sequence:
            if action in best_node.children:
                best_node = best_node.children[action]

        with torch.no_grad():
            final_logits = self.decoder(best_node.s_t)
            final_grid = final_logits.argmax(dim=1)

        # Calculate final accuracy
        final_accuracy = grid_accuracy(final_grid, target_grid)

        return {
            "action_sequence": action_sequence,
            "final_grid": final_grid.cpu(),
            "final_accuracy": final_accuracy,
            "root_node": root,
            "best_node": best_node,
            "stats": stats,
            "success": final_accuracy >= 1.0,
        }
