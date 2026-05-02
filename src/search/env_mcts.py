"""
RealEnvMCTS: proper UCT (Upper Confidence Trees) search in the live ARC environment.

Algorithm (AlphaZero-style PUCT, no world model required):
  Selection   — traverse tree via PUCT until a leaf or unexpanded node.
  Expansion   — add one child, priors from HeuristicAgent.propose().
  Simulation  — replay the prefix from env root, then HeuristicSolver rollout.
  Backprop    — propagate terminal score up the path.

On a terminal WIN, the winning action prefix is replayed once more and written
to disk as a verified episode via EpisodeWriter.

Q1 decision: replay full prefixes (safe, env-agnostic).
Q2 decision: UCT is a fallback — PortfolioSolver tries cheap passes first.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.arc_schema import action_uses_coordinates, available_indices
from src.data.episode_writer import EpisodeWriter, move_episode_to_outcome
from src.env.types import ARCAction, ARCObs
from src.search.node import EnvMCTSNode
from src.search.results import EpisodeResult

# Lazy import: avoids importing HeuristicPolicy at module load if unused
_HeuristicPolicy = None


def _get_heuristic_policy(exploration_rate: float = 0.0):
    global _HeuristicPolicy
    if _HeuristicPolicy is None:
        from src.data.heuristic_policy import ARCHeuristicPolicy
        _HeuristicPolicy = ARCHeuristicPolicy
    return _HeuristicPolicy(exploration_rate=exploration_rate)


class RealEnvMCTS:
    """
    UCT search entirely in the real ARC environment.

    Args:
        max_steps:        Maximum depth of any rollout / episode.
        num_simulations:  MCTS simulation budget per episode.
        c_puct:           Exploration constant (AlphaZero default 1.4).
        rollout_steps:    Steps in the random/heuristic rollout from leaf.
        seed:             RNG seed.
        policy_version:   Written into JSONL for data-lineage tracking.
    """

    def __init__(
        self,
        max_steps: int = 100,
        num_simulations: int = 200,
        c_puct: float = 1.4,
        rollout_steps: int = 10,
        seed: int = 0,
        policy_version: str = "env_uct_v1",
    ):
        self.max_steps = max_steps
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.rollout_steps = rollout_steps
        self.seed = seed
        self.policy_version = policy_version
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public interface (mirrors HeuristicSolver.solve / RandomLegalSolver.solve)
    # ------------------------------------------------------------------

    def solve(
        self,
        adapter,
        out_dir: str | Path,
        episode_id: Optional[str] = None,
        agent=None,
    ) -> EpisodeResult:
        out_dir = Path(out_dir)
        obs = adapter.reset()
        episode_id = episode_id or (
            f"{obs.game_id}-env_uct-{self._rng.randrange(10**12):012d}"
        )

        # Build the search tree
        winning_prefix = self._search(adapter, obs)

        # Replay the best prefix as a real recorded episode
        return self._replay_as_episode(
            adapter, winning_prefix, out_dir, episode_id, obs.game_id
        )

    # ------------------------------------------------------------------
    # UCT Search
    # ------------------------------------------------------------------

    def _search(self, adapter, root_obs: ARCObs) -> tuple[ARCAction, ...]:
        """Run MCTS and return the best action prefix found."""
        policy = _get_heuristic_policy(exploration_rate=0.0)
        root = EnvMCTSNode(prefix=(), parent=None, prior=1.0)
        best_win: tuple[ARCAction, ...] | None = None

        for _ in range(self.num_simulations):
            # --- Phase 1: Selection + Expansion ---
            path, node, obs = self._select_and_expand(root, root_obs, adapter, policy)

            # --- Phase 2: Simulation (rollout from leaf) ---
            score, success, sim_prefix = self._rollout(adapter, obs)

            # Early exit: record a real win
            if success and best_win is None:
                best_win = node.prefix + sim_prefix

            # --- Phase 3: Backpropagation ---
            self._backpropagate(path, score)

        if best_win is not None:
            return best_win

        # No win found — return greedy best prefix by Q-value
        return self._greedy_prefix(root)

    def _select_and_expand(
        self,
        root: EnvMCTSNode,
        root_obs: ARCObs,
        adapter,
        policy,
    ) -> tuple[list[EnvMCTSNode], EnvMCTSNode, ARCObs]:
        """Selection + Expansion. Returns (path, leaf_node, obs_at_leaf)."""
        node = root
        path = [node]

        # Replay prefix to get the obs at this node
        obs = self._replay_prefix(adapter, node.prefix)

        # --- Selection: follow PUCT until unexpanded or terminal ---
        while node.children and not node.terminal:
            # Only descend into children we haven't exceeded step budget for
            if len(node.prefix) >= self.max_steps:
                break
            node = node.select_child(self.c_puct)
            path.append(node)
            obs = self._replay_prefix(adapter, node.prefix)
            if obs.terminal:
                node.terminal = True
                node.success = obs.success
                break

        # --- Expansion: add one new child if node is not terminal ---
        if not node.terminal and len(node.prefix) < self.max_steps and obs.available_actions:
            action = self._select_expand_action(obs, node, policy)
            if action is not None:
                prior = self._action_prior(obs, action, policy)
                child = node.add_child(action, prior=prior)
                path.append(child)
                node = child
                obs = self._replay_prefix(adapter, node.prefix)
                if obs.terminal:
                    node.terminal = True
                    node.success = obs.success

        return path, node, obs

    def _select_expand_action(
        self,
        obs: ARCObs,
        node: EnvMCTSNode,
        policy,
    ) -> Optional[ARCAction]:
        """Pick an unexplored action for expansion. Prefer heuristic ordering."""
        explored_keys = set(node.children.keys())
        candidates = []

        available_names = [a.name for a in obs.available_actions]
        heuristic_action_enum, hx, hy = policy.select_action(
            obs.grid,
            available_actions=available_names,
            game_id=obs.game_id,
        )
        h_idx = policy._action_idx(heuristic_action_enum)

        # Build list: heuristic preferred first, then random shuffle of rest
        preferred = []
        rest = []
        for a in obs.available_actions:
            if a.action_id == 0:
                continue
            if action_uses_coordinates(a.action_id):
                # Sample a few coordinate points rather than the full grid
                coords = self._sample_coords(obs.grid, n=4)
                for x, y in coords:
                    arc_action = ARCAction(a.action_id, x, y)
                    if arc_action.key() in explored_keys:
                        continue
                    if a.action_id == h_idx:
                        preferred.append(arc_action)
                    else:
                        rest.append(arc_action)
            else:
                arc_action = ARCAction(a.action_id)
                if arc_action.key() in explored_keys:
                    continue
                if a.action_id == h_idx:
                    preferred.append(arc_action)
                else:
                    rest.append(arc_action)

        candidates = preferred + self._rng.sample(rest, len(rest))
        return candidates[0] if candidates else None

    def _action_prior(self, obs: ARCObs, action: ARCAction, policy) -> float:
        """Assign a prior probability to an action using heuristic guidance."""
        available_names = [a.name for a in obs.available_actions]
        h_enum, hx, hy = policy.select_action(
            obs.grid,
            available_actions=available_names,
            game_id=obs.game_id,
        )
        h_idx = policy._action_idx(h_enum)
        # Heuristic top pick gets high prior; coordinate match boosts further
        if action.action_id == h_idx:
            if action_uses_coordinates(action.action_id) and (action.x == hx and action.y == hy):
                return 0.8
            return 0.5
        return 0.1

    def _rollout(
        self, adapter, obs: ARCObs
    ) -> tuple[float, bool, tuple[ARCAction, ...]]:
        """Lightweight rollout from leaf: heuristic + random for `rollout_steps`."""
        from src.data.heuristic_policy import ARCHeuristicPolicy
        policy = ARCHeuristicPolicy(exploration_rate=0.3)
        actions_taken: list[ARCAction] = []

        for _ in range(self.rollout_steps):
            if obs.terminal or not obs.available_actions:
                break
            available_names = [a.name for a in obs.available_actions]
            action_enum, x, y = policy.select_action(
                obs.grid,
                available_actions=available_names,
                game_id=obs.game_id,
            )
            action_idx = policy._action_idx(action_enum)
            legal_ids = {a.action_id for a in obs.available_actions}
            if action_idx not in legal_ids:
                candidates = [a for a in obs.available_actions if a.action_id > 0]
                if not candidates:
                    break
                action_idx = self._rng.choice(candidates).action_id
                x, y = 0, 0
            if not action_uses_coordinates(action_idx):
                x, y = 0, 0
            arc_action = ARCAction(action_idx, int(x), int(y))
            result = adapter.step(arc_action)
            if result.valid_action:
                actions_taken.append(arc_action)
            obs = result.obs
            if obs.terminal:
                break

        return float(obs.score), bool(obs.success), tuple(actions_taken)

    @staticmethod
    def _backpropagate(path: list[EnvMCTSNode], value: float):
        for node in reversed(path):
            node.backpropagate(value)

    def _greedy_prefix(self, root: EnvMCTSNode) -> tuple[ARCAction, ...]:
        """Follow the highest-Q path in the tree."""
        node = root
        while node.children:
            node = max(node.children.values(), key=lambda c: c.q_value())
        return node.prefix

    # ------------------------------------------------------------------
    # Replay helpers
    # ------------------------------------------------------------------

    def _replay_prefix(self, adapter, prefix: tuple[ARCAction, ...]) -> ARCObs:
        """Reset the env and replay a sequence of actions. Returns final obs."""
        obs = adapter.reset()
        for action in prefix:
            result = adapter.step(action)
            obs = result.obs
            if obs.terminal:
                break
        return obs

    def _replay_as_episode(
        self,
        adapter,
        prefix: tuple[ARCAction, ...],
        out_dir: Path,
        episode_id: str,
        game_id: str,
    ) -> EpisodeResult:
        """Replay prefix in a fresh env, writing all transitions to JSONL."""
        temp_dir = out_dir / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{episode_id}.jsonl"

        obs = adapter.reset()
        valid_actions = 0
        invalid_actions = 0
        action_sequence = []
        terminal = obs.terminal
        success = obs.success
        final_score = obs.score

        with EpisodeWriter(
            temp_path,
            episode_id=episode_id,
            policy_version=self.policy_version,
            search_algorithm="env_uct",
            search_budget={
                "max_steps": self.max_steps,
                "num_simulations": self.num_simulations,
            },
        ) as writer:
            for step, action in enumerate(prefix):
                if not obs.available_actions or obs.terminal:
                    break
                before = obs
                result = adapter.step(action)
                writer.write_transition(step=step, before=before, result=result)
                action_sequence.append(action)
                if result.valid_action:
                    valid_actions += 1
                else:
                    invalid_actions += 1
                obs = result.obs
                terminal = obs.terminal
                success = obs.success
                final_score = obs.score
                if obs.terminal:
                    break

        outcome = "wins" if success else "failed" if terminal else "partial"
        final_path = move_episode_to_outcome(
            temp_path, out_dir, outcome, game_id, episode_id
        )
        total = valid_actions + invalid_actions
        return EpisodeResult(
            episode_id=episode_id,
            game_id=game_id,
            success=success,
            terminal=terminal,
            steps=len(action_sequence),
            valid_actions=valid_actions,
            invalid_actions=invalid_actions,
            valid_action_rate=valid_actions / max(1, total),
            path=final_path,
            action_sequence=action_sequence,
            final_score=final_score,
        )

    # ------------------------------------------------------------------
    # Coordinate sampling
    # ------------------------------------------------------------------

    def _sample_coords(self, grid: np.ndarray, n: int = 4) -> list[tuple[int, int]]:
        """
        Sample n (x, y) coordinates, biased toward foreground cells.
        Falls back to uniform random if grid is empty.
        """
        active = (grid != 0).nonzero()
        coords = []
        if len(active[0]) > 0:
            indices = self._np_rng.choice(len(active[0]), size=min(n, len(active[0])), replace=False)
            coords = [(int(active[1][i]), int(active[0][i])) for i in indices]
        while len(coords) < n:
            coords.append((int(self._rng.randrange(64)), int(self._rng.randrange(64))))
        return coords[:n]

    # Legacy search() method (kept for eval_agent.py compatibility)
    def search(self, adapter, agent=None, budget=None):
        obs = adapter.reset()
        return obs, []

    def best_action_sequence(self, root: EnvMCTSNode) -> list[tuple]:
        sequence = []
        node = root
        while node.children:
            node = max(node.children.values(), key=lambda c: c.visits)
            if node.prefix:
                sequence.append(node.prefix[-1])
        return sequence
