from typing import Any, Optional

import numpy as np

from src.data.arc_schema import (
    action_name,
    action_uses_coordinates,
    available_indices,
    encode_available_actions,
    game_family,
    parse_action_name,
)
from .types import ARCAction, ARCObs, ARCStepResult


class ARCEnvAdapter:
    def __init__(
        self,
        arcade: Any,
        game_id: str,
        save_recording: bool = False,
        game_action_enum: Any | None = None,
        game_state_enum: Any | None = None,
    ):
        self.arcade = arcade
        self.game_id = game_id
        self.save_recording = save_recording
        self.game_action_enum = game_action_enum
        self.game_state_enum = game_state_enum
        self.raw_env = None
        self.current_obs: ARCObs | None = None

    def reset(self) -> ARCObs:
        self.raw_env = self.arcade.make(self.game_id, save_recording=self.save_recording)
        raw_obs = self.raw_env.reset()
        self.current_obs = self._normalize_obs(raw_obs)
        return self.current_obs

    def step(self, action: ARCAction) -> ARCStepResult:
        if self.current_obs is None:
            self.reset()
        assert self.current_obs is not None

        legal_ids = {a.action_id for a in self.current_obs.available_actions}
        if action.action_id not in legal_ids:
            return ARCStepResult(
                obs=self.current_obs,
                action=action,
                reward=0.0,
                valid_action=False,
                invalid_reason="action_not_available",
            )

        try:
            raw_action, data = self.action_to_raw(action)
            if data is None:
                raw_obs = self.raw_env.step(raw_action)
            else:
                raw_obs = self.raw_env.step(raw_action, data=data)
            next_obs = self._normalize_obs(raw_obs)
            reward = next_obs.score - self.current_obs.score
            self.current_obs = next_obs
            return ARCStepResult(obs=next_obs, action=action, reward=reward, valid_action=True)
        except Exception as exc:
            return ARCStepResult(
                obs=self.current_obs,
                action=action,
                reward=0.0,
                valid_action=False,
                invalid_reason="env_step_exception",
                exception=str(exc),
            )

    def available_action_ids(self) -> set[int]:
        if self.current_obs is None:
            return set()
        return {a.action_id for a in self.current_obs.available_actions}

    def action_to_raw(self, action: ARCAction) -> tuple[Any, Optional[dict[str, int]]]:
        name = action_name(action.action_id)
        raw_action = getattr(self.game_action_enum, name, name) if self.game_action_enum is not None else name
        data = {"x": int(action.x), "y": int(action.y)} if action_uses_coordinates(action.action_id) else None
        return raw_action, data

    def raw_to_action(self, raw_action: Any, data: Optional[dict[str, Any]] = None) -> ARCAction:
        data = data or {}
        return ARCAction(parse_action_name(raw_action), int(data.get("x", 0)), int(data.get("y", 0)))

    def replay(self, actions: list[ARCAction]) -> tuple[ARCObs, list[ARCStepResult]]:
        obs = self.reset()
        results = []
        for action in actions:
            result = self.step(action)
            results.append(result)
            obs = result.obs
            if result.obs.terminal or not result.valid_action:
                break
        return obs, results

    def _normalize_obs(self, raw_obs: Any) -> ARCObs:
        grid = self._pad_grid(self._extract_grid(raw_obs))
        raw_available = self._extract_available_actions(raw_obs)
        available_mask = encode_available_actions(raw_available)
        actions = [ARCAction(idx) for idx in available_indices(available_mask)]
        state = self._extract_state(raw_obs)
        terminal, success = self._terminal_success(state, raw_obs)
        score = self._extract_score(raw_obs)
        return ARCObs(
            game_id=self.game_id,
            game_family=game_family(self.game_id),
            grid=grid,
            available_actions=actions,
            state=state,
            terminal=terminal,
            success=success,
            score=score,
            raw=raw_obs,
        )

    def _extract_grid(self, obj: Any):
        if obj is None:
            return np.zeros((1, 1), dtype=np.int64)
        if isinstance(obj, list) and obj and isinstance(obj[0], list):
            return obj
        if hasattr(obj, "grid"):
            return getattr(obj, "grid")
        if hasattr(obj, "frame"):
            return getattr(obj, "frame")
        if hasattr(obj, "board"):
            return getattr(obj, "board")
        if isinstance(obj, dict):
            for key in ("grid", "frame", "board", "observation", "obs", "state"):
                if key in obj:
                    found = self._extract_grid(obj[key])
                    if found is not None:
                        return found
            for value in obj.values():
                found = self._extract_grid(value)
                if found is not None:
                    return found
        return np.zeros((1, 1), dtype=np.int64)

    def _pad_grid(self, grid: Any) -> np.ndarray:
        arr = np.array(grid, dtype=np.int64)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim < 2 or arr.size == 0:
            arr = np.zeros((1, 1), dtype=np.int64)
        h, w = arr.shape
        padded = np.zeros((64, 64), dtype=np.int64)
        padded[:min(h, 64), :min(w, 64)] = arr[:64, :64]
        return padded

    def _extract_available_actions(self, raw_obs: Any):
        for name in ("available_actions", "availableActions", "valid_actions", "validActions"):
            if hasattr(raw_obs, name):
                return getattr(raw_obs, name)
        if isinstance(raw_obs, dict):
            for name in ("available_actions", "availableActions", "valid_actions", "validActions"):
                if name in raw_obs:
                    return raw_obs[name]
            for key in ("observation", "obs", "state"):
                nested = raw_obs.get(key)
                if isinstance(nested, dict):
                    for name in ("available_actions", "availableActions", "valid_actions", "validActions"):
                        if name in nested:
                            return nested[name]
        return None

    def _extract_state(self, raw_obs: Any) -> str:
        state = getattr(raw_obs, "state", None)
        if state is None and isinstance(raw_obs, dict):
            state = raw_obs.get("state") or raw_obs.get("status") or raw_obs.get("game_state")
        if hasattr(state, "name"):
            return str(state.name).upper()
        if state is None:
            return "UNKNOWN"
        return str(state).upper().rsplit(".", 1)[-1]

    def _terminal_success(self, state: str, raw_obs: Any) -> tuple[bool, bool]:
        terminal = bool(getattr(raw_obs, "terminal", False) or getattr(raw_obs, "done", False))
        success = bool(getattr(raw_obs, "success", False) or getattr(raw_obs, "win", False) or getattr(raw_obs, "won", False))
        terminal = terminal or state in {"WIN", "GAME_OVER", "LOSE", "LOSS", "DONE", "TERMINAL"}
        success = success or state == "WIN"
        return terminal, success

    def _extract_score(self, raw_obs: Any) -> float:
        for name in ("score", "reward", "rhae", "RHAE"):
            if hasattr(raw_obs, name):
                try:
                    return float(getattr(raw_obs, name))
                except (TypeError, ValueError):
                    return 0.0
        if isinstance(raw_obs, dict):
            for name in ("score", "reward", "rhae", "RHAE"):
                if name in raw_obs:
                    try:
                        return float(raw_obs[name])
                    except (TypeError, ValueError):
                        return 0.0
        return 0.0
