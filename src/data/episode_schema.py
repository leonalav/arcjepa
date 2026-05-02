import hashlib
import json
import time
import uuid
from typing import Any

import numpy as np

from src.data.arc_schema import action_name, game_family, parse_action_name
from src.env.types import ARCAction, ARCObs, ARCStepResult


def episode_id_for(game_id: str, policy_version: str, algorithm: str, seed: int | None = None) -> str:
    suffix = seed if seed is not None else uuid.uuid4().hex[:12]
    return f"{game_id}-{policy_version}-{algorithm}-{suffix}"


def action_to_json(action: ARCAction) -> dict[str, Any]:
    return {"action": action_name(action.action_id), "x": int(action.x), "y": int(action.y)}


def obs_to_json(obs: ARCObs) -> dict[str, Any]:
    return {
        "game_id": obs.game_id,
        "game_family": obs.game_family,
        "grid": _json_grid(obs.grid),
        "available_actions": [action_name(a.action_id) for a in obs.available_actions],
        "state": obs.state,
        "terminal": bool(obs.terminal),
        "success": bool(obs.success),
        "score": float(obs.score),
    }


def transition_to_row(
    *,
    episode_id: str,
    policy_version: str,
    search_algorithm: str,
    search_budget: dict[str, Any],
    step: int,
    before: ARCObs,
    result: ARCStepResult,
    **extras: Any,
) -> dict[str, Any]:
    action_payload = action_to_json(result.action)
    row = {
        "episode_id": episode_id,
        "game_id": before.game_id,
        "game_family": before.game_family or game_family(before.game_id),
        "policy_version": policy_version,
        "search_algorithm": search_algorithm,
        "search_budget": search_budget,
        "step": int(step),
        "grid_before": _json_grid(before.grid),
        "available_actions": [action_name(a.action_id) for a in before.available_actions],
        "action": action_payload["action"],
        "x": action_payload["x"],
        "y": action_payload["y"],
        "grid_after": _json_grid(result.obs.grid),
        "reward": float(result.reward),
        "score": float(result.obs.score),
        "terminal": bool(result.obs.terminal),
        "success": bool(result.obs.success),
        "invalid_action": not bool(result.valid_action),
        "invalid_reason": result.invalid_reason,
        "exception": result.exception,
        "created_at": time.time(),
    }
    row.update(extras)
    return row


def row_action(row: dict[str, Any]) -> ARCAction:
    return ARCAction(parse_action_name(row.get("action")), int(row.get("x", 0)), int(row.get("y", 0)))


def episode_success(rows: list[dict[str, Any]]) -> bool:
    return bool(rows and rows[-1].get("success", False))


def action_sequence_hash(rows: list[dict[str, Any]]) -> str:
    actions = [(row.get("action"), int(row.get("x", 0)), int(row.get("y", 0))) for row in rows]
    return hashlib.blake2b(json.dumps(actions, sort_keys=True).encode("utf-8"), digest_size=16).hexdigest()


def final_grid_hash(rows: list[dict[str, Any]]) -> str:
    final_grid = rows[-1].get("grid_after", []) if rows else []
    return hashlib.blake2b(json.dumps(final_grid, sort_keys=True).encode("utf-8"), digest_size=16).hexdigest()


def _json_grid(grid: Any) -> list[list[int]]:
    arr = np.array(grid, dtype=np.int64)
    return arr.tolist()
