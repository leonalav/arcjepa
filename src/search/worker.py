from pathlib import Path

import arc_agi
from arcengine import GameAction, GameState

from src.env.arc_env import ARCEnvAdapter
from src.search.algorithms.beam import BeamSearchSolver
from src.search.algorithms.portfolio import PortfolioSolver
from src.search.algorithms.random_legal import RandomLegalSolver


def mine_game_worker(args: dict):
    game_id = args["game_id"]
    env_dir = args.get("env_dir")
    arcade = arc_agi.Arcade(environments_dir=env_dir) if env_dir else arc_agi.Arcade()
    adapter = ARCEnvAdapter(
        arcade,
        game_id=game_id,
        save_recording=False,
        game_action_enum=GameAction,
        game_state_enum=GameState,
    )
    algorithm = args.get("algorithm", "random_legal")
    max_steps = int(args.get("max_steps", 100))
    seed = int(args.get("seed", 0))
    out_dir = Path(args["out_dir"])
    if algorithm == "beam":
        solver = BeamSearchSolver(max_steps=max_steps, seed=seed)
    elif algorithm == "portfolio":
        solver = PortfolioSolver(max_steps=max_steps, seed=seed)
    else:
        solver = RandomLegalSolver(max_steps=max_steps, seed=seed)
    result = solver.solve(adapter, out_dir=out_dir, episode_id=args.get("episode_id"))
    return {
        "game_id": game_id,
        "success": result.success,
        "terminal": result.terminal,
        "steps": result.steps,
        "valid_action_rate": result.valid_action_rate,
        "path": str(result.path),
        "final_score": result.final_score,
    }
