from pathlib import Path

import arc_agi

from src.env.arc_env import ARCEnvAdapter
from src.search.algorithms.beam import BeamSearchSolver
from src.search.algorithms.heuristic import HeuristicSolver
from src.search.algorithms.portfolio import PortfolioSolver
from src.search.algorithms.random_legal import RandomLegalSolver
from src.search.env_mcts import RealEnvMCTS


def mine_game_worker(args: dict):
    game_id = args["game_id"]
    env_dir = args.get("env_dir")
    arcade = arc_agi.Arcade(environments_dir=env_dir) if env_dir else arc_agi.Arcade()
    adapter = ARCEnvAdapter(arcade, game_id=game_id, save_recording=False)
    algorithm = args.get("algorithm", "random_legal")
    max_steps = int(args.get("max_steps", 100))
    seed = int(args.get("seed", 0))
    out_dir = Path(args["out_dir"])
    exploration_rate = float(args.get("exploration_rate", 0.15))
    uct_simulations = int(args.get("uct_simulations", 200))

    if algorithm == "beam":
        solver = BeamSearchSolver(max_steps=max_steps, seed=seed)
    elif algorithm == "heuristic":
        solver = HeuristicSolver(
            max_steps=max_steps,
            exploration_rate=exploration_rate,
            seed=seed,
        )
    elif algorithm == "env_uct":
        solver = RealEnvMCTS(
            max_steps=max_steps,
            num_simulations=uct_simulations,
            seed=seed,
        )
    elif algorithm == "portfolio":
        solver = PortfolioSolver(
            max_steps=max_steps,
            seed=seed,
            uct_simulations=uct_simulations,
        )
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
