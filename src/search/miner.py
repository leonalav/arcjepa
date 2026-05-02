import json
import multiprocessing as mp
import statistics
from dataclasses import dataclass
from pathlib import Path

from .budget import SearchBudget
from .worker import mine_game_worker


@dataclass
class MiningRunSummary:
    episodes: int
    wins: int
    valid_action_rate: float
    manifest_path: Path
    results: list[dict]


class WinMiner:
    def __init__(
        self,
        game_ids: list[str],
        out_dir: str | Path,
        algorithm: str = "random_legal",
        workers: int = 1,
        budget: SearchBudget | None = None,
        seed: int = 0,
        env_dir: str | None = None,
        exploration_rate: float = 0.15,
        uct_simulations: int = 200,
    ):
        self.game_ids = game_ids
        self.out_dir = Path(out_dir)
        self.algorithm = algorithm
        self.workers = max(1, int(workers))
        self.budget = budget or SearchBudget(seed=seed)
        self.seed = seed
        self.env_dir = env_dir
        self.exploration_rate = exploration_rate
        self.uct_simulations = uct_simulations

    def run(self) -> MiningRunSummary:
        jobs = []
        for game_idx, game_id in enumerate(self.game_ids):
            for episode_idx in range(self.budget.episodes_per_game):
                jobs.append({
                    "game_id": game_id,
                    "out_dir": str(self.out_dir),
                    "algorithm": self.algorithm,
                    "max_steps": self.budget.max_steps,
                    "seed": self.seed + game_idx * 100000 + episode_idx,
                    "env_dir": self.env_dir,
                    "episode_id": f"{game_id}-{self.algorithm}-{episode_idx:06d}",
                    "exploration_rate": self.exploration_rate,
                    "uct_simulations": self.uct_simulations,
                })
        if self.workers == 1:
            results = [mine_game_worker(job) for job in jobs]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(self.workers) as pool:
                results = list(pool.map(mine_game_worker, jobs))
        return self.write_manifest(results)

    def write_manifest(self, results: list[dict]) -> MiningRunSummary:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        wins = sum(1 for r in results if r.get("success"))
        valid_rates = [float(r.get("valid_action_rate", 0.0)) for r in results]
        valid_action_rate = statistics.mean(valid_rates) if valid_rates else 0.0
        manifest = {
            "episodes": len(results),
            "wins": wins,
            "win_rate": wins / max(1, len(results)),
            "valid_action_rate": valid_action_rate,
            "algorithm": self.algorithm,
            "budget": self.budget.to_json(),
            "results": results,
            "status": "failed_legality_gate" if valid_action_rate < 0.99 else "ok",
        }
        manifest_path = self.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return MiningRunSummary(len(results), wins, valid_action_rate, manifest_path, results)
