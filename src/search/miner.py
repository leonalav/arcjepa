import json
import multiprocessing as mp
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budget import SearchBudget
from .worker import mine_game_worker


@dataclass
class MiningRunSummary:
    episodes: int
    wins: int
    valid_action_rate: float
    manifest_path: Path
    results: list[dict]


# ---------------------------------------------------------------------------
# Progress bar helpers
# ---------------------------------------------------------------------------

def _game_label(game_id: str, width: int = 16) -> str:
    """Return a fixed-width label using the game family prefix (e.g. 'm0r0')."""
    return game_id[:width].ljust(width)


def _bar_desc(game_id: str, counters: dict) -> str:
    """Compact description: 'ls20  ✓12 ~3 ✗85'."""
    label = _game_label(game_id, width=16)
    w = counters["W"]
    p = counters["P"]
    f = counters["F"]
    return f"{label}  \033[32m✓{w:<4d}\033[0m\033[33m~{p:<3d}\033[0m\033[31m✗{f:<4d}\033[0m"


def _classify(result: dict) -> str:
    """Classify a result as 'W', 'P', or 'F'."""
    if result.get("success"):
        return "W"
    if result.get("terminal"):
        return "F"
    return "P"


# ---------------------------------------------------------------------------
# WinMiner
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> MiningRunSummary:
        jobs = self._build_jobs()
        try:
            from tqdm import tqdm as _tqdm  # noqa: F401 — just probe availability
            results = self._run_with_progress(jobs)
        except ImportError:
            results = self._run_plain(jobs)
        return self.write_manifest(results)

    # ------------------------------------------------------------------
    # Job builder
    # ------------------------------------------------------------------

    def _build_jobs(self) -> list[dict[str, Any]]:
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
        return jobs

    # ------------------------------------------------------------------
    # Execution: tqdm-enabled
    # ------------------------------------------------------------------

    def _run_with_progress(self, jobs: list[dict]) -> list[dict]:
        from tqdm import tqdm

        eps = self.budget.episodes_per_game
        n_games = len(self.game_ids)

        # One counter dict per game_id
        counters: dict[str, dict[str, int]] = {
            gid: {"W": 0, "P": 0, "F": 0} for gid in self.game_ids
        }

        # One tqdm bar per game, stacked vertically via position=
        # leave=True keeps finished bars visible; dynamic_ncols avoids wrapping
        bars: dict[str, tqdm] = {
            gid: tqdm(
                total=eps,
                position=i,
                leave=True,
                dynamic_ncols=True,
                bar_format=(
                    "{desc}: {bar} {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
                desc=_bar_desc(gid, counters[gid]),
                colour=None,
            )
            for i, gid in enumerate(self.game_ids)
        }

        results: list[dict] = []

        def _handle(result: dict):
            gid = result["game_id"]
            outcome = _classify(result)
            counters[gid][outcome] += 1
            bars[gid].set_description(_bar_desc(gid, counters[gid]))
            bars[gid].update(1)
            results.append(result)

        try:
            if self.workers == 1:
                for job in jobs:
                    _handle(mine_game_worker(job))
            else:
                ctx = mp.get_context("spawn")
                with ctx.Pool(self.workers) as pool:
                    for result in pool.imap_unordered(mine_game_worker, jobs):
                        _handle(result)
        finally:
            for bar in bars.values():
                bar.close()
            # Print a clean summary line below all bars
            total_w = sum(c["W"] for c in counters.values())
            total_p = sum(c["P"] for c in counters.values())
            total_f = sum(c["F"] for c in counters.values())
            total   = total_w + total_p + total_f
            tqdm.write(
                f"\n── Mining complete: "
                f"\033[32m✓ {total_w} wins\033[0m  "
                f"\033[33m~ {total_p} partial\033[0m  "
                f"\033[31m✗ {total_f} failed\033[0m  "
                f"({total} episodes, "
                f"{total_w / max(1, total) * 100:.1f}% win rate)"
            )

        return results

    # ------------------------------------------------------------------
    # Execution: plain fallback (no tqdm)
    # ------------------------------------------------------------------

    def _run_plain(self, jobs: list[dict]) -> list[dict]:
        if self.workers == 1:
            return [mine_game_worker(job) for job in jobs]
        ctx = mp.get_context("spawn")
        with ctx.Pool(self.workers) as pool:
            return list(pool.map(mine_game_worker, jobs))

    # ------------------------------------------------------------------
    # Manifest writer
    # ------------------------------------------------------------------

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
