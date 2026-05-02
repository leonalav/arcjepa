import json
import sqlite3
from pathlib import Path

from src.data.episode_reader import iter_episodes, read_episode
from src.data.episode_schema import action_sequence_hash, final_grid_hash


def connect(index_path: str | Path):
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            game_id TEXT NOT NULL,
            game_family TEXT NOT NULL,
            outcome TEXT NOT NULL,
            success INTEGER NOT NULL,
            terminal INTEGER NOT NULL,
            steps INTEGER NOT NULL,
            score REAL NOT NULL,
            policy_version TEXT,
            search_algorithm TEXT,
            search_budget_json TEXT,
            action_hash TEXT,
            final_grid_hash TEXT,
            created_at REAL
        )
    """)
    return conn


def upsert_episode(conn, path: str | Path):
    path = Path(path)
    rows = read_episode(path)
    if not rows:
        return
    first, last = rows[0], rows[-1]
    outcome = "wins" if last.get("success") else "failed" if last.get("terminal") else "partial"
    conn.execute(
        """
        INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            first.get("episode_id"),
            str(path),
            first.get("game_id"),
            first.get("game_family", "unknown"),
            outcome,
            int(bool(last.get("success"))),
            int(bool(last.get("terminal"))),
            len(rows),
            float(last.get("score", 0.0)),
            first.get("policy_version"),
            first.get("search_algorithm"),
            json.dumps(first.get("search_budget", {})),
            action_sequence_hash(rows),
            final_grid_hash(rows),
            first.get("created_at"),
        ),
    )


def build_index(raw_root: str | Path, index_path: str | Path, overwrite: bool = False):
    index_path = Path(index_path)
    if overwrite and index_path.exists():
        index_path.unlink()
    conn = connect(index_path)
    for path in iter_episodes(raw_root):
        upsert_episode(conn, path)
    conn.commit()
    return conn


def query_episodes(index_path: str | Path, mode: str = "pretrain") -> list[dict]:
    conn = connect(index_path)
    if mode == "expert_only":
        rows = conn.execute("SELECT * FROM episodes WHERE success=1").fetchall()
    elif mode == "mixed":
        rows = conn.execute("SELECT * FROM episodes WHERE success=1 OR terminal=1").fetchall()
    else:
        rows = conn.execute("SELECT * FROM episodes").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM episodes LIMIT 0").description]
    return [dict(zip(cols, row)) for row in rows]
