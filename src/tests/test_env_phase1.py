import json
from pathlib import Path

import numpy as np

from src.data.arc_schema import parse_action_name
from src.data.episode_reader import read_episode
from src.data.episode_writer import EpisodeWriter
from src.env.arc_env import ARCEnvAdapter
from src.env.types import ARCAction, ARCObs, ARCStepResult
from src.search.algorithms.random_legal import RandomLegalSolver


class FakeGameAction:
    ACTION1 = "ACTION1"
    ACTION2 = "ACTION2"
    ACTION6 = "ACTION6"
    SUBMIT = "SUBMIT"


class FakeState:
    READY = "READY"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class FakeRawObs:
    def __init__(self, grid, available_actions, state=FakeState.READY, score=0.0):
        self.grid = grid
        self.available_actions = available_actions
        self.state = state
        self.score = score


class FakeEnv:
    def __init__(self):
        self.steps = []
        self.state = FakeState.READY

    def reset(self):
        self.steps = []
        self.state = FakeState.READY
        return FakeRawObs([[0, 1], [0, 0]], [FakeGameAction.ACTION1, FakeGameAction.ACTION6])

    def step(self, action, data=None):
        self.steps.append((action, data))
        if len(self.steps) == 1:
            return FakeRawObs([[0, 1], [2, 0]], [FakeGameAction.SUBMIT], score=0.5)
        self.state = FakeState.WIN
        return FakeRawObs([[0, 1], [2, 0]], [], state=FakeState.WIN, score=1.0)


class FakeArcade:
    def __init__(self):
        self.env = FakeEnv()

    def make(self, game_id, save_recording=False):
        return self.env


def test_env_adapter_normalizes_observation_and_executes_legal_action():
    adapter = ARCEnvAdapter(FakeArcade(), game_id="ls20-abc", game_action_enum=FakeGameAction, game_state_enum=FakeState)

    obs = adapter.reset()

    assert isinstance(obs, ARCObs)
    assert obs.game_id == "ls20-abc"
    assert obs.game_family == "ls20"
    assert obs.grid.shape == (64, 64)
    assert parse_action_name("ACTION1") in [a.action_id for a in obs.available_actions]

    result = adapter.step(ARCAction(parse_action_name("ACTION6"), x=1, y=1))

    assert isinstance(result, ARCStepResult)
    assert result.valid_action
    assert result.obs.score == 0.5
    assert adapter.raw_env.steps[-1] == (FakeGameAction.ACTION6, {"x": 1, "y": 1})


def test_env_adapter_rejects_unavailable_action_before_step():
    adapter = ARCEnvAdapter(FakeArcade(), game_id="ls20-abc", game_action_enum=FakeGameAction, game_state_enum=FakeState)
    adapter.reset()

    result = adapter.step(ARCAction(parse_action_name("ACTION2")))

    assert not result.valid_action
    assert result.invalid_reason == "action_not_available"
    assert adapter.raw_env.steps == []


def test_episode_writer_persists_replayable_transitions(tmp_path):
    obs0 = ARCObs(
        game_id="ls20-abc",
        game_family="ls20",
        grid=np.zeros((64, 64), dtype=np.int64),
        available_actions=[ARCAction(parse_action_name("ACTION1"))],
        state="READY",
        terminal=False,
        success=False,
        score=0.0,
        raw=None,
    )
    obs1 = ARCObs(
        game_id="ls20-abc",
        game_family="ls20",
        grid=np.ones((64, 64), dtype=np.int64),
        available_actions=[],
        state="WIN",
        terminal=True,
        success=True,
        score=1.0,
        raw=None,
    )
    result = ARCStepResult(obs=obs1, action=ARCAction(parse_action_name("ACTION1")), reward=1.0, valid_action=True)

    path = tmp_path / "episode.jsonl"
    writer = EpisodeWriter(path, episode_id="ep1", policy_version="test", search_algorithm="unit", search_budget={"max_steps": 1})
    writer.write_transition(step=0, before=obs0, result=result)
    writer.close()

    rows = read_episode(path)
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "ep1"
    assert rows[0]["success"] is True
    assert rows[0]["action"] == "ACTION1"
    assert rows[0]["grid_after"][0][0] == 1


def test_random_legal_solver_records_verified_win(tmp_path):
    adapter = ARCEnvAdapter(FakeArcade(), game_id="ls20-abc", game_action_enum=FakeGameAction, game_state_enum=FakeState)
    solver = RandomLegalSolver(max_steps=4, seed=0)

    episode = solver.solve(adapter, out_dir=tmp_path, episode_id="ep-win")

    assert episode.success
    assert episode.valid_action_rate == 1.0
    assert episode.path.parent.name.startswith("game_id=")
    rows = read_episode(episode.path)
    assert rows[-1]["success"] is True
    assert rows[-1]["terminal"] is True
