import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from stable_baselines3.common.vec_env import SubprocVecEnv

from neuronoid.live import LiveReader
from neuronoid.vec import make_env_fn, make_vec_env

TESTS = Path(__file__).parent


@pytest.fixture
def tmp_live_name(monkeypatch):
    name = f"neuronoid_test_{uuid4().hex[:8]}"
    monkeypatch.setenv("NEURONOID_LIVE_NAME", name)
    return name


def test_vec_env_shapes():
    venv = make_vec_env(2, seed=0, live=False)
    obs = venv.reset()
    assert obs.shape == (2, 4, 93, 96) and obs.dtype == np.uint8
    obs, rew, done, infos = venv.step(np.array([1, 3]))
    assert rew.shape == (2,) and "known_score" in infos[0]
    venv.close()


def test_only_rank0_publishes(tmp_live_name):
    venv = make_vec_env(2, seed=0, live=True)
    venv.reset()
    venv.step(np.array([0, 0]))
    reader = LiveReader()
    frame, info, n = reader.read()
    assert n == 2  # reset + один step одной среды; две среды дали бы 4
    reader.close()
    venv.close()


def test_child_crash_surfaces(monkeypatch):
    # spawn-процессу нужен путь к tests/, чтобы импортировать _crash_env
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]))
    from _crash_env import crashing_env_fn

    venv = SubprocVecEnv([make_env_fn(0, 0, True, None), crashing_env_fn], start_method="spawn")
    venv.reset()
    with pytest.raises((EOFError, BrokenPipeError, RuntimeError, ConnectionResetError)):
        for _ in range(5):
            venv.step(np.array([0, 0]))
