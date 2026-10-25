import numpy as np
import pytest

from neuronoid.env import ArkanoidEnv
from neuronoid.evaluate import GameResult, Report, evaluate


@pytest.fixture(scope="module")
def eval_env(emu):
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, life_loss_terminal=False)
    yield env
    env.close()


def g(score, rnd):
    return GameResult(score, rnd, 100)


def test_milestone_rules():
    rep = Report.from_games([g(1600, 1)] * 7 + [g(100, 1)] * 3)
    assert rep.m1 and not rep.m2 and not rep.m3
    rep = Report.from_games([g(5000, 2)] * 6 + [g(100, 1)] * 4)
    assert not rep.m2
    rep = Report.from_games([g(30000, 5)] * 7 + [g(100, 1)] * 3)
    assert rep.m1 and rep.m2 and rep.m3


def test_evaluate_random_policy(eval_env):
    rng = np.random.default_rng(0)
    rep = evaluate(lambda obs: int(rng.integers(6)), eval_env, games=2)
    assert len(rep.games) == 2 and all(r.round == 1 and r.score >= 0 for r in rep.games)


def test_evaluate_deterministic_by_seed(eval_env):
    a = evaluate(lambda obs: 0, eval_env, games=2, seed=5).to_json()
    b = evaluate(lambda obs: 0, eval_env, games=2, seed=5).to_json()
    assert a == b
