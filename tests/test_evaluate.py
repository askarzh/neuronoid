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


def test_evaluate_stochastic_policy_reproducible_with_seed_hook(eval_env):
    # случайная политика воспроизводится, если её генератор пересевается в начале каждой игры
    state = {}

    def on_game_start(game_seed):
        state["rng"] = np.random.default_rng(game_seed)

    def policy(obs):
        return int(state["rng"].integers(6))

    a = evaluate(policy, eval_env, games=2, seed=7, on_game_start=on_game_start).to_json()
    b = evaluate(policy, eval_env, games=2, seed=7, on_game_start=on_game_start).to_json()
    assert a == b


def test_report_has_bricks_left(eval_env):
    rep = evaluate(lambda obs: 0, eval_env, games=1, seed=3)
    assert rep.games[0].bricks_left is not None and 0 <= rep.games[0].bricks_left <= 66


def test_env_kwargs_from_config(tmp_path):
    from neuronoid.evaluate import env_kwargs_from_config

    assert env_kwargs_from_config(tmp_path) == {}  # старый прогон без config.json
    (tmp_path / "config.json").write_text('{"n_envs": 10}')
    assert env_kwargs_from_config(tmp_path) == {}
    (tmp_path / "config.json").write_text('{"action_mode": "target", "obs_mode": "wide", "sticky_prob": 0.1}')
    assert env_kwargs_from_config(tmp_path) == {"action_mode": "target", "obs_mode": "wide", "sticky_prob": 0.1}


def test_classify_round1_exit():
    from neuronoid.evaluate import classify_round1_exit

    assert classify_round1_exit(1) and classify_round1_exit(0)
    assert not classify_round1_exit(60) and not classify_round1_exit(None)
