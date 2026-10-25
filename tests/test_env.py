import numpy as np
import pytest

from neuronoid.env import ArkanoidEnv, ScreenState, reward_between


@pytest.fixture(scope="module")
def env(emu):
    e = ArkanoidEnv(emulator=emu, start_state=emu.round1_state)
    yield e
    e.close()  # чужой эмулятор среда не закрывает


def run_episode_prefix(env, seed, steps=80):
    obs, _ = env.reset(seed=seed)
    obs_list, rewards = [obs], []
    for a in ([3, 3, 1, 5, 5, 2, 0, 4] * 20)[:steps]:
        o, r, *_ = env.step(a)
        obs_list.append(o)
        rewards.append(r)
    return obs_list, rewards


def test_spaces(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert env.action_space.n == 6 and info["playing"] and info["score"] == 0


def test_reset_seed_reproducible(env):
    (oa, ra), (ob, rb) = run_episode_prefix(env, 123), run_episode_prefix(env, 123)
    assert all(np.array_equal(x, y) for x, y in zip(oa, ob)) and ra == rb


def test_different_seeds_diverge(env):
    # случайный старт и залипающие действия: при тех же действиях игра идёт по-разному
    runs = [run_episode_prefix(env, s, steps=160)[0][-1] for s in (1, 2, 3)]
    assert not all(np.array_equal(runs[0], r) for r in runs[1:])


def test_score_reward_when_ball_in_play(env):
    env.reset(seed=0)
    total = sum(env.step(1 if i < 3 else 0)[1] for i in range(400))
    assert total != 0


def test_game_over_terminates(env):
    env.reset(seed=0)
    terminated = False
    for _ in range(20_000):
        *_, terminated, truncated, info = env.step(0)
        if terminated:
            break
    assert terminated and not info["playing"]


def test_reward_between():
    def s(sc, lv=2, rd=1):
        return ScreenState(True, sc, rd, lv)

    assert reward_between(s(100), s(None)) == 0
    assert reward_between(s(None), s(12000)) == 0
    assert reward_between(s(12000), s(0)) == 0
    assert reward_between(s(100), s(200)) == pytest.approx(1.0)
    assert reward_between(s(100, lv=2), s(100, lv=1)) == pytest.approx(-1.0)
    assert reward_between(s(100, rd=1), s(100, rd=2)) == pytest.approx(10.0)
    assert reward_between(s(100), s(100 + 6000)) == 0  # неправдоподобный скачок


def test_check_env(env):
    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=True)
