from pathlib import Path

import numpy as np
import pytest

from neuronoid.env import ArkanoidEnv, ScreenState, reward_between, update_known


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


def test_known_score_is_monotonic():
    # ложное завышенное чтение не должно оплачиваться дважды
    known, total = ScreenState(True, 100, 1, 2), 0.0
    for sc in (300, 200, 300):
        cur = ScreenState(True, sc, 1, 2)
        total += reward_between(known, cur)
        known = update_known(known, cur)
    assert total == pytest.approx(2.0)


def _force_frames(env, monkeypatch, frame):
    def fake_advance(mouse):
        env._frame = frame
        return np.zeros((93, 96), np.uint8)

    monkeypatch.setattr(env, "_advance", fake_advance)


def test_blank_screen_does_not_end_episode(env, monkeypatch):
    # переход между раундами может гасить поле: это не Game Over
    env.reset(seed=0)
    _force_frames(env, monkeypatch, np.zeros((200, 320, 3), np.uint8))
    assert not any(env.step(0)[2] for _ in range(50))


def test_title_screen_ends_episode(env, monkeypatch):
    from PIL import Image
    from pathlib import Path

    env.reset(seed=0)
    title = np.asarray(Image.open(Path(__file__).parent / "data/frames/idle_0099_not.png").convert("RGB"))
    _force_frames(env, monkeypatch, title)
    assert env.step(0)[2]


def test_known_score_ignores_implausible_jump():
    # ложный скачок не должен «съесть» награду за следующие настоящие очки
    known = ScreenState(True, 100, 1, 2)
    known = update_known(known, ScreenState(True, 100 + 6000, 1, 2))
    assert known.score == 100
    assert reward_between(known, ScreenState(True, 200, 1, 2)) == pytest.approx(1.0)


def test_info_reports_points_gained_since_reset(env):
    obs, info = env.reset(seed=0)
    assert info["gained_score"] == 0
    for i in range(400):
        *_, info = env.step(1 if i < 3 else 0)
        if info["known_score"] and info["known_score"] > 0:
            break
    start = info["known_score"] - info["gained_score"]
    assert start >= 0 and info["gained_score"] == info["known_score"] - start


def test_confirmed_score_jump_becomes_new_baseline():
    # выход «B» даёт настоящий бонус > MAX_SCORE_JUMP: после 3 устойчивых чтений это новая база
    from neuronoid.env import JumpConfirm

    j = JumpConfirm()
    assert j.update(1220, 7930) is None
    assert j.update(1220, 7930) is None
    assert j.update(1220, 7950) == 7950


def test_single_score_spike_is_not_confirmed():
    from neuronoid.env import JumpConfirm

    j = JumpConfirm()
    assert j.update(1220, 7930) is None
    assert j.update(1220, 1300) is None  # обычное чтение сбрасывает подозрение
    assert j.update(1300, 7930) is None
    assert j.update(1300, 7930) is None
    assert j.update(1300, 1400) is None


def test_life_loss_episode_continues_same_game(emu):
    # как EpisodicLife в Atari: потеря жизни — конец эпизода для обучения,
    # но следующий reset продолжает ту же игру, а не грузит свежий раунд 1
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, life_loss_terminal=True)
    _, info = env.reset(seed=0)
    lives0 = info["lives"]
    for _ in range(5000):
        *_, terminated, truncated, info = env.step(0)
        if terminated:
            break
    assert terminated and info["known_lives"] == lives0 - 1
    score_at_loss = info["known_score"]
    _, info = env.reset(seed=1)
    assert info["known_lives"] == lives0 - 1 and info["known_score"] == score_at_loss
    assert info["gained_score"] == 0


def test_game_over_resets_to_round1_in_life_loss_mode(emu):
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, life_loss_terminal=True)
    env.reset(seed=0)
    for _ in range(20):  # три жизни и Game Over
        for _ in range(5000):
            *_, terminated, truncated, info = env.step(0)
            if terminated:
                break
        if not info["playing"] or info["known_lives"] == 0 and terminated and not info["playing"]:
            break
        env.reset()
    _, info = env.reset(seed=2)
    assert info["lives"] == 2 and info["score"] == 0  # свежая игра


from neuronoid.curriculum import StartMeta, StartPool  # noqa: E402


def test_env_offers_snapshots(emu, tmp_path):
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, start_pool=tmp_path,
                      curriculum_prob=0.0, snapshot_every=5, snapshot_confirm_steps=1)
    env.reset(seed=0)
    for i in range(60):
        env.step(1 if i < 3 else 0)
    assert StartPool(tmp_path).stats().get(1, 0) >= 1


def test_reset_from_pool_continues_that_position(emu, tmp_path):
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, start_pool=tmp_path,
                      curriculum_prob=0.0, snapshot_every=5, snapshot_confirm_steps=1)
    env.reset(seed=0)
    for i in range(200):
        *_, info = env.step(1 if i < 3 else 0)
        if info["known_score"]:
            break
    env.curriculum_prob = 1.0
    obs, info = env.reset(seed=1)
    assert info["curriculum_start"] and info["gained_score"] == 0
    assert info["known_score"] is not None
    _, r, *_ = env.step(0)
    assert r <= 2.0  # не «выстрел» разницей со старым эпизодом


def test_broken_pool_entry_falls_back_to_round1(emu, tmp_path):
    pool = StartPool(tmp_path)
    pool.offer(b"not a savestate", StartMeta(1, 999, 999, 2))
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, start_pool=tmp_path, curriculum_prob=1.0)
    _, info = env.reset(seed=0)
    assert not info["curriculum_start"] and info["score"] == 0
    assert pool.stats() == {}


def test_without_pool_always_round1(emu):
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, life_loss_terminal=False)
    for s in range(3):
        _, info = env.reset(seed=s)
        assert info["score"] == 0 and info["round"] == 1 and not info["curriculum_start"]


def _frame(name):
    from PIL import Image

    return np.asarray(Image.open(Path(__file__).parent / "data/frames" / name).convert("RGB")).copy()


def test_new_game_after_skipped_title_ends_episode(emu, monkeypatch):
    # клик на титульном экране сразу начинает новую игру: счёт упал до 0 — это Game Over
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, life_loss_terminal=True)
    env.reset(seed=0)
    _force_frames(env, monkeypatch, _frame("play_01699_score.png"))  # 1270 очков
    assert not env.step(0)[2]
    _force_frames(env, monkeypatch, _frame("play_start.png"))  # новая игра, 0 очков
    *_, terminated, _, info = env.step(0)
    assert terminated
    assert not env._continue_game  # следующий reset — новая игра, а не продолжение


def test_snapshots_wait_for_survival(emu, tmp_path):
    # позиция попадает в пул, только если после неё не было потери жизни и Game Over
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, start_pool=tmp_path,
                      curriculum_prob=0.0, snapshot_every=5, snapshot_confirm_steps=100_000)
    env.reset(seed=0)
    for _ in range(20_000):
        if env.step(0)[2]:
            break
    assert StartPool(tmp_path).stats() == {}


def test_snapshot_offered_after_confirmation(emu, tmp_path):
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, start_pool=tmp_path,
                      curriculum_prob=0.0, snapshot_every=5, snapshot_confirm_steps=1)
    env.reset(seed=0)
    for i in range(60):
        env.step(1 if i < 3 else 0)
    assert StartPool(tmp_path).stats().get(1, 0) >= 1
