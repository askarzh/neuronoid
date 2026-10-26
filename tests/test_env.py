import json
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
    assert reward_between(s(100, rd=1), s(100, rd=2)) == pytest.approx(50.0)
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


def test_capsule_points_are_capped():
    # пойманная капсула (+1000 за шаг) не должна стоить как прохождение раунда
    def s(sc):
        return ScreenState(True, sc, 1, 2)

    assert reward_between(s(1000), s(2000)) == pytest.approx(2.0)
    assert reward_between(s(1000), s(1120)) == pytest.approx(1.2)  # кирпич — как раньше


from neuronoid.bricks import BrickEvents  # noqa: E402
from neuronoid.env import ROUND_BONUS, brick_reward  # noqa: E402


def test_brick_reward_function():
    assert brick_reward(BrickEvents(1, 0, 30)) == pytest.approx(0.5)
    assert brick_reward(BrickEvents(1, 0, 9)) == pytest.approx(2.5)
    assert brick_reward(BrickEvents(0, 0, None)) == 0


def test_round1_brick_reward_is_bounded():
    total = sum(brick_reward(BrickEvents(1, 0, left)) for left in range(65, -1, -1))
    assert total <= 60 and total < 2 * ROUND_BONUS


def test_env_reports_bricks_left(env):
    env.reset(seed=0)
    for _ in range(10):
        *_, info = env.step(0)
    assert 60 <= info["bricks_left"] <= 66  # мяч вылетает сам и может успеть разбить кирпич-другой


def test_no_brick_reward_on_round_change(env, monkeypatch):
    # стена сменилась целиком (раунд 1 -> раунд 2): «разрушений» нет
    env.reset(seed=0)
    _force_frames(env, monkeypatch, _frame("play_start.png"))
    for _ in range(6):
        env.step(0)
    _force_frames(env, monkeypatch, _frame("idle_0924_play.png"))
    rewards = [env.step(0)[1] for _ in range(6)]
    assert max(rewards) <= ROUND_BONUS + 1  # бонус за раунд — да, «выстрел» за 66 кирпичей — нет


def test_no_snapshot_without_known_bricks(emu, tmp_path):
    # снимок с «0 кирпичей» во время прогрева трекера отравил бы пул
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, start_pool=tmp_path,
                      curriculum_prob=0.0, snapshot_every=1, snapshot_confirm_steps=1)
    env.reset(seed=0)
    for i in range(30):
        env.step(1 if i < 3 else 0)
    metas = [json.loads(p.read_text()) for p in tmp_path.rglob("*.json")]
    assert metas and all(m["bricks_left"] not in (None, 0) for m in metas)


from neuronoid.env import N_TARGETS, controller_dx, paddle_center, target_x  # noqa: E402


def test_paddle_center_on_frames():
    assert paddle_center(_frame("play_start.png")) == pytest.approx(106, abs=1)
    assert paddle_center(_frame("play_01699_score.png")) == pytest.approx(194, abs=1)
    assert paddle_center(np.zeros((200, 320, 3), np.uint8)) is None


def test_controller_idle_without_paddle():
    assert controller_dx(100.0, None) == 0


def test_controller_saturates():
    assert controller_dx(200.0, 60.0) == 14 and controller_dx(60.0, 200.0) == -14
    assert controller_dx(105.0, 100.0) == 5


def test_target_action_moves_paddle_to_target(emu):
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, action_mode="target", obs_mode="wide",
                      sticky_prob=0.0, max_noop=0, use_brick_reward=False)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 93, 192) and env.observation_space.contains(obs)
    assert env.action_space.n == N_TARGETS + 1
    for _ in range(15):
        env.step(5)
    left = paddle_center(env._frame)
    for _ in range(15):
        env.step(34)
    right = paddle_center(env._frame)
    assert abs(left - target_x(5)) <= 4 and abs(right - target_x(34)) <= 4


def _row_frame(runs):
    f = np.zeros((200, 320, 3), np.uint8)
    for x, n in runs:
        f[178, x:x + n] = 170
    return f


def test_paddle_center_ignores_capsule_on_paddle_row():
    # капсула (15 пикселей) левее ракетки не должна перехватывать её положение
    f = _row_frame([(56, 15), (123, 4), (128, 15), (144, 4)])
    assert paddle_center(f) == pytest.approx(135.0, abs=0.6)


def test_paddle_center_enlarged_paddle():
    f = _row_frame([(90, 4), (95, 27), (123, 4)])
    assert paddle_center(f) == pytest.approx(108.0, abs=0.6)
