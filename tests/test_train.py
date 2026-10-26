import json
from pathlib import Path

import pytest
from stable_baselines3 import PPO

from neuronoid.train import train


def event_tags(tb_dir: Path) -> set[str]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tags: set[str] = set()
    for f in tb_dir.rglob("events.out.tfevents.*"):
        acc = EventAccumulator(str(f))
        acc.Reload()
        tags |= set(acc.Tags()["scalars"])
    return tags


def test_train_smoke_and_resume(tmp_path):
    p = train(tmp_path, total_steps=512, n_envs=2, checkpoint_every=256, live=False)
    assert p.exists() and (tmp_path / "config.json").exists()
    m1 = PPO.load(p)
    assert m1.num_timesteps >= 512
    train(tmp_path, total_steps=1024, n_envs=2, resume=True, live=False)  # --steps — общая цель
    m2 = PPO.load(tmp_path / "checkpoints" / "latest.zip")
    assert m2.num_timesteps >= 1024


def test_interrupt_saves_latest(tmp_path, monkeypatch):
    def boom(self, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(PPO, "learn", boom)
    with pytest.raises(KeyboardInterrupt):
        train(tmp_path, total_steps=512, n_envs=2, live=False)
    assert (tmp_path / "checkpoints" / "latest.zip").exists()


def test_score_callback_logs_metrics(tmp_path):
    train(tmp_path, total_steps=2048, n_envs=2, live=False)
    assert {"neuronoid/score_max", "neuronoid/score_mean", "rollout/ep_rew_mean"} <= event_tags(tmp_path / "tb")


def test_resume_uses_newest_periodic_checkpoint(tmp_path):
    # жёсткая остановка: latest.zip не успел записаться, есть только периодические
    train(tmp_path, total_steps=512, n_envs=2, checkpoint_every=256, live=False)
    ckpts = tmp_path / "checkpoints"
    (ckpts / "latest.zip").unlink()
    newest = max(ckpts.glob("ppo_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    steps_before = PPO.load(newest).num_timesteps
    # цель уже достигнута: продолжение не учит (веса те же), а старт с нуля дал бы другие веса
    train(tmp_path, total_steps=steps_before, n_envs=2, seed=1, resume=True, live=False)
    a = PPO.load(newest).policy.state_dict()
    b = PPO.load(ckpts / "latest.zip").policy.state_dict()
    assert all((a[k] == b[k]).all() for k in a)


def test_resume_without_checkpoints_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        train(tmp_path, total_steps=512, n_envs=2, resume=True, live=False)


def test_refuses_to_overwrite_without_resume(tmp_path):
    train(tmp_path, total_steps=256, n_envs=2, live=False)
    with pytest.raises(FileExistsError):
        train(tmp_path, total_steps=256, n_envs=2, live=False)


def test_steps_is_total_target_on_resume(tmp_path):
    train(tmp_path, total_steps=512, n_envs=2, live=False)
    train(tmp_path, total_steps=1024, n_envs=2, resume=True, live=False)
    n = PPO.load(tmp_path / "checkpoints" / "latest.zip").num_timesteps
    assert 1024 <= n < 1024 + 512


def test_lr_schedule_continues_globally():
    from neuronoid.train import LEARNING_RATE, global_linear

    sched = global_linear(done=500, target=1000)  # половина пути уже пройдена
    assert sched(1.0) == pytest.approx(LEARNING_RATE * 0.5)
    assert sched(0.0) == pytest.approx(0.0)


def test_latest_save_is_atomic(tmp_path, monkeypatch):
    train(tmp_path, total_steps=256, n_envs=2, live=False)
    latest = tmp_path / "checkpoints" / "latest.zip"
    good = latest.read_bytes()

    def broken_save(self, path, *a, **k):
        open(path, "wb").write(b"half")  # запись оборвалась
        raise KeyboardInterrupt

    monkeypatch.setattr(PPO, "save", broken_save)
    with pytest.raises(KeyboardInterrupt):
        train(tmp_path, total_steps=512, n_envs=2, resume=True, live=False)
    assert latest.read_bytes() == good


def test_watchdog_interrupts_when_steps_stop():
    import time

    from neuronoid.train import Watchdog

    fired = []
    dog = Watchdog(timeout=0.3, on_timeout=lambda: fired.append(1))
    dog.start()
    time.sleep(1.0)
    dog.stop()
    assert fired


def test_watchdog_quiet_while_steps_flow():
    import time

    from neuronoid.train import Watchdog

    fired = []
    dog = Watchdog(timeout=0.5, on_timeout=lambda: fired.append(1))
    dog.start()
    for _ in range(10):
        dog.feed()
        time.sleep(0.1)
    dog.stop()
    assert not fired


def test_init_from_starts_new_run_with_old_weights(tmp_path):
    src = train(tmp_path / "a", total_steps=256, n_envs=2, live=False)
    dst = train(tmp_path / "b", total_steps=512, n_envs=2, live=False, init_from=src)
    m = PPO.load(dst)
    assert m.num_timesteps >= 512 and (tmp_path / "a" / "checkpoints" / "latest.zip").exists()


def test_init_from_with_resume_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        train(tmp_path, total_steps=256, n_envs=2, live=False, resume=True, init_from=tmp_path / "x.zip")


def test_curriculum_run_fills_pool(tmp_path):
    train(tmp_path, total_steps=2048, n_envs=2, live=False, curriculum=True, snapshot_confirm_steps=1)
    assert any((tmp_path / "starts").rglob("*.state"))


def test_score_callback_logs_fresh_metrics(tmp_path):
    train(tmp_path, total_steps=2048, n_envs=2, live=False, curriculum=True)
    tags = event_tags(tmp_path / "tb")
    assert {"neuronoid/points_mean_fresh", "neuronoid/score_max_fresh"} & tags
    assert "neuronoid/bricks_left_min_fresh" in tags


def test_resume_keeps_curriculum_settings(tmp_path):
    train(tmp_path, total_steps=256, n_envs=2, live=False, curriculum=True)
    with pytest.raises(ValueError, match="curriculum"):
        train(tmp_path, total_steps=512, n_envs=2, live=False, resume=True)  # забыли --curriculum


def test_train_smoke_target_scheme(tmp_path):
    p = train(tmp_path, total_steps=512, n_envs=2, live=False, action_mode="target", obs_mode="wide",
              sticky_prob=0.1)
    m = PPO.load(p)
    assert m.action_space.n == 41 and m.observation_space.shape == (4, 93, 192)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["action_mode"] == "target" and cfg["obs_mode"] == "wide" and cfg["sticky_prob"] == 0.1
