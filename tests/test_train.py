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
    train(tmp_path, total_steps=512, n_envs=2, resume=True, live=False)
    m2 = PPO.load(tmp_path / "checkpoints" / "latest.zip")
    assert m2.num_timesteps >= m1.num_timesteps + 512


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
