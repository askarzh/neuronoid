"""Обучение PPO на параллельных ArkanoidEnv: чекпоинты, возобновление, графики."""

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from neuronoid.vec import make_vec_env

# В духе Atari-настроек RL Zoo.
PPO_KWARGS: dict = {
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 4,
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
}
LEARNING_RATE = 2.5e-4


def linear(start: float):
    """Линейно от start к 0 за время одного вызова learn()."""
    return lambda progress_remaining: progress_remaining * start


class ScoreCallback(BaseCallback):
    """Счёт и раунд завершённых эпизодов (прочитанные с экрана) — в TensorBoard."""

    def __init__(self):
        super().__init__()
        self._scores: list[int] = []
        self._rounds: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is not None:
                if ep.get("known_score") is not None:
                    self._scores.append(ep["known_score"])
                if ep.get("known_round") is not None:
                    self._rounds.append(ep["known_round"])
        return True

    def _on_rollout_end(self) -> None:
        if self._scores:
            self.logger.record("neuronoid/score_mean", float(np.mean(self._scores)))
            self.logger.record("neuronoid/score_max", float(np.max(self._scores)))
        if self._rounds:
            self.logger.record("neuronoid/round_max", float(np.max(self._rounds)))
        self._scores.clear()
        self._rounds.clear()


def train(
    run_dir: Path,
    total_steps: int,
    n_envs: int = 10,
    seed: int = 0,
    resume: bool = False,
    device: str = "auto",
    checkpoint_every: int = 200_000,
    live: bool = True,
) -> Path:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.zip"
    (run_dir / "config.json").write_text(
        json.dumps(
            {"total_steps": total_steps, "n_envs": n_envs, "seed": seed, "resume": resume,
             "device": device, "learning_rate": LEARNING_RATE, **PPO_KWARGS},
            indent=1,
        )
    )
    venv = make_vec_env(n_envs, seed=seed, life_loss_terminal=True, live=live)
    model = None
    try:
        if resume and latest.exists():
            model = PPO.load(latest, env=venv, device=device, tensorboard_log=str(run_dir / "tb"))
        else:
            model = PPO(
                "CnnPolicy", venv, learning_rate=linear(LEARNING_RATE), seed=seed, device=device,
                tensorboard_log=str(run_dir / "tb"), verbose=1, **PPO_KWARGS,
            )
        callbacks = CallbackList([
            CheckpointCallback(max(checkpoint_every // n_envs, 1), str(ckpt_dir), name_prefix="ppo"),
            ScoreCallback(),
        ])
        model.learn(total_steps, callback=callbacks, reset_num_timesteps=not (resume and latest.exists()),
                    tb_log_name="ppo")
    finally:
        if model is not None:
            model.save(latest)  # и при Ctrl+C, и при падении среды
        venv.close()
    return latest
