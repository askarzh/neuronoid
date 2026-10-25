"""Параллельные среды для обучения: одна ArkanoidEnv на процесс."""

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env_fn(
    rank: int,
    seed: int,
    life_loss_terminal: bool,
    live_rank: int | None = 0,
    start_pool: Path | None = None,
    curriculum_prob: float = 0.5,
) -> Callable[[], gym.Env]:
    """Фабрика для SubprocVecEnv; сид раздаёт make_vec_env через venv.seed()."""
    def make() -> gym.Env:
        from neuronoid.env import ArkanoidEnv
        from neuronoid.quiet import silence_native_stdout

        silence_native_stdout()
        env = ArkanoidEnv(life_loss_terminal=life_loss_terminal, live=rank == live_rank,
                          start_pool=start_pool, curriculum_prob=curriculum_prob)
        return Monitor(env, info_keywords=("known_score", "known_round", "gained_score", "curriculum_start"))

    return make


def make_vec_env(
    n_envs: int,
    seed: int = 0,
    life_loss_terminal: bool = True,
    live: bool = True,
    start_pool: Path | None = None,
    curriculum_prob: float = 0.5,
) -> SubprocVecEnv:
    """`live=True` — живое окно показывает среду №0 (и только её); `start_pool` — учебная программа."""
    fns = [
        make_env_fn(i, seed, life_loss_terminal, 0 if live else None, start_pool, curriculum_prob)
        for i in range(n_envs)
    ]
    venv = SubprocVecEnv(fns, start_method="spawn")
    venv.seed(seed)  # среда i получит seed + i при первом reset
    return venv
