"""Среда, падающая на втором step (для теста падения дочернего процесса)."""

import gymnasium as gym
import numpy as np


class CrashEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (4, 93, 96), np.uint8)
    action_space = gym.spaces.Discrete(6)

    def __init__(self):
        self.n = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((4, 93, 96), np.uint8), {}

    def step(self, action):
        self.n += 1
        if self.n >= 2:
            raise RuntimeError("среда упала")
        return np.zeros((4, 93, 96), np.uint8), 0.0, False, False, {}


def crashing_env_fn():
    return CrashEnv()
