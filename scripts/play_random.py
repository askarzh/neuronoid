"""Случайный агент в ArkanoidEnv с живым окном — проверить всё глазами.

Запуск: uv run python scripts/play_random.py [--steps N] [--realtime]
Параллельно: uv run python scripts/watch.py
"""

import argparse
import time

import numpy as np

from neuronoid.config import ConfigError
from neuronoid.env import ArkanoidEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--realtime", action="store_true", help="замедлить до скорости игры")
    args = ap.parse_args()

    env = ArkanoidEnv(live=True)
    rng = np.random.default_rng()
    try:
        obs, info = env.reset()
    except ConfigError as e:
        raise SystemExit(str(e))
    step_time = env.frame_skip / 70.087
    episode, total, t0, score = 1, 0.0, time.perf_counter(), 0
    for i in range(1, args.steps + 1):
        tick = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(int(rng.integers(env.action_space.n)))
        total += reward
        if info["score"] is not None:
            score = info["score"]  # на титульном экране после Game Over счёта уже нет
        if terminated or truncated:
            print(f"эпизод {episode}: награда {total:.1f}, счёт {score}")
            episode, total = episode + 1, 0.0
            env.reset()
        if args.realtime:
            time.sleep(max(0.0, step_time - (time.perf_counter() - tick)))
    print(f"шагов в секунду: {args.steps / (time.perf_counter() - t0):.0f}")
    env.close()


if __name__ == "__main__":
    main()
