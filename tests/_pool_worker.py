"""Воркер для теста одновременной записи в пул (импортируется spawn-процессом)."""

import sys
from pathlib import Path

import numpy as np

from neuronoid.curriculum import StartMeta, StartPool


def run(root: str, seed: int) -> None:
    pool = StartPool(Path(root), per_round=5)
    rng = np.random.default_rng(seed)
    for i in range(30):
        pool.offer(bytes([seed]) * 1000, StartMeta(1, int(rng.integers(0, 10_000)), 0, 2))
        pool.sample(rng)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
