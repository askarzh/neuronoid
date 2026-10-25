"""Оценка агента по честным правилам зачёта: полные игры с раунда 1, без продолжений."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from neuronoid.env import ArkanoidEnv

GAMES = 10
PASS_COUNT = 7
MILESTONES: dict[str, Callable[["GameResult"], bool]] = {
    "m1": lambda g: g.score >= 1500,  # стабильно отбивает мяч и набирает очки
    "m2": lambda g: g.round >= 2,
    "m3": lambda g: g.round >= 5,
}


@dataclass
class GameResult:
    score: int
    round: int
    steps: int


@dataclass
class Report:
    games: list[GameResult]
    m1: bool
    m2: bool
    m3: bool

    @classmethod
    def from_games(cls, games: list[GameResult]) -> "Report":
        flags = {name: sum(map(rule, games)) >= PASS_COUNT for name, rule in MILESTONES.items()}
        return cls(games=games, **flags)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)


def evaluate(
    policy: Callable[[np.ndarray], int],
    env: ArkanoidEnv,
    games: int = GAMES,
    seed: int = 1000,
    on_game_start: Callable[[int], None] | None = None,
) -> Report:
    """Игра i начинается с reset(seed=seed + i) и идёт до Game Over или лимита шагов.

    `on_game_start(seed + i)` — чтобы пересеять случайность самой политики.
    """
    results = []
    for i in range(games):
        if on_game_start is not None:
            on_game_start(seed + i)
        obs, info = env.reset(seed=seed + i)
        steps, done = 0, False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            steps += 1
            done = terminated or truncated
        results.append(GameResult(int(info["known_score"] or 0), int(info["known_round"] or 1), steps))
    return Report.from_games(results)
