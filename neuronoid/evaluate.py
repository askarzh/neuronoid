"""Оценка агента по честным правилам зачёта: полные игры с раунда 1, без продолжений."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

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
    bricks_left: int | None = None  # сколько кирпичей осталось в последнем раунде игры
    cleared_round1: bool = False  # раунд 1 очищен (а не пройден через выход «B»)


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


SCHEME_KEYS = ("action_mode", "obs_mode", "sticky_prob")
CLEARED_MAX_BRICKS = 2


def env_kwargs_from_config(run_dir: Path) -> dict:
    """Схема среды прогона из config.json; у старых прогонов полей нет — значения по умолчанию."""
    path = Path(run_dir) / "config.json"
    if not path.exists():
        return {}
    cfg = json.loads(path.read_text())
    return {k: cfg[k] for k in SCHEME_KEYS if k in cfg}


def classify_round1_exit(bricks_before: int | None) -> bool:
    """Переход в раунд 2 — очистка, если на шаг раньше кирпичей почти не оставалось; иначе выход «B»."""
    return bricks_before is not None and bricks_before <= CLEARED_MAX_BRICKS


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
        steps, done, cleared, bricks_r1 = 0, False, False, None
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            steps += 1
            done = terminated or truncated
            if info.get("known_round") == 1:
                bricks_r1 = info.get("bricks_left", bricks_r1)
            elif info.get("known_round") == 2 and not cleared and bricks_r1 is not None:
                cleared = classify_round1_exit(bricks_r1)
                bricks_r1 = None
        results.append(GameResult(int(info["known_score"] or 0), int(info["known_round"] or 1), steps,
                                  info.get("bricks_left"), cleared))
    return Report.from_games(results)
