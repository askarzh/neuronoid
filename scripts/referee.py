"""Судья для внешнего игрока: uv run python scripts/referee.py путь/к/player.py [--games 10] [--seed 1000]

player.py должен содержать класс Player с методом act(frame) -> (keys, Mouse).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from neuronoid.referee import judge


def load_player(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("external_player", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Player


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("player", type=Path)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-minutes", type=float, default=20)
    args = ap.parse_args()
    result = judge(load_player(args.player), args.games, args.seed, int(70 * 60 * args.max_minutes))
    for i, g in enumerate(result["games"], 1):
        print(f"  игра {i:2d}: счёт {g['score']:6d}, раунд {g['round']}, кадров {g['frames']}, "
              f"{'ПОБЕДА' if g['won'] else 'Game Over' if g['game_over'] else ('лимит времени' if g['started'] else 'игра не началась')}")
    print(f"средний счёт {result['mean_score']:.0f}, максимальный раунд {result['max_round']}, побед {result['wins']} из {len(result['games'])}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
