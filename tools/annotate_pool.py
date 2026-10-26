"""Досчитывает bricks_left позициям пула, у которых его нет (пулы до этапа 4).

Запуск: uv run python tools/annotate_pool.py runs/ppo4/starts
"""

import sys
from pathlib import Path

from neuronoid.config import core_path, game_path
from neuronoid.curriculum import StartPool, annotate_bricks
from neuronoid.emulator import LibretroEmulator


def main() -> None:
    root = Path(sys.argv[1])
    with LibretroEmulator(core_path(), game_path()) as emu:
        n = annotate_bricks(root, emu)
    print("обновлено позиций:", n, "| пул:", StartPool(root).stats())


if __name__ == "__main__":
    main()
