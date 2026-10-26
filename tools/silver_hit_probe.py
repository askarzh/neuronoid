"""Одноразовая проверка: видно ли на экране первый удар по серому кирпичу.

Берёт позиции раунда 1 из пула ppo3 (там серый ряд обычно цел), ведёт ракетку
следящим алгоритмом и покадрово сравнивает серые ячейки. Изменение ячейки, после
которого она остаётся кирпичом, — видимый удар. Кадры до/после пишутся в runs/silver/.
Запуск: uv run python tools/silver_hit_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from capture_frames import find_ball_x, paddle_x  # noqa: E402

from neuronoid.config import ROOT, core_path, game_path  # noqa: E402
from neuronoid.emulator import LibretroEmulator, Mouse  # noqa: E402

OUT = ROOT / "runs" / "silver"
SILVER = (170, 170, 170)


def cells(frame: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    out = {}
    for r in range(20):
        y = 9 + 8 * r
        for c in range(11):
            x = 40 + 16 * c
            cell = frame[y : y + 7, x + 2 : x + 16]
            if (cell.max(axis=2) > 0).mean() > 0.8 and tuple(cell[3, 7]) == SILVER:
                out[(r, c)] = cell.copy()
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = sorted((ROOT / "runs" / "ppo3" / "starts" / "1").glob("*.json"),
                  key=lambda p: -json.loads(p.read_text())["progress"])[:6]
    found = 0
    with LibretroEmulator(core_path(), game_path()) as emu:
        for meta in pool:
            emu.load_state(meta.with_suffix(".state").read_bytes())
            prev = cells(emu.frame())
            for t in range(8000):
                f = emu.frame()
                bx, px = find_ball_x(f), paddle_x(f)
                dx = 0 if bx is None or px is None else int(np.clip((bx - px) * 2, -14, 14))
                emu.step(mouse=Mouse(dx=dx, left=t % 40 == 0))
                f2 = emu.frame()
                now = cells(f2)
                for key, before in (prev.items() if t > 300 else []):
                    if key in now and not np.array_equal(before, now[key]):
                        Image.fromarray(np.concatenate([before, now[key]], axis=1)).resize((112, 28), Image.NEAREST).save(
                            OUT / f"hit_{meta.stem[:8]}_{t}_{key[0]}_{key[1]}.png")
                        Image.fromarray(f2).save(OUT / f"hit_{meta.stem[:8]}_{t}_frame.png")
                        found += 1
                    if key not in now:
                        Image.fromarray(f2).save(OUT / f"gone_{meta.stem[:8]}_{t}_{key[0]}_{key[1]}.png")
                prev = now
                if found >= 6:
                    break
            print(meta.name, "серых ячеек:", len(cells(emu.frame())), "изменений:", found)
            if found >= 6:
                break


if __name__ == "__main__":
    main()
