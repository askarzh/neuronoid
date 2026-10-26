"""Одноразовая проверка: зависит ли угол отскока от точки удара по ракетке.

Когда мяч летит вниз, сохраняется состояние. Для каждого смещения состояние
загружается заново, ракетка заранее ставится так, чтобы мяч попал в точку
«центр + смещение», и замеряется направление вылета. Один и тот же падающий мяч —
меняется только точка удара. Запуск: uv run python tools/bounce_probe.py
"""

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from capture_frames import paddle_x  # noqa: E402

from neuronoid.boot import boot_to_round1  # noqa: E402
from neuronoid.config import core_path, game_path  # noqa: E402
from neuronoid.emulator import LibretroEmulator, Mouse  # noqa: E402

OFFSETS = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
BALLS = 10


def ball(frame: np.ndarray) -> tuple[float, float] | None:
    m = frame[85:175, 40:216].max(axis=2) > 200
    ys, xs = np.nonzero(m)
    if 0 < len(xs) < 20:
        return float(xs.mean()) + 40, float(ys.mean()) + 85
    return None


def follow(emu, target_x: float) -> None:
    px = paddle_x(emu.frame())
    dx = 0 if px is None else int(np.clip((target_x - px) * 2, -14, 14))
    emu.step(mouse=Mouse(dx=dx))


def trial(emu, state: bytes, land_x: float, offset: int) -> tuple[float, float] | None:
    """(фактическое смещение удара, угол вылета) или None, если мяч потерян."""
    emu.load_state(state)
    track = []
    for _ in range(80):
        follow(emu, land_x - offset)
        b = ball(emu.frame())
        if b is None:
            continue
        track.append((b, paddle_x(emu.frame())))
        if len(track) >= 3 and track[-2][0][1] > track[-1][0][1] and track[-2][0][1] > 155:
            (cx, _), cpx = track[-2]
            for _ in range(8):
                emu.step()
            after = ball(emu.frame())
            if after is None or cpx is None:
                return None
            x0, y0 = track[-1][0]
            return cx - cpx, math.degrees(math.atan2(y0 - after[1], after[0] - x0))
    return None


def main() -> None:
    results: dict[int, list[float]] = defaultdict(list)
    balls = 0
    with LibretroEmulator(core_path(), game_path()) as emu:
        boot_to_round1(emu)
        emu.step(mouse=Mouse(left=True), frames=3)
        prev = None
        for t in range(40_000):
            b = ball(emu.frame())
            if b is not None and prev is not None and b[1] > prev[1] and 125 < b[1] < 140:
                vx, vy = b[0] - prev[0], b[1] - prev[1]
                land = b[0] + vx * (176 - b[1]) / vy
                if 50 < land < 205:
                    state = emu.save_state()
                    for o in OFFSETS:
                        r = trial(emu, state, land, o)
                        if r is not None:
                            results[int(round(r[0] / 3) * 3)].append(r[1])
                    emu.load_state(state)
                    balls += 1
                    if balls >= BALLS:
                        break
            prev = b
            follow(emu, b[0] if b else 128)
    print("смещение удара от центра ракетки (пикс., + = правее) → угол вылета (90° = вверх, <90 — вправо)")
    for off in sorted(results):
        a = results[off]
        print(f"{off:+4d}: {np.mean(a):6.1f}°  (от {np.min(a):5.1f} до {np.max(a):5.1f}, n={len(a)})")


if __name__ == "__main__":
    main()
