"""Одноразовый сборщик кадров для разметки тестов ScreenReader.

Это инструмент, а не агент: ракетку ведёт простой следящий алгоритм по
пикселям, чтобы набрать очки и увидеть все цифры счёта.
Запуск: uv run python tools/capture_frames.py
"""

import numpy as np
from PIL import Image

from neuronoid.boot import BOOT_SEQUENCE, FRAMES_TO_ROUND1, KEY_HOLD_FRAMES
from neuronoid.config import ROOT, core_path, game_path
from neuronoid.emulator import LibretroEmulator, Mouse

OUT = ROOT / "tests" / "data" / "frames"
SCORE = (slice(63, 69), slice(241, 289))
LIVES = (slice(192, 197), slice(44, 120))
ROUND = (slice(120, 135), slice(250, 285))


def find_ball_x(f: np.ndarray) -> int | None:
    m = f[85:175, 40:216].max(axis=2) > 200
    ys, xs = np.nonzero(m)
    return int(xs.mean()) + 40 if 0 < len(xs) < 20 else None


def paddle_x(f: np.ndarray) -> int | None:
    cols = np.where((f[176:182, 40:216].max(axis=2) > 100).any(axis=0))[0]
    return int(cols.mean()) + 40 if len(cols) else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved = 0

    def save(f: np.ndarray, name: str) -> None:
        nonlocal saved
        Image.fromarray(f).save(OUT / f"{name}.png")
        saved += 1

    with LibretroEmulator(core_path(), game_path()) as emu:
        for i, (wait, key) in enumerate(BOOT_SEQUENCE):
            emu.step(frames=wait)
            save(emu.frame(), f"menu_{i}")
            emu.step([key], frames=KEY_HOLD_FRAMES)
        emu.step(frames=600)
        save(emu.frame(), "intro")
        emu.step(frames=FRAMES_TO_ROUND1 - 600)
        start = emu.save_state()
        save(emu.frame(), "play_start")

        last = {k: None for k in ("score", "lives", "round")}
        boxes = {"score": SCORE, "lives": LIVES, "round": ROUND}
        rng = np.random.default_rng(0)
        checkpoints: list[bytes] = []
        lives_px = None
        for t in range(40_000):
            if t % 100 == 0:
                checkpoints = (checkpoints + [emu.save_state()])[-5:]
            f = emu.frame()
            bx, px = find_ball_x(f), paddle_x(f)
            dx = 0 if bx is None or px is None else int(np.clip((bx - px) * 2, -14, 14))
            if rng.random() < 0.02:
                dx = int(rng.integers(-14, 15))
            emu.step(mouse=Mouse(dx=dx, left=t % 50 == 0))
            f = emu.frame()
            if f.shape != (200, 320, 3):
                continue
            n = int((f[LIVES].max(axis=2) > 60).sum())
            if lives_px is not None and n < lives_px and len(checkpoints) >= 2:
                emu.load_state(checkpoints[-2])  # жизнь потеряна — откат и другая попытка
                checkpoints = checkpoints[:-1]
                lives_px = None
                continue
            lives_px = n
            for k, (ys, xs) in boxes.items():
                region = f[ys, xs].tobytes()
                if region != last[k]:
                    last[k] = region
                    save(f, f"play_{t:05d}_{k}")
            if saved >= 70:
                break

        # Отдельная ветка: бездействие — теряем жизни до Game Over
        emu.load_state(start)
        emu.step(mouse=Mouse(left=True), frames=3)
        prev = None
        for t in range(6000):
            emu.step(frames=10)
            f = emu.frame()
            playing = f.shape == (200, 320, 3) and (f[5:189, 34:37].max(axis=2) > 100).mean() > 0.5
            key = (f.shape, playing, f[LIVES].tobytes() if playing else None)
            if key != prev:
                prev = key
                save(f, f"idle_{t:04d}_{'play' if playing else 'not'}")
            if not playing and t > 50 and saved > 90:
                break
    print("сохранено кадров:", saved)


if __name__ == "__main__":
    main()
