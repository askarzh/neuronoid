"""Одноразовая проверка: DOSBox Pure + Arkanoid через libretro.py.

Запуск: uv run python scripts/spike.py
Кадры ключевых экранов пишутся в runs/spike/, цифры печатаются в консоль.
"""

import time

import numpy as np
from libretro import ArrayVideoDriver, Session
from libretro.api.input import Key, MouseState
from PIL import Image

from neuronoid.config import ROOT, core_path, game_path

OUT = ROOT / "runs" / "spike"
OPTIONS = {"dosbox_pure_cycles": "4720"}


def frame_np(video: ArrayVideoDriver) -> np.ndarray:
    d = video._frame_dims
    buf = np.frombuffer(video._frame, dtype=np.uint8, count=d.length)
    buf = buf.reshape(d.height, d.pitch // 4, 4)
    return buf[:, : d.width, 2::-1].copy()  # XRGB8888 (BGRX в памяти) -> RGB


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mouse = [None]

    def inputs():
        while True:
            yield mouse[0]

    video = ArrayVideoDriver()
    with Session(core_path(), game_path(), video=video, perf=None, input=inputs, options=OPTIONS) as s:
        kb = s.input.keyboard_callback

        def frames(n: int) -> None:
            for _ in range(n):
                s.run()

        def tap(key: Key, ch: int = 0) -> None:
            kb(True, key, ch, 0)
            frames(5)
            kb(False, key, ch, 0)

        def shot(name: str) -> np.ndarray:
            f = frame_np(video)
            Image.fromarray(f).save(OUT / f"{name}.png")
            return f

        av = s.core.get_system_av_info()
        print("fps ядра:", av.timing.fps)

        # Последовательность до раунда 1 (подготовка, не управление агентом)
        frames(150)
        shot("01_video_menu")
        tap(Key.RETURN, 13)  # EGA (выбран по умолчанию)
        frames(60)
        shot("02_input_menu")
        tap(Key.M, ord("m"))  # мышь
        frames(700)
        shot("03_title")
        tap(Key.SPACE, 32)  # старт игры, дальше вступление
        frames(600)
        shot("04_intro")
        frames(880)
        f = shot("05_round1_ready")
        print("кадр игры:", f.shape)

        state = bytearray(s.core.serialize_size())
        s.core.serialize(state)
        state = bytes(state)
        print("размер savestate:", len(state))

        # Детерминизм: одно состояние + одни действия -> одинаковые кадры
        outs = []
        for _ in range(2):
            s.core.unserialize(state)
            mouse[0] = MouseState(x=3)
            frames(60)
            mouse[0] = MouseState(left=True)
            frames(3)
            mouse[0] = None
            frames(120)
            outs.append(frame_np(video))
        print("детерминизм:", np.array_equal(*outs))
        shot("06_ball_in_play")

        # Ничего не делаем — теряем жизни до Game Over
        s.core.unserialize(state)
        mouse[0] = MouseState(left=True)
        frames(3)
        mouse[0] = None
        for i in range(1, 4000):
            s.run()
            if i % 200 == 0:
                shot(f"07_idle_{i:05d}")

        # Скорость прогона без вывода
        s.core.unserialize(state)
        t = time.perf_counter()
        frames(10000)
        print("кадров/с (без вывода):", round(10000 / (time.perf_counter() - t)))
        s.core.unserialize(state)
        t = time.perf_counter()
        for i in range(10000):
            s.run()
            if i % 4 == 0:
                frame_np(video)
        print("кадров/с (кадр каждые 4):", round(10000 / (time.perf_counter() - t)))


if __name__ == "__main__":
    main()
