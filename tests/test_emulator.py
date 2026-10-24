import os
import subprocess
import sys
import time

import numpy as np
import pytest

from neuronoid.config import core_path, game_path
from neuronoid.emulator import LibretroEmulator, Mouse

PADDLE_ROWS = slice(176, 182)
FIELD_COLS = slice(40, 216)


def paddle_left(frame: np.ndarray) -> int:
    cols = np.where((frame[PADDLE_ROWS, FIELD_COLS].max(axis=2) > 100).any(axis=0))[0]
    assert len(cols), "ракетка не найдена"
    return int(cols.min()) + FIELD_COLS.start


def test_frame_shape(emu, round1):
    f = emu.frame()
    assert f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 3
    assert f.shape == (200, 320, 3)


def test_frame_is_copy(emu, round1):
    f = emu.frame()
    emu.step(mouse=Mouse(dx=5), frames=10)
    assert not np.array_equal(f, emu.frame())


def test_fps_is_fixed(emu):
    assert 50 <= emu.fps <= 75


def test_boot_reaches_paddle(emu, round1):
    paddle_left(emu.frame())


def test_deterministic_from_state(emu, round1):
    emu.step(mouse=Mouse(dx=3), frames=60)
    emu.step(mouse=Mouse(left=True), frames=3)
    emu.step(frames=120)
    a = emu.frame()
    emu.load_state(round1)
    emu.step(mouse=Mouse(dx=3), frames=60)
    emu.step(mouse=Mouse(left=True), frames=3)
    emu.step(frames=120)
    assert np.array_equal(a, emu.frame())


def test_mouse_moves_paddle(emu, round1):
    x0 = paddle_left(emu.frame())
    emu.step(mouse=Mouse(dx=4), frames=20)
    right = paddle_left(emu.frame())
    emu.load_state(round1)
    emu.step(mouse=Mouse(dx=-4), frames=20)
    left = paddle_left(emu.frame())
    assert right > x0 + 5 and left < x0 - 5


def test_mouse_released_after_step(emu, round1):
    emu.step(mouse=Mouse(dx=4), frames=10)
    emu.step(frames=30)  # игра догоняет накопленное смещение с задержкой
    x = paddle_left(emu.frame())
    emu.step(frames=20)
    assert paddle_left(emu.frame()) == x


def test_frame_fresh_after_load_state(emu, round1):
    # кадр сразу после load_state не зависит от того, где эмулятор был до загрузки
    emu.step(mouse=Mouse(dx=-6), frames=80)
    emu.load_state(round1)
    a = emu.frame()
    emu.step(mouse=Mouse(dx=6), frames=80)
    emu.step(mouse=Mouse(left=True), frames=3)
    emu.step(frames=60)
    emu.load_state(round1)
    assert np.array_equal(a, emu.frame())


def test_audio_drained(emu, round1):
    emu.step(frames=600)
    assert emu.audio().shape == (0, 2)
    assert len(emu._session.audio.buffer) < 10000  # буфер не копится


def test_second_instance_in_process_fails(emu):
    with pytest.raises(RuntimeError, match="one emulator per process"):
        LibretroEmulator(core_path(), game_path())


REOPEN = """
from neuronoid.config import core_path, game_path
from neuronoid.emulator import LibretroEmulator
LibretroEmulator(core_path(), game_path()).close()
try:
    LibretroEmulator(core_path(), game_path())
except RuntimeError as e:
    print("refused:", e)
"""


def test_reopen_after_close_refused_not_hung():
    # повторная инициализация ядра в процессе зависает внутри libretro.py
    r = subprocess.run(
        [sys.executable, "-c", REOPEN], capture_output=True, encoding="utf-8", timeout=120,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert "refused: one emulator per process" in r.stdout


def test_missing_game_file_rejected(emu, tmp_path):
    with pytest.raises(FileNotFoundError):
        LibretroEmulator(core_path(), tmp_path / "missing.zip")


def replay(emu, state, sleep_rng=None, steps=300):
    """Кадр после каждого шага повтора с одинаковой случайной мышью."""
    rng = np.random.default_rng(1)
    emu.load_state(state)
    out = []
    for i in range(steps):
        emu.step(mouse=Mouse(dx=int(rng.choice([-10, -4, 0, 4, 10])), left=i == 5), frames=4)
        out.append(emu.frame())
        if sleep_rng is not None:
            time.sleep(float(sleep_rng.random()) * 0.003)
    return out


def test_replay_deterministic_every_step(emu, round1):
    ref = replay(emu, round1)
    for sleeps in (None, np.random.default_rng(7)):
        other = replay(emu, round1, sleeps)
        bad = [i for i, (a, b) in enumerate(zip(ref, other)) if not np.array_equal(a, b)]
        assert not bad, f"{len(bad)} шагов из {len(ref)} разошлись (паузы: {sleeps is not None})"


CHILD = """
import sys, numpy as np
from neuronoid.config import core_path, game_path
from neuronoid.emulator import LibretroEmulator, Mouse
state = open(sys.argv[1], 'rb').read()
with LibretroEmulator(core_path(), game_path()) as e:
    e.load_state(state)
    e.step(mouse=Mouse(dx=3), frames=60)
    e.step(mouse=Mouse(left=True), frames=3)
    e.step(frames=120)
    np.save(sys.argv[2], e.frame())
"""


def test_state_loads_in_other_process(emu, round1, tmp_path):
    (tmp_path / "s.state").write_bytes(round1)
    subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path / "s.state"), str(tmp_path / "f.npy")],
        check=True, capture_output=True, timeout=120,
    )
    emu.step(mouse=Mouse(dx=3), frames=60)
    emu.step(mouse=Mouse(left=True), frames=3)
    emu.step(frames=120)
    assert np.array_equal(np.load(tmp_path / "f.npy"), emu.frame())


def test_frame_fresh_after_load_state_random_history(emu, round1):
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(40):
        emu.step(mouse=Mouse(dx=int(rng.integers(-9, 10)), left=bool(rng.random() < 0.3)),
                 frames=int(rng.integers(1, 200)))
        time.sleep(float(rng.random()) * 0.03)  # поток эмуляции ядра чувствителен ко времени
        emu.load_state(round1)
        frames.append(emu.frame())
    assert all(np.array_equal(frames[0], f) for f in frames)
