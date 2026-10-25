"""Чтение состояния игры с кадра 320×200: только пиксели, никакой памяти эмулятора.

Цифры нарисованы фиксированным шрифтом в ячейках 8×6, поэтому сравнение с
образцами (neuronoid/glyphs.npz) точное: незнакомая ячейка — значит «не прочитал».
"""

from pathlib import Path

import numpy as np

GAME_SHAPE = (200, 320, 3)
CELL = 8
GLYPH_ROWS = 6
INK = 100  # пиксель «горит», если max(R, G, B) > INK

SCORE_BOX = (63, 69, 241, 289)  # (y0, y1, x0, x1): 6 ячеек по 8 пикселей
# Номер раунда: одна цифра в ячейке x 265–272 (проверено на раундах 1–8).
# Двузначные номера (10+) ещё не видели: предполагаем две ячейки с x 261.
ROUND_Y = (126, 132)
ROUND_ONE_X = 265
ROUND_TWO_X = 261

_GLYPHS: dict[str, np.ndarray] = dict(np.load(Path(__file__).with_name("glyphs.npz")))


def _ink(frame: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    return frame[y0:y1, x0:x1].max(axis=2) > INK


def _digit(cell: np.ndarray) -> str | None:
    """Цифра, ' ' для пустой ячейки, None для незнакомой."""
    if not cell.any():
        return " "
    for ch, glyph in _GLYPHS.items():
        if np.array_equal(cell, glyph):
            return ch
    return None


def read_number(frame: np.ndarray, box: tuple[int, int, int, int], digits: int) -> int | None:
    """Число из `digits` ячеек по 8 пикселей; пустые ячейки допустимы только слева."""
    if frame.shape != GAME_SHAPE:
        return None
    y0, y1, x0, _ = box
    mask = _ink(frame, y0, y1, x0, x0 + digits * CELL)
    text = ""
    for i in range(digits):
        ch = _digit(mask[:, i * CELL : (i + 1) * CELL])
        if ch is None or (ch == " " and text.strip()):
            return None
        text += ch
    text = text.strip()
    return int(text) if text else None


def read_score(frame: np.ndarray) -> int | None:
    return read_number(frame, SCORE_BOX, 6)


def read_round(frame: np.ndarray) -> int | None:
    y0, y1 = ROUND_Y
    two = read_number(frame, (y0, y1, ROUND_TWO_X, ROUND_TWO_X + 2 * CELL), 2)
    if two is not None and two >= 10:
        return two
    return read_number(frame, (y0, y1, ROUND_ONE_X, ROUND_ONE_X + CELL), 1)
