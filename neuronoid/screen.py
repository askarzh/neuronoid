"""Чтение состояния игры с кадра 320×200: только пиксели, никакой памяти эмулятора.

Цифры нарисованы фиксированным шрифтом в ячейках 8×6, поэтому сравнение с
образцами (neuronoid/glyphs.npz) точное: незнакомая ячейка — значит «не прочитал».
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

GAME_SHAPE = (200, 320, 3)
CELL = 8
GLYPH_ROWS = 6
INK = 100  # пиксель «горит», если max(R, G, B) > INK

SCORE_BOX = (63, 69, 241, 289)  # (y0, y1, x0, x1): 6 ячеек по 8 пикселей
# Номер раунда: одна цифра в ячейке x 265–272 (раунды 1–9); двузначный —
# две ячейки с x 257 (проверено на кадре раунда 18).
ROUND_Y = (126, 132)
ROUND_ONE_X = 265
ROUND_TWO_X = 257

_MASKS = dict(np.load(Path(__file__).with_name("glyphs.npz")))
_GLYPHS: dict[str, np.ndarray] = {k: v for k, v in _MASKS.items() if k.isdigit()}
# Титульный экран (после Game Over): слово PLAYER в заголовке таблицы рекордов.
TITLE_BOX = (81, 87, 57, 105)
_TITLE = _MASKS["TITLE"]
# Финал после победы над боссом раунда 33: корабль на звёздном фоне под текстом истории.
SHIP_BOX = (100, 170, 100, 220)
_SHIP = _MASKS["SHIP"]


def _ink(frame: np.ndarray, y0: int, y1: int, x0: int, x1: int, threshold: int = INK) -> np.ndarray:
    return frame[y0:y1, x0:x1].max(axis=2) > threshold


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


# Иконки запасных жизней: ракетки 12×3 в строках 193–195, шаг 16, первая с x 48.
LIVES_Y = (193, 196)
LIVES_X0 = 48
LIVES_PITCH = 16
LIVES_SLOT = 12
LIVES_MAX_X = 216  # дальше — правая стена
_LIFE_ICON = np.array(
    [[0] + [1] * 10 + [0], [1] * 12, [0] + [1] * 10 + [0]], dtype=bool
)

# Стены поля: если они видны, идёт игра (или демо-режим, который похож на игру).
LEFT_WALL_X = (34, 37)
RIGHT_WALL_X = (217, 220)
WALL_Y = (5, 189)


@dataclass(frozen=True)
class ScreenState:
    playing: bool
    score: int | None
    round: int | None
    lives: int | None


def is_playing(frame: np.ndarray) -> bool:
    if frame.shape != GAME_SHAPE:
        return False
    y0, y1 = WALL_Y
    # у стен есть тёмные элементы, поэтому считаем любой не чёрный пиксель:
    # на игровых кадрах покрытие 0.96, на титульном экране и меню — не больше 0.38
    left = _ink(frame, y0, y1, *LEFT_WALL_X, threshold=0).any(axis=1).mean()
    right = _ink(frame, y0, y1, *RIGHT_WALL_X, threshold=0).any(axis=1).mean()
    return bool(left > 0.9 and right > 0.9)


def read_lives(frame: np.ndarray) -> int | None:
    """Число иконок запасных жизней; None, если кадр не игровой или иконку что-то закрыло."""
    if not is_playing(frame):
        return None
    y0, y1 = LIVES_Y
    count, seen_empty = 0, False
    x = LIVES_X0
    while x + LIVES_SLOT <= LIVES_MAX_X:
        slot = _ink(frame, y0, y1, x, x + LIVES_SLOT, threshold=0)  # у иконки есть тёмно-серые пиксели
        if np.array_equal(slot, _LIFE_ICON):
            if seen_empty:
                return None  # иконка после пустого места — что-то не так
            count += 1
        elif slot.any():
            return None
        else:
            seen_empty = True
        x += LIVES_PITCH
    return count


def read_state(frame: np.ndarray) -> ScreenState:
    if not is_playing(frame):
        return ScreenState(False, None, None, None)
    return ScreenState(True, read_score(frame), read_round(frame), read_lives(frame))


def is_title_screen(frame: np.ndarray) -> bool:
    """Титульный экран с таблицей рекордов — сюда игра уходит после Game Over."""
    if frame.shape != GAME_SHAPE:
        return False
    return bool(np.array_equal(_ink(frame, *TITLE_BOX), _TITLE))


def is_ending_screen(frame: np.ndarray) -> bool:
    """Финальная история после победы. Похожий корабль есть во вступлении, поэтому верить — только после начала партии."""
    if frame.shape != GAME_SHAPE:
        return False
    return bool(np.array_equal(_ink(frame, *SHIP_BOX), _SHIP))


# Сетка кирпичей: ячейки 16×8, столбцы x = 40 + 16·c (11 штук), ряды y = 9 + 8·r.
# Ниже BRICK_ROWS кирпичей не бывает — там ходит ракетка.
BRICK_COLS = 11
BRICK_ROWS = 18
BRICK_X0, BRICK_Y0 = 40, 9
BRICK_W, BRICK_H = 16, 8
BRICK_FILL = 0.8  # доля не чёрных пикселей внутри ячейки (14×7), чтобы считать её кирпичом


def brick_cells(frame: np.ndarray) -> np.ndarray | None:
    """Сетка «есть кирпич» (BRICK_ROWS × BRICK_COLS) по одному кадру; None для не игрового кадра."""
    if not is_playing(frame):
        return None
    y1 = BRICK_Y0 + BRICK_ROWS * BRICK_H
    x1 = BRICK_X0 + BRICK_COLS * BRICK_W
    ink = frame[BRICK_Y0:y1, BRICK_X0:x1].max(axis=2) > 0
    cells = ink.reshape(BRICK_ROWS, BRICK_H, BRICK_COLS, BRICK_W)[:, : BRICK_H - 1, :, 2:]
    return cells.mean(axis=(1, 3)) >= BRICK_FILL
