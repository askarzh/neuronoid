"""Отслеживание кирпичей по пикселям поля: сколько осталось и сколько разрушено за шаг.

Ячейка становится кирпичом, только если держится STABLE_STEPS шагов подряд, —
падающая капсула, мяч или враг проходят быстрее. Разрушен — устойчивый кирпич
исчез на VANISH_STEPS шагов. Если за шаг исчезло больше MASS_VANISH кирпичей,
это смена экрана (новый раунд, перерисовка поля), а не разрушения.
"""

from dataclasses import dataclass

import numpy as np

from neuronoid.screen import brick_cells

STABLE_STEPS = 4
VANISH_STEPS = 2
MASS_VANISH = 4


@dataclass(frozen=True)
class BrickEvents:
    destroyed: int
    silver_hits: int  # удар по серому кирпичу на экране не виден (журнал этапа 4) — всегда 0
    remaining: int | None


class BrickTracker:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._on: np.ndarray | None = None  # шагов подряд «есть кирпич»
        self._off: np.ndarray | None = None  # шагов подряд «нет» у устойчивого кирпича
        self._stable: np.ndarray | None = None

    @property
    def remaining(self) -> int | None:
        return None if self._stable is None else int(self._stable.sum())

    def update(self, frame: np.ndarray) -> BrickEvents:
        cells = brick_cells(frame)
        if cells is None:
            return BrickEvents(0, 0, None)
        if self._stable is None:
            self._on = np.zeros(cells.shape, np.int32)
            self._off = np.zeros(cells.shape, np.int32)
            self._stable = np.zeros(cells.shape, bool)
        self._on = np.where(cells, self._on + 1, 0)
        self._off = np.where(~cells & self._stable, self._off + 1, 0)
        self._stable |= self._on >= STABLE_STEPS
        vanished = self._stable & (self._off >= VANISH_STEPS)
        n = int(vanished.sum())
        self._stable &= ~vanished
        self._off[vanished] = 0
        destroyed = 0 if n > MASS_VANISH else n
        return BrickEvents(destroyed, 0, self.remaining)
