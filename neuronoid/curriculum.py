"""Учебная программа: общий пул стартовых позиций, до которых агент дошёл сам.

Позиции лежат на диске (`<root>/<раунд>/<прогресс>-<uuid>.state` + `.json`) и
общие для всех процессов сред. Блокировок нет: запись через временный файл и
os.replace, позиция видна только когда есть её .json; гонки при удалении
безвредны (файл уже удалён другим процессом — и ладно).
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PER_ROUND = 40
FRONT_ROUND = 1  # раунд, на котором агент застревает: туда — заметная доля стартов
FRONT_SHARE = 0.5
# Не больше стольких позиций с одинаковым числом оставшихся кирпичей: иначе корзина
# застывает на «1 кирпич», и середина конца раунда (5–15 кирпичей) в неё не попадает.
PER_BRICKS_VALUE = 5


@dataclass(frozen=True)
class StartMeta:
    round: int
    progress: int  # очки, набранные в этом раунде: грубая мера «сколько стены разбито»
    score: int
    lives: int
    bricks_left: int | None = None  # с экрана; у позиций до этапа 4 его нет


def rank(meta: StartMeta) -> tuple:
    """Чем больше, тем ценнее позиция: меньше кирпичей осталось, иначе — больше прогресс.

    Позиции без числа кирпичей (старые) уступают любой позиции с ним.
    """
    if meta.bricks_left is not None:
        return (1, -meta.bricks_left, meta.progress)
    return (0, meta.progress)


def _remove(path: Path) -> None:
    for p in (path.with_suffix(".json"), path.with_suffix(".state")):
        try:
            p.unlink()
        except OSError:
            # уже удалён другим процессом или (Windows) сейчас читается им —
            # удалит следующая чистка корзины
            pass


class StartPool:
    def __init__(self, root: Path, per_round: int = PER_ROUND, front_round: int = FRONT_ROUND,
                 front_share: float = FRONT_SHARE, per_bricks_value: int = PER_BRICKS_VALUE):
        self.root = Path(root)
        self.per_round = per_round
        self.front_round = front_round
        self.front_share = front_share
        self.per_bricks_value = per_bricks_value
        self.root.mkdir(parents=True, exist_ok=True)

    def _entries(self, bucket: Path) -> list[tuple[StartMeta, Path]]:
        out = []
        for meta_path in bucket.glob("*.json"):
            try:
                meta = StartMeta(**json.loads(meta_path.read_text()))
            except (OSError, ValueError, TypeError):
                continue
            out.append((meta, meta_path))
        return sorted(out, key=lambda e: rank(e[0]))  # худшие в начале

    def _buckets(self) -> list[Path]:
        return sorted(p for p in self.root.iterdir() if p.is_dir() and p.name.isdigit())

    def offer(self, state: bytes, meta: StartMeta) -> bool:
        """Сохраняет позицию, если корзина раунда не полна или позиция лучше худшей."""
        bucket = self.root / str(meta.round)
        bucket.mkdir(exist_ok=True)
        entries = self._entries(bucket)
        if meta.bricks_left is not None:
            same = [e for e in entries if e[0].bricks_left == meta.bricks_left]
            if len(same) >= self.per_bricks_value and rank(meta) <= rank(same[0][0]):
                return False  # таких позиций уже хватает, а эта не лучше худшей из них
        if len(entries) >= self.per_round and rank(meta) <= rank(entries[0][0]):
            return False
        stem = bucket / f"{meta.progress}-{uuid.uuid4().hex}"
        try:
            for suffix, data in ((".state", state), (".json", json.dumps(asdict(meta)).encode())):
                tmp = stem.with_name(stem.name + suffix + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, stem.with_suffix(suffix))  # .state раньше .json
        except OSError:
            # диск, антивирус или индексатор держит файл: позицию просто не берём,
            # обучение из-за этого падать не должно
            for leftover in (stem.with_name(stem.name + ".state.tmp"), stem.with_name(stem.name + ".json.tmp"),
                             stem.with_suffix(".state")):
                try:
                    leftover.unlink()
                except OSError:
                    pass
            return False
        self.prune(meta.round)
        return True

    def prune(self, round_: int | None = None) -> None:
        """Удаляет худшие позиции сверх лимита (на Windows занятый файл удалится при следующей чистке)."""
        buckets = [self.root / str(round_)] if round_ is not None else self._buckets()
        for bucket in buckets:
            groups: dict[int, list] = {}
            for meta, path in self._entries(bucket):
                if meta.bricks_left is not None:
                    groups.setdefault(meta.bricks_left, []).append(path)
            for paths in groups.values():  # худшие в начале каждой группы
                for path in paths[: max(0, len(paths) - self.per_bricks_value)]:
                    _remove(path)
            entries = self._entries(bucket)
            for _, path in entries[: max(0, len(entries) - self.per_round)]:
                _remove(path)

    def sample(self, rng: np.random.Generator) -> tuple[bytes, StartMeta, Path] | None:
        """Корзина фронта — с вероятностью front_share, иначе равновероятно из остальных;
        позиция — из лучшей половины корзины."""
        while True:
            buckets = [(b, e) for b in self._buckets() if (e := self._entries(b))]
            if not buckets:
                return None
            front = [x for x in buckets if x[0].name == str(self.front_round)]
            rest = [x for x in buckets if x[0].name != str(self.front_round)]
            if front and (not rest or rng.random() < self.front_share):
                _, entries = front[0]
            else:
                _, entries = rest[int(rng.integers(len(rest)))]
            top = entries[len(entries) // 2 :]
            meta, path = top[int(rng.integers(len(top)))]
            try:
                return path.with_suffix(".state").read_bytes(), meta, path
            except OSError:
                _remove(path)  # недописанная или чужая гонка — выбросить и выбрать другую

    def forget(self, meta_path: Path) -> None:
        _remove(meta_path)

    def stats(self) -> dict[int, int]:
        return {int(b.name): n for b in self._buckets() if (n := len(self._entries(b)))}


def annotate_bricks(root: Path, emu) -> int:
    """Досчитывает bricks_left позициям без него (пул до этапа 4): грузит позицию и считает кирпичи.

    Возвращает число обновлённых позиций.
    """
    from neuronoid.screen import brick_cells

    pool = StartPool(root)
    done = 0
    for bucket in pool._buckets():
        for meta, path in pool._entries(bucket):
            if meta.bricks_left is not None:
                continue
            try:
                emu.load_state(path.with_suffix(".state").read_bytes())
            except (OSError, RuntimeError):
                _remove(path)
                continue
            cells = brick_cells(emu.frame())
            if cells is None:
                _remove(path)  # позиция не игровая — в пуле ей не место
                continue
            data = asdict(meta) | {"bricks_left": int(cells.sum())}
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, path)
            done += 1
    return done
