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


@dataclass(frozen=True)
class StartMeta:
    round: int
    progress: int  # очки, набранные в этом раунде: грубая мера «сколько стены разбито»
    score: int
    lives: int


def _remove(path: Path) -> None:
    for p in (path.with_suffix(".json"), path.with_suffix(".state")):
        try:
            p.unlink()
        except OSError:
            # уже удалён другим процессом или (Windows) сейчас читается им —
            # удалит следующая чистка корзины
            pass


class StartPool:
    def __init__(self, root: Path, per_round: int = PER_ROUND):
        self.root = Path(root)
        self.per_round = per_round
        self.root.mkdir(parents=True, exist_ok=True)

    def _entries(self, bucket: Path) -> list[tuple[StartMeta, Path]]:
        out = []
        for meta_path in bucket.glob("*.json"):
            try:
                meta = StartMeta(**json.loads(meta_path.read_text()))
            except (OSError, ValueError, TypeError):
                continue
            out.append((meta, meta_path))
        return sorted(out, key=lambda e: e[0].progress)

    def _buckets(self) -> list[Path]:
        return sorted(p for p in self.root.iterdir() if p.is_dir() and p.name.isdigit())

    def offer(self, state: bytes, meta: StartMeta) -> bool:
        """Сохраняет позицию, если корзина раунда не полна или позиция лучше худшей."""
        bucket = self.root / str(meta.round)
        bucket.mkdir(exist_ok=True)
        entries = self._entries(bucket)
        if len(entries) >= self.per_round and meta.progress <= entries[0][0].progress:
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
            entries = self._entries(bucket)
            for _, path in entries[: max(0, len(entries) - self.per_round)]:
                _remove(path)

    def sample(self, rng: np.random.Generator) -> tuple[bytes, StartMeta, Path] | None:
        """Корзина равновероятно среди непустых, позиция — из лучшей половины корзины."""
        while True:
            buckets = [(b, e) for b in self._buckets() if (e := self._entries(b))]
            if not buckets:
                return None
            _, entries = buckets[int(rng.integers(len(buckets)))]
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
