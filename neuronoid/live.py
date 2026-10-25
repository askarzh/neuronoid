"""Живой просмотр: последний кадр игры в общей памяти.

Среда (с live=True) пишет кадр после каждого шага, окно scripts/watch.py читает.
Блокировок нет: счётчик кадров — seqlock (нечётный — запись идёт), так что
читатель отбрасывает рваный кадр, а писатель никогда не ждёт читателя.
"""

import json
import os
import struct
import sys
from multiprocessing import shared_memory

import numpy as np

LIVE_NAME = "neuronoid_live"


def live_name() -> str:
    """Имя сегмента; NEURONOID_LIVE_NAME позволяет развести тесты и запуски."""
    return os.environ.get("NEURONOID_LIVE_NAME", LIVE_NAME)
FRAME_SHAPE = (200, 320, 3)
INFO_BYTES = 256
_HEADER = struct.Struct("<QI")  # счётчик, длина JSON
_FRAME_OFFSET = _HEADER.size + INFO_BYTES
SIZE = _FRAME_OFFSET + int(np.prod(FRAME_SHAPE))


def _attach(name: str) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(name=name)
    if sys.platform != "win32":
        # иначе resource_tracker читателя удалит чужой сегмент при выходе
        from multiprocessing import resource_tracker

        resource_tracker.unregister(shm._name, "shared_memory")
    return shm


class LivePublisher:
    def __init__(self, name: str | None = None):
        name = name or live_name()
        try:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=SIZE)
            self._owner = True
        except FileExistsError:  # остался от упавшего процесса — пишем в него
            self._shm = _attach(name)
            self._owner = False
        self._frame = np.ndarray(FRAME_SHAPE, np.uint8, self._shm.buf, offset=_FRAME_OFFSET)
        self._counter = _HEADER.unpack_from(self._shm.buf, 0)[0] & ~1

    def publish(self, frame: np.ndarray, info: dict) -> None:
        data = json.dumps(info, default=str).encode()
        if len(data) > INFO_BYTES:
            data = b"{}"
        buf = self._shm.buf
        _HEADER.pack_into(buf, 0, self._counter + 1, len(data))  # нечётный: пишем
        buf[_HEADER.size : _HEADER.size + len(data)] = data
        if frame.shape == FRAME_SHAPE:
            self._frame[:] = frame
        self._counter += 2
        _HEADER.pack_into(buf, 0, self._counter, len(data))

    def close(self) -> None:
        if self._shm is None:
            return
        del self._frame
        self._shm.close()
        if self._owner:
            self._shm.unlink()
        self._shm = None


class LiveReader:
    def __init__(self, name: str | None = None):
        self._name = name or live_name()
        self._shm: shared_memory.SharedMemory | None = None

    def read(self) -> tuple[np.ndarray, dict, int] | None:
        """Кадр, состояние и номер кадра; None, если публикатора нет или кадр рвётся."""
        if self._shm is None:
            try:
                self._shm = _attach(self._name)
            except FileNotFoundError:
                return None
        buf = self._shm.buf
        for _ in range(5):
            c1, n = _HEADER.unpack_from(buf, 0)
            if c1 == 0 or c1 & 1:
                continue
            info_raw = bytes(buf[_HEADER.size : _HEADER.size + n])
            frame = np.ndarray(FRAME_SHAPE, np.uint8, buf, offset=_FRAME_OFFSET).copy()
            if _HEADER.unpack_from(buf, 0)[0] == c1:
                try:
                    info = json.loads(info_raw)
                except ValueError:
                    info = {}
                return frame, info, c1 // 2
        return None

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            self._shm = None
