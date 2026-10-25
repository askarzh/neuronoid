"""Глушит служебный вывод ядра DOSBox, оставляя print() Python на консоли."""

import os
import sys


def silence_native_stdout() -> None:
    """Перенаправляет дескрипторы 1 и 2 в devnull; sys.stdout/sys.stderr — на исходную консоль.

    Ядро пишет в stdout напрямую, мимо Python, и десяток процессов сред залил бы консоль.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    out, err = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    sys.stdout = os.fdopen(out, "w", buffering=1, encoding="utf-8", errors="replace")
    sys.stderr = os.fdopen(err, "w", buffering=1, encoding="utf-8", errors="replace")
