"""Пути к файлам игры и ядра эмулятора."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Не хватает файла, без которого проект не запустится."""


def core_filename() -> str:
    if sys.platform == "win32":
        return "dosbox_pure_libretro.dll"
    if sys.platform == "darwin":
        return "dosbox_pure_libretro.dylib"
    raise ConfigError(f"Платформа {sys.platform} не поддерживается (нужна Windows или macOS)")


def game_path(root: Path = ROOT) -> Path:
    path = root / "games" / "arkanoid" / "DOS_arkanoid.zip"
    if not path.is_file():
        raise ConfigError(f"Нет игры: {path}. Положи туда свою копию игры (DOS_arkanoid.zip).")
    return path


def core_path(root: Path = ROOT) -> Path:
    path = root / "cores" / core_filename()
    if not path.is_file():
        raise ConfigError(
            f"Нет ядра эмулятора: {path}. Запусти `uv run python scripts/fetch_core.py`."
        )
    return path
