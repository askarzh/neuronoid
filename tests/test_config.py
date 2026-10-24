import sys

import pytest

from neuronoid.config import ConfigError, core_path, game_path


def test_game_path_missing(tmp_path):
    with pytest.raises(ConfigError, match="DOS_arkanoid.zip"):
        game_path(tmp_path)


def test_core_path_missing_mentions_fetch(tmp_path):
    with pytest.raises(ConfigError, match="fetch_core"):
        core_path(tmp_path)


def test_paths_found(tmp_path):
    (tmp_path / "games/arkanoid").mkdir(parents=True)
    (tmp_path / "games/arkanoid/DOS_arkanoid.zip").write_bytes(b"x")
    (tmp_path / "cores").mkdir()
    name = "dosbox_pure_libretro.dll" if sys.platform == "win32" else "dosbox_pure_libretro.dylib"
    (tmp_path / "cores" / name).write_bytes(b"x")
    assert game_path(tmp_path).name == "DOS_arkanoid.zip"
    assert core_path(tmp_path).name == name
