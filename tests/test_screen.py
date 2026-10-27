import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from neuronoid.screen import ScreenState, is_ending_screen, is_playing, is_title_screen, read_lives, read_round, read_score, read_state

FRAMES = Path(__file__).parent / "data" / "frames"
LABELS = json.loads((FRAMES / "labels.json").read_text())


def load(name: str) -> np.ndarray:
    return np.asarray(Image.open(FRAMES / name).convert("RGB")).copy()


@pytest.mark.parametrize("name,label", LABELS.items())
def test_read_score_matches_label(name, label):
    assert read_score(load(name)) == label["score"]


@pytest.mark.parametrize("name,label", LABELS.items())
def test_read_round_matches_label(name, label):
    assert read_round(load(name)) == label["round"]


def test_read_number_rejects_menu_frame():
    assert read_score(np.zeros((400, 640, 3), np.uint8)) is None


def test_read_number_rejects_garbage_cell():
    f = load("play_01699_score.png")
    f[64, 243] ^= 255  # испортить один пиксель цифры
    assert read_score(f) is None


@pytest.mark.parametrize("name,label", LABELS.items())
def test_read_lives_matches_label(name, label):
    assert read_lives(load(name)) == label["lives"]


@pytest.mark.parametrize("name,label", LABELS.items())
def test_is_playing_matches_label(name, label):
    assert is_playing(load(name)) == label["playing"]


def test_read_state_on_menu_frame_is_empty():
    s = read_state(np.zeros((400, 640, 3), np.uint8))
    assert s == ScreenState(False, None, None, None)


def test_read_lives_unknown_when_capsule_covers_icon():
    f = load("play_start.png")
    f[193:196, 52:56] = 0  # иконку частично закрыл чужой объект
    assert read_lives(f) is None


TITLE_FRAMES = {n for n in LABELS if n.startswith("idle_") and n.endswith("_not.png")} | {"menu_2.png"}


@pytest.mark.parametrize("name", sorted(LABELS))
def test_is_title_screen(name):
    assert is_title_screen(load(name)) == (name in TITLE_FRAMES)


@pytest.mark.parametrize("name", sorted(LABELS))
def test_is_ending_screen(name):
    # Корабль после победы над боссом; во вступлении корабль похож, но не совпадает пиксель в пиксель.
    assert is_ending_screen(load(name)) == (name == "ending_story.png")
