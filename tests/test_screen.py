import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from neuronoid.screen import read_round, read_score

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
