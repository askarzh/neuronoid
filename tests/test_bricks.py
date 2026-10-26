import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from neuronoid.bricks import BrickEvents, BrickTracker
from neuronoid.screen import brick_cells

FRAMES = Path(__file__).parent / "data" / "frames"
LABELS = json.loads((FRAMES / "labels.json").read_text())


def load(name: str) -> np.ndarray:
    return np.asarray(Image.open(FRAMES / name).convert("RGB")).copy()


@pytest.mark.parametrize("name,label", [(n, l) for n, l in LABELS.items() if l.get("bricks") is not None])
def test_brick_count_matches_label(name, label):
    assert int(brick_cells(load(name)).sum()) == label["bricks"]


def test_brick_cells_none_on_menu_frame():
    assert brick_cells(np.zeros((400, 640, 3), np.uint8)) is None


def settled(frame, steps=5):
    t = BrickTracker()
    for _ in range(steps):
        t.update(frame)
    return t


def test_tracker_counts_destroyed_brick():
    f = load("play_start.png")
    t = settled(f)
    g = f.copy()
    g[73:80, 42:56] = 0  # нижний левый кирпич исчез
    events = [t.update(g) for _ in range(3)]
    assert sum(e.destroyed for e in events) == 1 and events[-1].remaining == 65


def test_tracker_ignores_passing_object():
    f = load("play_start.png")
    t = settled(f)
    g = f.copy()
    g[97:104, 42:56] = 200  # «капсула» в пустой ячейке на 2 шага
    t.update(g)
    t.update(g)
    events = [t.update(f) for _ in range(3)]
    assert sum(e.destroyed for e in events) == 0 and events[-1].remaining == 66


def test_tracker_ignores_mass_vanish():
    # смена экрана: целый ряд исчез за один шаг — это не разрушения
    f = load("play_start.png")
    t = settled(f)
    g = f.copy()
    g[33:40, 40:216] = 0
    events = [t.update(g) for _ in range(3)]
    assert sum(e.destroyed for e in events) == 0


def test_tracker_quiet_on_non_game_frames():
    t = settled(load("play_start.png"))
    e = t.update(load("idle_0099_not.png"))
    assert e == BrickEvents(0, 0, None)


def test_remaining_unknown_during_warmup():
    t = BrickTracker()
    f = load("play_start.png")
    assert [t.update(f).remaining for _ in range(5)] == [None, None, None, 66, 66]
