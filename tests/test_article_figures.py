"""Данные и графики для статьи."""

import sys
from pathlib import Path

import pytest

ARTICLE = Path(__file__).resolve().parents[1] / "docs" / "article"
sys.path.insert(0, str(ARTICLE))

from figures import plot_progress, researcher_progress, rl_curve  # noqa: E402

CSV = ARTICLE / "data" / "researcher_progress.csv"


def test_rl_curve_reads_step_from_filename(tmp_path):
    d = tmp_path / "eval"
    d.mkdir()
    (d / "13000320-stoch-s1000.json").write_text('{"games":[{"score":7000},{"score":9000}]}')
    assert rl_curve([d]) == [(13000320, 8000.0)]


def test_rl_curve_pools_seeds_and_sorts(tmp_path):
    d = tmp_path / "eval"
    d.mkdir()
    (d / "2000-stoch-s1000.json").write_text('{"games":[{"score":1000}]}')
    (d / "2000-stoch-s2000.json").write_text('{"games":[{"score":3000}]}')
    (d / "1000-stoch-s1000.json").write_text('{"games":[{"score":500}]}')
    assert rl_curve([d]) == [(1000, 500.0), (2000, 2000.0)]


def test_researcher_hours():
    first = researcher_progress(CSV)[1]
    assert first == (pytest.approx(0.3), 101820)


def test_progress_two_separate_panels(tmp_path):
    fig = plot_progress([(1, 1.0)], [(0.3, 101820)], tmp_path / "p.png")
    a, b = fig.axes
    assert len(fig.axes) == 2 and not a.get_shared_x_axes().joined(a, b) and not a.get_shared_y_axes().joined(a, b)
    assert (tmp_path / "p.png").exists()


def test_cycle_png_written(tmp_path):
    from figures import plot_cycle
    plot_cycle(tmp_path / "c.png", "en")
    assert (tmp_path / "c.png").stat().st_size > 0
