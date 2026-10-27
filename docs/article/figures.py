"""Иллюстрации к статье: прогресс RL и исследователя, зоны отскока, цикл исследователя, кадр финала.

Запуск: python docs/article/figures.py — пишет img/progress*, bounce*, cycle* (ru и en) и final.png.
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
START = "11:58"  # коммит c01c024: исследователь получил «тело» и задание

INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
RL_COLOR, RES_COLOR = "#8250df", "#0969da"

TEXT = {
    "ru": {
        "rl_title": "RL-агент: средний счёт за партию",
        "rl_x": "шаги обучения, млн",
        "res_title": "Исследователь на LLM: счёт за 20 минут игры",
        "res_x": "часы от получения задания",
        "res_note": "стенд исследователя",
        "y": "очки",
        "bounce_title": "Куда летит мяч: зависит только от точки удара",
        "bounce_x": "смещение точки удара от центра ракетки, пикс.",
        "bounce_note": "замеры исследователя; угол подлёта не влияет",
        "cycle": ["наблюдение\nкадры экрана", "гипотеза", "эксперимент\nс сохранения", "код игрока",
                  "проверка\nпо партиям", "разбор\nпотерь"],
        "cycle_center": "LLM-исследователь",
        "human": "человек ставит задачу\nи запускает починку",
    },
    "en": {
        "rl_title": "RL agent: mean score per game",
        "rl_x": "training steps, millions",
        "res_title": "LLM researcher: score in 20 minutes of play",
        "res_x": "hours since the task was given",
        "res_note": "researcher's own test bench",
        "y": "points",
        "bounce_title": "Where the ball goes depends only on the hit point",
        "bounce_x": "hit offset from paddle centre, px",
        "bounce_note": "researcher's measurements; incoming angle doesn't matter",
        "cycle": ["observe\nscreen frames", "hypothesis", "experiment\nfrom a save state", "player code",
                  "check\nover games", "analyse\nlosses"],
        "cycle_center": "LLM researcher",
        "human": "a human sets the task\nand triggers each fix",
    },
}

# Зоны отскока по замерам исследователя (neuronoid-strategist/LOG.md §7): (от, до, vx, vy).
BOUNCE_ZONES = [(-12, -9, -2.4, -0.75), (-8, -5, -1.75, -1.75), (-4, -1, -0.65, -2.45),
                (0, 3, 0.65, -2.45), (4, 7, 1.75, -1.75), (8, 12, 2.4, -0.75)]


def rl_curve(eval_dirs: list[Path]) -> list[tuple[int, float]]:
    """(шаг, средний счёт) по всем партиям оценки на этом шаге; файлы <шаг>-stoch-s<сид>.json."""
    by_step: dict[int, list[float]] = {}
    for d in eval_dirs:
        for f in d.glob("*-stoch-s*.json"):
            step = int(re.match(r"(\d+)-", f.name).group(1))
            by_step.setdefault(step, []).extend(g["score"] for g in json.loads(f.read_text())["games"])
    return [(s, sum(v) / len(v)) for s, v in sorted(by_step.items())]


def researcher_progress(path: Path) -> list[tuple[float, int]]:
    """(часы от START, счёт) по коммитам исследователя."""
    t0 = datetime.strptime(START, "%H:%M")
    with path.open(encoding="utf-8") as f:
        return [((datetime.strptime(r["time"], "%H:%M") - t0).seconds / 3600, int(r["score"]))
                for r in csv.DictReader(f)]


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace(",", " "))


def plot_progress(rl: list[tuple[int, float]], res: list[tuple[float, int]], out: Path, lang: str = "ru") -> Figure:
    """Две отдельные панели: шкалы времени несопоставимы, поэтому общих осей нет."""
    t = TEXT[lang]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    a.plot([s / 1e6 for s, _ in rl], [v for _, v in rl], color=RL_COLOR, linewidth=2, marker="o", markersize=4)
    _style(a, t["rl_title"], t["rl_x"], t["y"])
    a.set_ylim(bottom=0)
    b.plot([h for h, _ in res], [v for _, v in res], color=RES_COLOR, linewidth=2, marker="o", markersize=4)
    _style(b, t["res_title"], t["res_x"], t["y"])
    b.set_ylim(bottom=0)
    b.text(0.98, 0.04, t["res_note"], transform=b.transAxes, ha="right", color=MUTED, fontsize=9)
    fig.savefig(out, dpi=150)
    return fig


def plot_bounce(out: Path, lang: str = "ru") -> Figure:
    """Ракетка, шесть зон и направление вылета мяча из каждой."""
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    for lo, hi, vx, vy in BOUNCE_ZONES:
        mid = (lo + hi) / 2
        ax.add_patch(plt.Rectangle((lo - 0.4, -0.6), hi - lo + 0.8, 0.6, color=GRID, ec="white", lw=2))
        ax.annotate("", xy=(mid + vx * 1.6, -vy * 1.6), xytext=(mid, 0),
                    arrowprops={"arrowstyle": "-|>", "color": RES_COLOR, "lw": 2})
        ax.text(mid, -1.1, f"{lo}…{hi}", ha="center", color=MUTED, fontsize=9)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-1.6, 4.4)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(t["bounce_title"], loc="left", fontsize=12, color=INK)
    ax.set_xlabel(t["bounce_x"], color=MUTED)
    ax.text(15, 4.2, t["bounce_note"], ha="right", color=MUTED, fontsize=9)
    fig.savefig(out, dpi=150)
    return fig


def plot_cycle(out: Path, lang: str = "ru") -> Figure:
    """Цикл исследователя по кругу; человек снаружи — ставит задачу."""
    import math
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    steps = t["cycle"]
    pts = [(math.cos(math.pi / 2 - 2 * math.pi * i / len(steps)), math.sin(math.pi / 2 - 2 * math.pi * i / len(steps)))
           for i in range(len(steps))]
    for i, ((x, y), label) in enumerate(zip(pts, steps)):
        nx, ny = pts[(i + 1) % len(pts)]
        ax.annotate("", xy=(x + (nx - x) * 0.72, y + (ny - y) * 0.72), xytext=(x + (nx - x) * 0.28, y + (ny - y) * 0.28),
                    arrowprops={"arrowstyle": "-|>", "color": RES_COLOR, "lw": 2})
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color=INK,
                bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": GRID, "lw": 1.5})
    ax.text(0, 0, t["cycle_center"], ha="center", va="center", fontsize=12, color=RES_COLOR, weight="bold")
    ax.text(0, -1.55, t["human"], ha="center", va="center", fontsize=10, color=MUTED,
            bbox={"boxstyle": "round,pad=0.5", "fc": "#f6f8fa", "ec": GRID, "ls": "--"})
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.9, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out, dpi=150)
    return fig


def final_frame(out: Path) -> None:
    """Кадр финала (победа над DOH, сид 3002), увеличенный в 3 раза без сглаживания."""
    from PIL import Image
    im = Image.open(ROOT / "runs" / "verify_doh" / "3002" / "after_t0203140.png")
    im.resize((im.width * 3, im.height * 3), Image.NEAREST).save(out)


def main() -> None:
    rl = rl_curve(sorted((ROOT / "runs").glob("ppo[1-4]/eval")))
    res = researcher_progress(HERE / "data" / "researcher_progress.csv")
    img = HERE / "img"
    img.mkdir(exist_ok=True)
    for lang, suffix in (("ru", ""), ("en", "_en")):
        plot_progress(rl, res, img / f"progress{suffix}.png", lang)
        plot_bounce(img / f"bounce{suffix}.png", lang)
        plot_cycle(img / f"cycle{suffix}.png", lang)
    final_frame(img / "final.png")


if __name__ == "__main__":
    main()
