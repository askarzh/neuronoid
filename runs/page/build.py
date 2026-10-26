"""Собирает страницу «Исследователь Арканоида» из шаблона: график, зоны отскока, кадр, итоги судьи."""

import base64
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

POINTS = [  # (время, счёт, подпись) — счёт за 20 минут на стенде исследователя
    ("12:07", 28950, "v0 · р. 3"),
    ("12:16", 101820, "v1 · р. 9"),
    ("12:43", 117670, "v2 · р. 13"),
    ("12:53", 140340, "капсулы"),
    ("13:05", 150320, "лазер · р. 14"),
    ("13:14", 172390, "темп · р. 15"),
    ("14:04", 176805, "мультимяч · среднее 4"),
]
RL_BEST = 8301


def minutes(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m


def chart() -> str:
    W, H, L, R, T, B = 900, 330, 64, 24, 24, 40
    x0, x1 = minutes("12:00"), minutes("14:15")
    ymax = 200_000
    sx = lambda m: L + (m - x0) / (x1 - x0) * (W - L - R)
    sy = lambda v: T + (1 - v / ymax) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    for v in range(0, ymax + 1, 50_000):
        y = sy(v)
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--rule)" stroke-width="1"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end">{v // 1000} тыс.</text>')
    for t in ("12:00", "12:30", "13:00", "13:30", "14:00"):
        x = sx(minutes(t))
        out.append(f'<text x="{x:.1f}" y="{H - 14}" text-anchor="middle">{t}</text>')
    yb = sy(RL_BEST)
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{yb:.1f}" y2="{yb:.1f}" stroke="var(--second)" stroke-width="1.5" stroke-dasharray="5 4"/>')
    out.append(f'<text x="{W - R}" y="{yb - 7:.1f}" text-anchor="end" fill="var(--second)" style="fill:var(--second)">RL-агент, лучшее среднее за 17 млн шагов: 8 301</text>')
    pts = [(sx(minutes(t)), sy(v), v, lab) for t, v, lab in POINTS]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _, _) in enumerate(pts))
    out.append(f'<path d="{path}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round"/>')
    for i, (x, y, v, lab) in enumerate(pts):
        below = i in (2, 4)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="var(--accent)" stroke="var(--surface)" stroke-width="2"/>')
        ty = y + 22 if below else y - 22
        anchor = "end" if i == len(pts) - 1 else "middle"
        out.append(f'<text class="lbl" x="{x:.1f}" y="{ty:.1f}" text-anchor="{anchor}">{v:,}'.replace(",", " ") + "</text>")
        out.append(f'<text x="{x:.1f}" y="{ty + 13:.1f}" text-anchor="{anchor}">{lab}</text>')
    out.append("</svg>")
    return "".join(out)


def zones() -> str:
    data = [("−12…−9", (-2.4, -0.75), "пологий влево"), ("−8…−5", (-1.75, -1.75), "45° влево"),
            ("−4…−1", (-0.65, -2.45), "крутой влево"), ("0…3", (0.65, -2.45), "крутой вправо"),
            ("4…7", (1.75, -1.75), "45° вправо"), ("8…12", (2.4, -0.75), "пологий вправо")]
    out = []
    for off, (vx, vy), name in data:
        x2, y2 = 27 + vx * 9.5, 31 + vy * 9.5
        out.append(
            f'<div><svg viewBox="0 0 54 34" aria-hidden="true">'
            f'<rect x="15" y="30" width="24" height="3" rx="1" fill="var(--muted)"/>'
            f'<line x1="27" y1="30" x2="{x2:.1f}" y2="{y2:.1f}" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round"/>'
            f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="2.6" fill="var(--accent)"/></svg>'
            f"<b>{off}</b><span>{name}</span></div>")
    return "".join(out)


def image() -> str:
    im = Image.open(ROOT / "tests" / "data" / "frames" / "round18_play.png").convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def judge() -> tuple[str, str]:
    games = []
    for log in sorted(ROOT.glob("runs/referee_strategist_*.log")):
        last = [l for l in log.read_text(encoding="utf-8").splitlines() if l.startswith("{")][-1]
        games += json.loads(last)["games"]
    scores = [g["score"] for g in games]
    rounds = sorted(g["round"] for g in games)
    mean = f"{sum(scores) / len(scores):,.0f}".replace(",", " ")
    return f"{mean} (10 игр)", f"{rounds[0]}–{rounds[-1]}"


def main() -> None:
    html = (HERE / "researcher.template.html").read_text(encoding="utf-8")
    mean, rounds = judge()
    for key, val in {"CHART": chart(), "ZONES": zones(), "IMG": image(), "JUDGE_MEAN": mean, "JUDGE_ROUNDS": rounds}.items():
        html = html.replace("{{" + key + "}}", val)
    assert not re.search(r"\{\{[A-Z_]+\}\}", html)
    (HERE / "researcher.html").write_text(html, encoding="utf-8")
    print("ok", mean, rounds, len(html))


if __name__ == "__main__":
    main()
