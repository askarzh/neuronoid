"""Собирает образцы цифр (маски 6×8) из размеченных кадров счёта.

Запуск: uv run python tools/make_glyphs.py  ->  neuronoid/glyphs.npz
"""

import json

import numpy as np
from PIL import Image

from neuronoid.config import ROOT

FRAMES = ROOT / "tests" / "data" / "frames"
Y0, Y1, X0, CELL, DIGITS = 63, 69, 241, 8, 6


def main() -> None:
    labels = json.loads((FRAMES / "labels.json").read_text())
    glyphs: dict[str, np.ndarray] = {}
    for name, lab in labels.items():
        if lab["score"] is None:
            continue
        frame = np.asarray(Image.open(FRAMES / name).convert("RGB"))
        mask = frame[Y0:Y1].max(axis=2) > 100
        for i, ch in enumerate(f"{lab['score']:06d}"):
            cell = mask[:, X0 + i * CELL : X0 + (i + 1) * CELL]
            if ch in glyphs and not np.array_equal(glyphs[ch], cell):
                raise SystemExit(f"цифра {ch} в {name} не совпадает с уже собранной — проверь разметку")
            glyphs[ch] = cell
    missing = sorted(set("0123456789") - set(glyphs))
    if missing:
        raise SystemExit(f"не хватает цифр: {missing}")
    np.savez(ROOT / "neuronoid" / "glyphs.npz", **glyphs)
    print("цифры собраны:", "".join(sorted(glyphs)))


if __name__ == "__main__":
    main()
