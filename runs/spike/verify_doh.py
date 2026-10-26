"""Повтор партии судьи (тот же сид) с кадрами после раунда 33: победа над боссом или Game Over."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from neuronoid.config import core_path, game_path
from neuronoid.emulator import LibretroEmulator
from neuronoid.screen import is_title_screen, read_state

seed = int(sys.argv[1])
OUT = Path(f"runs/verify_doh/{seed}")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, r"../neuronoid-strategist")
spec = importlib.util.spec_from_file_location("p", r"../neuronoid-strategist/player.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rng = np.random.default_rng(seed)
with LibretroEmulator(core_path(), game_path()) as emu:
    power_on = emu.save_state()
    emu.load_state(power_on)
    emu.step(frames=int(rng.integers(0, 60)))
    player, started, r33, saved, last = mod.Player(), False, None, 0, None
    for t in range(70 * 60 * 70):
        keys, mouse = player.act(emu.frame())
        emu.step(list(keys), mouse=mouse)
        f = emu.frame()
        s = read_state(f)
        if s.playing:
            started = True
            last = s
            if s.round == 33 and r33 is None:
                r33 = t
                print("round 33 at", t, "score", s.score, "lives", s.lives, flush=True)
        elif r33 is not None and t % 140 == 0 and saved < 40:
            Image.fromarray(f).save(OUT / f"after_t{t:07d}.png")
            saved += 1
        if started and is_title_screen(f):
            Image.fromarray(f).save(OUT / f"title_t{t:07d}.png")
            print("title at", t, "last playing", last, flush=True)
            break
