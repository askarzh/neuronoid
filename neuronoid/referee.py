"""Судья для внешнего игрока (стратега): игра с включения питания, итог — по экрану.

Игрок — класс с методом `act(frame) -> (keys, Mouse)`, который вызывается на каждом кадре.
Судья включает игру с самого начала (как после включения), ждёт случайное число кадров
(чтобы игры различались), отдаёт управление игроку и по проверенным датчикам экрана
следит за счётом и раундом до конца игры или лимита кадров. Конец игры — титульный экран;
победа — если перед ним был финал с кораблём (после босса раунда 33).
"""

from collections.abc import Callable

import numpy as np

from neuronoid.config import core_path, game_path
from neuronoid.emulator import LibretroEmulator
from neuronoid.screen import is_ending_screen, is_title_screen, read_state

CONFIRM_FRAMES = 3  # счёт засчитывается, если одно значение прочитано столько кадров подряд


def judge(player_factory: Callable, games: int = 10, seed: int = 1000, max_frames: int = 70 * 60 * 20) -> dict:
    rng = np.random.default_rng(seed)
    results = []
    with LibretroEmulator(core_path(), game_path()) as emu:
        power_on = emu.save_state()
        for _ in range(games):
            emu.load_state(power_on)
            emu.step(frames=int(rng.integers(0, 60)))
            player = player_factory()
            started = over = won = False
            best_score, best_round, streak, last = 0, 1, 0, None
            frames = 0
            while frames < max_frames:
                frame = emu.frame()
                keys, mouse = player.act(frame)
                emu.step(list(keys), mouse=mouse)
                frames += 1
                frame = emu.frame()
                state = read_state(frame)
                if state.playing:
                    started = True
                    streak = streak + 1 if state.score == last else 1
                    last = state.score
                    if state.score is not None and streak >= CONFIRM_FRAMES:
                        best_score = max(best_score, state.score)
                    if state.round is not None:
                        best_round = max(best_round, state.round)
                elif started and not won and is_ending_screen(frame):
                    won = True
                elif started and is_title_screen(frame):
                    over = True
                    break
            results.append({"started": started, "game_over": over, "won": won, "score": best_score,
                            "round": best_round, "frames": frames})
    scores = [g["score"] for g in results]
    return {"games": results, "mean_score": float(np.mean(scores)) if scores else 0.0,
            "max_round": max(g["round"] for g in results) if results else 0,
            "wins": sum(g["won"] for g in results)}
