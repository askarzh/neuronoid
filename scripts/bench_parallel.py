"""Замер суммарной скорости эмуляции при N параллельных процессах.

Запуск: uv run python scripts/bench_parallel.py --procs 1 4 8 12 14 --frames 5000
Каждый процесс доходит до раунда 1, ждёт остальных и гоняет кадры со случайной
мышью, забирая кадр каждые 4 кадра (как при обучении с frame skip 4).
"""

import argparse
import multiprocessing as mp
import time

import numpy as np

FRAME_SKIP = 4
ACTIONS_DX = (-10, -4, 0, 4, 10)


def worker(seed: int, frames: int, barrier, results) -> None:
    from neuronoid.boot import boot_to_round1
    from neuronoid.config import core_path, game_path
    from neuronoid.emulator import LibretroEmulator, Mouse

    rng = np.random.default_rng(seed)
    with LibretroEmulator(core_path(), game_path()) as emu:
        boot_to_round1(emu)
        start = emu.save_state()
        barrier.wait()
        t = time.perf_counter()
        for i in range(frames // FRAME_SKIP):
            mouse = Mouse(dx=int(rng.choice(ACTIONS_DX)), left=bool(rng.random() < 0.05))
            emu.step(mouse=mouse, frames=FRAME_SKIP)
            emu.frame()
            if i % 500 == 499:
                emu.load_state(start)  # как сброс эпизода
        results.put(time.perf_counter() - t)


def run(procs: int, frames: int) -> tuple[float, float]:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(procs)
    results = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(i, frames, barrier, results)) for i in range(procs)]
    for p in ps:
        p.start()
    times = [results.get() for _ in ps]
    for p in ps:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"процесс упал с кодом {p.exitcode}")
    per_proc = frames / float(np.mean(times))
    total = procs * frames / max(times)
    return total, per_proc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 4, 8, 12, 14])
    ap.add_argument("--frames", type=int, default=5000)
    args = ap.parse_args()
    print("| процессов | суммарно кадров/с | на процесс кадров/с |")
    print("|---|---|---|")
    for n in args.procs:
        total, per = run(n, args.frames)
        print(f"| {n} | {total:.0f} | {per:.0f} |", flush=True)


if __name__ == "__main__":
    main()
