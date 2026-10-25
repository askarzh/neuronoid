"""Сводка графиков обучения из TensorBoard без браузера.

Запуск: uv run python scripts/tb_summary.py runs/ppo1 [--points 8]
"""

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ("neuronoid/points_mean", "neuronoid/score_max", "rollout/ep_rew_mean", "rollout/ep_len_mean", "time/fps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--points", type=int, default=8)
    args = ap.parse_args()
    series: dict[str, list] = {}
    for f in sorted(Path(args.run, "tb").rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(f))
        acc.Reload()
        for tag in TAGS:
            if tag in acc.Tags()["scalars"]:
                series.setdefault(tag, []).extend(acc.Scalars(tag))
    for tag in TAGS:
        events = sorted(series.get(tag, []), key=lambda e: e.step)
        if not events:
            continue
        idx = sorted({round(i * (len(events) - 1) / max(args.points - 1, 1)) for i in range(args.points)})
        print(f"{tag:24s} " + "  ".join(f"{events[i].step // 1000}k:{events[i].value:.1f}" for i in idx))


if __name__ == "__main__":
    main()
