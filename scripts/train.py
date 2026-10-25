"""Обучение агента.

Запуск: uv run python scripts/train.py --run ppo1 --steps 5000000 [--envs 10] [--resume]
Смотреть игру: uv run python scripts/watch.py
Графики: uv run tensorboard --logdir runs/ppo1/tb
"""

import argparse

from neuronoid.config import ROOT, ConfigError, core_path, game_path
from neuronoid.train import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=5_000_000, help="сколько шагов всего (с --resume — цель)")
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--checkpoint-every", type=int, default=200_000)
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()
    try:
        core_path(), game_path()
    except ConfigError as e:
        raise SystemExit(str(e))
    run_dir = ROOT / "runs" / args.run
    print(f"Обучение: {run_dir}")
    print("  окно:    uv run python scripts/watch.py")
    print(f"  графики: uv run tensorboard --logdir {run_dir / 'tb'}")
    print("  остановить: Ctrl+C — модель сохранится; продолжить: та же команда с --resume")
    try:
        latest = train(run_dir, args.steps, n_envs=args.envs, seed=args.seed, resume=args.resume,
                       device=args.device, checkpoint_every=args.checkpoint_every, live=not args.no_live)
    except (FileExistsError, FileNotFoundError) as e:
        raise SystemExit(str(e))
    print(f"Сохранено: {latest}")


if __name__ == "__main__":
    main()
