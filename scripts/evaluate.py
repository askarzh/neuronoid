"""Оценка агента и зачёт ступеней М1–М3.

Запуск: uv run python scripts/evaluate.py runs/ppo1 [--games 10] [--deterministic]
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from neuronoid.config import ConfigError
from neuronoid.env import ArkanoidEnv
from neuronoid.evaluate import GAMES, MILESTONES, PASS_COUNT, evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="папка запуска (runs/NAME) или файл чекпоинта .zip")
    ap.add_argument("--games", type=int, default=GAMES)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    target = Path(args.target)
    ckpt = target if target.suffix == ".zip" else target / "checkpoints" / "latest.zip"
    run_dir = ckpt.parent.parent
    if not ckpt.exists():
        raise SystemExit(f"Нет чекпоинта: {ckpt}")
    model = PPO.load(ckpt, device="cpu")
    env = ArkanoidEnv(life_loss_terminal=False)
    try:
        report = evaluate(
            lambda obs: int(model.predict(obs, deterministic=args.deterministic)[0]),
            env, args.games, seed=args.seed, on_game_start=model.set_random_seed,
        )
    except ConfigError as e:
        raise SystemExit(str(e))
    finally:
        env.close()
    print(f"Чекпоинт: {ckpt} ({model.num_timesteps} шагов)")
    for i, g in enumerate(report.games, 1):
        print(f"  игра {i:2d}: счёт {g.score:6d}, раунд {g.round}, шагов {g.steps}")
    for name in MILESTONES:
        print(f"  {name.upper()}: {'зачёт' if getattr(report, name) else 'нет'} (нужно {PASS_COUNT} из {args.games})")
    mode = "det" if args.deterministic else "stoch"
    out = run_dir / "eval" / f"{model.num_timesteps}-{mode}-s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_json(), encoding="utf-8")
    print(f"Отчёт: {out}")


if __name__ == "__main__":
    main()
