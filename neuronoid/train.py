"""Обучение PPO на параллельных ArkanoidEnv: чекпоинты, возобновление, графики."""

import _thread
import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from neuronoid.curriculum import StartPool
from neuronoid.vec import make_vec_env

# В духе Atari-настроек RL Zoo.
PPO_KWARGS: dict = {
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 4,
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
}
LEARNING_RATE = 2.5e-4
WATCHDOG_SECONDS = 300  # столько без единого шага — считаем, что среда зависла


def global_linear(done: int, target: int):
    """Скорость обучения линейно от LEARNING_RATE к 0 за весь путь до `target` шагов.

    SB3 считает прогресс от 1 к 0 в пределах одного learn(); при возобновлении
    с `done` шагов этот отрезок — оставшаяся часть общего пути.
    """
    remaining = max(0.0, 1.0 - done / target)
    return lambda progress_remaining: LEARNING_RATE * remaining * progress_remaining


class ScoreCallback(BaseCallback):
    """Счёт, очки за эпизод и раунд (прочитанные с экрана) — в TensorBoard.

    При обучении эпизод — одна жизнь (life_loss_terminal), поэтому это «за жизнь».
    """

    def __init__(self, pool_dir: Path | None = None):
        super().__init__()
        self._pool_dir = pool_dir
        self._scores: list[int] = []
        self._points: list[int] = []
        self._rounds: list[int] = []
        self._curriculum: list[bool] = []
        self._fresh: dict[str, list] = {"points": [], "score": [], "round": []}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is None:
                continue
            for key, bucket in (("known_score", self._scores), ("gained_score", self._points),
                                ("known_round", self._rounds), ("curriculum_start", self._curriculum)):
                if ep.get(key) is not None:
                    bucket.append(ep[key])
            if not ep.get("curriculum_start"):  # только игры со свежего раунда 1 — сравнимо с ppo1
                for key, name in (("gained_score", "points"), ("known_score", "score"), ("known_round", "round")):
                    if ep.get(key) is not None:
                        self._fresh[name].append(ep[key])
        return True

    def _on_rollout_end(self) -> None:
        if self._scores:
            self.logger.record("neuronoid/score_mean", float(np.mean(self._scores)))
            self.logger.record("neuronoid/score_max", float(np.max(self._scores)))
        if self._points:
            self.logger.record("neuronoid/points_mean", float(np.mean(self._points)))
        if self._rounds:
            self.logger.record("neuronoid/round_max", float(np.max(self._rounds)))
        if self._curriculum:
            # доля эпизодов-жизней (не игр), начатых с позиции из пула
            self.logger.record("neuronoid/curriculum_share", float(np.mean(self._curriculum)))
        if self._fresh["points"]:
            self.logger.record("neuronoid/points_mean_fresh", float(np.mean(self._fresh["points"])))
        if self._fresh["score"]:
            self.logger.record("neuronoid/score_max_fresh", float(np.max(self._fresh["score"])))
        if self._fresh["round"]:
            self.logger.record("neuronoid/round_max_fresh", float(np.max(self._fresh["round"])))
        for bucket in self._fresh.values():
            bucket.clear()
        if self._pool_dir is not None:
            stats = StartPool(self._pool_dir).stats()
            if stats:
                self.logger.record("neuronoid/pool_round_max", float(max(stats)))
                self.logger.record("neuronoid/pool_size", float(sum(stats.values())))
        for bucket in (self._scores, self._points, self._rounds, self._curriculum):
            bucket.clear()


def save_atomic(model: PPO, path: Path) -> None:
    """Через временный файл: оборванная запись не портит прежний чекпоинт."""
    tmp = path.with_name(path.stem + ".tmp.zip")
    model.save(tmp)
    os.replace(tmp, path)
    path.with_suffix(".json").write_text(json.dumps({"num_timesteps": model.num_timesteps}))


def checkpoint_steps(path: Path) -> int:
    meta = path.with_suffix(".json")
    if meta.exists():
        return json.loads(meta.read_text())["num_timesteps"]
    return int(path.stem.split("_")[1]) if path.stem.startswith("ppo_") else -1


def find_resume_checkpoint(ckpt_dir: Path) -> Path | None:
    """Самый свежий из latest.zip и периодических ppo_<N>_steps.zip."""
    candidates = [p for p in ckpt_dir.glob("*.zip") if not p.name.endswith(".tmp.zip")]
    return max(candidates, key=checkpoint_steps, default=None)


class PeriodicSave(BaseCallback):
    """Каждые `every` шагов: ppo_<N>_steps.zip и свежий latest.zip (на случай жёсткой остановки)."""

    def __init__(self, every_calls: int, ckpt_dir: Path):
        super().__init__()
        self.every_calls = every_calls
        self.ckpt_dir = ckpt_dir

    def _on_step(self) -> bool:
        if self.n_calls % self.every_calls == 0:
            path = self.ckpt_dir / f"ppo_{self.num_timesteps}_steps.zip"
            save_atomic(self.model, path)
            latest = self.ckpt_dir / "latest.zip"
            shutil.copyfile(path, latest.with_name("latest.tmp.zip"))
            os.replace(latest.with_name("latest.tmp.zip"), latest)
            latest.with_suffix(".json").write_text(json.dumps({"num_timesteps": self.num_timesteps}))
        return True


class Watchdog:
    """Если шагов нет `timeout` секунд — зовёт on_timeout (по умолчанию прерывает обучение)."""

    def __init__(self, timeout: float = WATCHDOG_SECONDS, on_timeout=None):
        self.timeout = timeout
        self.on_timeout = on_timeout or self._interrupt
        self._last = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _interrupt() -> None:
        print(f"\nНет шагов обучения {WATCHDOG_SECONDS} с — похоже, зависла среда. Останавливаю с сохранением.", flush=True)
        _thread.interrupt_main()

    def feed(self) -> None:
        self._last = time.monotonic()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(min(1.0, self.timeout / 4)):
            if time.monotonic() - self._last > self.timeout:
                self.on_timeout()
                return


class FeedWatchdog(BaseCallback):
    def __init__(self, dog: Watchdog):
        super().__init__()
        self.dog = dog

    def _on_step(self) -> bool:
        self.dog.feed()
        return True


def train(
    run_dir: Path,
    total_steps: int,
    n_envs: int = 10,
    seed: int = 0,
    resume: bool = False,
    device: str = "auto",
    checkpoint_every: int = 200_000,
    live: bool = True,
    torch_threads: int = 4,
    curriculum: bool = False,
    curriculum_prob: float = 0.5,
    init_from: Path | None = None,
    snapshot_confirm_steps: int = 60,
) -> Path:
    """Учит до `total_steps` шагов всего (при resume — досчитывает до этой цели).

    `torch_threads`: на ноутбуке 4 потока быстрее 16 — иначе torch отнимает ядра у сред.
    `curriculum`: пул стартовых позиций в `run_dir/starts`. `init_from`: новый прогон
    с весов чужого чекпоинта (счётчик шагов продолжается с него).
    """
    if init_from is not None and resume:
        raise ValueError("--init-from начинает новый прогон, --resume продолжает старый: выбери одно")
    torch.set_num_threads(torch_threads)
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.zip"
    start = find_resume_checkpoint(ckpt_dir)
    if resume and start is None:
        raise FileNotFoundError(f"Нечего продолжать: в {ckpt_dir} нет чекпоинтов")
    if not resume and start is not None:
        raise FileExistsError(f"В {ckpt_dir} уже есть чекпоинты: добавь --resume или возьми другое имя --run")
    config_path = run_dir / "config.json"
    if resume and config_path.exists():
        old = json.loads(config_path.read_text())
        for key, value in (("curriculum", curriculum), ("curriculum_prob", curriculum_prob)):
            if key in old and old[key] != value:
                raise ValueError(f"--resume с другим {key}: было {old[key]}, сейчас {value} (curriculum-настройки)")
    config_path.write_text(
        json.dumps(
            {"total_steps": total_steps, "n_envs": n_envs, "seed": seed, "resume": resume,
             "device": device, "learning_rate": LEARNING_RATE, "curriculum": curriculum,
             "curriculum_prob": curriculum_prob, "init_from": str(init_from) if init_from else None,
             **PPO_KWARGS},
            indent=1,
        )
    )
    pool_dir = run_dir / "starts" if curriculum else None
    if init_from is not None:
        start = Path(init_from)
        if not start.exists():
            raise FileNotFoundError(f"Нет чекпоинта: {start}")
    venv = make_vec_env(n_envs, seed=seed, life_loss_terminal=True, live=live,
                        start_pool=pool_dir, curriculum_prob=curriculum_prob,
                        snapshot_confirm_steps=snapshot_confirm_steps)
    model = None
    dog = Watchdog()
    try:
        if start is not None:
            model = PPO.load(start, env=venv, device=device, tensorboard_log=str(run_dir / "tb"))
            done = model.num_timesteps
            model.learning_rate = global_linear(done, total_steps)
            model._setup_lr_schedule()
        else:
            done = 0
            model = PPO(
                "CnnPolicy", venv, learning_rate=global_linear(0, total_steps), seed=seed, device=device,
                tensorboard_log=str(run_dir / "tb"), verbose=1, **PPO_KWARGS,
            )
        if done < total_steps:
            callbacks = CallbackList([
                PeriodicSave(max(checkpoint_every // n_envs, 1), ckpt_dir),
                ScoreCallback(pool_dir),
                FeedWatchdog(dog),
            ])
            dog.start()
            model.learn(total_steps - done, callback=callbacks, reset_num_timesteps=start is None, tb_log_name="ppo")
    finally:
        dog.stop()
        try:
            if model is not None:
                save_atomic(model, latest)  # и при Ctrl+C, и при падении среды
        finally:
            venv.close()
    return latest
