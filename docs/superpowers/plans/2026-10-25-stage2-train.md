# Этап 2: обучение PPO — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Обучать агента PPO на параллельных `ArkanoidEnv` с чекпоинтами, возобновлением и графиками, оценивать его по честным правилам зачёта и довести до ступени М1 (стабильно отбивает мяч и набирает очки).

**Architecture:** Stable-Baselines3 `PPO` с `CnnPolicy` (NatureCNN) поверх `SubprocVecEnv` из N процессов `ArkanoidEnv`; живое окно публикует только среда №0. Обучение — с `life_loss_terminal=True` (эпизод до первой потери жизни), оценка — полные игры с раунда 1 без продолжений. `scripts/train.py` запускает и возобновляет обучение, `scripts/evaluate.py` считает зачёт ступеней.

**Tech Stack:** Python 3.12, stable-baselines3, torch (CPU), tensorboard, gymnasium.

**Spec:** `docs/superpowers/specs/2026-10-24-neuronoid-design.md` (цели, ступени М1–М3, правила зачёта); прошлые этапы: `docs/spike-report.md`, `neuronoid/env.py`.

## Global Constraints

- Только пиксели на входе, награда только с экрана (как в `ArkanoidEnv`); память эмулятора не читается.
- Зачёт ступени — только полные игры **с раунда 1**, обычный запас жизней, без продолжений; серия оценочных игр, «≥ 7 из 10».
- В оценке тоже есть случайность против заучивания (случайный старт, залипающие действия) — та же среда.
- Один эмулятор на процесс; среды — в отдельных процессах.
- Число процессов задаётся параметром (нагрев ноутбука); по умолчанию 10.
- Обучение должно переживать остановку: чекпоинты и `--resume`.
- Код кроссплатформенный (Windows сейчас, MacBook M4 потом: устройство `cpu`/`mps` выбирается параметром).

## Пороги ступеней (решение исполнителя, спека оставила точный порог за ним)

- **М1:** в ≥ 7 из 10 оценочных игр счёт ≥ 1500 (около 15 кирпичей; случайный агент набирает 100–400).
- **М2:** в ≥ 7 из 10 игр агент дошёл до раунда 2.
- **М3:** в ≥ 7 из 10 игр — до раунда 5.

## Review Focus

1. Остановка посреди обучения (Ctrl+C, сон ноутбука, падение) → последний чекпоинт цел, `--resume` продолжает с того же числа шагов и не затирает графики. — тест в Task 2.
2. Падение одного процесса среды (зависание ядра) не должно молча повесить всё обучение: падение видно в логе и обучение останавливается с чекпоинтом. — тест в Task 1 (дочерний процесс, падающий на `step`).
3. Консоль не заливается служебными строками DOSBox из 10 процессов — видно только прогресс обучения. — проверка в Task 1.
4. Оценка детерминирована по сиду: один чекпоинт и один сид → одинаковый отчёт. — тест в Task 3.
5. Живое окно показывает ровно одну среду, а не мигает между несколькими. — тест в Task 1.

---

### Task 1: Параллельные среды

**Files:**
- Modify: `pyproject.toml` (+ `stable-baselines3`, `torch`, `tensorboard`)
- Modify: `neuronoid/env.py` — в `info` добавить `known_score`, `known_round`, `known_lives` (последние прочитанные значения: на титульном экране после Game Over обычные поля уже `None`)
- Create: `neuronoid/vec.py`
- Create: `neuronoid/quiet.py`
- Test: `tests/test_vec.py`

**Interfaces:**
- Consumes: `ArkanoidEnv`.
- Produces:
  - `neuronoid.quiet.silence_native_stdout() -> None` — перенаправляет файловые дескрипторы 1 и 2 процесса в `os.devnull` (ядро DOSBox пишет туда напрямую, мимо Python), а `sys.stdout`/`sys.stderr` Python оставляет на исходной консоли через `os.dup` сохранённых дескрипторов. Вызывается в дочерних процессах сред.
  - `neuronoid.vec.make_env_fn(rank: int, seed: int, life_loss_terminal: bool, live_rank: int | None = 0) -> Callable[[], gym.Env]` — фабрика для `SubprocVecEnv`: в дочернем процессе вызывает `silence_native_stdout()`, создаёт `ArkanoidEnv(life_loss_terminal=..., live=(rank == live_rank))`, оборачивает в `stable_baselines3.common.monitor.Monitor(env, info_keywords=("known_score", "known_round"))`, сид `seed + rank`.
  - `neuronoid.vec.make_vec_env(n_envs: int, seed: int = 0, life_loss_terminal: bool = True, live: bool = True) -> SubprocVecEnv` — `start_method="spawn"`; `live=False` — ни одна среда не публикует.

- [ ] **Step 1: Тесты** (среды в дочерних процессах, поэтому с session-эмулятором тестов не конфликтуют)

```python
def test_vec_env_shapes():
    venv = make_vec_env(2, seed=0, live=False)
    obs = venv.reset()
    assert obs.shape == (2, 4, 93, 96) and obs.dtype == np.uint8
    obs, rew, done, infos = venv.step(np.array([1, 3]))
    assert rew.shape == (2,) and "known_score" in infos[0]
    venv.close()

def test_only_rank0_publishes(tmp_live_name, monkeypatch):
    # LIVE_NAME подменяется через переменную окружения NEURONOID_LIVE_NAME (читается в neuronoid.live)
    venv = make_vec_env(2, seed=0, live=True)
    venv.reset(); venv.step(np.array([0, 0]))
    frame, info, n = LiveReader(tmp_live_name).read()
    assert n == 2   # reset + один step одной среды; две среды дали бы 4
    venv.close()

def test_child_crash_surfaces():
    venv = SubprocVecEnv([make_env_fn(0, 0, True, None), crashing_env_fn], start_method="spawn")
    venv.reset()
    with pytest.raises((EOFError, BrokenPipeError, RuntimeError)):
        for _ in range(5):
            venv.step(np.array([0, 0]))
```

  `crashing_env_fn` (в `tests/_crash_env.py`, импортируемом модуле — `spawn` требует) — `gym.Env`, бросающий исключение на втором `step`. `tmp_live_name` — фикстура, ставит `NEURONOID_LIVE_NAME` в уникальное имя через `monkeypatch.setenv` (дочерние процессы наследуют окружение).
  Для этого в `neuronoid/live.py`: `LIVE_NAME = os.environ.get("NEURONOID_LIVE_NAME", "neuronoid_live")`, а значения по умолчанию у `LivePublisher`/`LiveReader` читают его при вызове, не при импорте.
- [ ] **Step 2:** Run `uv run pytest tests/test_vec.py -v` → FAIL.
- [ ] **Step 3:** Реализовать `quiet.py`, `vec.py`, правки `env.py` и `live.py`.
- [ ] **Step 4:** Run `uv run pytest -v` → PASS. Проверка глазами: `uv run python -c "from neuronoid.vec import make_vec_env; v=make_vec_env(4, live=False); v.reset(); v.close()"` — в консоли нет строк `[DOSBOX]`/`[DBP]`.
- [ ] **Step 5:** Commit `feat: параллельные среды для обучения`.

---

### Task 2: Скрипт обучения

**Files:**
- Create: `neuronoid/train.py` — логика (тестируемая)
- Create: `scripts/train.py` — CLI
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `make_vec_env`.
- Produces:
  - `neuronoid.train.PPO_KWARGS: dict` — значения по умолчанию в духе Atari-настроек RL Zoo: `n_steps=128`, `batch_size=256`, `n_epochs=4`, `learning_rate` — линейно от `2.5e-4` к 0, `clip_range=0.1`, `ent_coef=0.01`, `vf_coef=0.5`, `gamma=0.99`, `gae_lambda=0.95`.
  - `neuronoid.train.train(run_dir: Path, total_steps: int, n_envs: int = 10, seed: int = 0, resume: bool = False, device: str = "auto", checkpoint_every: int = 200_000, live: bool = True) -> Path` — создаёт или загружает модель (`run_dir/checkpoints/latest.zip`), учит `model.learn(total_steps, reset_num_timesteps=not resume, tb_log_name="ppo")`, логи TensorBoard в `run_dir/tb`. Колбэки: `CheckpointCallback` каждые `checkpoint_every` шагов (суммарно по средам) и сохранение `latest.zip` при любом выходе (`try/finally`, включая `KeyboardInterrupt`). Пишет `run_dir/config.json` (параметры запуска). Возвращает путь к `latest.zip`.
  - Метрики в TensorBoard: стандартные `rollout/ep_rew_mean`, `rollout/ep_len_mean` + собственный колбэк `neuronoid/score_max`, `neuronoid/round_max` и `neuronoid/score_mean` по завершённым эпизодам за rollout (из `known_score`/`known_round` в `info["episode"]`-эпизодах Monitor).
  - `scripts/train.py --run NAME [--steps N] [--envs 10] [--resume] [--device auto] [--no-live]` — `run_dir = runs/NAME`; печатает, как открыть окно и TensorBoard.

- [ ] **Step 1: Тесты** (маленький прогон; обучение в дочерних процессах сред, модель в тестовом процессе)

```python
def test_train_smoke_and_resume(tmp_path):
    p = train(tmp_path, total_steps=512, n_envs=2, checkpoint_every=256, live=False)
    assert p.exists() and (tmp_path / "config.json").exists()
    m1 = PPO.load(p); assert m1.num_timesteps >= 512
    train(tmp_path, total_steps=512, n_envs=2, resume=True, live=False)
    m2 = PPO.load(tmp_path / "checkpoints" / "latest.zip")
    assert m2.num_timesteps >= m1.num_timesteps + 512

def test_interrupt_saves_latest(tmp_path, monkeypatch):
    # KeyboardInterrupt внутри learn -> latest.zip всё равно сохранён
    calls = {"n": 0}
    def boom(self, *a, **k):
        calls["n"] += 1; raise KeyboardInterrupt
    monkeypatch.setattr(PPO, "learn", boom)
    with pytest.raises(KeyboardInterrupt):
        train(tmp_path, total_steps=512, n_envs=2, live=False)
    assert (tmp_path / "checkpoints" / "latest.zip").exists()

def test_score_callback_logs_metrics(tmp_path):
    train(tmp_path, total_steps=2048, n_envs=2, live=False)
    tags = event_tags(tmp_path / "tb")    # хелпер: читает теги из tfevents через tensorboard EventAccumulator
    assert {"neuronoid/score_max", "rollout/ep_rew_mean"} <= tags
```

- [ ] **Step 2:** Run `uv run pytest tests/test_train.py -v` → FAIL.
- [ ] **Step 3:** Реализовать `neuronoid/train.py` и `scripts/train.py`.
- [ ] **Step 4:** Run `uv run pytest -v` → PASS.
- [ ] **Step 5:** Commit `feat: обучение PPO с чекпоинтами и возобновлением`.

---

### Task 3: Оценка и зачёт ступеней

**Files:**
- Create: `neuronoid/evaluate.py`
- Create: `scripts/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `ArkanoidEnv` (`life_loss_terminal=False`), модель PPO.
- Produces:
  - `@dataclass class GameResult: score: int; round: int; steps: int`
  - `@dataclass class Report: games: list[GameResult]; m1: bool; m2: bool; m3: bool` + `to_json()`.
  - `MILESTONES = {"m1": lambda g: g.score >= 1500, "m2": lambda g: g.round >= 2, "m3": lambda g: g.round >= 5}`, `PASS_COUNT = 7`, `GAMES = 10`.
  - `evaluate(policy: Callable[[np.ndarray], int], env: ArkanoidEnv, games: int = GAMES, seed: int = 1000) -> Report` — игра `i` начинается с `env.reset(seed=seed + i)` и идёт до `terminated` (Game Over) или `truncated`; счёт и раунд — `info["known_score"]`, `info["known_round"]`. Принимает любую политику (функция «наблюдение → действие»), чтобы тестироваться без нейросети.
  - `scripts/evaluate.py RUN_OR_CHECKPOINT [--games 10] [--deterministic]` — грузит `latest.zip`, политика `model.predict(obs, deterministic=...)`, печатает таблицу игр и вердикт по М1–М3, пишет `runs/NAME/eval/<steps>.json`.

- [ ] **Step 1: Тесты** (среда в тестовом процессе — через session-эмулятор: `ArkanoidEnv(emulator=emu, start_state=emu.round1_state, life_loss_terminal=False)`)

```python
def test_milestone_rules():
    g = lambda s, r: GameResult(s, r, 100)
    rep = Report.from_games([g(1600, 1)] * 7 + [g(100, 1)] * 3)
    assert rep.m1 and not rep.m2
    rep = Report.from_games([g(5000, 2)] * 6 + [g(100, 1)] * 4)
    assert not rep.m2

def test_evaluate_random_policy(eval_env):
    rng = np.random.default_rng(0)
    rep = evaluate(lambda obs: int(rng.integers(6)), eval_env, games=2)
    assert len(rep.games) == 2 and all(g.round == 1 and g.score >= 0 for g in rep.games)

def test_evaluate_deterministic_by_seed(eval_env):
    policy = lambda obs: 0
    a = evaluate(policy, eval_env, games=2, seed=5).to_json()
    b = evaluate(policy, eval_env, games=2, seed=5).to_json()
    assert a == b
```

  `Report.from_games(games)` — считает флаги по `MILESTONES`/`PASS_COUNT`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** реализовать. **Step 4:** Run `uv run pytest -v` → PASS.
- [ ] **Step 5:** Commit `feat: оценка агента и зачёт ступеней`.

---

### Task 4: Первый настоящий прогон обучения

Исследовательская задача: код не пишется (кроме исправлений по находкам — каждое с тестом), результат — цифры и вывод.

**Files:**
- Create: `docs/training-log.md` — журнал прогонов: параметры, скорость, кривая счёта, оценки.

- [ ] **Step 1: Замер скорости:** `uv run python scripts/train.py --run speed --steps 50000 --envs N` для N = 8, 10, 12 (в фоне, с таймаутом 20 минут на каждый). Записать шагов/с (`time/fps` в TensorBoard или вывод SB3) и выбрать N.
- [ ] **Step 2: Прогон:** `uv run python scripts/train.py --run ppo1 --steps 5000000 --envs <N>` в фоне. Каждые ~30 минут смотреть TensorBoard-метрики (`neuronoid/score_mean`, `score_max`, `ep_rew_mean`) — читать файлы событий скриптом, а не глазами — и записывать в журнал. Признак обучения — рост `score_mean` выше уровня случайного агента (100–400).
- [ ] **Step 3:** На каждом миллионе шагов — `uv run python scripts/evaluate.py runs/ppo1` и запись в журнал. Скриншот живого окна (`scripts/watch.py --screenshot`) в `docs/img/ppo1_<steps>.png`.
- [ ] **Step 4: Контрольная точка.** Если через ~2 часа `score_mean` не растёт — остановить, разобрать (награда, наблюдение, гиперпараметры) и записать вывод; не жечь ночь впустую. Если растёт — дать дойти до 5 млн шагов или до М1.
- [ ] **Step 5:** Commit журнала `docs: журнал первого прогона обучения`. **Остановиться и показать пользователю:** кривую, оценку по М1, скриншот окна, предложение — продолжать ли обучение ночью.

---

## После этого плана

По результату: план учебной программы на savestates и движение к М2/М3 (этап 5 спеки) и режим показа (этап 4).
