# Этап 3: учебная программа на savestates агента — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Довести агента до ступени М2 (в 7 из 10 полных игр с раунда 1 дойти до раунда 2). Для этого часть эпизодов обучения начинается не со свежего раунда 1, а с позиций, до которых агент **сам** дошёл раньше: поздние стадии раунда 1, где осталось мало кирпичей, и раунд 2 и дальше.

**Architecture:** Общий для всех процессов пул стартовых позиций на диске (`runs/NAME/starts/`). Среда во время игры периодически предлагает пулу savestate текущей позиции с метаданными (раунд, прогресс в раунде). При начале новой игры среда с вероятностью `curriculum_prob` берёт старт из пула, иначе — свежий раунд 1. Оценка не меняется: только полные игры с раунда 1.

**Tech Stack:** как в этапе 2 (SB3 PPO, gymnasium, numpy).

**Spec:** `docs/superpowers/specs/2026-10-24-neuronoid-design.md` — раздел «Учебная программа через savestates агента» и этап 5; итоги прогона ppo1 — `docs/training-log.md`.

## Global Constraints

- Savestates делает только сам агент во время обучения. Ручных сохранений человека нет (требование спеки — «чистый» подход).
- Метаданные для выбора старта берутся только с экрана (раунд, счёт); память эмулятора не читается.
- Зачёт ступеней не меняется: полные игры с раунда 1, обычный запас жизней, ≥ 7 из 10.
- Пул общий для процессов, но процессы не блокируют друг друга (файлы + атомарное переименование, без блокировок).
- Продолжаем с весов ppo1 (`runs/ppo1/checkpoints/latest.zip`) в новом прогоне `ppo2`; ppo1 не изменяется.

## Решения исполнителя (спека оставила их за ним)

- **Прогресс в раунде** = известный счёт − счёт на момент входа в этот раунд. Кирпичи дают очки, поэтому это грубая, но честная мера «сколько стены разбито», без нового распознавания.
- **Политика пула:** корзина на каждый раунд, в корзине не больше `PER_ROUND = 40` позиций. При переполнении выбрасывается позиция с наименьшим прогрессом. Выбор старта: сначала равновероятно корзина среди непустых (раунд 2+ получает практику сразу, как только агент туда попал), затем позиция из лучшей половины корзины по прогрессу, равновероятно.
- **Снимки:** каждые `SNAPSHOT_EVERY = 100` шагов, только когда идёт игра (`playing`) и в этом шаге не потеряна жизнь, и только если позиция лучше худшей в корзине (или корзина не полна).
- **`curriculum_prob = 0.5`** — половина новых игр со свежего раунда 1, чтобы не разучиться начинать.

## Review Focus

1. Повреждённый или недописанный файл позиции (процесс убит во время записи) не должен ронять среды: при ошибке загрузки позиция удаляется из пула, а среда стартует со свежего раунда 1. — тест в Task 1/Task 2.
2. 10 процессов одновременно пишут и читают пул: ни исключений, ни «чужих полуфайлов» (запись во временный файл и `os.replace`). — тест в Task 1.
3. Пул не растёт без предела на диске (~525 КБ на позицию): ограничение по корзинам действительно удаляет файлы. — тест в Task 1.
4. Старт из пула — это «продолжение чужой игры»: `known_*`, `start_score`, отметка входа в раунд и счётчики эпизода корректно устанавливаются по кадру после загрузки; награда на первом шаге не «выстреливает» разницей со старым эпизодом. — тест в Task 2.
5. Оценка (`life_loss_terminal=False`, без пула) по-прежнему всегда начинается со свежего раунда 1. — тест в Task 2.

---

### Task 1: Пул стартовых позиций

**Files:**
- Create: `neuronoid/curriculum.py`
- Test: `tests/test_curriculum.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class StartMeta: round: int; progress: int; score: int; lives: int`
  - `class StartPool`:
    - `__init__(self, root: Path, per_round: int = 40)` — каталог создаётся при необходимости.
    - `offer(self, state: bytes, meta: StartMeta) -> bool` — сохраняет позицию, если корзина раунда не полна или `meta.progress` больше худшего в корзине; тогда же удаляет худший. Запись: `<round>/<progress>-<uuid>.state` и `.json` рядом, через временное имя и `os.replace` (сначала `.state`, потом `.json`; позиция видна, только когда есть `.json`). Возвращает, сохранена ли позиция.
    - `sample(self, rng: np.random.Generator) -> tuple[bytes, StartMeta] | None` — политика из раздела «Решения»; `None`, если пул пуст. Файл, который не читается или не соответствует `.json`, удаляется и выбирается другой.
    - `forget(self, meta_path: Path) -> None` — удалить позицию (для «не загрузилась в эмулятор»).
    - `stats(self) -> dict[int, int]` — раунд → число позиций.

- [ ] **Step 1: Тесты**

```python
def test_empty_pool_samples_none(tmp_path):
    assert StartPool(tmp_path).sample(np.random.default_rng(0)) is None

def test_offer_and_sample_roundtrip(tmp_path):
    pool = StartPool(tmp_path)
    assert pool.offer(b"state-A", StartMeta(1, 500, 500, 2))
    state, meta = pool.sample(np.random.default_rng(0))
    assert state == b"state-A" and meta.round == 1 and meta.progress == 500

def test_bucket_cap_evicts_lowest_progress(tmp_path):
    pool = StartPool(tmp_path, per_round=3)
    for p in (100, 200, 300):
        pool.offer(b"x", StartMeta(1, p, p, 2))
    assert not pool.offer(b"x", StartMeta(1, 50, 50, 2))      # хуже всех — не берём
    assert pool.offer(b"x", StartMeta(1, 400, 400, 2))
    assert pool.stats() == {1: 3}
    assert len(list((tmp_path / "1").glob("*.state"))) == 3     # файлы реально удалены

def test_sample_prefers_top_half_and_all_rounds(tmp_path):
    pool = StartPool(tmp_path, per_round=10)
    for p in range(10):
        pool.offer(b"x", StartMeta(1, p * 100, p * 100, 2))
    pool.offer(b"y", StartMeta(2, 10, 6000, 1))
    rng = np.random.default_rng(0)
    metas = [pool.sample(rng)[1] for _ in range(200)]
    assert 60 < sum(m.round == 2 for m in metas) < 140           # корзины равновероятны
    assert min(m.progress for m in metas if m.round == 1) >= 500  # только лучшая половина

def test_corrupt_file_is_dropped(tmp_path):
    pool = StartPool(tmp_path)
    pool.offer(b"good", StartMeta(1, 100, 100, 2))
    bad = next((tmp_path / "1").glob("*.json"))
    bad.with_suffix(".state").unlink()                            # «процесс убит во время записи»
    assert pool.sample(np.random.default_rng(0)) is None
    assert pool.stats() == {}

def test_concurrent_writers(tmp_path):
    # 4 процесса × 30 offer/sample одновременно — ни одного исключения, корзины в пределах
    ...  # multiprocessing spawn, воркер в tests/_pool_worker.py, PYTHONPATH как в test_vec
```

  Последний тест: дописать по комментарию; ожидание — все воркеры завершились с кодом 0, `pool.stats()[1] <= per_round`.
- [ ] **Step 2:** Run `uv run pytest tests/test_curriculum.py -v` → FAIL.
- [ ] **Step 3:** Реализовать `neuronoid/curriculum.py`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: общий пул стартовых позиций на диске`.

---

### Task 2: Учебная программа в среде

**Files:**
- Modify: `neuronoid/env.py`
- Modify: `tests/test_env.py`

**Interfaces:**
- Consumes: `StartPool`, `StartMeta`.
- Produces:
  - `ArkanoidEnv(..., start_pool: Path | None = None, curriculum_prob: float = 0.5, snapshot_every: int = 100)`. Без `start_pool` поведение не меняется.
  - Среда хранит `self._round_entry_score` — известный счёт в момент, когда прочитанный раунд последний раз вырос (или в момент старта игры/загрузки позиции).
  - **Снимки:** в `step`, если `start_pool` задан, `self._steps % snapshot_every == 0`, `state.playing`, жизнь в этом шаге не потеряна и известны раунд, счёт и жизни — `pool.offer(emu.save_state(), StartMeta(round, known_score - round_entry_score, known_score, lives))`.
  - **Старт:** в `reset` при начале новой игры (не продолжение после потери жизни) с вероятностью `curriculum_prob` — `pool.sample(self.np_random)`; если вернулась позиция — `emu.load_state(state)`, кадр, `read_state`; если загрузка бросила исключение или кадр не игровой — `pool.forget(...)` и свежий раунд 1. `known` = прочитанное состояние, `start_score` = прочитанный счёт, `round_entry_score` = `score - meta.progress`.
  - `info["curriculum_start"]: bool` — эпизод начат с позиции из пула (для графиков).

- [ ] **Step 1: Тесты** (session-эмулятор, пул во `tmp_path`)

```python
def test_env_offers_snapshots(emu, tmp_path):
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, start_pool=tmp_path,
                      curriculum_prob=0.0, snapshot_every=5)
    env.reset(seed=0)
    for i in range(60):
        env.step(1 if i < 3 else 0)
    assert StartPool(tmp_path).stats().get(1, 0) >= 1

def test_reset_from_pool_continues_that_position(emu, tmp_path):
    pool = StartPool(tmp_path)
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, start_pool=tmp_path,
                      curriculum_prob=0.0, snapshot_every=5)
    env.reset(seed=0)
    for i in range(200):
        *_, info = env.step(1 if i < 3 else 0)
        if info["known_score"]:
            break
    env.curriculum_prob = 1.0
    obs, info = env.reset(seed=1)
    assert info["curriculum_start"] and info["gained_score"] == 0
    assert info["known_score"] is not None
    _, r, *_ = env.step(0)
    assert r <= 2.0   # не «выстрел» разницей со старым эпизодом

def test_broken_pool_entry_falls_back_to_round1(emu, tmp_path):
    pool = StartPool(tmp_path)
    pool.offer(b"not a savestate", StartMeta(1, 999, 999, 2))
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, start_pool=tmp_path, curriculum_prob=1.0)
    _, info = env.reset(seed=0)
    assert not info["curriculum_start"] and info["score"] == 0
    assert pool.stats() == {}

def test_without_pool_always_round1(emu):
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, life_loss_terminal=False)
    for s in range(3):
        _, info = env.reset(seed=s)
        assert info["score"] == 0 and info["round"] == 1 and not info["curriculum_start"]
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** реализовать. **Step 4:** Run `uv run pytest -v` → PASS.
- [ ] **Step 5:** Commit `feat: учебная программа — старты с позиций, до которых дошёл агент`.

---

### Task 3: Обучение с учебной программой и стартом с весов ppo1

**Files:**
- Modify: `neuronoid/vec.py`, `neuronoid/train.py`, `scripts/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Produces:
  - `make_env_fn(..., start_pool: Path | None = None, curriculum_prob: float = 0.5)` и `make_vec_env(..., start_pool=None, curriculum_prob=0.5)` — пробрасывают в `ArkanoidEnv`.
  - `train(..., curriculum: bool = False, curriculum_prob: float = 0.5, init_from: Path | None = None)`: `curriculum=True` → пул `run_dir/"starts"`; `init_from` — для **нового** прогона берёт веса и состояние оптимизатора из чекпоинта, а счётчик шагов продолжает с него (`num_timesteps` сохраняется, TensorBoard нового прогона начинается с этого числа). Вместе с `resume=True` — ошибка `ValueError`.
  - `ScoreCallback` дополнительно логирует `neuronoid/curriculum_share` (доля эпизодов со старта из пула) и `neuronoid/pool_round_max` (максимальный раунд в пуле).
  - CLI: `--curriculum`, `--curriculum-prob 0.5`, `--init-from PATH`.

- [ ] **Step 1: Тесты**

```python
def test_init_from_starts_new_run_with_old_weights(tmp_path):
    src = train(tmp_path / "a", total_steps=256, n_envs=2, live=False)
    dst = train(tmp_path / "b", total_steps=512, n_envs=2, live=False, init_from=src)
    m = PPO.load(dst)
    assert m.num_timesteps >= 512 and (tmp_path / "a" / "checkpoints" / "latest.zip").exists()

def test_init_from_with_resume_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        train(tmp_path, total_steps=256, n_envs=2, live=False, resume=True, init_from=tmp_path / "x.zip")

def test_curriculum_run_fills_pool(tmp_path):
    train(tmp_path, total_steps=2048, n_envs=2, live=False, curriculum=True)
    assert any((tmp_path / "starts").rglob("*.state"))
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** реализовать. **Step 4:** Run `uv run pytest -v` → PASS.
- [ ] **Step 5:** Commit `feat: обучение с учебной программой и стартом с весов прошлого прогона`.

---

### Task 4: Прогон ppo2 к ступени М2

Исследовательская задача: код — только исправления по находкам, каждое с тестом.

- [ ] **Step 1:** `uv run python scripts/train.py --run ppo2 --init-from runs/ppo1/checkpoints/latest.zip --curriculum --steps 10000000 --envs 10` в фоне (это ещё 5 млн шагов сверх 5 млн ppo1, около 4,5 часов).
- [ ] **Step 2:** Каждые ~25 минут — сводка (`scripts/tb_summary.py`, плюс `curriculum_share`, `pool_round_max`, число позиций в пуле по раундам) в `docs/training-log.md`.
- [ ] **Step 3:** На каждом миллионе — `scripts/evaluate.py` (чистая оценка с раунда 1) и запись в журнал.
- [ ] **Step 4: Контрольные точки.**
  - Через ~1 млн шагов в пуле должны появиться позиции раунда 2. Если нет — разобрать (снимки, политика пула) и записать вывод.
  - Если оценка полными играми **падает** ниже ppo1 (средний счёт < 5000 на двух оценках подряд) — учебная программа вредит. Остановить, понизить `curriculum_prob` или разобрать, записать вывод.
- [ ] **Step 5:** Commit журнала. **Остановиться и показать пользователю:** графики, оценки, достигнута ли М2, предложение по следующему шагу.

---

## После этого плана

М2 достигнута → движение к М3 (раунд 5) той же учебной программой и режим показа. Не достигнута → разбор причин (награда за очистку раунда, последние кирпичи, длительность обучения) по журналу.
