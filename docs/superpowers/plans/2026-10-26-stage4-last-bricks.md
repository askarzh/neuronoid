# Этап 4: добить раунд — награда за кирпичи и учебная программа на конец раунда

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Научить агента очищать раунд 1 и тем взять М2 (7 из 10 полных игр с раунда 1 доходят до раунда 2). Сейчас (ppo3, 12,4 млн шагов) раунд 1 не очищается ни разу. Остаются верхний ряд серых кирпичей и одиночные кирпичи в углах, награда за них редкая.

**Architecture:** Новый считыватель кирпичей по пикселям поля (`BrickTracker`) даёт число оставшихся кирпичей и события «кирпич разрушен» (и «серый кирпич ударен», если это видно на экране). Награда получает плотный сигнал за кирпичи, усиленный, когда кирпичей осталось мало. Пул учебной программы хранит число оставшихся кирпичей и чаще стартует с конца раунда 1. Обучение продолжается с весов ppo3.

**Tech Stack:** как в этапах 2–3.

**Spec:** `docs/superpowers/specs/2026-10-24-neuronoid-design.md`; выводы прошлых прогонов — `docs/training-log.md` (раздел «Итог этапа 3»).

## Global Constraints

- Только пиксели экрана: кирпичи считаются по кадру, память эмулятора не читается.
- Зачёт ступеней не меняется: полные игры с раунда 1, «≥ 7 из 10» на сиде 1000; прогресс отслеживается по 20 играм (сиды 1000 и 2000).
- Savestates делает только агент.
- Обучение продолжается с весов ppo3 (`runs/ppo3/checkpoints/latest.zip`); пул копируется из ppo3, старые позиции без числа кирпичей остаются рабочими.

## Решения исполнителя

- **Геометрия сетки** (измерена на раунде 1): ячейки 16×8 пикселей, столбцы `x = 40 + 16·c`, c = 0…10; ряды `y = 9 + 8·r`. Число рядов исполнитель уточняет по кадрам поздних раундов из пула. Кирпич занимает ячейку почти целиком (14×7 пикселей); «есть кирпич» — не меньше 80% внутренних пикселей ячейки не чёрные.
- **Устойчивость:** ячейка считается кирпичом, только если «есть кирпич» держится `STABLE_STEPS = 4` шага подряд. Падающая капсула или мяч проходят ячейку быстрее и кирпичом не считаются. «Разрушен» — устойчивый кирпич исчез и не вернулся за 2 шага.
- **Награда (добавляется к существующей):**
  - `BRICK_REWARD = 0.5` за каждый разрушенный кирпич;
  - если после разрушения осталось не больше `FEW_BRICKS = 10` кирпичей — ещё `LAST_BRICKS_BONUS = 2.0` за кирпич;
  - `SILVER_HIT_REWARD = 0.3` за удар по серому кирпичу без разрушения, **только если** проверка (Task 1) покажет, что удар видно на экране; иначе награда не вводится, и это фиксируется в журнале.
- **Учебная программа:** среди стартов из пула корзина раунда 1 выбирается с вероятностью `FRONT_SHARE = 0.5`, остальные корзины — поровну из оставшейся половины. Внутри корзины позиции ранжируются по `bricks_left` по возрастанию (меньше кирпичей — лучше), при отсутствии — по прогрессу, как раньше. Выбирается лучшая половина.

## Review Focus

1. Капсулы, мяч, враги и выстрелы лазера, пролетающие через пустые ячейки, не должны считаться ни кирпичами, ни разрушениями (ложная награда). — тест в Task 1.
2. Переход на новый раунд (стена меняется целиком) и загрузка позиции из пула не должны давать «разрушение» всех кирпичей разом: трекер сбрасывается при смене раунда, старте и продолжении игры. — тест в Task 2.
3. На экранах без поля (ROUND N, GAME OVER, титул) трекер не ломается и награды не даёт. — тест в Task 1.
4. Сумма новой награды за раунд ограничена: 66 кирпичей × 0.5 + 10 × 2.0 = 53, это сопоставимо с бонусом за раунд (+50), а не на порядок больше. — проверка в Task 2.
5. Старые позиции пула без `bricks_left` продолжают загружаться и выбираться. — тест в Task 3.

---

### Task 1: Считыватель кирпичей (`BrickTracker`) и проверка серых кирпичей

**Files:**
- Modify: `neuronoid/screen.py` — `brick_cells(frame) -> np.ndarray` (bool, ряды × 11)
- Create: `neuronoid/bricks.py` — `BrickTracker`
- Create: `tools/silver_hit_probe.py` — одноразовая проверка: видно ли удар по серому кирпичу
- Modify: `tests/data/frames/labels.json` — поле `bricks` (число кирпичей) для игровых кадров
- Test: `tests/test_bricks.py`

**Interfaces:**
- Produces:
  - `neuronoid.screen.BRICK_ROWS: int`, `brick_cells(frame: np.ndarray) -> np.ndarray | None` — сетка «есть кирпич» по одному кадру; `None` для не игрового кадра.
  - `class BrickTracker`: `reset() -> None`; `update(frame) -> BrickEvents`; `remaining: int | None`.
  - `@dataclass(frozen=True) class BrickEvents: destroyed: int; silver_hits: int; remaining: int | None`.

- [ ] **Step 1: Проверка (исследование).** `tools/silver_hit_probe.py`: с позиции раунда 1 провести мяч в серый ряд (следящий алгоритм из `tools/capture_frames.py`), записать кадры ячеек серого ряда до и после первого удара. Посмотреть глазами (Read на PNG): меняется ли ячейка после первого удара (мигание, трещина, смена цвета) и как долго. Записать вывод в `docs/training-log.md` («Серые кирпичи: удар виден / не виден»). От вывода зависит `silver_hits`: если удар не виден, `silver_hits` всегда 0.
- [ ] **Step 2: Разметка.** Для игровых кадров из `tests/data/frames/` записать в `labels.json` поле `bricks` (посчитать по картинке; для `play_start.png` — 66). Добавить 2–3 кадра позднего раунда из пула ppo3 (`runs/ppo3/starts/`) с подсчитанным числом кирпичей.
- [ ] **Step 3: Тесты**

```python
@pytest.mark.parametrize("name,label", [(n, l) for n, l in LABELS.items() if l.get("bricks") is not None])
def test_brick_count_matches_label(name, label):
    assert int(brick_cells(load(name)).sum()) == label["bricks"]

def test_brick_cells_none_on_menu_frame():
    assert brick_cells(np.zeros((400, 640, 3), np.uint8)) is None

def test_tracker_counts_destroyed_brick():
    f = load("play_start.png"); t = BrickTracker()
    for _ in range(5): t.update(f)
    g = f.copy(); g[73:80, 42:56] = 0                      # нижний левый кирпич исчез
    events = [t.update(g) for _ in range(3)]
    assert sum(e.destroyed for e in events) == 1 and events[-1].remaining == 65

def test_tracker_ignores_passing_object():
    f = load("play_start.png"); t = BrickTracker()
    for _ in range(5): t.update(f)
    g = f.copy(); g[97:104, 74:88] = 200                    # «капсула» в пустой ячейке на 2 шага
    t.update(g); t.update(g)
    events = [t.update(f) for _ in range(3)]
    assert sum(e.destroyed for e in events) == 0 and events[-1].remaining == 66

def test_tracker_quiet_on_non_game_frames():
    f = load("play_start.png"); t = BrickTracker()
    for _ in range(5): t.update(f)
    e = t.update(load("idle_0099_not.png"))
    assert e.destroyed == 0 and e.remaining is None
```

  Координаты в тестах — по геометрии из раздела «Решения»; исполнитель сверяет их с кадром перед запуском.
- [ ] **Step 4:** Run `uv run pytest tests/test_bricks.py -v` → FAIL. **Step 5:** реализовать. **Step 6:** Run `uv run pytest -v` → PASS.
- [ ] **Step 7:** Commit `feat: считыватель кирпичей по пикселям поля`.

---

### Task 2: Награда за кирпичи в среде

**Files:**
- Modify: `neuronoid/env.py`
- Modify: `tests/test_env.py`

**Interfaces:**
- Consumes: `BrickTracker`, `BrickEvents`.
- Produces:
  - Константы `BRICK_REWARD = 0.5`, `FEW_BRICKS = 10`, `LAST_BRICKS_BONUS = 2.0`, `SILVER_HIT_REWARD` (0.3 или 0.0 по выводу Task 1).
  - `brick_reward(events: BrickEvents) -> float` — чистая функция: `destroyed × BRICK_REWARD`, плюс `LAST_BRICKS_BONUS` за каждый разрушенный, если `remaining ≤ FEW_BRICKS`, плюс `silver_hits × SILVER_HIT_REWARD`.
  - `ArkanoidEnv(..., brick_reward: bool = True)`. Трекер сбрасывается в `reset` (новая игра, старт из пула, продолжение после потери жизни) и при росте прочитанного раунда. Награда шага = прежняя + `brick_reward(events)`.
  - `info["bricks_left"]: int | None`.

- [ ] **Step 1: Тесты**

```python
def test_brick_reward_function():
    assert brick_reward(BrickEvents(1, 0, 30)) == pytest.approx(0.5)
    assert brick_reward(BrickEvents(1, 0, 9)) == pytest.approx(2.5)
    assert brick_reward(BrickEvents(0, 0, None)) == 0

def test_round1_brick_reward_is_bounded():
    total = sum(brick_reward(BrickEvents(1, 0, left)) for left in range(65, -1, -1))
    assert total <= 60   # сопоставимо с ROUND_BONUS, а не на порядок больше

def test_env_reports_bricks_left(env):
    _, info = env.reset(seed=0)
    assert info["bricks_left"] in (None, 66)
    for _ in range(10):
        *_, info = env.step(0)
    assert info["bricks_left"] == 66

def test_no_brick_reward_on_round_change_or_reset(env, monkeypatch):
    # кадр раунда 2 после кадров раунда 1 — стена сменилась целиком, «разрушений» нет
    ...  # _force_frames: 6 шагов play_start.png, затем 6 шагов idle_0924_play.png (раунд 2) — сумма brick-награды 0
```

  Последний тест: дописать по комментарию через `_force_frames` (как в `test_title_screen_ends_episode`) и `info`/награду; ожидание — ни на одном шаге награда не превышает обычную (нет «выстрела» на 30+).
- [ ] **Step 2:** Run → FAIL. **Step 3:** реализовать. **Step 4:** Run `uv run pytest -v` → PASS.
- [ ] **Step 5:** Commit `feat: награда за кирпичи, усиленная на последних кирпичах`.

---

### Task 3: Учебная программа на конец раунда

**Files:**
- Modify: `neuronoid/curriculum.py`, `neuronoid/env.py`
- Modify: `tests/test_curriculum.py`

**Interfaces:**
- Produces:
  - `StartMeta` получает поле `bricks_left: int | None = None` (старые `.json` без него читаются).
  - `StartPool(root, per_round=40, front_round=1, front_share=0.5)`: в `sample` корзина `front_round` выбирается с вероятностью `front_share` (если не пуста), иначе — равновероятно из остальных; ранжирование внутри корзины — по `bricks_left` по возрастанию, если оно есть у всех позиций корзины, иначе по `progress` по убыванию. Вытеснение при переполнении — по тому же ранжированию (выбрасывается худшая).
  - Среда пишет `bricks_left` из трекера в метаданные снимка.

- [ ] **Step 1: Тесты**

```python
def test_front_round_gets_its_share(tmp_path):
    pool = StartPool(tmp_path, front_share=0.5)
    pool.offer(b"a", StartMeta(1, 100, 100, 2, bricks_left=20))
    for r in (2, 3, 4):
        pool.offer(b"x", StartMeta(r, 100, 9000, 2, bricks_left=40))
    rng = np.random.default_rng(0)
    rounds = [pool.sample(rng)[1].round for _ in range(400)]
    assert 160 < rounds.count(1) < 240

def test_fewer_bricks_ranked_higher(tmp_path):
    pool = StartPool(tmp_path, per_round=10)
    for left in range(10, 0, -1):
        pool.offer(b"x", StartMeta(1, 1000 - left, 1000, 2, bricks_left=left))
    rng = np.random.default_rng(0)
    assert max(pool.sample(rng)[1].bricks_left for _ in range(100)) <= 5

def test_old_meta_without_bricks_left_still_loads(tmp_path):
    pool = StartPool(tmp_path)
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "5-old.state").write_bytes(b"s")
    (tmp_path / "1" / "5-old.json").write_text('{"round": 1, "progress": 5, "score": 5, "lives": 2}')
    state, meta, _ = pool.sample(np.random.default_rng(0))
    assert state == b"s" and meta.bricks_left is None
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** реализовать. **Step 4:** Run `uv run pytest -v` → PASS.
- [ ] **Step 5:** Commit `feat: учебная программа чаще стартует с конца раунда 1`.

---

### Финальное ревью кода — до долгого прогона

Независимый ревьюер на всей ветке, с Review Focus выше. Исправления — по TDD, одним проходом.

---

### Task 4: Прогон ppo4

- [ ] **Step 1:** Скопировать пул `runs/ppo3/starts` → `runs/ppo4/starts`. `uv run python scripts/train.py --run ppo4 --init-from runs/ppo3/checkpoints/latest.zip --curriculum --steps 17400000 --envs 10` в фоне (ещё 5 млн шагов, около 4,5 часов).
- [ ] **Step 2:** Сводка каждые ~25 минут; в `scripts/tb_summary.py` добавить `neuronoid/bricks_left_min_fresh` (минимум оставшихся кирпичей в свежих играх — главный признак успеха: должен падать к 0) — колбэк логирует его из `info["bricks_left"]` эпизодов со свежего старта.
- [ ] **Step 3:** На каждом миллионе — оценка 20 игр; дополнительно в отчёт оценки — `bricks_left` в конце каждой игры раунда 1.
- [ ] **Step 4: Контрольные точки.**
  - Через 1 млн шагов `bricks_left_min_fresh` должен опуститься заметно ниже уровня ppo3 (тот измерить на 20 играх перед стартом). Если нет — разобрать.
  - Если средний счёт по 20 играм < 5000 дважды подряд — стоп и разбор.
- [ ] **Step 5:** Commit журнала. **Остановиться и показать пользователю** графики, оценки, достигнута ли М2.

---

## После этого плана

М2 взята → М3 (раунд 5) и режим показа. Не взята → разбор по журналу (сколько кирпичей остаётся, где именно), следующий шаг — по данным.
