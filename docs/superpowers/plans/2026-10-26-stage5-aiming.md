# Этап 5: прицельное управление и чёткое зрение — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Дать агенту возможность целиться ракеткой: угол отскока в этой версии меняется от ~20° до ~160° при сдвиге точки удара на ±12 пикселей (`docs/training-log.md`, «Проверка механизма отскока»), а текущая схема видит поле с точностью 2 пикселя и двигает ракетку шагами 4–10 пикселей. Проверить, начнёт ли агент с новой схемой очищать раунд 1.

**Architecture:** Новый вариант среды (флаги, а не замена): действие — **целевая позиция ракетки** (одна из N по ширине поля) плюс «огонь»; контроллер внутри среды на каждом кадре довозит мышь к цели по положению ракетки **на экране**. Наблюдение — поле без уменьшения по горизонтали (93×192). Обучение с нуля (`ppo5`), сравнение с прежними прогонами на тех же оценках.

**Tech Stack:** как раньше.

**Spec:** `docs/superpowers/specs/2026-10-24-neuronoid-design.md`; данные — `docs/training-log.md`.

## Global Constraints

- Только пиксели: положение ракетки для контроллера читается с экрана, память не читается.
- Управление по-прежнему мышью; «целевая позиция» — это как человек ведёт мышь к нужной точке, а не подсказка о мяче.
- Прежняя схема остаётся рабочей по умолчанию (старые модели и тесты не ломаются); новая включается параметрами.
- Зачёт ступеней не меняется; прогресс — по 20 играм; главная метрика этапа — оставшиеся кирпичи и честные очистки раунда 1 (без «B»).

## Решения исполнителя

- **Действия:** `Discrete(N_TARGETS + 1)`, `N_TARGETS = 40`: действие `i < 40` — центр ракетки к `x = 52 + i·(152/39)` (диапазон центра ракетки 52…204 уточнить по кадрам у стен), без огня; действие `40` — огонь, ракетка стоит. Шаг цели ≈ 3,9 пикселя — меньше трети ракетки, достаточно для выбора угла.
- **Контроллер:** на каждом из `frame_skip` кадров `dx = clip(2·(цель − центр_ракетки), −14, 14)`; центр ракетки — по пикселям строк 176–181 (как в `tools/capture_frames.py`, поле x 40–216); нет ракетки на экране — `dx = 0`.
- **Наблюдение:** `obs_mode="wide"` — поле 186×192 → оттенки серого → среднее по парам строк → **93×192** (по горизонтали без уменьшения), максимум двух последних кадров, стек 4. NatureCNN SB3 справляется с этой формой.
- **Залипание:** `sticky_prob = 0.1` в новой схеме (повтор цели вредит меньше, чем повтор толчка мыши, но 0.25 всё равно слишком грубо для прицеливания).
- **Награда, учебная программа:** как в этапе 4 (кирпичи, капсула ≤ 200 очков за шаг, раунд +50, пул с фронтом раунда 1). Пул — копия пула ppo4 (позиции не зависят от схемы действий).
- **Честная очистка раунда:** в отчёт оценки — для каждой игры «раунд 1 очищен» = переход в раунд 2 при `bricks_left ≤ 2` за шаг до перехода (иначе — выход «B»).

## Review Focus

1. Контроллер не дёргает ракетку, когда её нет на экране (смерть, ROUND N, GAME OVER) и когда цель у стены (ракетка упирается). — тест в Task 1.
2. Старые модели (`ppo1`–`ppo4`) продолжают загружаться и оцениваться в старой схеме; флаги схемы записываются в `config.json` и подхватываются `evaluate.py`. — тест в Task 2.
3. Скорость: новая схема не медленнее старой больше чем в 3 раза на шаг (иначе 5 млн шагов не влезут в разумное время). — замер в Task 3.
4. Признак «честная очистка» не путает выход «B» с очисткой. — тест в Task 2.

---

### Task 1: Прицельные действия и широкое наблюдение в среде

**Files:**
- Modify: `neuronoid/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Produces:
  - `ArkanoidEnv(..., action_mode: str = "nudge", obs_mode: str = "half")`; `action_mode="target"` → `Discrete(N_TARGETS + 1)`; `obs_mode="wide"` → `Box(0, 255, (stack, 93, 192), uint8)`.
  - `target_x(i: int) -> float`, `paddle_center(frame) -> float | None`, `controller_dx(target: float, center: float | None) -> int` — чистые функции.
- [ ] **Step 1: Тесты**

```python
def test_target_action_moves_paddle_to_target(emu):
    env = ArkanoidEnv(emulator=emu, start_state=emu.round1_state, action_mode="target", obs_mode="wide",
                      sticky_prob=0.0, max_noop=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 93, 192) and env.action_space.n == N_TARGETS + 1
    for _ in range(15):
        env.step(5)                                   # цель слева
    left = paddle_center(env._frame)
    for _ in range(15):
        env.step(34)                                  # цель справа
    right = paddle_center(env._frame)
    assert abs(left - target_x(5)) <= 4 and abs(right - target_x(34)) <= 4

def test_controller_idle_without_paddle():
    assert controller_dx(100.0, None) == 0

def test_controller_saturates():
    assert controller_dx(200.0, 60.0) == 14 and controller_dx(60.0, 200.0) == -14
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** реализовать. **Step 4:** Run `uv run pytest -v` → PASS (старые тесты среды не трогаются: по умолчанию прежняя схема).
- [ ] **Step 5:** Commit `feat: прицельное управление и широкое наблюдение (новая схема среды)`.

### Task 2: Обучение и оценка в новой схеме

**Files:**
- Modify: `neuronoid/vec.py`, `neuronoid/train.py`, `scripts/train.py`, `neuronoid/evaluate.py`, `scripts/evaluate.py`
- Test: `tests/test_train.py`, `tests/test_evaluate.py`

**Interfaces:**
- `train(..., action_mode="nudge", obs_mode="half", sticky_prob=0.25)` → в `config.json`; CLI `--action-mode`, `--obs-mode`, `--sticky`.
- `scripts/evaluate.py` читает схему из `config.json` прогона (нет файла или полей — старая схема).
- `GameResult.cleared_round1: bool` — переход 1 → 2 при `bricks_left ≤ 2` на предыдущем шаге.
- [ ] **Step 1: Тесты:** смоук-обучение 512 шагов в новой схеме; `evaluate` старой модели без полей в `config.json` работает; `cleared_round1` — False при переходе с 60 кирпичами, True при переходе с 1 (через `_force_frames`-подобную подстановку `info` в чистой функции `classify_round1_exit(bricks_before) -> bool`).
- [ ] **Step 2–4:** FAIL → реализовать → PASS. **Step 5:** Commit `feat: обучение и оценка в прицельной схеме`.

### Финальное ревью кода — до долгого прогона

### Task 3: Прогон ppo5 с нуля

- [ ] **Step 1: Замер скорости:** 50 тыс. шагов новой схемы (10 сред) → шагов/с; сравнить со старой (~330–380).
- [ ] **Step 2:** Пул ppo4 → `runs/ppo5/starts`. `train.py --run ppo5 --action-mode target --obs-mode wide --sticky 0.1 --curriculum --steps 5000000 --envs 10` в фоне.
- [ ] **Step 3:** Оценка 20 игр на каждом миллионе: средний счёт, кирпичи в конце, честные очистки раунда 1, выходы «B». Сравнение с ppo1 на тех же шагах (1–5 млн).
- [ ] **Step 4: Контрольные точки:** на 2 млн новая схема не хуже ppo1 по среднему счёту (3407) — иначе разбор; появление хотя бы одной честной очистки раунда 1 к 5 млн — главный признак успеха.
- [ ] **Step 5:** Журнал, графики, **показать пользователю**.
