# Этап 0: проверка и слой эмулятора — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Доказать, что Arkanoid запускается в DOSBox Pure из Python достаточно быстро, и получить надёжный класс `LibretroEmulator`, на котором будет стоять вся остальная система.

**Architecture:** Ядро `dosbox_pure_libretro` загружается фронтендом `libretro.py` (запасной вариант — свой фронтенд на `ctypes`). Тонкая обёртка `LibretroEmulator` даёт кадры как `numpy`-массивы, нажатия клавиш, savestates. Ядро libretro хранит глобальное состояние, поэтому **один эмулятор на процесс**, а параллельность — через процессы.

**Tech Stack:** Python 3.12 (uv), libretro.py, numpy, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-10-24-neuronoid-design.md`

**Почему план только на этап 0 и эмулятор:** раскладка экрана, клавиши, последовательность меню и реальная скорость станут известны только после проверки. План для среды, `ScreenReader` и обучения пишется следующим, по данным отчёта этого этапа.

## Global Constraints

- Python 3.12 в отдельном окружении uv (системный 3.14 не используется).
- Память эмулятора не читается нигде в `neuronoid/` (исключение — будущие `tools/` и `tests/` для разметки кадров).
- Файлы игры (`games/`) и ядра (`cores/`) не коммитятся.
- Код кроссплатформенный: Windows x86_64 (`.dll`) и macOS arm64 (`.dylib`).
- Критерий перехода дальше: игра работает, и один экземпляр без окна даёт ≥ 300 кадров/с.
- Игра: `games/arkanoid/DOS_arkanoid.zip` (внутри `Arkanoid/ARKANOID.COM` — загрузчик, `Arkanoid/ARKANOID.DAT` — игра).

## Review Focus

1. Нет архива игры или ядра → понятное сообщение «что положить и куда» (или «запусти fetch_core»), а не трассировка стека. — тест в Task 1.
2. Второй `LibretroEmulator` в том же процессе → явная ошибка `RuntimeError`, а не тихая порча состояния ядра. — тест в Task 3.
3. Детерминизм: одно и то же состояние + одни и те же нажатия → побайтно одинаковые кадры. Без этого нельзя отлаживать обучение. — тест в Task 3.
4. Savestate, сохранённый в одном процессе, загружается в другом (учебная программа будет передавать состояния между процессами). — тест в Task 3.
5. Ускоренный прогон не меняет игровое время: число эмулированных кадров на секунду игры одинаково при любой скорости хоста (фиксированные циклы CPU, а не «max»). — проверка и вывод в отчёте Task 2, тест на `fps` ядра в Task 3.

---

### Task 1: Каркас проекта, загрузка ядра, конфиг путей

**Files:**
- Create: `pyproject.toml`, `.python-version` (`3.12`)
- Create: `neuronoid/__init__.py`, `neuronoid/config.py`
- Create: `scripts/fetch_core.py`
- Create: `tests/test_config.py`
- Create: `README.md` (кратко: что за проект, как поставить, как скачать ядро, куда положить игру)

**Interfaces:**
- Produces:
  - `neuronoid.config.ROOT: Path` — корень репозитория.
  - `neuronoid.config.ConfigError(Exception)`.
  - `neuronoid.config.game_path(root: Path = ROOT) -> Path` — `root/games/arkanoid/DOS_arkanoid.zip`; если файла нет — `ConfigError`, в тексте путь и «положи туда свою копию игры».
  - `neuronoid.config.core_path(root: Path = ROOT) -> Path` — `root/cores/dosbox_pure_libretro.dll` (Windows) или `.dylib` (macOS); если нет — `ConfigError` с текстом «запусти `uv run python scripts/fetch_core.py`».
  - `scripts/fetch_core.py` — скачивает zip ядра с buildbot и распаковывает в `cores/`:
    - Windows: `https://buildbot.libretro.com/nightly/windows/x86_64/latest/dosbox_pure_libretro.dll.zip`
    - macOS arm64: `https://buildbot.libretro.com/nightly/apple/osx/arm64/latest/dosbox_pure_libretro.dylib.zip`

- [ ] **Step 1:** `uv init --package`-подобный `pyproject.toml` вручную: пакет `neuronoid`, `requires-python = ">=3.12,<3.13"`, зависимости `libretro.py`, `numpy`, `pillow`; dev-группа `pytest`. `uv sync` создаёт `.venv` на 3.12.
  Run: `uv run python -c "import sys; print(sys.version)"` → `3.12.x`.
- [ ] **Step 2: Тесты**

```python
def test_game_path_missing(tmp_path):
    with pytest.raises(ConfigError, match="DOS_arkanoid.zip"):
        game_path(tmp_path)

def test_core_path_missing_mentions_fetch(tmp_path):
    with pytest.raises(ConfigError, match="fetch_core"):
        core_path(tmp_path)

def test_paths_found(tmp_path):
    (tmp_path / "games/arkanoid").mkdir(parents=True)
    (tmp_path / "games/arkanoid/DOS_arkanoid.zip").write_bytes(b"x")
    (tmp_path / "cores").mkdir()
    name = "dosbox_pure_libretro.dll" if sys.platform == "win32" else "dosbox_pure_libretro.dylib"
    (tmp_path / "cores" / name).write_bytes(b"x")
    assert game_path(tmp_path).name == "DOS_arkanoid.zip"
    assert core_path(tmp_path).name == name
```

- [ ] **Step 3:** Run `uv run pytest tests/test_config.py -v` → FAIL (нет модуля).
- [ ] **Step 4:** Реализовать `neuronoid/config.py` и `scripts/fetch_core.py` (urllib + zipfile, без новых зависимостей; неподдерживаемая платформа → понятная ошибка).
- [ ] **Step 5:** Run `uv run pytest tests/test_config.py -v` → PASS. Run `uv run python scripts/fetch_core.py` → в `cores/` появляется ядро.
- [ ] **Step 6:** Commit `feat: каркас проекта, конфиг путей, загрузка ядра DOSBox Pure`.

---

### Task 2: Проверка (spike) — запуск игры, ввод, savestates, скорость

Исследовательская задача: код одноразовый, без тестов; результат — отчёт и скриншоты. **Контрольная точка: по окончании остановиться и показать пользователю отчёт и скриншоты.**

**Files:**
- Create: `scripts/spike.py`
- Create: `docs/spike-report.md`
- Output (не в git): `runs/spike/*.png`

**Interfaces:**
- Consumes: `config.game_path()`, `config.core_path()`.
- Produces: `docs/spike-report.md` с разделами, которые читают Task 3 и следующий план:
  1. **Фронтенд:** работает ли `libretro.py` с DOSBox Pure (загрузка, кадры, клавиатура, savestates, звук). Если нет — что именно не работает и решение «свой ctypes-фронтенд».
  2. **Запуск игры:** как DOSBox Pure стартует `ARKANOID.COM` из zip (авто-старт или меню выбора — какие опции ядра `dosbox_pure_*` выставлены).
  3. **Опции ядра:** итоговый набор (`dosbox_pure_cycles` и др.), видеорежим игры (CGA/EGA/VGA), размер кадра, `fps` из av_info. Отдельно — проверка, что скорость игры (движение мяча за N эмулированных кадров) не зависит от скорости прогона.
  4. **Клавиши:** какие клавиши двигают ракетку, запускают мяч, проходят меню; работает ли «влево+огонь» одновременно.
  5. **Последовательность до игры:** точная последовательность «кадры ожидания + клавиши» от включения до «раунд 1, мяч на ракетке».
  6. **Экран:** скриншоты меню, игрового поля, «ROUND N», потери жизни, Game Over; где на экране счёт и жизни (координаты прямоугольников).
  7. **Защита от копирования:** есть ли вопрос на старте.
  8. **Скорость:** кадров/с одного экземпляра без вывода; размер savestate в байтах.
  9. **Вердикт:** идём дальше / запасной путь (порядок из спеки: другое ядро → DOSBox-X/Staging → C → A).

- [ ] **Step 1:** `scripts/spike.py`: загрузить ядро и игру через `libretro.py`, прогнать ~20 с эмулированного времени, сохранять кадр каждые 30 кадров в `runs/spike/`. Посмотреть скриншоты (Read на PNG), понять, что происходит.
- [ ] **Step 2:** Добиться старта игры и пройти меню до раунда 1; записать последовательность. Проверить клавиши движения и запуска мяча (скриншоты до/после).
- [ ] **Step 3:** Savestate: сохранить на раунде 1, прогнать 300 кадров, загрузить, прогнать те же 300 кадров с теми же клавишами, сравнить кадры побайтно.
- [ ] **Step 4:** Замер: 10 000 кадров без сохранения картинок, кадры/с. Проверить, что скорость мяча в пикселях за кадр одинакова при разных настройках прогона.
- [ ] **Step 5:** Если `libretro.py` не справляется с клавиатурой или savestates — сделать минимальный ctypes-фронтенд прямо в спайке, повторить шаги 2–4. Написать `docs/spike-report.md` (разделы выше), приложить ключевые скриншоты в `docs/img/` (маленькие PNG; это кадры игры пользователя для личного репозитория).
- [ ] **Step 6:** Commit `docs: отчёт о проверке DOSBox Pure + Arkanoid`. **Остановиться, показать пользователю вердикт и скриншоты.** Если вердикт «не идём» — дальше не продолжать, вернуться к выбору запасного пути вместе с пользователем.

---

### Task 3: `LibretroEmulator`

**Files:**
- Create: `neuronoid/emulator.py`
- Create: `tests/test_emulator.py`
- Create: `tests/conftest.py` (фикстура: пропуск тестов, если нет игры или ядра — `pytest.skip` с текстом `ConfigError`)

**Interfaces:**
- Consumes: `config.game_path()`, `config.core_path()`; опции ядра и клавиши из `docs/spike-report.md` (разделы 3–4); фронтенд, выбранный в отчёте (раздел 1).
- Produces:
  - `class Key(enum.IntEnum)` — коды клавиш libretro (`RETROK_*`): `LEFT=276`, `RIGHT=275`, `UP=273`, `DOWN=274`, `SPACE=32`, `RETURN=13`, `ESCAPE=27`, а также `A`..`Z` (97..122) и `D0`..`D9` (48..57).
  - `class LibretroEmulator`:
    - `__init__(self, core: Path, game: Path, options: dict[str, str] | None = None, audio: bool = False)` — опции по умолчанию берутся из `DEFAULT_OPTIONS: dict[str, str]` (значения из отчёта). Второй экземпляр в том же процессе → `RuntimeError("one emulator per process")`.
    - `step(self, keys: Iterable[Key] = (), frames: int = 1) -> None` — держит `keys` нажатыми все `frames` кадров; клавиши, которых нет в `keys`, отпущены.
    - `frame(self) -> np.ndarray` — последний кадр, `uint8`, форма `(H, W, 3)`, RGB, копия.
    - `save_state(self) -> bytes`, `load_state(self, state: bytes) -> None`.
    - `audio(self) -> np.ndarray` — накопленные с прошлого вызова сэмплы `int16`, форма `(N, 2)`; при `audio=False` всегда пустой массив.
    - `fps: float` — частота кадров ядра (из av_info).
    - `close(self) -> None`; поддержка `with`.

- [ ] **Step 1: Тесты** (все используют фикстуру `emu`, которая создаёт эмулятор и прогоняет загрузку на 300 кадров)

```python
def test_frame_shape(emu):
    f = emu.frame()
    assert f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 3
    assert f.shape[0] >= 200 and f.shape[1] >= 320

def test_fps_is_fixed(emu):
    assert 50 <= emu.fps <= 75

def test_deterministic_from_state(emu):
    s = emu.save_state()
    emu.step([Key.RIGHT], 120); a = emu.frame()
    emu.load_state(s)
    emu.step([Key.RIGHT], 120); b = emu.frame()
    assert np.array_equal(a, b)

def test_keys_change_picture(emu):
    s = emu.save_state()
    emu.step([], 60); idle = emu.frame()
    emu.load_state(s)
    emu.step([Key.RIGHT], 60); moved = emu.frame()   # клавишу движения взять из отчёта
    assert not np.array_equal(idle, moved)

def test_second_instance_in_process_fails(emu):
    with pytest.raises(RuntimeError, match="one emulator per process"):
        LibretroEmulator(core_path(), game_path())

def test_state_loads_in_other_process(emu, tmp_path):
    # сохранить state и кадр после 120 кадров RIGHT; в дочернем процессе
    # (multiprocessing, spawn) создать эмулятор, load_state, step 120 RIGHT,
    # вернуть кадр; сравнить с кадром этого процесса после того же
    ...
```

  Последний тест: дописать тело по комментарию; ожидание — `np.array_equal`.
- [ ] **Step 2:** Run `uv run pytest tests/test_emulator.py -v` → FAIL (нет модуля).
- [ ] **Step 3:** Реализовать `neuronoid/emulator.py` поверх выбранного в отчёте фронтенда. Флаг «экземпляр уже создан» — на уровне модуля, сбрасывается в `close()`.
- [ ] **Step 4:** Run `uv run pytest -v` → PASS.
- [ ] **Step 5:** Commit `feat: LibretroEmulator — кадры, клавиши, savestates`.

---

### Task 4: Параллельная пропускная способность

**Files:**
- Create: `scripts/bench_parallel.py`
- Modify: `docs/spike-report.md` (раздел «Параллельный прогон»)

**Interfaces:**
- Consumes: `LibretroEmulator`, `Key`.
- Produces: таблица в отчёте: число процессов (1, 4, 8, 12, 14) → суммарные кадры/с, кадры/с на процесс, загрузка CPU. Рекомендованное число экземпляров для обучения.

- [ ] **Step 1:** `scripts/bench_parallel.py --procs 1 4 8 12 14 --frames 5000`: каждый процесс (spawn) создаёт эмулятор, проходит до раунда 1 по последовательности из отчёта, затем 5000 кадров со случайными клавишами из `{(), (LEFT,), (RIGHT,), (SPACE,)}` и вызовом `frame()` каждые 4 кадра (как при обучении с frame skip 4); печатает таблицу.
- [ ] **Step 2:** Run `uv run python scripts/bench_parallel.py --procs 1 4 8 12 14 --frames 5000` → таблица напечатана, ни один процесс не упал.
- [ ] **Step 3:** Внести таблицу и рекомендацию в отчёт, сравнить с оценкой из спеки (1500–3000 кадров/с суммарно).
- [ ] **Step 4:** Commit `docs: замер параллельной скорости эмулятора`.

---

## После этого плана

Показать пользователю итоги (скорость, скриншоты, вердикт) и написать следующий план: `ArkanoidEnv` + `ScreenReader` (+ офлайн-разметка кадров по RAM в `tools/`) + живое окно.
