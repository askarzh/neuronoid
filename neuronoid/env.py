"""Gymnasium-среда: агент видит только пиксели, награда читается с экрана.

Память эмулятора не читается: счёт, жизни, раунд и конец игры берутся из
neuronoid.screen по кадру.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pathlib import Path

from neuronoid.boot import boot_to_round1
from neuronoid.curriculum import StartMeta, StartPool
from neuronoid.config import core_path, game_path
from neuronoid.emulator import LibretroEmulator, Mouse
from neuronoid.screen import ScreenState, is_title_screen, read_state

__all__ = ["ACTIONS", "ArkanoidEnv", "JumpConfirm", "ScreenState", "reward_between", "update_known"]

# Ничего, огонь (запуск мяча / лазер), медленно и быстро влево/вправо.
# Скорость ракетки игра ограничивает сама: около 1 и 2,5 пикселя за кадр.
ACTIONS: tuple[Mouse, ...] = (
    Mouse(),
    Mouse(left=True),
    Mouse(dx=-4),
    Mouse(dx=4),
    Mouse(dx=-12),
    Mouse(dx=12),
)

SCORE_SCALE = 0.01
MAX_SCORE_JUMP = 5000  # больший прирост за шаг — ошибка чтения, а не игра
LIFE_PENALTY = -1.0
ROUND_BONUS = 10.0
# Конец игры — титульный экран. Отсутствие поля само по себе не конец: переход
# между раундами может гасить экран. На случай зависания — предохранитель.
MAX_NOT_PLAYING_STEPS = 1000

FIELD = (slice(4, 190), slice(32, 224))  # игровое поле внутри стен, 186×192
OBS_SHAPE = (93, 96)


def reward_between(prev: ScreenState, cur: ScreenState) -> float:
    """Награда за переход между последними известными значениями и новым кадром."""
    reward = 0.0
    if prev.score is not None and cur.score is not None:
        gain = cur.score - prev.score
        if 0 < gain <= MAX_SCORE_JUMP:
            reward += gain * SCORE_SCALE
    if prev.lives is not None and cur.lives is not None and cur.lives < prev.lives:
        reward += LIFE_PENALTY
    if prev.round is not None and cur.round is not None and cur.round > prev.round:
        reward += ROUND_BONUS
    return reward


def update_known(known: ScreenState, cur: ScreenState) -> ScreenState:
    """Последние известные значения: None не затирает, счёт только растёт."""
    score = known.score
    if cur.score is not None:
        if known.score is None:
            score = cur.score
        elif 0 <= cur.score - known.score <= MAX_SCORE_JUMP:
            score = cur.score
        # падение счёта и неправдоподобный скачок — ошибки чтения: известный счёт
        # не меняется, иначе ложное завышение оплатилось бы дважды или «съело» бы
        # награду за следующие настоящие очки
    return ScreenState(
        playing=cur.playing,
        score=score,
        round=cur.round if cur.round is not None else known.round,
        lives=cur.lives if cur.lives is not None else known.lives,
    )


JUMP_CONFIRM_STEPS = 3


class JumpConfirm:
    """Большой скачок счёта, устойчивый JUMP_CONFIRM_STEPS шагов, — настоящий бонус.

    Так выход в следующий раунд через капсулу «B» (+6710 очков) не принимается
    за ошибку чтения навсегда. Бонус становится новой базой без выплаты:
    за сам переход и так начисляется ROUND_BONUS.
    """

    def __init__(self):
        self._reads: list[int] = []

    def update(self, known: int | None, read: int | None) -> int | None:
        """Новая база счёта, если скачок подтвердился, иначе None."""
        if known is None or read is None or read - known <= MAX_SCORE_JUMP:
            self._reads.clear()
            return None
        if self._reads and read < self._reads[-1]:
            self._reads.clear()
        self._reads.append(read)
        if len(self._reads) >= JUMP_CONFIRM_STEPS:
            self._reads.clear()
            return read
        return None


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Поле -> оттенки серого -> среднее по блокам 2×2: uint8 (93, 96)."""
    field = frame[FIELD].astype(np.float32)
    gray = field @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    small = gray.reshape(OBS_SHAPE[0], 2, OBS_SHAPE[1], 2).mean(axis=(1, 3))
    return small.astype(np.uint8)


class ArkanoidEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        frame_skip: int = 4,
        frame_stack: int = 4,
        sticky_prob: float = 0.25,
        max_noop: int = 30,
        life_loss_terminal: bool = False,
        max_steps: int = 27_000,
        live: bool = False,
        emulator: LibretroEmulator | None = None,
        start_state: bytes | None = None,
        start_pool: Path | None = None,
        curriculum_prob: float = 0.5,
        snapshot_every: int = 100,
        snapshot_confirm_steps: int = 60,
    ):
        """`emulator`/`start_state` — для тестов: чужой эмулятор, уже знающий старт.

        Без них эмулятор создаётся при первом reset и проходит загрузку до раунда 1.
        `start_pool` — учебная программа: среда сама кладёт туда позиции, до которых
        дошёл агент, и с вероятностью `curriculum_prob` начинает новую игру с одной из них.
        """
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.sticky_prob = sticky_prob
        self.max_noop = max_noop
        self.life_loss_terminal = life_loss_terminal
        self.max_steps = max_steps
        self.live = live
        self._emu = emulator
        self._owns_emu = emulator is None
        self._start = start_state
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(0, 255, (frame_stack, *OBS_SHAPE), np.uint8)
        self._stack = np.zeros((frame_stack, *OBS_SHAPE), np.uint8)
        self._known = ScreenState(False, None, None, None)
        self._prev_action = 0
        self._steps = 0
        self._not_playing = 0
        self._start_score: int | None = None
        self._jump = JumpConfirm()
        self._continue_game = False  # после потери жизни reset продолжает ту же игру
        self._pool = StartPool(start_pool) if start_pool is not None else None
        self.curriculum_prob = curriculum_prob
        self.snapshot_every = snapshot_every
        # позиция уходит в пул, только если после неё ещё столько шагов не было ни
        # потери жизни, ни Game Over: иначе пул забивается уже проигранными позициями
        # (мяч пролетел мимо ракетки, на поле надпись GAME OVER)
        self.snapshot_confirm_steps = snapshot_confirm_steps
        self._pending: list[tuple[int, bytes, StartMeta]] = []
        self._round_entry_score: int | None = None  # счёт при входе в текущий раунд
        self._curriculum_start = False
        self._frame = None
        self._live_pub = None

    def _ensure_emulator(self) -> LibretroEmulator:
        if self._emu is None:
            self._emu = LibretroEmulator(core_path(), game_path())
        if self._start is None:
            boot_to_round1(self._emu)
            self._start = self._emu.save_state()
        return self._emu

    def _advance(self, mouse: Mouse) -> np.ndarray:
        """frame_skip кадров; наблюдение — максимум двух последних (против мерцания)."""
        emu = self._emu
        if self.frame_skip > 1:
            emu.step(mouse=mouse, frames=self.frame_skip - 1)
            before = emu.frame()
            emu.step(mouse=mouse)
        else:
            emu.step(mouse=mouse)
            before = None
        self._frame = emu.frame()
        obs = preprocess(self._frame)
        if before is not None and before.shape == self._frame.shape:
            obs = np.maximum(obs, preprocess(before))
        return obs

    def _info(self, state: ScreenState) -> dict:
        # known_* — последние прочитанные значения: на титульном экране после
        # Game Over обычные поля уже None, а итог эпизода нужен для графиков
        k = self._known
        return {
            "score": state.score, "round": state.round, "lives": state.lives, "playing": state.playing,
            "known_score": k.score, "known_round": k.round, "known_lives": k.lives,
            # очки, набранные с начала эпизода (без набранных за случайный старт)
            "gained_score": (k.score - self._start_score)
            if k.score is not None and self._start_score is not None else 0,
            "curriculum_start": self._curriculum_start,
        }

    def _publish(self, state: ScreenState) -> None:
        if self.live:
            if self._live_pub is None:
                from neuronoid.live import LivePublisher

                self._live_pub = LivePublisher()
            self._live_pub.publish(self._frame, self._info(state))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        emu = self._ensure_emulator()
        if self._continue_game:
            # как EpisodicLife в Atari: эпизод кончился потерей жизни, а игра идёт дальше —
            # агент учится и на второй-третьей жизни, с полуразбитой стеной и т. д.
            self._continue_game = False
            state = read_state(self._frame)
            self._known = update_known(self._known, state)
        else:
            state = self._start_from_pool()
            if state is None:
                self._curriculum_start = False
                emu.load_state(self._start)
                for _ in range(int(self.np_random.integers(0, self.max_noop + 1))):
                    emu.step(mouse=ACTIONS[0], frames=self.frame_skip)
                self._frame = emu.frame()
                state = read_state(self._frame)
                self._round_entry_score = state.score
            self._known = state
        self._stack[:] = preprocess(self._frame)
        self._pending.clear()
        self._start_score = self._known.score
        self._jump = JumpConfirm()
        self._prev_action = 0
        self._steps = 0
        self._not_playing = 0
        self._publish(state)
        return self._stack.copy(), self._info(state)

    def _start_from_pool(self) -> ScreenState | None:
        """Позиция из пула или None (пула нет, не выпало, пусто, позиция битая)."""
        if self._pool is None or self.np_random.random() >= self.curriculum_prob:
            return None
        try:
            got = self._pool.sample(self.np_random)
        except OSError:
            return None
        if got is None:
            return None
        data, meta, path = got
        try:
            self._emu.load_state(data)
            self._frame = self._emu.frame()
            state = read_state(self._frame)
            if not state.playing or state.score is None:
                raise ValueError("позиция не игровая")
        except (RuntimeError, ValueError):
            self._pool.forget(path)
            return None
        self._curriculum_start = True
        self._round_entry_score = state.score - meta.progress
        return state

    def _maybe_snapshot(self, state: ScreenState, lost_life: bool) -> None:
        k = self._known
        if (
            self._pool is None
            or self._steps % self.snapshot_every
            or not state.playing
            or lost_life
            or None in (k.score, k.round, k.lives, self._round_entry_score)
        ):
            return
        progress = max(0, k.score - self._round_entry_score)
        self._pending.append((self._steps, self._emu.save_state(), StartMeta(k.round, progress, k.score, k.lives)))

    def _confirm_snapshots(self, bad: bool) -> None:
        if self._pool is None:
            return
        if bad:
            self._pending.clear()
            return
        while self._pending and self._steps - self._pending[0][0] >= self.snapshot_confirm_steps:
            _, data, meta = self._pending.pop(0)
            self._pool.offer(data, meta)

    def step(self, action):
        action = int(action)
        if self.np_random.random() < self.sticky_prob:
            action = self._prev_action
        self._prev_action = action
        obs = self._advance(ACTIONS[action])
        self._stack = np.roll(self._stack, -1, axis=0)
        self._stack[-1] = obs
        self._steps += 1

        state = read_state(self._frame)
        k = self._known
        # Новая игра без титульного экрана (клик в момент Game Over сразу начинает
        # игру заново): прочитанный счёт меньше известного, а жизней не меньше.
        # В одной игре счёт не убывает, а точное сравнение глифов не читает меньшее.
        new_game = (
            state.playing and state.score is not None and k.score is not None and state.score < k.score
            and (state.lives is None or k.lives is None or state.lives >= k.lives)
        )
        reward = reward_between(self._known, state)
        lost_life = (
            self._known.lives is not None and state.lives is not None and state.lives < self._known.lives
        )
        prev_round = self._known.round
        self._known = update_known(self._known, state)
        base = self._jump.update(self._known.score, state.score)
        if base is not None:
            # подтверждённый бонус: новая база; очки за эпизод считаем без него
            if self._start_score is not None:
                self._start_score += base - self._known.score
            if self._round_entry_score is not None:
                self._round_entry_score += base - self._known.score  # бонус — не разбитые кирпичи
            self._known = ScreenState(self._known.playing, base, self._known.round, self._known.lives)
        if prev_round is not None and self._known.round is not None and self._known.round > prev_round:
            self._round_entry_score = self._known.score  # новый раунд: прогресс с нуля
        self._not_playing = 0 if state.playing else self._not_playing + 1
        game_over = new_game or is_title_screen(self._frame) or self._not_playing >= MAX_NOT_PLAYING_STEPS
        self._confirm_snapshots(bad=lost_life or game_over)
        if not game_over:
            self._maybe_snapshot(state, lost_life)
        terminated = game_over or (self.life_loss_terminal and lost_life)
        truncated = self._steps >= self.max_steps
        self._continue_game = self.life_loss_terminal and lost_life and not game_over and not truncated
        self._publish(state)
        return self._stack.copy(), reward, terminated, truncated, self._info(state)

    def close(self) -> None:
        if self._live_pub is not None:
            self._live_pub.close()
            self._live_pub = None
        if self._owns_emu and self._emu is not None:
            self._emu.close()
            self._emu = None
