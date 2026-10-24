"""Подготовка: от включения до «раунд 1, мяч на ракетке» (это не управление агентом)."""

from neuronoid.emulator import Key, LibretroEmulator

# (кадров ждать, клавиша) — из docs/spike-report.md, раздел 5
BOOT_SEQUENCE: list[tuple[int, Key]] = [
    (150, Key.RETURN),  # видеорежим: EGA (по умолчанию)
    (60, Key.M),  # устройство ввода: мышь
    (700, Key.SPACE),  # титульный экран -> старт, дальше вступление
]
FRAMES_TO_ROUND1 = 1480  # после старта: вступление и «ROUND 1», мяч на ракетке
KEY_HOLD_FRAMES = 5


def boot_to_round1(emu: LibretroEmulator) -> None:
    for wait, key in BOOT_SEQUENCE:
        emu.step(frames=wait)
        emu.step([key], frames=KEY_HOLD_FRAMES)
    emu.step(frames=FRAMES_TO_ROUND1)
