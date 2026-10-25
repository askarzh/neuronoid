import pytest

from neuronoid.config import ConfigError, core_path, game_path


@pytest.fixture(scope="session")
def emu():
    """Один эмулятор на всю сессию тестов (ядро — одно на процесс), уже на раунде 1.

    Состояние «раунд 1, мяч на ракетке» снимается сразу после загрузки: `emu.round1_state`.
    """
    try:
        core, game = core_path(), game_path()
    except ConfigError as e:
        pytest.skip(str(e))
    from neuronoid.boot import boot_to_round1
    from neuronoid.emulator import LibretroEmulator

    with LibretroEmulator(core, game) as e:
        boot_to_round1(e)
        e.round1_state = e.save_state()
        yield e


@pytest.fixture
def round1(emu):
    """Каждый тест эмулятора начинает с состояния «раунд 1»."""
    emu.load_state(emu.round1_state)
    return emu.round1_state
