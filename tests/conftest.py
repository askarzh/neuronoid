import pytest

from neuronoid.config import ConfigError, core_path, game_path


@pytest.fixture(scope="session")
def emu():
    """Один эмулятор на всю сессию тестов (ядро — одно на процесс), уже на раунде 1."""
    try:
        core, game = core_path(), game_path()
    except ConfigError as e:
        pytest.skip(str(e))
    from neuronoid.boot import boot_to_round1
    from neuronoid.emulator import LibretroEmulator

    with LibretroEmulator(core, game) as e:
        boot_to_round1(e)
        yield e


@pytest.fixture
def round1(emu):
    """Состояние «раунд 1, мяч на ракетке»; каждый тест начинает с него."""
    if not hasattr(emu, "_test_round1"):
        emu._test_round1 = emu.save_state()
    emu.load_state(emu._test_round1)
    return emu._test_round1
