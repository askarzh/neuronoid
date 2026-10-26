import subprocess
import sys
from pathlib import Path

import numpy as np

from neuronoid.curriculum import StartMeta, StartPool

TESTS = Path(__file__).parent


def test_empty_pool_samples_none(tmp_path):
    assert StartPool(tmp_path).sample(np.random.default_rng(0)) is None


def test_offer_and_sample_roundtrip(tmp_path):
    pool = StartPool(tmp_path)
    assert pool.offer(b"state-A", StartMeta(1, 500, 500, 2))
    state, meta, _ = pool.sample(np.random.default_rng(0))
    assert state == b"state-A" and meta.round == 1 and meta.progress == 500


def test_bucket_cap_evicts_lowest_progress(tmp_path):
    pool = StartPool(tmp_path, per_round=3)
    for p in (100, 200, 300):
        pool.offer(b"x", StartMeta(1, p, p, 2))
    assert not pool.offer(b"x", StartMeta(1, 50, 50, 2))  # хуже всех — не берём
    assert pool.offer(b"x", StartMeta(1, 400, 400, 2))
    assert pool.stats() == {1: 3}
    assert len(list((tmp_path / "1").glob("*.state"))) == 3  # файлы реально удалены


def test_sample_prefers_top_half_and_all_rounds(tmp_path):
    pool = StartPool(tmp_path, per_round=10)
    for p in range(10):
        pool.offer(b"x", StartMeta(1, p * 100, p * 100, 2))
    pool.offer(b"y", StartMeta(2, 10, 6000, 1))
    rng = np.random.default_rng(0)
    metas = [pool.sample(rng)[1] for _ in range(200)]
    assert 60 < sum(m.round == 2 for m in metas) < 140  # корзины равновероятны
    assert min(m.progress for m in metas if m.round == 1) >= 500  # только лучшая половина


def test_corrupt_file_is_dropped(tmp_path):
    pool = StartPool(tmp_path)
    pool.offer(b"good", StartMeta(1, 100, 100, 2))
    bad = next((tmp_path / "1").glob("*.json"))
    bad.with_suffix(".state").unlink()  # «процесс убит во время записи»
    assert pool.sample(np.random.default_rng(0)) is None
    assert pool.stats() == {}


def test_forget_removes_entry(tmp_path):
    pool = StartPool(tmp_path)
    pool.offer(b"x", StartMeta(1, 100, 100, 2))
    _, _, path = pool.sample(np.random.default_rng(0))
    pool.forget(path)
    assert pool.stats() == {}


def test_concurrent_writers(tmp_path):
    procs = [
        subprocess.Popen([sys.executable, str(TESTS / "_pool_worker.py"), str(tmp_path), str(i)])
        for i in range(4)
    ]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    pool = StartPool(tmp_path, per_round=5)
    pool.prune()  # отложенные удаления (файл был занят читателем) дочищаются
    assert pool.stats()[1] <= 5


def test_offer_survives_write_error(tmp_path, monkeypatch):
    pool = StartPool(tmp_path)

    def locked(self, data):
        raise PermissionError("файл занят антивирусом")

    monkeypatch.setattr(Path, "write_bytes", locked)
    assert pool.offer(b"x", StartMeta(1, 100, 100, 2)) is False


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


def test_eviction_keeps_fewest_bricks(tmp_path):
    pool = StartPool(tmp_path, per_round=3)
    for left in (30, 20, 10):
        pool.offer(b"x", StartMeta(1, 100, 100, 2, bricks_left=left))
    assert not pool.offer(b"x", StartMeta(1, 9999, 100, 2, bricks_left=40))  # больше кирпичей — хуже
    assert pool.offer(b"x", StartMeta(1, 1, 100, 2, bricks_left=5))


def test_old_meta_without_bricks_left_still_loads(tmp_path):
    pool = StartPool(tmp_path)
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "5-old.state").write_bytes(b"s")
    (tmp_path / "1" / "5-old.json").write_text('{"round": 1, "progress": 5, "score": 5, "lives": 2}')
    state, meta, _ = pool.sample(np.random.default_rng(0))
    assert state == b"s" and meta.bricks_left is None
