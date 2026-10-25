from uuid import uuid4

import numpy as np
import pytest

from neuronoid.live import LivePublisher, LiveReader


@pytest.fixture
def tmp_name():
    return f"neuronoid_test_{uuid4().hex[:8]}"


def test_roundtrip(tmp_name):
    pub = LivePublisher(tmp_name)
    f = np.random.default_rng(0).integers(0, 255, (200, 320, 3), np.uint8)
    pub.publish(f, {"score": 120, "round": 1, "lives": 2})
    reader = LiveReader(tmp_name)
    frame, info, n = reader.read()
    assert np.array_equal(frame, f) and info["score"] == 120 and n == 1
    reader.close()
    pub.close()


def test_frame_counter_advances(tmp_name):
    pub = LivePublisher(tmp_name)
    reader = LiveReader(tmp_name)
    for _ in range(3):
        pub.publish(np.zeros((200, 320, 3), np.uint8), {})
    assert reader.read()[2] == 3
    reader.close()
    pub.close()


def test_reader_without_publisher_returns_none(tmp_name):
    reader = LiveReader(tmp_name)
    assert reader.read() is None
    reader.close()


def test_reader_connects_after_publisher_appears(tmp_name):
    reader = LiveReader(tmp_name)
    assert reader.read() is None
    pub = LivePublisher(tmp_name)
    pub.publish(np.zeros((200, 320, 3), np.uint8), {"score": 5})
    assert reader.read()[1]["score"] == 5
    reader.close()
    pub.close()


def test_publisher_survives_reader_gone(tmp_name):
    pub = LivePublisher(tmp_name)
    reader = LiveReader(tmp_name)
    reader.read()
    reader.close()
    pub.publish(np.zeros((200, 320, 3), np.uint8), {})
    pub.close()


def test_publisher_reuses_stale_segment(tmp_name):
    stale = LivePublisher(tmp_name)  # «упал», не закрыл
    pub2 = LivePublisher(tmp_name)  # не должен падать на FileExistsError
    pub2.publish(np.zeros((200, 320, 3), np.uint8), {})
    pub2.close()
    stale.close()


def test_long_info_is_truncated_not_crashing(tmp_name):
    pub = LivePublisher(tmp_name)
    pub.publish(np.zeros((200, 320, 3), np.uint8), {"note": "x" * 1000})
    reader = LiveReader(tmp_name)
    assert reader.read() is not None
    reader.close()
    pub.close()
