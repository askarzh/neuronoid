"""Пересчёт стоимости сессий Claude Code по ценам API."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "article"))

from cost import cost_usd, usage_records  # noqa: E402


def test_dedup_by_message_and_request(tmp_path):
    line = '{"requestId":"r1","message":{"id":"m1","usage":{"output_tokens":10}}}'
    p = tmp_path / "s.jsonl"
    p.write_text(line + "\n" + line + "\n", encoding="utf-8")
    assert len(usage_records(p)) == 1


def test_cost_one_million_each():
    rec = {"input_tokens": 10**6, "output_tokens": 10**6, "cache_read_input_tokens": 10**6,
           "cache_creation": {"ephemeral_1h_input_tokens": 10**6, "ephemeral_5m_input_tokens": 0}}
    assert cost_usd([rec]) == pytest.approx(4 + 20 + 0.2 + 8)
