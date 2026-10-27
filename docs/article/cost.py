"""Пересчёт расходов сессий Claude Code на цены API (Claude Opus 5.5).

Подписка денег за токены не берёт; это оценка «сколько стоило бы по API».
Запуск: python docs/article/cost.py <session.jsonl>... — токены и сумма по фазам работы исследователя.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# $ за 1 млн токенов; запись в кэш на 5 минут — 1,25× ввода, на 1 час — 2× ввода.
PRICES = {"input": 4.0, "output": 20.0, "cache_write_1h": 8.0, "cache_write_5m": 5.0, "cache_read": 0.20}

# Границы фаз (UTC) — по коммитам neuronoid-strategist: v1…мультимяч, победа над DOH, починки раунда 25.
PHASES = [("игрок с нуля", "2026-10-26T09:40:00Z"),
          ("победа над DOH", "2026-10-26T12:55:00Z"),
          ("починки раунда 25", "2026-10-27T05:10:00Z"),
          ("после", "9999")]


def usage_records(path: Path) -> list[dict]:
    """usage из журнала сессии; одно сообщение пишется несколькими строками — берём его один раз."""
    seen, records = set(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = entry.get("message")
        if not isinstance(msg, dict) or not msg.get("usage"):
            continue
        key = (msg.get("id"), entry.get("requestId"))
        if key in seen:
            continue
        seen.add(key)
        records.append({**msg["usage"], "timestamp": entry.get("timestamp", "")})
    return records


def cost_usd(records: list[dict], prices: dict = PRICES) -> float:
    total = 0.0
    for r in records:
        cc = r.get("cache_creation") or {}
        w1h = cc.get("ephemeral_1h_input_tokens") or 0
        w5m = cc.get("ephemeral_5m_input_tokens") or 0
        if not cc:
            w5m = r.get("cache_creation_input_tokens") or 0
        total += ((r.get("input_tokens") or 0) * prices["input"]
                  + (r.get("output_tokens") or 0) * prices["output"]
                  + (r.get("cache_read_input_tokens") or 0) * prices["cache_read"]
                  + w1h * prices["cache_write_1h"] + w5m * prices["cache_write_5m"])
    return total / 1e6


def main() -> None:
    phases = defaultdict(list)
    for f in sys.argv[1:]:
        for r in usage_records(Path(f)):
            phases[next(name for name, end in PHASES if r["timestamp"] < end)].append(r)
    for name, _ in PHASES:
        recs = phases.get(name, [])
        out = sum(r.get("output_tokens") or 0 for r in recs)
        read = sum(r.get("cache_read_input_tokens") or 0 for r in recs)
        print(f"{name:20} вызовов {len(recs):5}  вывод {out / 1e3:7.0f} тыс.  чтение кэша {read / 1e6:6.1f} млн"
              f"  ≈ ${cost_usd(recs):.0f}")


if __name__ == "__main__":
    main()
