"""Проверка текста статьи: число слов и числа, у которых нет источника в facts.md.

Запуск: python docs/article/check.py habr.md --min 1800 --max 2200
Код выхода 1, если слов вне диапазона или в тексте есть числа, которых нет в таблице фактов.
"""

import argparse
import re
import sys
from pathlib import Path

FACTS = Path(__file__).with_name("facts.md")
SMALL = 10  # числа 0..SMALL не проверяем: «два-три», порядковые и т. п.

# Группы по три цифры через пробел, неразрывный пробел или запятую — разделитель тысяч;
# затем необязательная дробная часть через точку или запятую.
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:[   ,]\d{3})+(?!\d)|\d+)(?:[.,](\d+))?")


def _clean(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"https?://\S+", " ", text)


def word_count(text: str) -> int:
    return sum(1 for w in _clean(text).split() if re.search(r"\w", w))


# Малое число всё же проверяем, если оно — сумма, часть диапазона, счёт «N из M» или стоит перед множителем.
_SMALL_BEFORE = re.compile(r"(\$|(?:\bиз|\bof)\s)$", re.I)
_SMALL_AFTER = re.compile(r"^(?:[–-]\d+)?\s*(?:тыс|млн|thousand|million|%)|^[–-]\d|^\s+(?:из|of|out of)\s+\d", re.I)


def _numbers(text: str):
    """(число в нормальной форме, нужно ли проверять малое число) для каждого числа текста."""
    text = _clean(text)
    for m in _NUMBER.finditer(text):
        n = re.sub(r"[   ,]", "", m.group(1))
        n = f"{n}.{m.group(2)}" if m.group(2) else n
        weighty = bool(_SMALL_BEFORE.search(text[max(0, m.start() - 4):m.start()])
                       or _SMALL_AFTER.search(text[m.end():m.end() + 16])
                       or re.search(r"\d[–-]$", text[:m.start()]))
        yield n, weighty


def numbers_in(text: str) -> set[str]:
    return {n for n, _ in _numbers(text)}


def load_facts(path: Path) -> set[str]:
    facts = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = line.strip().strip("|").split("|")
        if line.startswith("|") and cells:
            facts |= numbers_in(cells[0])
    return facts


def unverified(text: str, facts: set[str]) -> set[str]:
    return {n for n, weighty in _numbers(text)
            if n not in facts and (weighty or not (n.isdigit() and int(n) <= SMALL))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--min", type=int, default=0)
    ap.add_argument("--max", type=int, default=10**9)
    ap.add_argument("--facts", type=Path, default=FACTS)
    args = ap.parse_args()
    text = args.file.read_text(encoding="utf-8")
    words = word_count(text)
    missing = sorted(unverified(text, load_facts(args.facts)))
    print(f"слов: {words} (нужно {args.min}–{args.max})")
    print("числа без источника:", ", ".join(missing) if missing else "нет")
    return 0 if args.min <= words <= args.max and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
