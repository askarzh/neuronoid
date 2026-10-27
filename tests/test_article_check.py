"""Проверка текста статьи: подсчёт слов и числа без источника."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "article"))

from check import load_facts, numbers_in, unverified, word_count  # noqa: E402


def test_numbers_normalized():
    assert numbers_in("набрал 101 820 очков за 2,5 часа") == {"101820", "2.5"}


def test_english_thousands_separator():
    assert numbers_in("mean score 343,977 in 2.5 hours") == {"343977", "2.5"}


def test_small_numbers_ignored():
    assert unverified("за 3 партии и 17 млн шагов", {"17"}) == set()


def test_unverified_found():
    assert unverified("средний счёт 343 977", {"343970"}) == {"343977"}


def test_word_count_skips_markup_and_urls():
    assert word_count("## Заголовок\n\nДва слова ![подпись](img/a.png) https://x.y") == 3


def test_load_facts(tmp_path):
    p = tmp_path / "facts.md"
    p.write_text("| Число | Что | Источник |\n|---|---|---|\n| 343 977 | среднее | log |\n", encoding="utf-8")
    assert load_facts(p) == {"343977"}


def test_small_numbers_with_multiplier_checked():
    assert unverified("плато около 7–8 тыс. очков", set()) == {"7", "8"}
    assert unverified("about $5 and 2 million steps", set()) == {"5", "2"}


def test_small_numbers_in_scores_checked():
    assert unverified("в 10 партиях из 10", set()) == {"10"}
    assert unverified("won 9 of 10, 10 out of 10", {"10"}) == {"9"}


def test_article_texts_match_facts():
    root = Path(__file__).resolve().parents[1] / "docs" / "article"
    facts = load_facts(root / "facts.md")
    for name in ("habr.md", "linkedin.md", "linkedin-post.md"):
        assert unverified((root / name).read_text(encoding="utf-8"), facts) == set(), name
