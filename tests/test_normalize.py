"""The search index is only as good as its normalisation."""

from __future__ import annotations

import pytest

from poimae.normalize import bracket_contents, dice, normalize, normalize_query, strip_brackets


@pytest.mark.parametrize(
    "a,b",
    [
        ("ペットボトル", "ぺっとぼとる"),          # katakana vs hiragana
        ("ＰＥＴボトル", "PETボトル"),             # full-width vs half-width
        ("乾電池", "乾電池"),                      # identity
        ("ストロー・スプーン", "ストロースプーン"),  # separator noise
    ],
)
def test_variants_normalise_together(a: str, b: str) -> None:
    assert normalize(a) == normalize(b)


def test_brackets_are_removable_but_recoverable() -> None:
    raw = "カイロ（使い捨て）"
    assert strip_brackets(raw) == "カイロ"
    assert "使い捨て" in bracket_contents(raw)


def test_query_normalisation_keeps_meaning() -> None:
    assert normalize_query("  ペット ボトル  ") == normalize("ペットボトル")


def test_dice_is_symmetric_and_bounded() -> None:
    assert dice("かん", "かん") == pytest.approx(1.0)
    assert 0.0 <= dice("かん", "びん") <= 1.0
    assert dice("かん", "びん") == pytest.approx(dice("びん", "かん"))
