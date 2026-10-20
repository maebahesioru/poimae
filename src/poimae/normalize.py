"""Japanese text normalisation used by the dictionary lookup.

The same item is typed a dozen ways in real life:

    "ペットボトル"  "ペットボトル（500ml）"  "ぺっとぼとる"  "ﾍﾟｯﾄﾎﾞﾄﾙ"
    "アイスの棒"    "アイスキャンデーの棒（木製・プラ製）"

Nothing here is language-model magic; it is the small, auditable set of
folding rules that a clerk at a counter would apply.  Every rule has a test.
"""

from __future__ import annotations

import re
import unicodedata

_KATAKANA = re.compile(r"[\u30a1-\u30f6]")
# Distinguishing marks that people drop when typing but that the dictionary uses.
_NOISE = re.compile(r"[\s\u3000・･,，、。\.\|/／\\\-–—ー~〜\"'（）()\[\]【】<>〈〉《》「」『』]")
_BRACKET = re.compile(r"[（(\[【].*?[）)\]】]")


def to_hiragana(text: str) -> str:
    """Katakana -> hiragana. Leaves kanji, latin and digits untouched."""
    return _KATAKANA.sub(lambda m: chr(ord(m.group()) - 0x60), text)


def strip_brackets(text: str) -> str:
    """Drop parenthetical material/size qualifiers: 棒（木製・プラ製） -> 棒."""
    return _BRACKET.sub("", text)


def bracket_contents(text: str) -> str:
    """The inverse: only what was inside the brackets, joined by spaces."""
    return " ".join(_BRACKET.findall(text))


def normalize(text: str) -> str:
    """Fold a display name or a user query into the comparison form.

    NFKC first so that half-width katakana (ﾍﾟｯﾄﾎﾞﾄﾙ) and full-width latin
    collapse; then katakana->hiragana so that カン and かん meet; then drop the
    noise characters.  Order matters: NFKC can reintroduce spaces.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = to_hiragana(text)
    text = _NOISE.sub("", text)
    return text


def normalize_query(text: str) -> str:
    """Normalise a search box entry, also dropping a trailing quantity/unit."""
    text = normalize(text)
    # "ぺっとぼとる500" -> "ぺっとぼとる" (people paste the volume off a label)
    text = re.sub(r"[0-9]+(ml|l|g|kg|cm|mm|枚|本|個|袋)$", "", text)
    return text


def bigrams(text: str) -> set[str]:
    """Character bigrams - a tokenizer-free handle on Japanese compound nouns."""
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def dice(a: str, b: str) -> float:
    """Sørensen-Dice coefficient over character bigrams, 0.0 - 1.0."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ga, gb = bigrams(a), bigrams(b)
    if not ga or not gb:
        return 0.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))
