"""Loading a municipality rule file and answering "where does this go?".

A municipality file is a plain JSON document (see data/municipalities/*.json).
The engine treats every file the same way, so adding a city is a data drop-in,
not a code change.  The dummy file is loaded through this exact path, which is
how the tests prove the plugin claim instead of asserting it.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import dice, normalize, normalize_query, strip_brackets

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "municipalities"

#: Minimum fuzzy similarity for a candidate to be shown at all.  Below this the
#: engine says "not found" rather than guessing: a wrong bin is worse than no
#: answer, and the UI is built around asking the user to confirm.
FUZZY_FLOOR = 0.42

#: A partial match only counts as a confident answer when the query is long
#: enough to be a word.  "かん" is a substring of "やかん" and "アメリカンレーキ";
#: calling either of those an answer would be a fabrication.
MIN_CONFIDENT_LEN = 3


class MunicipalityError(ValueError):
    """Raised when a rule file is malformed - never silently ignored."""


def _lcs_ratio(a: str, b: str) -> float:
    """difflib's 2M/T match ratio.

    Character bigrams alone miss the case where the dictionary stores a word in
    kanji and the user types it in kana (大学 vs だいがく keep no shared bigram
    once the kanji differ).  The longest-matching-block ratio degrades far more
    gracefully there, so the fuzzy tier takes the maximum of the two.
    """
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass(frozen=True)
class Item:
    name: str
    category: str
    generic: str | None
    fee: str
    note: str


@dataclass(frozen=True)
class Category:
    label: str
    generic: str | None
    generic_label: str | None
    default_fee: str
    item_count: int


@dataclass
class Municipality:
    id: str
    display_name: str
    prefecture: str
    verified: bool
    data_kind: str
    source: dict
    categories: list[Category]
    items: list[Item]
    #: display name -> normalised form, built once at load time
    _index: dict[str, str] = field(default_factory=dict, repr=False)
    #: (item, normalised name, normalised name without parentheticals)
    _keys: list[tuple[Item, str, str]] = field(default_factory=list, repr=False)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict) -> "Municipality":
        for key in ("id", "display_name", "categories", "items"):
            if key not in payload:
                raise MunicipalityError(f"missing key {key!r}")
        if payload.get("schema_version") != 1:
            raise MunicipalityError(
                f"{payload.get('id')}: unsupported schema_version "
                f"{payload.get('schema_version')!r}"
            )

        categories = [Category(**c) for c in payload["categories"]]
        labels = {c.label for c in categories}
        items = [Item(**i) for i in payload["items"]]
        for item in items:
            if item.category not in labels:
                raise MunicipalityError(
                    f"{payload['id']}: item {item.name!r} points at undeclared "
                    f"category {item.category!r}"
                )
        if not items:
            raise MunicipalityError(f"{payload['id']}: item list is empty")

        municipality = cls(
            id=payload["id"],
            display_name=payload["display_name"],
            prefecture=payload.get("prefecture", ""),
            # `verified` is tri-state: True (checked against the official page),
            # False (the file says it is not verified) and None (the file does
            # not say).  `False or None` would erase a declared False, which is
            # exactly the flag a sample file must never lose.
            verified=payload.get("verified"),
            data_kind=payload.get("data_kind", "unknown"),
            source=payload.get("source", {}),
            categories=categories,
            items=items,
        )
        municipality._build_index()
        return municipality

    def _build_index(self) -> None:
        index: dict[str, str] = {}
        keys: list[tuple[Item, str, str]] = []
        for item in self.items:
            full = normalize(item.name)
            # Also index the name without its parenthetical qualifier, so
            # "アイスの棒" reaches "アイスキャンデーの棒（木製・プラ製）".
            stripped = normalize(strip_brackets(item.name))
            keys.append((item, full, stripped))
            index.setdefault(full, item.name)
            if stripped:
                index.setdefault(stripped, item.name)
        self._index = index
        self._keys = keys

    # -- queries -----------------------------------------------------------
    @property
    def category_labels(self) -> list[str]:
        return [c.label for c in self.categories]

    def category(self, label: str) -> Category | None:
        for category in self.categories:
            if category.label == label:
                return category
        return None

    def find_exact(self, query: str) -> Item | None:
        name = self._index.get(normalize(query))
        if name is None:
            name = self._index.get(normalize_query(query))
        if name is None:
            return None
        return next(i for i in self.items if i.name == name)

    def search(self, query: str, limit: int = 8) -> list[tuple[Item, float, str]]:
        """Rank items for a query. Returns (item, score, how) triples.

        `how` names the tier that produced the hit, so the UI can be honest
        about it and the tests can assert on the mechanism rather than on vibes.
        Tiers, strongest first:

            exact      the dictionary has this word
            prefix     the word starts with the query
            contains   the query appears inside a longer entry
            fuzzy      character-overlap similarity (bigram Dice / LCS ratio)
        """
        q = normalize_query(query)
        if not q:
            return []

        scored: list[tuple[Item, float, str]] = []
        for item, full, stripped in self._keys:
            best, how = 0.0, ""
            for key in (full, stripped):
                if not key:
                    continue
                if key == q:
                    best, how = 1.0, "exact"
                    break
                if key.startswith(q):
                    if 0.95 > best:
                        best, how = 0.95, "prefix"
                elif q in key:
                    # contained but not at the start: weaker, and the extra
                    # length is evidence that this is a different word.
                    ratio = len(q) / len(key)
                    if 0.90 * ratio > best:
                        best, how = 0.90 * ratio, "contains"
                similarity = max(dice(q, key), _lcs_ratio(q, key))
                # The LCS ratio rewards a short entry that happens to sit inside
                # a longer query: "やかん" scores 0.86 against "かんでんち".
                # Scaling by the length ratio kills that - 2/5 leaves 0.34,
                # under the show-anything floor - without touching a genuine
                # near-miss of similar length.
                length_ratio = min(len(q), len(key)) / max(len(q), len(key))
                similarity *= length_ratio
                if similarity * 0.89 > best:
                    best, how = similarity * 0.89, "fuzzy"
            if best >= FUZZY_FLOOR:
                scored.append((item, round(best, 4), how))

        scored.sort(key=lambda t: (-t[1], len(t[0].name), t[0].name))
        return scored[:limit]

    def lookup(self, query: str) -> dict:
        """The engine's answer for one query, including honest uncertainty.

        `confident` is true only for an exact hit, or a prefix hit long enough
        to be a word, and never when a near-tied candidate points at a
        different bin.  Everything else is returned as a candidate list for the
        user to confirm - PoiMae abstains instead of guessing.
        """
        candidates = self.search(query)
        if not candidates:
            return {
                "query": query,
                "found": False,
                "confident": False,
                "item": None,
                "category": None,
                "generic": None,
                "fee": None,
                "note": "",
                "match": "none",
                "score": 0.0,
                "candidates": [],
            }

        item, score, how = candidates[0]
        confident = how == "exact" or (
            how == "prefix" and len(normalize_query(query)) >= MIN_CONFIDENT_LEN
        )
        # A near tie between different bins means the word itself is ambiguous
        # in this dictionary; say so rather than picking one at random.
        if confident and len(candidates) > 1:
            second_item, second_score, _ = candidates[1]
            if second_score >= score - 1e-6 and second_item.category != item.category:
                confident = False

        return {
            "query": query,
            "found": True,
            "confident": confident,
            "item": item.name,
            "category": item.category,
            "generic": item.generic,
            "fee": item.fee,
            "note": item.note,
            "match": how,
            "score": score,
            "candidates": [
                {"item": c.name, "category": c.category, "score": s, "match": h}
                for c, s, h in candidates
            ],
        }

    def to_summary(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "prefecture": self.prefecture,
            "verified": self.verified,
            "data_kind": self.data_kind,
            "item_count": len(self.items),
            "source": self.source,
            "categories": [
                {
                    "label": c.label,
                    "generic": c.generic,
                    "generic_label": c.generic_label,
                    "item_count": c.item_count,
                    "default_fee": c.default_fee,
                }
                for c in self.categories
            ],
        }


class Registry:
    """All bundled municipality files, loaded once."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or DATA_DIR)
        self._municipalities: dict[str, Municipality] = {}
        self._load()

    def _load(self) -> None:
        paths = sorted(self.data_dir.glob("*.json"))
        if not paths:
            raise MunicipalityError(f"no municipality files under {self.data_dir}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            municipality = Municipality.from_dict(payload)
            self._municipalities[municipality.id] = municipality

    def __len__(self) -> int:
        return len(self._municipalities)

    def ids(self) -> list[str]:
        return list(self._municipalities)

    def get(self, municipality_id: str) -> Municipality:
        try:
            return self._municipalities[municipality_id]
        except KeyError:
            raise MunicipalityError(
                f"unknown municipality {municipality_id!r}; "
                f"available: {sorted(self._municipalities)}"
            ) from None

    def all(self) -> list[Municipality]:
        return list(self._municipalities.values())
