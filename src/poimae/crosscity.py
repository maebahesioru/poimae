"""Cross-municipality comparisons, computed from the two official dictionaries.

This module is the evidence behind the pitch.  Every number it returns is
derived from `data/municipalities/*.json`, which is itself parsed from the
cities' own pages, so a judge can re-run it and get the same answer.

Two things are deliberately *not* here:

* no equivalence is guessed for categories without a clean counterpart
  (`雑がみ`, `資源ごみ`, `びん・缶・ペットボトル`, `リサイクルプラ` ...).  Items
  in those categories are dropped from the differing-bin statistics rather than
  forced into an analogy that might be wrong;
* no comparison is made for the dummy sample city.  It is excluded by
  `data_kind`, not by an if-statement naming it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .municipalities import Item, Municipality
from .normalize import normalize

#: Wording that makes an item conditional in the municipality's own note.
#: These are the entries where "which bin" genuinely depends on state, size or
#: cleanliness - the exact situations where people get it wrong.
CONDITION_PATTERNS = {
    "汚れ": r"汚れ|よごれ|すすいでも|洗っても",
    "大きさ": r"30cm|指定ごみ袋に入らない|袋に入らない|大きさ",
    "中身": r"中身|使い切|使いき|残さず",
    "分解": r"はずし|外し|取り除|切り離|割って",
    "材質": r"木製|金属製|プラスチック製|陶磁器|ガラス製|紙製",
}


@dataclass(frozen=True)
class Difference:
    name: str
    category_a: str
    category_b: str
    generic_a: str
    generic_b: str


def _index(municipality: Municipality) -> dict[str, Item]:
    """Normalised name -> first item.  Later duplicates lose, deterministically."""
    index: dict[str, Item] = {}
    for item in municipality.items:
        index.setdefault(normalize(item.name), item)
    return index


def compare(a: Municipality, b: Municipality) -> dict:
    """Overlap and disagreement between two official dictionaries."""
    if a.data_kind != "official" or b.data_kind != "official":
        raise ValueError(
            "compare() only accepts official dictionaries; "
            f"got {a.id}={a.data_kind}, {b.id}={b.data_kind}"
        )

    index_a, index_b = _index(a), _index(b)
    shared = sorted(set(index_a) & set(index_b))

    comparable: list[tuple[Item, Item]] = []
    unmapped: list[str] = []
    for key in shared:
        item_a, item_b = index_a[key], index_b[key]
        if item_a.generic is None or item_b.generic is None:
            unmapped.append(item_a.name)
            continue
        comparable.append((item_a, item_b))

    differences = [
        Difference(
            name=item_a.name,
            category_a=item_a.category,
            category_b=item_b.category,
            generic_a=item_a.generic or "",
            generic_b=item_b.generic or "",
        )
        for item_a, item_b in comparable
        if item_a.generic != item_b.generic
    ]
    differences.sort(key=lambda d: d.name)

    return {
        "a": {"id": a.id, "name": a.display_name, "items": len(a.items)},
        "b": {"id": b.id, "name": b.display_name, "items": len(b.items)},
        "shared_names": len(shared),
        "comparable": len(comparable),
        "unmapped": len(unmapped),
        "identical_bin": len(comparable) - len(differences),
        "different_bin": len(differences),
        "different_bin_rate": (
            round(len(differences) / len(comparable), 4) if comparable else 0.0
        ),
        "examples": [d.__dict__ for d in differences[:12]],
        "differences": [d.__dict__ for d in differences],
    }


def conditional_items(municipality: Municipality) -> dict:
    """Items whose own note makes the answer conditional, grouped by reason."""
    buckets: Counter[str] = Counter()
    examples: dict[str, list[dict]] = {}
    for item in municipality.items:
        note = item.note or ""
        if not note:
            continue
        for label, pattern in CONDITION_PATTERNS.items():
            if re.search(pattern, note):
                buckets[label] += 1
                examples.setdefault(label, [])
                if len(examples[label]) < 4:
                    examples[label].append(
                        {"name": item.name, "category": item.category, "note": note}
                    )
    return {
        "municipality": municipality.display_name,
        "total": sum(buckets.values()),
        "by_reason": dict(buckets.most_common()),
        "examples": examples,
    }


def category_shape(municipality: Municipality) -> dict:
    """How many bins the city actually asks a household to keep separate."""
    generic = {c.generic for c in municipality.categories if c.generic}
    return {
        "municipality": municipality.display_name,
        "verified": municipality.verified,
        "data_kind": municipality.data_kind,
        "categories": len(municipality.categories),
        "generic_bins": len(generic),
        "items": len(municipality.items),
        "labels": [c.label for c in municipality.categories],
    }


def ambiguity_report(registry) -> dict:
    """The whole picture: per-city shape + pairwise disagreements."""
    official = [m for m in registry.all() if m.data_kind == "official"]
    official.sort(key=lambda m: m.id)

    pairs = []
    for i, a in enumerate(official):
        for b in official[i + 1 :]:
            pairs.append(compare(a, b))

    return {
        "cities": [category_shape(m) for m in registry.all()],
        "conditional": [conditional_items(m) for m in official],
        "pairs": pairs,
    }
