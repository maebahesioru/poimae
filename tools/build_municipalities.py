#!/usr/bin/env python3
"""Rebuild data/municipalities/*.json from the official pages cached in evidence/sources/.

Why this script exists
----------------------
PoiMae does not invent waste rules.  Every item -> category pair in the shipped
JSON is parsed from a page the municipality itself publishes.  Keeping the raw
HTML in `evidence/sources/` and the parser here means a reviewer can re-run

    .venv/bin/python tools/build_municipalities.py

and get byte-identical JSON back.  If a city changes its dictionary, the diff
shows up in `git diff data/municipalities/`.

Inputs  : evidence/sources/sapporo_bunbetsu.html
          evidence/sources/hiroshima_*.html
Outputs : data/municipalities/sapporo.json
          data/municipalities/hiroshima.json

The third bundled municipality (data/municipalities/sample-dummy.json) is NOT
produced here: it is a hand-written fake kept only to prove that a municipality
file is a drop-in plugin.  It is labelled `"data_kind": "dummy"` everywhere.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "evidence" / "sources"
OUT = ROOT / "data" / "municipalities"

RETRIEVED_AT = "2026-10-20"

# ---------------------------------------------------------------------------
# Generic bin axis
# ---------------------------------------------------------------------------
# The two cities do not share a taxonomy, which is the whole point of the app.
# To compare them at all we need a small common axis.  This mapping is OUR
# editorial reduction, not a statement by either municipality: it only covers
# the four pairs whose correspondence is unambiguous.  Categories left out
# (雜がみ, 資源ごみ, びん・缶・ペットボトル, ...) are excluded from every
# cross-city statistic so that no number in this repo rests on a shaky analogy.
GENERIC_BINS = {
    # id -> (Japanese label used in the UI, colour)
    "burnable": ("もやす", "#e2725b"),
    "non_burnable": ("もやさない", "#4a6fa5"),
    "oversized": ("粗大", "#8a6cc8"),
    "plastic_packaging": ("プラ容器包装", "#e3a008"),
    "recyclable": ("資源", "#2f9e6f"),
    "hazardous": ("危険・有害", "#c0392b"),
    "not_collected": ("市では収集しない", "#6b7280"),
    "other": ("その他", "#9ca3af"),
}

# ---------------------------------------------------------------------------
# Per-city category tables.  `generic` is the reduction above (None = excluded).
# ---------------------------------------------------------------------------
SAPPORO_CATEGORIES = {
    "燃やせるごみ": ("burnable", "有料"),
    "燃やせないごみ": ("non_burnable", "有料"),
    "大型ごみ": ("oversized", "有料"),
    "容器包装プラスチック": ("plastic_packaging", "無料"),
    "雑がみ": (None, "無料"),
    "びん・缶・ペットボトル": (None, "無料"),
    "枝・葉・草": (None, "無料"),
    "スプレー缶・カセットボンベ": ("hazardous", "無料"),
    "加熱式たばこ・電子たばこ、ライター": ("hazardous", "無料"),
    "筒型乾電池": ("hazardous", "無料"),
    "市で収集しないもの": ("not_collected", "-"),
    "市で収集しないもの ※ステーション収集はしていません": ("not_collected", "-"),
    "集団資源回収など": ("other", "-"),
    "（備考欄参照）": (None, "-"),
}

HIROSHIMA_CATEGORIES = {
    "可燃ごみ": ("burnable", "-"),
    "不燃ごみ": ("non_burnable", "-"),
    "大型ごみ": ("oversized", "-"),
    "その他プラ": ("plastic_packaging", "-"),
    "リサイクルプラ": (None, "-"),          # Hiroshima splits plastics in two
    "資源ごみ": (None, "-"),                # mixed resource bin, no clean pair
    "ペットボトル": (None, "-"),
    "有害ごみ": ("hazardous", "-"),
    "家電リサイクル法対象機器": (None, "-"),
    "家電リサイクル法 対象機器": (None, "-"),
}

# Rows whose second column is a sentence rather than a category (Hiroshima uses
# these as cross-references).  Anything not in the allowlist above is dropped.
SAPPORO_ALLOW = set(SAPPORO_CATEGORIES)
HIROSHIMA_ALLOW = set(HIROSHIMA_CATEGORIES)

_TAG = re.compile(r"(?is)<[^>]+>")
_TR = re.compile(r"(?is)<tr.*?</tr>")
_TD = re.compile(r"(?is)<t[dh].*?</t[dh]>")
_TABLE = re.compile(r"(?is)<table.*?</table>")


def cell_text(chunk: str) -> str:
    """Tag-strip one table cell and collapse the whitespace the CMS sprinkles in."""
    text = _TAG.sub("", chunk)
    text = html.unescape(text)
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    return " ".join(text.split())


def rows_of(document: str):
    """Yield the 4-column (name, category, fee, note) rows of every table."""
    for table in _TABLE.findall(document):
        for row in _TR.findall(table):
            cells = _TD.findall(row)
            if len(cells) != 4:
                continue
            yield [cell_text(c) for c in cells]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(city_id, display_name, prefecture, sources, categories, allow, meta):
    """Parse a list of cached pages into the municipality JSON shape."""
    see_source_label = set()
    items = []
    dropped = 0
    for path in sources:
        document = path.read_text(encoding="utf-8", errors="replace")
        for name, label, fee, note in rows_of(document):
            if not name or name in {"品目", "名称"}:
                continue
            if label not in allow:
                dropped += 1
                continue
            generic, default_fee = categories[label]
            see_source_label.add(label)
            items.append(
                {
                    "name": name,
                    "category": label,
                    "generic": generic,
                    "fee": fee or default_fee,
                    "note": note,
                }
            )

    missing = sorted(allow - see_source_label)
    if missing:
        raise SystemExit(
            f"{city_id}: declared categories that the pages never used: {missing}. "
            "The page structure changed - update the mapping before shipping."
        )

    # A municipality that silently shrinks is a bug, not a data update.
    if len(items) < 500:
        raise SystemExit(f"{city_id}: only {len(items)} items parsed - refusing to write")

    payload = {
        "schema_version": 1,
        "id": city_id,
        "display_name": display_name,
        "prefecture": prefecture,
        "verified": True,
        "data_kind": "official",
        "source": {
            "title": meta["title"],
            "publisher": display_name,
            "url": meta["url"],
            "retrieved_at": RETRIEVED_AT,
            "raw_files": [f"evidence/sources/{p.name}" for p in sources],
            "raw_sha256": {p.name: sha256(p) for p in sources},
        },
        "categories": [
            {
                "label": label,
                "generic": generic,
                "generic_label": GENERIC_BINS[generic][0] if generic else None,
                "default_fee": fee,
                "item_count": sum(1 for i in items if i["category"] == label),
            }
            for label, (generic, fee) in categories.items()
            if label in see_source_label
        ],
        "item_count": len(items),
        "items": items,
    }
    return payload, dropped


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    sapporo, s_drop = build(
        "sapporo",
        "札幌市",
        "北海道",
        [SRC / "sapporo_bunbetsu.html"],
        SAPPORO_CATEGORIES,
        SAPPORO_ALLOW,
        {
            "title": "家庭ごみ50音分別辞典",
            "url": "https://www.city.sapporo.jp/seiso/bunbetsu/",
        },
    )
    hiro_pages = sorted(SRC.glob("hiroshima_*.html"))
    if len(hiro_pages) != 8:
        raise SystemExit(f"expected 8 Hiroshima pages, found {len(hiro_pages)}")
    hiroshima, h_drop = build(
        "hiroshima",
        "広島市",
        "広島県",
        hiro_pages,
        HIROSHIMA_CATEGORIES,
        HIROSHIMA_ALLOW,
        {
            "title": "家庭ごみ分別50音事典",
            "url": (
                "https://www.city.hiroshima.lg.jp/living/gomi-kankyo/"
                "1021277/1003072/1026094/1008419.html"
            ),
        },
    )

    for payload in (sapporo, hiroshima):
        path = OUT / f"{payload['id']}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(
            f"{path.relative_to(ROOT)}: {payload['item_count']} items, "
            f"{len(payload['categories'])} categories"
        )

    print(f"dropped non-category rows: sapporo={s_drop} hiroshima={h_drop}")


if __name__ == "__main__":
    main()
