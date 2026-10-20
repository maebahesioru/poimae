"""The decision layer: photo -> candidate words -> the city's own rule.

PoiMae refuses to convert a photo directly into a bin.  What it does instead:

    photo --OpenCV--> features --rules--> a few material words
          --dictionary--> the city's own entries for those words
          --user--       "yes, that's the thing" -> the bin

The last arrow is the product.  "捨てる前に、ひと呼吸" means the app asks one
short question, from the *municipality's* dictionary, before it tells you where
to put something.  That is slower than a confident guess and right far more
often; the measurements in docs/METRICS.md show what the guess alone is worth.
"""

from __future__ import annotations

from .classifier import classify, candidate_keywords
from .crosscity import CONDITION_PATTERNS, _index
from .features import Features, analyze
from .municipalities import Municipality, Registry
from .normalize import normalize_query

#: How many dictionary candidates to surface in one answer.
MAX_CANDIDATES = 6

#: Words people type that a city prints under a different name.
#:
#: A 50音 dictionary is written in the city's own vocabulary: 広島市 lists
#: 「電池 その他」 and 「アルカリ電池」, never 「乾電池」, so a person holding a
#: battery was told the item did not exist.  This table is a *fallback*: it is
#: consulted only when the dictionary returned nothing at all, so it can never
#: override a real entry, and the answer says which word actually matched.
#: Keep it short and only add pairs verified against a loaded dictionary.
SYNONYM_FALLBACK: tuple[tuple[str, str], ...] = (
    ("乾電池", "電池"),
    ("アルカリ乾電池", "アルカリ電池"),
    ("マンガン乾電池", "電池"),
    ("ボタン電池", "ボタン型電池"),
    ("蛍光灯", "蛍光管"),
    ("スプレー缶", "スプレー"),
    ("ペットボトルキャップ", "キャップ"),
)


def _synonym_for(text: str) -> tuple[str, str] | None:
    """Return ``(typed, matched)`` when a fallback word applies to ``text``."""
    typed = normalize_query(text)
    for word, alternate in SYNONYM_FALLBACK:
        if typed == normalize_query(word):
            return word, alternate
    return None


def _other_city_view(registry: Registry, municipality: Municipality, item_name: str) -> list[dict]:
    """Do other real cities put this same item in a different bin?"""
    if municipality.data_kind != "official":
        return []
    views = []
    for other in registry.all():
        if other.id == municipality.id or other.data_kind != "official":
            continue
        item = _index(other).get(_normalise(item_name))
        if item is None:
            continue
        if item.generic is None or item.category is None:
            continue
        views.append(
            {
                "municipality": other.display_name,
                "category": item.category,
                "generic": item.generic,
                "same_generic": None,
            }
        )
    return views


def _normalise(text: str) -> str:
    from .normalize import normalize

    return normalize(text)


def _condition_hints(note: str) -> list[str]:
    """Turn an official note into the reason this item is easy to get wrong."""
    import re

    hints = []
    for label, pattern in CONDITION_PATTERNS.items():
        if note and re.search(pattern, note):
            hints.append(label)
    return hints


def build_answer(
    municipality: Municipality,
    lookup: dict,
    *,
    source: str,
    registry: Registry | None = None,
) -> dict:
    """Wrap a raw dictionary hit in the warnings the user needs to see."""
    answer = {
        "source": source,                       # "text" or "photo"
        "municipality": municipality.display_name,
        "municipality_id": municipality.id,
        "verified": municipality.verified,
        "data_kind": municipality.data_kind,
        "dictionary_url": municipality.source.get("url", ""),
        "lookup": lookup,
        "question": None,
        "warnings": [],
        "other_cities": [],
    }

    if not lookup.get("found"):
        answer["question"] = (
            "この品目は辞典に見つかりませんでした。名前を変えて試すか、"
            f"{municipality.display_name}のごみ分別辞典で確認してください。"
        )
        return answer

    if not lookup.get("confident"):
        # One near-miss is a different sentence from three: "候補が複数あります"
        # in front of a single candidate reads like a bug.
        candidates = lookup.get("candidates") or []
        if len(candidates) <= 1:
            answer["question"] = (
                f"「{lookup['query']}」そのものは辞典にありません。"
                f"いちばん近い「{lookup['item']}」で合っていますか？"
            )
        else:
            answer["question"] = (
                f"「{lookup['query']}」に近い候補が複数あります。"
                "どれが近いか選んでください。"
            )
    else:
        answer["question"] = (
            f"「{lookup['item']}」で合っていますか？ 合っていれば"
            f"「{lookup['category']}」です。"
        )

    note = lookup.get("note") or ""
    hints = _condition_hints(note)
    if hints:
        answer["warnings"].append(
            {
                "code": "conditional",
                "text": "公式の注記に条件があります（"
                + "・".join(hints)
                + "）。状態によって行き先が変わります。",
                "note": note,
            }
        )
    if lookup.get("category") == "大型ごみ" or lookup.get("generic") == "oversized":
        answer["warnings"].append(
            {
                "code": "oversized",
                "text": "「大型ごみ」は予約と手数料が必要な区分です。"
                "指定袋に入れば通常のごみになる場合があります。",
                "note": note,
            }
        )
    if municipality.data_kind != "official":
        answer["warnings"].append(
            {
                "code": "dummy",
                "text": "この自治体ファイルはサンプル（ダミー）です。"
                "実在の制度ではありません。",
                "note": "",
            }
        )

    if registry is not None and lookup.get("item"):
        for view in _other_city_view(registry, municipality, lookup["item"]):
            view["same_generic"] = view["generic"] == lookup.get("generic")
            answer["other_cities"].append(view)
        disagree = [v for v in answer["other_cities"] if v["same_generic"] is False]
        if disagree:
            answer["warnings"].append(
                {
                    "code": "cross_city",
                    "text": "同じ品目でも、"
                    + "・".join(v["municipality"] for v in disagree)
                    + "では違う区分になります。引っ越し・旅行先では特に注意してください。",
                    "note": "",
                }
            )

    return answer


def advise_text(registry: Registry, municipality_id: str, text: str) -> dict:
    municipality = registry.get(municipality_id)
    lookup = municipality.lookup(text)
    retried: tuple[str, str] | None = None

    if not lookup["found"]:
        candidate = _synonym_for(text)
        if candidate is not None:
            retried = candidate
            lookup = municipality.lookup(candidate[1])

    answer = build_answer(municipality, lookup, source="text", registry=registry)
    if retried is not None:
        typed, matched = retried
        answer["asked"] = text
        answer["retried_with"] = matched
        answer["warnings"].insert(
            0,
            {
                "code": "synonym",
                "text": f"「{typed}」という品名はこの辞典にありません。"
                f"同じものを指す「{matched}」で探した結果です。",
            },
        )
    return answer


def advise_photo(
    registry: Registry,
    municipality_id: str,
    image_bytes: bytes,
) -> dict:
    """Photo -> material words -> dictionary entries, as a confirmable shortlist."""
    municipality = registry.get(municipality_id)
    features: Features = analyze(image_bytes)
    evidence = classify(features)
    keywords = candidate_keywords(evidence)

    shortlist: list[dict] = []
    seen: set[str] = set()
    for rank, word in enumerate(keywords):
        for item, score, how in municipality.search(word, limit=6):
            # A photo keyword is a category noun ("缶", "びん").  Letting a fuzzy
            # or mid-word hit in ("やかん" for "かん") would fill the shortlist
            # with items that share no meaning, so only the two strong tiers
            # are allowed to name a candidate.
            if how not in ("exact", "prefix"):
                continue
            if item.name in seen:
                continue
            seen.add(item.name)
            shortlist.append(
                {
                    "item": item.name,
                    "category": item.category,
                    "generic": item.generic,
                    "fee": item.fee,
                    "note": item.note,
                    "via_keyword": word,
                    "score": score,
                    "match": how,
                    "_keyword_rank": rank,
                }
            )
    # Order by *which keyword found it* first, not by score alone.  Every
    # keyword below the first is a weaker guess about the material, so a hit on
    # the second or third keyword must never outrank a hit on the first - that
    # is how "プラ" used to drag プラモデル ahead of ペットボトル for a photo of
    # a bottle.  Within one keyword, the stronger match tier and the higher
    # score win, and the shorter name wins a true tie because a shorter entry is
    # usually the general item and the longer one a special case.
    tier = {"exact": 0, "prefix": 1}
    shortlist.sort(
        key=lambda c: (c["_keyword_rank"], tier.get(c["match"], 2), -c["score"], len(c["item"]))
    )
    shortlist = shortlist[:MAX_CANDIDATES]
    for entry in shortlist:
        entry.pop("_keyword_rank", None)

    best = shortlist[0] if shortlist else None
    lookup = {
        "query": ", ".join(keywords),
        "found": bool(shortlist),
        "confident": False,     # a photo is never a confident answer
        "item": best["item"] if best else None,
        "category": best["category"] if best else None,
        "generic": best["generic"] if best else None,
        "fee": best["fee"] if best else None,
        "note": best["note"] if best else "",
        "match": "photo",
        "score": best["score"] if best else 0.0,
        "candidates": [
            {
                "item": c["item"],
                "category": c["category"],
                "score": c["score"],
                "match": c["match"],
            }
            for c in shortlist
        ],
    }

    answer = build_answer(municipality, lookup, source="photo", registry=registry)
    answer["features"] = features.as_dict()
    answer["material_evidence"] = [e.as_dict() for e in evidence]
    answer["keywords"] = keywords
    answer["shortlist"] = shortlist
    if shortlist:
        answer["question"] = (
            "写真からは「"
            + "／".join(e.label for e in evidence[:2])
            + "」らしいと分かりました。この中から近いものを選んでください。"
        )
    else:
        answer["question"] = (
            "写真からは候補を絞れませんでした。品目名を入力してください。"
        )
    return answer
