"""Material guess from the OpenCV feature vector - an explainable rule table.

Why rules and not a CNN
-----------------------
We ship no trained weights, and we have no labelled photo corpus to train or
validate one on.  A small explainable rule table has three properties that
matter more here than a few points of accuracy on an unknown distribution:

  1. every guess comes with the reasons that produced it, so a user can tell
     when the app is guessing badly;
  2. it cannot silently degrade - the code is the model;
  3. it runs in ~3ms on a laptop with no download, no GPU and no telemetry.

The output is a *ranked list of candidate materials*, not a verdict.  The
municipality dictionary is what decides the bin (see `advisor.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import Features

#: Material -> the words PoiMae will try in the municipality dictionary when
#: this material wins.  Ordered by how specific the word is.
MATERIAL_KEYWORDS: dict[str, list[str]] = {
    "pet_bottle": ["ペットボトル", "ボトル", "プラ"],
    "metal_can": ["かん", "缶", "空き缶", "アルミ缶", "スチール缶"],
    "glass_bottle": ["びん", "ビン", "瓶"],
    "paper": ["紙", "新聞", "雑がみ", "紙くず"],
    "cardboard": ["段ボール", "紙", "雑がみ"],
    "plastic_film": ["トレー", "パック", "ラップ", "袋", "容器包装プラスチック"],
    "food_waste": ["生ごみ", "食べ残し", "調理くず"],
    "battery": ["電池", "乾電池", "充電池"],
    "textile": ["衣類", "布", "タオル"],
    "ceramic": ["陶磁器", "茶わん", "皿", "せともの"],
    "unknown": [],
}

MATERIAL_LABELS_JA: dict[str, str] = {
    "pet_bottle": "ペットボトル",
    "metal_can": "金属缶",
    "glass_bottle": "ガラスびん",
    "paper": "紙",
    "cardboard": "段ボール",
    "plastic_film": "プラ容器・フィルム",
    "food_waste": "生ごみ・食品",
    "battery": "電池",
    "textile": "布・衣類",
    "ceramic": "陶磁器",
    "unknown": "判別できない",
}


@dataclass
class Evidence:
    material: str
    label: str
    score: float
    reasons: list[str]
    contradictions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "material": self.material,
            "label": self.label,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "contradictions": self.contradictions,
        }


def _between(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def _hue_reliable(f: Features) -> bool:
    """Hue is undefined for a grey pixel; without this gate every white card
    "is" brown cardboard at hue 30, because OpenCV returns a confident-looking
    number for a colourless pixel."""
    return f.saturation_mean >= 0.12


def _rules() -> dict[str, list[tuple[str, object, float]]]:
    """material -> [(reason shown to the user, predicate, weight)].

    Where the numbers come from
    ---------------------------
    The thresholds are set from the *development split* of the synthetic corpus
    (`tools/synth.py`, 6 images per class), by taking each feature's per-class
    median and placing the cut between the two classes it has to separate.  That
    is a documented, reproducible procedure - and it is fitting: a rule table
    tuned on drawings will not transfer perfectly to photographs.  The test
    split and the real-photo set are never used to choose a number, and
    `docs/EVALUATION.md` reports all three.

    A negative weight is *contradicting* evidence: a dark frame argues against
    film, a glossy surface argues against paper.  Without them a flat grey card
    scored 1.00 for ceramic, paper and plastic film at once - three different
    answers, each equally "certain".
    """
    return {
        "pet_bottle": [
            # dev medians: aspect 0.31 (film 0.92, ceramic 1.63), texture 0.89
            ("縦長のボトル形状", lambda f: f.aspect <= 0.55, 2.0),
            ("明るい（中身が透ける）", lambda f: f.value_mean >= 0.80, 1.5),
            ("ラベルやしわの細部がある", lambda f: f.texture >= 0.40, 2.0),
            ("輪郭が多い", lambda f: f.edge_density >= 0.045, 1.0),
            ("色が淡い", lambda f: f.saturation_mean <= 0.35, 1.0),
            ("暗い（ボトルらしくない）", lambda f: f.value_mean <= 0.45, -1.5),
        ],
        "metal_can": [
            # dev medians: value 0.75, saturation 0.05, band_contrast 0.084
            ("金属的な無彩色", lambda f: f.saturation_mean <= 0.22, 2.0),
            ("中くらいの明るさ", lambda f: _between(f.value_mean, 0.55, 0.92), 1.5),
            ("円筒に近い形状", lambda f: _between(f.aspect, 0.35, 1.15), 1.0),
            ("帯状の塗り分け", lambda f: f.band_contrast >= 0.05, 1.0),
            ("表面がなめらか", lambda f: f.texture <= 0.45, 1.0),
            ("暗すぎる（缶らしくない）", lambda f: f.value_mean <= 0.45, -1.5),
        ],
        "glass_bottle": [
            # dev medians: value 0.38, saturation 0.61, solidity 0.68
            ("暗い（色ガラス）", lambda f: f.value_mean <= 0.55, 2.5),
            ("色がついている", lambda f: f.saturation_mean >= 0.40, 2.0),
            ("縦長のボトル形状", lambda f: _between(f.aspect, 0.20, 0.90), 1.5),
            ("曲面で輪郭がくずれる", lambda f: f.solidity <= 0.88, 1.0),
            ("明るすぎる（色ガラスらしくない）", lambda f: f.value_mean >= 0.80, -1.5),
        ],
        "paper": [
            # dev medians: value 0.96, texture 0.64, edge_density 0.056
            ("明るい", lambda f: f.value_mean >= 0.88, 2.0),
            ("ほぼ無彩色", lambda f: f.saturation_mean <= 0.18, 1.5),
            ("印刷・折り目の細部", lambda f: f.texture >= 0.35, 2.0),
            ("輪郭が多い", lambda f: f.edge_density >= 0.040, 1.0),
            ("平面的な比率", lambda f: _between(f.aspect, 0.70, 1.40), 0.5),
            ("ざらつきすぎ（紙らしくない）", lambda f: f.texture >= 2.5, -1.0),
        ],
        "cardboard": [
            # dev medians: hue 14, saturation 0.58, texture 0.07
            ("茶系の色", lambda f: _hue_reliable(f) and _between(f.hue_median, 5.0, 30.0), 2.5),
            ("しっかり色がついている", lambda f: f.saturation_mean >= 0.40, 2.0),
            ("板らしく平ら", lambda f: f.texture <= 0.25, 1.0),
            ("箱らしい比率", lambda f: _between(f.aspect, 0.60, 1.60), 0.5),
            ("茶色ではない", lambda f: _hue_reliable(f) and not _between(f.hue_median, 5.0, 35.0), -2.5),
        ],
        "plastic_film": [
            # dev medians: texture 0.10, edge_density 0.012 - the flattest class
            ("表面が非常になめらか", lambda f: f.texture <= 0.30, 2.5),
            ("輪郭がほとんど無い", lambda f: f.edge_density <= 0.030, 1.5),
            ("色が淡い", lambda f: f.saturation_mean <= 0.22, 1.5),
            ("明るい", lambda f: f.value_mean >= 0.80, 1.0),
            ("かたい輪郭", lambda f: f.solidity >= 0.92, 0.5),
            ("暗い（フィルムらしくない）", lambda f: f.value_mean <= 0.55, -2.0),
            ("細部が多い", lambda f: f.texture >= 0.55, -1.5),
        ],
        "food_waste": [
            # dev medians: solidity 0.72, extent 0.45, colorfulness 35
            ("輪郭がいびつ", lambda f: f.solidity <= 0.88, 2.0),
            ("形がまとまらない", lambda f: f.extent <= 0.72, 1.5),
            ("彩度が高い（有機物らしい）", lambda f: f.saturation_mean >= 0.35, 1.5),
            ("色の幅が広い", lambda f: f.colorfulness >= 20.0, 1.0),
            ("質感が不規則", lambda f: f.texture >= 0.15, 0.5),
            ("四角く整いすぎている", lambda f: f.solidity >= 0.95 and f.extent >= 0.88, -1.5),
        ],
        "battery": [
            # dev medians: value 0.23, foreground_ratio 0.11, aspect 0.36
            ("暗い", lambda f: f.value_mean <= 0.35, 2.5),
            ("小さい", lambda f: f.foreground_ratio <= 0.20, 1.0),
            ("細長い", lambda f: _between(f.aspect, 0.20, 0.75), 1.0),
            ("表面がなめらか", lambda f: f.texture <= 0.60, 0.5),
            ("明るい（乾電池らしくない）", lambda f: f.value_mean >= 0.60, -1.5),
        ],
        "textile": [
            # dev medians: solidity 0.98, extent 0.90, saturation 0.43
            ("しわで面が広い", lambda f: f.solidity >= 0.90, 1.5),
            ("画面の大半を占める", lambda f: f.extent >= 0.78, 1.5),
            ("中間の彩度", lambda f: _between(f.saturation_mean, 0.18, 0.70), 1.5),
            ("中くらいの明るさ", lambda f: _between(f.value_mean, 0.40, 0.80), 1.0),
            ("織り目の細部", lambda f: _between(f.texture, 0.08, 0.80), 1.0),
            ("茶色っぽい（段ボールの可能性）", lambda f: _hue_reliable(f) and _between(f.hue_median, 5.0, 30.0), -1.0),
        ],
        "ceramic": [
            # dev medians: aspect 1.63, value 0.93, texture 0.18
            ("横長の器", lambda f: f.aspect >= 1.20, 2.0),
            ("明るい", lambda f: f.value_mean >= 0.85, 1.5),
            ("ほぼ無彩色", lambda f: f.saturation_mean <= 0.15, 1.5),
            ("かたい輪郭", lambda f: f.solidity >= 0.95, 1.0),
            ("細部が少なくなめらか", lambda f: f.texture <= 0.35, 1.5),
        ],
    }
_RULES = _rules()


def classify(features: Features, top_k: int = 4) -> list[Evidence]:
    """Score every material, return the strongest `top_k` with their reasons."""
    results: list[Evidence] = []
    for material, rules in _RULES.items():
        positive = sum(w for _, _, w in rules if w > 0)
        earned = 0.0
        reasons: list[str] = []
        contradictions: list[str] = []
        for reason, predicate, weight in rules:
            if predicate(features):          # type: ignore[operator]
                earned += weight
                (reasons if weight > 0 else contradictions).append(reason)
        # Normalising by the positive weight keeps scores comparable between
        # materials with different numbers of rules; contradictions pull the
        # score down and the clamp keeps it a probability-shaped number.
        score = max(0.0, earned / positive) if positive else 0.0
        results.append(
            Evidence(
                material=material,
                label=MATERIAL_LABELS_JA[material],
                score=score,
                reasons=reasons,
                contradictions=contradictions,
            )
        )

    results.sort(key=lambda e: (-e.score, e.material))
    chosen = results[:top_k]
    # A top score this low means the frame did not look like any of the tables;
    # saying "unknown" is more honest than showing a 3% winner.
    if not chosen or chosen[0].score < 0.34:
        return [
            Evidence(
                material="unknown",
                label=MATERIAL_LABELS_JA["unknown"],
                score=chosen[0].score if chosen else 0.0,
                reasons=["どの材質の特徴にも当てはまらなかった"],
            )
        ] + chosen
    return chosen


def candidate_keywords(evidence: list[Evidence], limit: int = 3) -> list[str]:
    """Flatten the winners' dictionary keywords, most likely first."""
    words: list[str] = []
    for entry in evidence[:2]:
        for word in MATERIAL_KEYWORDS.get(entry.material, []):
            if word not in words:
                words.append(word)
    return words[:limit]
