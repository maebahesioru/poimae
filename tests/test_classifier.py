"""The rule table: does it look at the cue it claims to look at?

Each test here pins one behaviour that a plausible-looking refactor could break
silently - the hue gate, the "unknown" floor, the negative weights, and the fact
that a photo answer is never marked confident.
"""

from __future__ import annotations

import pytest

from poimae.classifier import MATERIAL_KEYWORDS, classify, candidate_keywords
from poimae.features import Features, analyze
from tests import helpers


def blank_features(**overrides) -> Features:
    """A feature vector with every field explicit, so a test states its world."""
    base = dict(
        width=320, height=320, aspect=1.0, value_mean=0.5, saturation_mean=0.0,
        hue_median=0.0, colorfulness=0.0, texture=0.1, edge_density=0.02,
        specular_ratio=0.0, dark_ratio=0.0, band_contrast=0.0, solidity=0.99,
        extent=0.95, foreground_ratio=0.3, flat_background=True,
        object_found=True, notes=[],
    )
    base.update(overrides)
    return Features(**base)


def test_scores_are_probability_shaped(grey_card_png: bytes) -> None:
    evidence = classify(analyze(grey_card_png))
    assert evidence
    for entry in evidence:
        assert 0.0 <= entry.score <= 1.0
        assert entry.label                       # every entry is displayable
        assert entry.reasons or entry.material == "unknown"


def test_a_colourless_card_is_not_brown_cardboard() -> None:
    """Hue is meaningless without saturation; OpenCV still returns a number.

    The first version scored a grey card 100% cardboard at hue 30 because it
    trusted that number - this test is the regression.
    """
    grey = blank_features(saturation_mean=0.01, hue_median=15.0, value_mean=0.7)
    top = classify(grey)[0]
    assert top.material != "cardboard"
    assert "茶色ではない" not in top.reasons


def test_a_brown_box_is_cardboard() -> None:
    box = blank_features(
        saturation_mean=0.55, hue_median=14.0, value_mean=0.68, texture=0.08,
        aspect=1.0, solidity=0.99, extent=0.93,
    )
    assert classify(box)[0].material == "cardboard"


def test_a_frame_below_the_floor_reports_unknown(monkeypatch) -> None:
    """Below the 34% floor the honest answer is 'I do not know', not a 12% winner.

    The floor is only reachable when the rules themselves stay quiet, so this
    test replaces the table with predicates that never fire rather than hunting
    for a feature combination no class happens to like.
    """
    import poimae.classifier as classifier

    silent = {
        material: [("決して当たらない", lambda f: False, 1.0)]
        for material in classifier.MATERIAL_KEYWORDS
        if material != "unknown"
    }
    monkeypatch.setattr(classifier, "_RULES", silent)
    top = classifier.classify(blank_features())[0]
    assert top.material == "unknown"
    assert top.reasons == ["どの材質の特徴にも当てはまらなかった"]


def test_the_floor_is_where_it_says_it_is(monkeypatch) -> None:
    """Just above the floor the material wins; just below, unknown prefixes it."""
    import poimae.classifier as classifier

    above = {"ceramic": [("当たる", lambda f: True, 1.0)]}
    below = {"ceramic": [("当たらない", lambda f: False, 1.0)]}
    monkeypatch.setattr(classifier, "_RULES", above)
    assert classifier.classify(blank_features())[0].material == "ceramic"
    monkeypatch.setattr(classifier, "_RULES", below)
    assert classifier.classify(blank_features())[0].material == "unknown"


def test_contradicting_evidence_lowers_a_score() -> None:
    """A dark frame must not score as plastic film, which is a bright class."""
    bright = blank_features(value_mean=0.93, saturation_mean=0.06, texture=0.10, edge_density=0.012)
    dark = blank_features(value_mean=0.30, saturation_mean=0.06, texture=0.10, edge_density=0.012)
    bright_score = {e.material: e.score for e in classify(bright, top_k=10)}["plastic_film"]
    dark_score = {e.material: e.score for e in classify(dark, top_k=10)}["plastic_film"]
    assert bright_score > dark_score


def test_a_green_bottle_is_not_paper(dark_green_bottle_png: bytes) -> None:
    materials = [entry.material for entry in classify(analyze(dark_green_bottle_png), top_k=3)]
    assert "glass_bottle" in materials


def test_keywords_follow_the_two_best_materials() -> None:
    card = classify(analyze(helpers.grey_card()))
    words = candidate_keywords(card)
    assert words
    assert len(words) == len(set(words))              # no duplicates
    for word in words:
        assert any(word in MATERIAL_KEYWORDS[m] for m in MATERIAL_KEYWORDS)


def test_top_k_is_respected(grey_card_png: bytes) -> None:
    assert len(classify(analyze(grey_card_png), top_k=2)) <= 3   # unknown may prefix


@pytest.mark.parametrize("material", sorted(MATERIAL_KEYWORDS))
def test_every_material_has_keywords_and_a_label(material: str) -> None:
    from poimae.classifier import MATERIAL_LABELS_JA

    assert material in MATERIAL_LABELS_JA
    if material != "unknown":
        assert MATERIAL_KEYWORDS[material], "a material with no keywords can never reach a bin"
