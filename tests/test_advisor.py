"""The advisor: text answers, photo shortlists, and what it refuses to claim."""

from __future__ import annotations

import pytest

from poimae.advisor import advise_photo, advise_text
from tests import helpers


def test_text_lookup_answers_from_the_dictionary(registry) -> None:
    answer = advise_text(registry, "sapporo", "ペットボトル")
    assert answer["source"] == "text"
    assert answer["lookup"]["found"]
    assert answer["lookup"]["item"] == "ペットボトル"
    assert answer["lookup"]["category"] == "びん・缶・ペットボトル"


def test_text_lookup_marks_uncertainty_instead_of_guessing(registry) -> None:
    answer = advise_text(registry, "sapporo", "よくわからないもの")
    assert answer["lookup"]["found"] is False or answer["lookup"]["confident"] is False
    assert answer["lookup"]["candidates"] == [] or answer["lookup"]["confident"] is False


def test_a_photo_is_never_a_confident_answer(registry) -> None:
    """The whole product claim: a photo gives candidates to confirm, not a verdict."""
    answer = advise_photo(registry, "sapporo", helpers.brown_box())
    assert answer["source"] == "photo"
    assert answer["lookup"]["confident"] is False
    assert answer["question"]
    assert answer["keywords"]
    assert answer["material_evidence"]


def test_photo_keywords_only_match_exact_or_prefix_entries(registry) -> None:
    """A category word must not drag in a word that merely contains it.

    "かん" used to return やかん (an electric kettle) and カーテン, which are
    neither cans nor related to the photo.
    """
    answer = advise_photo(registry, "hiroshima", helpers.grey_card())
    assert answer["shortlist"], "the fixture should reach at least one entry"
    for candidate in answer["shortlist"]:
        assert candidate["match"] in ("exact", "prefix"), candidate
        assert candidate["via_keyword"] in answer["keywords"]


def test_the_shortlist_is_bounded(registry) -> None:
    answer = advise_photo(registry, "sapporo", helpers.brown_box())
    assert len(answer["shortlist"]) <= 6
    assert len(answer["lookup"]["candidates"]) <= 6


def test_a_different_city_gives_a_different_bin_for_the_same_item(registry) -> None:
    """The same object maps to different bins per municipality - the whole point."""
    same = "電池"
    bins = {
        city: advise_text(registry, city, same)["lookup"]["category"]
        for city in ("hiroshima", "sapporo")
    }
    assert bins["hiroshima"] == "有害ごみ"
    assert bins["hiroshima"] != bins["sapporo"] or bins["sapporo"] is not None


def test_the_answer_carries_a_cross_city_view(registry) -> None:
    answer = advise_text(registry, "hiroshima", "陶磁器")
    other = answer.get("other_cities")
    assert other, "a user moving house needs to see the other city's rule"
    for entry in other:
        assert {"municipality", "category"} <= set(entry)


def test_the_dummy_city_is_labelled_as_a_sample(registry) -> None:
    """A sample rule set must never be able to pass itself off as a real city."""
    answer = advise_text(registry, "sample-dummy", "ペットボトル")
    assert answer["verified"] is False
    assert answer["data_kind"] == "dummy"
    assert "ダミー" in answer["municipality"]
    codes = {warning["code"] for warning in answer["warnings"]}
    assert "dummy" in codes, "a sample rule set must carry a visible warning"


@pytest.mark.parametrize("payload", [b"", b"definitely not an image"])
def test_bad_image_bytes_raise(registry, payload: bytes) -> None:
    from poimae.features import ImageError

    with pytest.raises(ImageError):
        advise_photo(registry, "sapporo", payload)


def test_unknown_municipality_is_a_typed_error(registry) -> None:
    from poimae.municipalities import MunicipalityError

    with pytest.raises(MunicipalityError):
        advise_text(registry, "atlantis", "ペットボトル")


def test_shortlist_never_lets_a_weaker_keyword_outrank_a_stronger_one(registry) -> None:
    """Regression: a photo of a bottle once answered "プラモデル".

    The classifier ranks its material guesses, so the first keyword is the one
    it believes most.  Ties were broken by item-name length, which let a
    "プラ" prefix hit (プラモデル, short) beat the "ペットボトル" prefix hit
    (long).  Ordering must follow the keyword's rank first.
    """
    image = (_samples() / "pet_bottle.png").read_bytes()
    answer = advise_photo(registry, "hiroshima", image)

    assert answer["keywords"][0] == "ペットボトル"
    assert answer["shortlist"][0]["via_keyword"] == "ペットボトル"
    assert answer["lookup"]["item"] is not None
    assert answer["lookup"]["item"].startswith("ペットボトル")


def test_shortlist_is_capped_and_ordered_within_a_keyword(registry) -> None:
    image = (_samples() / "ceramic.png").read_bytes()
    answer = advise_photo(registry, "hiroshima", image)

    assert len(answer["shortlist"]) <= 6
    first = answer["shortlist"][0]
    assert first["item"] == "陶磁器"
    assert first["match"] == "exact", "an exact entry must lead its keyword group"
    # Scores must be non-increasing inside one keyword's group.
    scores = [c["score"] for c in answer["shortlist"] if c["via_keyword"] == first["via_keyword"]]
    assert scores == sorted(scores, reverse=True)


def test_shortlist_entries_do_not_leak_internal_keys(registry) -> None:
    answer = advise_photo(registry, "sapporo", (_samples() / "metal_can.png").read_bytes())
    for entry in answer["shortlist"]:
        assert "_keyword_rank" not in entry


def test_a_photo_answer_is_never_confident(registry) -> None:
    """The design promise: a camera guess is a question, not a verdict."""
    for stem in ("pet_bottle", "metal_can", "ceramic"):
        answer = advise_photo(registry, "hiroshima", (_samples() / f"{stem}.png").read_bytes())
        assert answer["lookup"]["confident"] is False
        assert "?" in answer["question"] or "ください" in answer["question"]


def _samples():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "static" / "samples"


def test_a_word_the_dictionary_lacks_falls_back_to_a_synonym(registry) -> None:
    """広島市 prints 「電池 その他」, never 「乾電池」.

    Typing 乾電池 used to answer "見つかりませんでした" while a person stood
    there holding a battery.
    """
    answer = advise_text(registry, "hiroshima", "乾電池")

    assert answer["lookup"]["found"] is True
    assert answer["lookup"]["category"] == "有害ごみ"
    assert answer["retried_with"] == "電池"
    codes = [warning["code"] for warning in answer["warnings"]]
    assert codes[0] == "synonym", "the substitution must be the first thing shown"


def test_a_synonym_never_overrides_a_real_entry(registry) -> None:
    """The fallback only runs when the dictionary returned nothing."""
    answer = advise_text(registry, "sapporo", "乾電池")
    assert answer["lookup"]["item"].startswith("乾電池")
    assert "retried_with" not in answer


def test_an_unknown_word_still_reports_not_found(registry) -> None:
    answer = advise_text(registry, "hiroshima", "宇宙船")
    assert answer["lookup"]["found"] is False
    assert answer["lookup"]["confident"] is False
