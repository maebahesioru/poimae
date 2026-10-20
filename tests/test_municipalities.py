"""Loading rule files, and refusing to answer when the answer is a guess."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poimae.municipalities import MunicipalityError, Registry

GENERIC_ALLOWED = {
    "burnable", "non_burnable", "plastic_packaging", "resource",
    "hazardous", "oversized", "not_collected", "other",
}


def test_three_rule_files_load(registry: Registry) -> None:
    assert set(registry.ids()) == {"sapporo", "hiroshima", "sample-dummy"}


def test_official_files_declare_provenance(registry: Registry) -> None:
    for municipality in registry.all():
        if municipality.data_kind != "official":
            continue
        assert municipality.verified, municipality.id
        assert municipality.source["url"].startswith("http"), municipality.id
        assert municipality.source["raw_files"], municipality.id


def test_dummy_file_is_labelled_dummy(registry: Registry) -> None:
    dummy = registry.get("sample-dummy")
    assert dummy.data_kind == "dummy"
    # The file declares `verified: false`; that declared False must survive the
    # load (only a *missing* flag becomes None).
    assert dummy.verified is False
    assert "実在" in dummy.source["disclaimer"]


def test_every_category_maps_to_a_known_generic_bin(registry: Registry) -> None:
    """None is allowed and means "this bin has no honest counterpart elsewhere";
    anything else must be one of the vocabulary words the app understands."""
    for municipality in registry.all():
        for category in municipality.categories:
            assert category.generic is None or category.generic in GENERIC_ALLOWED, (
                municipality.id,
                category.label,
                category.generic,
            )


def test_exact_hit_is_confident(registry: Registry) -> None:
    answer = registry.get("sapporo").lookup("傘")
    assert answer["found"] and answer["confident"]
    assert answer["category"] == "燃やせないごみ"
    assert answer["match"] == "exact"


def test_short_query_is_never_confident(registry: Registry) -> None:
    """"かん" is a substring of やかん and アメリカンレーキ; guessing would be a lie."""
    answer = registry.get("sapporo").lookup("かん")
    assert answer["found"] is True
    assert answer["confident"] is False
    assert answer["candidates"]


def test_unknown_item_is_not_found_rather_than_wrong(registry: Registry) -> None:
    answer = registry.get("sapporo").lookup("存在しない品目xyz")
    assert answer["found"] is False
    assert answer["category"] is None


def test_kana_only_query_never_answers_confidently_wrong(registry: Registry) -> None:
    """The dictionary stores 乾電池 in kanji; a kana query has no reading index.

    What matters is that the app does not answer *confidently* with a different
    item ("やかん" used to win this query on raw LCS similarity).
    """
    answer = registry.get("sapporo").lookup("かんでんち")
    if answer["confident"]:
        assert answer["item"] == "乾電池（筒型）", answer["item"]


def test_kanji_query_is_exact(registry: Registry) -> None:
    answer = registry.get("sapporo").lookup("乾電池")
    assert answer["confident"]
    assert answer["item"].startswith("乾電池")


def test_registry_rejects_unknown_id(registry: Registry) -> None:
    with pytest.raises(MunicipalityError):
        registry.get("nowhere")


def test_malformed_file_is_rejected(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"id": "broken"}), encoding="utf-8")
    with pytest.raises(MunicipalityError):
        Registry(data_dir=tmp_path)


def test_bundled_samples_are_documented(registry) -> None:
    """The sample photos shipped for the UI must say what they are."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "static" / "samples"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "合成画像" in manifest["note"]
    for entry in manifest["samples"]:
        assert (root / entry["file"]).is_file(), entry["file"]
