"""The HTTP surface: what a browser can do, and what it must be refused.

These go through FastAPI's TestClient, so they exercise the real routes,
validation and error handlers - not an internal function that happens to be
named the same.
"""

from __future__ import annotations

import pytest

from tests import helpers

PNG = "image/png"


def test_the_page_serves_without_any_external_asset(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "<html" in body.lower()
    # No CDN, no webfont, no analytics: the app must work offline, because a
    # privacy claim that breaks without wifi is not a privacy claim.
    for marker in ("http://", "https://", "cdn.", "googleapis", "analytics"):
        assert marker not in body, marker


def test_health_reports_the_loaded_dictionaries(client) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert len(payload["municipalities"]) >= 3
    # The claim the whole design rests on, asserted where a user can see it.
    assert "no outbound" in payload["network"]


def test_municipality_list_marks_the_sample(client) -> None:
    payload = client.get("/api/municipalities").json()
    ids = {entry["id"]: entry for entry in payload["municipalities"]}
    assert "sample-dummy" in ids
    assert ids["sample-dummy"]["data_kind"] == "dummy"


def test_lookup_returns_a_bin(client) -> None:
    payload = client.get("/api/lookup", params={"municipality": "sapporo", "q": "ペットボトル"}).json()
    assert payload["lookup"]["category"] == "びん・缶・ペットボトル"
    assert payload["question"]


def test_lookup_rejects_an_empty_query(client) -> None:
    assert client.get("/api/lookup", params={"municipality": "sapporo", "q": ""}).status_code == 422


def test_unknown_municipality_is_a_404(client) -> None:
    response = client.get("/api/lookup", params={"municipality": "atlantis", "q": "ペットボトル"})
    assert response.status_code == 404


def test_photo_endpoint_answers_with_candidates(client) -> None:
    response = client.post(
        "/api/photo",
        params={"municipality": "hiroshima"},
        files={"image": ("box.png", helpers.brown_box(), PNG)},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "photo"
    assert payload["lookup"]["confident"] is False
    assert payload["material_evidence"]


def test_photo_endpoint_refuses_a_non_image(client) -> None:
    response = client.post(
        "/api/photo",
        params={"municipality": "hiroshima"},
        files={"image": ("x.png", b"this is not an image", PNG)},
    )
    assert response.status_code == 400


def test_photo_endpoint_refuses_a_wrong_content_type(client) -> None:
    response = client.post(
        "/api/photo",
        params={"municipality": "hiroshima"},
        files={"image": ("x.svg", b"<svg/>", "image/svg+xml")},
    )
    assert response.status_code == 415


def test_photo_endpoint_refuses_an_oversized_upload(client) -> None:
    """A 20 MB payload is refused before OpenCV ever sees it."""
    from poimae.app import MAX_UPLOAD_BYTES

    payload = b"\x89PNG\r\n\x1a\n" + b"0" * (MAX_UPLOAD_BYTES + 1024)
    response = client.post(
        "/api/photo",
        params={"municipality": "hiroshima"},
        files={"image": ("big.png", payload, PNG)},
    )
    assert response.status_code == 413


def test_week_endpoint_plans_the_days(client) -> None:
    response = client.post(
        "/api/week",
        json={
            "municipality": "sapporo",
            "today": "2026-10-20",
            "schedule": {"燃やせるごみ": [2, 5], "びん・缶・ペットボトル": [3]},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["today"] == "2026-10-20"          # a Tuesday
    assert payload["next_by_category"]["燃やせるごみ"]["in_days"] == 0   # today is one of its days
    assert payload["next_by_category"]["びん・缶・ペットボトル"]["in_days"] == 1
    assert [day["date"] for day in payload["due"]] == ["2026-10-20", "2026-10-21", "2026-10-23"]
    assert payload["headline"].startswith("今日は")


def test_week_endpoint_rejects_a_category_the_city_does_not_have(client) -> None:
    response = client.post(
        "/api/week",
        json={"municipality": "sapporo", "today": "2026-10-20", "schedule": {"燃えないごみ": [1]}},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("value", ["2026-13-01", "yesterday", "2026-02-30"])
def test_week_endpoint_rejects_a_bad_date(client, value: str) -> None:
    response = client.post(
        "/api/week",
        json={"municipality": "sapporo", "today": value, "schedule": {}},
    )
    assert response.status_code == 400


def test_ambiguity_report_lists_items_that_differ_between_cities(client) -> None:
    payload = client.get("/api/ambiguity").json()
    assert payload["cities"]
    assert payload["pairs"], "items that land in different bins must be reported"
    entry = payload["pairs"][0]
    assert isinstance(entry, dict)
