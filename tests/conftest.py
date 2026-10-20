"""Fixtures are thin wrappers around `helpers` so nothing is duplicated."""

from __future__ import annotations

import pytest

from tests import helpers


@pytest.fixture(scope="session")
def registry():
    from poimae.municipalities import Registry

    return Registry()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from poimae.app import app

    return TestClient(app)


@pytest.fixture
def grey_card_png() -> bytes:
    return helpers.grey_card()


@pytest.fixture
def dark_green_bottle_png() -> bytes:
    return helpers.dark_green_bottle()


@pytest.fixture
def brown_box_png() -> bytes:
    return helpers.brown_box()
