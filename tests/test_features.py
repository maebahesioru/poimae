"""OpenCV feature extraction: does it measure what the rules assume it measures?"""

from __future__ import annotations

import pytest

from poimae.features import ImageError, analyze

FEATURE_NAMES = {
    "width", "height", "aspect", "value_mean", "saturation_mean", "hue_median",
    "colorfulness", "texture", "edge_density", "specular_ratio", "dark_ratio",
    "band_contrast", "solidity", "extent", "foreground_ratio", "flat_background",
    "object_found", "notes",
}


def test_all_documented_features_are_present(grey_card_png: bytes) -> None:
    features = analyze(grey_card_png)
    assert FEATURE_NAMES <= set(features.as_dict())


def test_values_are_in_their_declared_range(grey_card_png: bytes) -> None:
    f = analyze(grey_card_png)
    assert 0.0 <= f.value_mean <= 1.0
    assert 0.0 <= f.saturation_mean <= 1.0
    assert 0.0 <= f.edge_density <= 1.0
    assert 0.0 <= f.specular_ratio <= 1.0
    assert 0.0 <= f.dark_ratio <= 1.0
    assert 0.0 <= f.solidity <= 1.01
    assert 0.0 <= f.extent <= 1.01
    assert 0.0 <= f.foreground_ratio <= 1.0
    assert f.texture >= 0.0


def test_grey_card_reads_as_grey_and_flat(grey_card_png: bytes) -> None:
    """Colour is measured on the object, so a grey card must read as grey."""
    f = analyze(grey_card_png)
    assert f.object_found
    assert f.saturation_mean < 0.10        # the card itself is grey
    assert f.value_mean > 0.40             # mid grey, not black
    # `aspect` is the object's bounding box (100 wide x 180 tall), not the frame
    # (which is square) - this is the assertion that fails if someone reverts to
    # measuring the whole image.
    assert f.aspect == pytest.approx(100 / 180, abs=0.02)

def test_object_features_ignore_a_white_background(grey_card_png: bytes,
                                                  dark_green_bottle_png: bytes) -> None:
    """Both fixtures sit on a near-white background; it must not leak in."""
    for payload in (grey_card_png, dark_green_bottle_png):
        f = analyze(payload)
        assert f.foreground_ratio < 0.60
        # A white sweep is bright and unsaturated; if it leaked into the object
        # statistics the bottle would read as pale instead of dark green.
    assert analyze(dark_green_bottle_png).value_mean < 0.45


def test_dark_green_bottle_is_darker_and_more_saturated(dark_green_bottle_png: bytes) -> None:
    bottle = analyze(dark_green_bottle_png)
    card = analyze_from_fixture_name("grey_card")
    assert bottle.aspect < card.aspect            # taller than wide
    assert bottle.saturation_mean > card.saturation_mean
    assert bottle.value_mean < card.value_mean   # the bottle is the dark part


def test_noisy_brown_box_has_more_texture(brown_box_png: bytes) -> None:
    box = analyze(brown_box_png)
    card = analyze_from_fixture_name("grey_card")
    assert box.texture > card.texture
    assert box.saturation_mean > card.saturation_mean
    assert box.hue_median < 40                     # brown lives low in OpenCV hue


def test_foreground_is_the_object_not_the_frame(dark_green_bottle_png: bytes) -> None:
    """The mask polarity bug showed up as foreground_ratio ~= 1.0 on everything."""
    f = analyze(dark_green_bottle_png)
    assert 0.05 < f.foreground_ratio < 0.60, f.foreground_ratio
    assert f.extent < 0.99


def test_oversized_input_is_scaled_not_rejected() -> None:
    """A 12 MP phone photo must not blow up the working set."""
    from tests.helpers import make_large_png

    big = make_large_png(1600, 1200)
    features = analyze(big)
    assert max(features.width, features.height) <= 1024


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"0" * 30])
def test_bad_bytes_raise_a_typed_error(payload: bytes) -> None:
    with pytest.raises(ImageError):
        analyze(payload)


def analyze_from_fixture_name(name: str):
    """Rebuild one of the helper images and analyse it."""
    from tests import helpers

    return analyze(getattr(helpers, name)())
