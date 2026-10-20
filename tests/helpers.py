"""Deterministic image builders.

These are plain functions, not fixtures, so both the tests and the evaluation
tool can build the exact same image.  Nothing binary is committed to the repo
for the synthetic set: the generator *is* the fixture.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def png_bytes(array: np.ndarray) -> bytes:
    """Write RGB bytes.

    Callers build arrays in RGB; OpenCV decodes to BGR.  Use `bgr_png_bytes`
    when the colour values themselves matter to the test.
    """
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def bgr_png_bytes(bgr: np.ndarray) -> bytes:
    """Write an array whose channel order is already BGR, as OpenCV reads it."""
    return png_bytes(bgr[:, :, ::-1].copy())


def solid(width: int, height: int, bgr: tuple[int, int, int]) -> np.ndarray:
    """A canvas in BGR order (what OpenCV returns and what we write)."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :] = bgr
    return array


def grey_card() -> bytes:
    """A flat mid-grey card on a light background."""
    array = solid(320, 320, (238, 238, 236))
    array[70:250, 110:210] = (150, 150, 152)
    return bgr_png_bytes(array)


def dark_green_bottle() -> bytes:
    """A dark green tall shape on a white background - a green glass bottle."""
    array = solid(320, 480, (245, 245, 243))
    array[40:440, 120:200] = (26, 74, 40)
    array[40:90, 140:180] = (24, 70, 38)
    return bgr_png_bytes(array)


def brown_box() -> bytes:
    """A brown, noisy rectangle - cardboard."""
    rng = np.random.default_rng(20261020)
    array = solid(400, 320, (240, 240, 238))
    patch = np.zeros((220, 280, 3), dtype=np.uint8)
    # BGR order: a warm brown, i.e. high red, low blue.
    patch[:, :, 0] = 74
    patch[:, :, 1] = 122
    patch[:, :, 2] = 172
    noise = rng.normal(0, 16, patch.shape).astype(np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    array[50:270, 60:340] = patch
    return bgr_png_bytes(array)


def make_large_png(width: int, height: int) -> bytes:
    """A structured (not noise) image, so a resize still means something."""
    yy, xx = np.mgrid[0:height, 0:width]
    channels = [
        ((xx * 255) // max(width - 1, 1)).astype(np.uint8),
        ((yy * 255) // max(height - 1, 1)).astype(np.uint8),
        np.full((height, width), 120, dtype=np.uint8),
    ]
    array = np.stack(channels, axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()
