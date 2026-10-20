"""Local image analysis: decode a photo, describe the object in it with OpenCV.

Nothing leaves the process.  There is no cache directory, no temp file and no
network call anywhere below this line - `tests/test_privacy.py` asserts that
these modules never import a networking library, which is the weakest honest
form of that guarantee available without a sandbox.

Two design decisions were forced by measurement rather than taste:

1. **Measure the object, not the tablecloth.**  A photo is mostly background.
   The first version averaged colour and brightness over the whole frame, so a
   white table made every object look white and "glossy" (the background is
   bright and unsaturated, which is exactly the specular test).  Every colour
   and texture statistic below is computed over the foreground mask only, and
   the shape features come from the object's own bounding box.

2. **The mask polarity has to be decided, not assumed.**  Otsu splits the frame
   in two but does not say which half is the subject.  The border of a photo is
   background by construction, so its polarity decides.  Without this the
   largest "blob" is the whole frame and solidity/extent/foreground are all 1.0.

The features are deliberately low-level (colour, light, texture, shape).  We do
not have a labelled photo corpus, so we do not pretend to run a deep classifier:
the numbers here feed an explainable rule table in `classifier.py`, and the
results are shown to the user as *candidates to confirm*, never as a verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

#: Working resolution.  A phone photo is 4000px wide and carries nothing extra
#: for these statistics, while the resize costs ~2ms instead of ~200ms.
WORK_SIZE = 320

#: Below this share of the frame, the "foreground" is not a subject we can
#: describe; the statistics fall back to the whole frame and a note says so.
MIN_OBJECT_RATIO = 0.015


class ImageError(ValueError):
    """The bytes were not a decodable image."""


@dataclass
class Features:
    width: int
    height: int
    aspect: float              # object bounding box width / height
    value_mean: float          # HSV V over the object, 0..1
    saturation_mean: float     # HSV S over the object, 0..1
    hue_median: float          # OpenCV H over the object, 0..180
    colorfulness: float        # std of Lab a/b over the object
    texture: float             # Laplacian variance over the object / 1000
    edge_density: float        # Canny edge pixels / object area
    specular_ratio: float      # highlight: much brighter than the object's median
    dark_ratio: float          # V <= 0.18, over the object
    band_contrast: float       # std of row brightness inside the object box
    solidity: float            # object area / convex hull area
    extent: float              # object area / bounding box area
    foreground_ratio: float    # object area / image area
    flat_background: bool      # is the border a single colour? (studio shot)
    object_found: bool         # did segmentation find a subject at all
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Segment:
    mask: np.ndarray           # uint8 0/255, filled object
    area: float
    solidity: float
    extent: float
    ratio: float
    box: tuple[int, int, int, int]   # x, y, w, h


def _segment(gray: np.ndarray) -> _Segment:
    """Otsu-threshold the frame, decide polarity from the border, keep the blob."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if float((border > 0).mean()) > 0.5:
        mask = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty = np.zeros_like(gray)
    if not contours:
        return _Segment(empty, 0.0, 0.0, 0.0, 0.0, (0, 0, gray.shape[1], gray.shape[0]))

    biggest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(biggest))
    if area <= 0:
        return _Segment(empty, 0.0, 0.0, 0.0, 0.0, (0, 0, gray.shape[1], gray.shape[0]))

    filled = np.zeros_like(gray)
    cv2.drawContours(filled, [biggest], -1, 255, thickness=cv2.FILLED)

    hull_area = float(cv2.contourArea(cv2.convexHull(biggest))) or 1.0
    x, y, w, h = cv2.boundingRect(biggest)
    box_area = float(w * h) or 1.0
    return _Segment(
        mask=filled,
        area=area,
        solidity=area / hull_area,
        extent=area / box_area,
        ratio=area / float(gray.size),
        box=(x, y, w, h),
    )


def _masked_mean(values: np.ndarray, mask: np.ndarray, fallback: float = 0.0) -> float:
    selected = values[mask]
    return float(selected.mean()) if selected.size else fallback


def analyze(image_bytes: bytes) -> Features:
    """Decode bytes and compute the feature vector.  Raises ImageError."""
    if not image_bytes:
        raise ImageError("empty image payload")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageError("not a decodable image (expected png/jpeg/webp)")

    height, width = image.shape[:2]
    scale = WORK_SIZE / float(max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    height, width = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    val_f = value.astype(np.float32) / 255.0
    sat_f = saturation.astype(np.float32) / 255.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segment = _segment(gray)

    object_found = segment.ratio >= MIN_OBJECT_RATIO
    if object_found:
        mask = segment.mask > 0
        x, y, w, h = segment.box
        object_area = float(mask.sum())
    else:
        mask = np.ones(gray.shape, dtype=bool)
        x, y, w, h = 0, 0, width, height
        object_area = float(mask.sum())

    # --- colour, over the object only -----------------------------------
    value_mean = _masked_mean(val_f, mask, float(val_f.mean()))
    saturation_mean = _masked_mean(sat_f, mask, float(sat_f.mean()))
    hue_median = (
        float(np.median(hue[mask])) if mask.any() else float(np.median(hue))
    )
    # Gloss is a *small region much brighter than the object*, not "bright and
    # unsaturated" - that older test counted a sheet of white paper as 90%
    # gloss, because white paper is bright and unsaturated everywhere.  Anchoring
    # the threshold to the object's own median makes it measure highlight.
    median_value = float(np.median(val_f[mask])) if mask.any() else float(np.median(val_f))
    highlight_floor = min(0.97, median_value + 0.08)
    specular = float(((val_f >= highlight_floor) & (sat_f <= 0.20))[mask].mean())
    dark = float((val_f <= 0.18)[mask].mean())

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a_axis, b_axis = cv2.split(lab)
    a_f = a_axis.astype(np.float32)[mask]
    b_f = b_axis.astype(np.float32)[mask]
    colorfulness = float(a_f.std() + b_f.std()) if a_f.size else 0.0

    # --- texture and edges, over the object only ------------------------
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = float(laplacian[mask].var()) / 1000.0 if mask.any() else 0.0
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.count_nonzero(edges[mask])) / object_area if object_area else 0.0

    # --- banding: a label or stripe shows as row-brightness spread -------
    box_value = val_f[y : y + h, x : x + w]
    box_mask = mask[y : y + h, x : x + w]
    if box_value.size and box_mask.any():
        rows = []
        for row_index in range(box_value.shape[0]):
            row_mask = box_mask[row_index]
            if row_mask.any():
                rows.append(float(box_value[row_index][row_mask].mean()))
        band_contrast = float(np.std(rows)) if rows else 0.0
    else:
        band_contrast = 0.0

    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    flat_background = bool(np.std(border.astype(np.float32)) < 12.0)

    notes: list[str] = []
    if flat_background:
        notes.append("背景が単色（撮影は整っている）")
    else:
        notes.append("背景に模様がある（判定がぶれやすい）")
    if not object_found:
        notes.append("被写体を切り出せなかった（全面の統計で代用）")
    elif segment.ratio < 0.06:
        notes.append("被写体が小さい（枠いっぱいに写すと精度が上がる）")
    if texture > 0.5:
        notes.append("細かい模様が多い（布・段ボール・食品の可能性）")
    if specular > 0.04:
        notes.append("つやのある反射がある（プラ・金属・ガラス）")

    return Features(
        width=int(width),
        height=int(height),
        aspect=round(w / float(h), 4) if h else 0.0,
        value_mean=round(value_mean, 4),
        saturation_mean=round(saturation_mean, 4),
        hue_median=round(hue_median, 1),
        colorfulness=round(colorfulness, 2),
        texture=round(texture, 3),
        edge_density=round(edge_density, 4),
        specular_ratio=round(specular, 4),
        dark_ratio=round(dark, 4),
        band_contrast=round(band_contrast, 4),
        solidity=round(float(segment.solidity), 4),
        extent=round(float(segment.extent), 4),
        foreground_ratio=round(float(segment.ratio), 4),
        flat_background=flat_background,
        object_found=bool(object_found),
        notes=notes,
    )
