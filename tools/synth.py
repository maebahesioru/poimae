"""Generate a labelled synthetic photo corpus, so accuracy can be measured.

Honest framing: these are **drawings**, not photographs.  A score on them says
the rule table reacts to the cues it was written for; it says nothing about a
real kitchen table.  The real-photo set in `eval/real/` (fetched separately) is
the honest counterweight, and `docs/EVALUATION.md` reports both numbers side by
side rather than the flattering one.

Each class is drawn with randomised colour jitter, position, size, background,
shadow, blur and noise, and half the images get distractor clutter, because a
photo of a can on a clean white sweep is the easy case and not the normal one.

Usage:  python -m tools.synth --out eval/synthetic --per-class 12
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = [
    "pet_bottle",
    "metal_can",
    "glass_bottle",
    "paper",
    "cardboard",
    "plastic_film",
    "food_waste",
    "battery",
    "textile",
    "ceramic",
]

FRAME = 420


# --------------------------------------------------------------------------
# background / finishing
# --------------------------------------------------------------------------
def _background(rng: random.Random) -> np.ndarray:
    kind = rng.choice(["white", "table", "tile", "cloth"])

    if kind == "white":                      # studio sweep
        base = rng.randint(228, 250)
        image = np.full((FRAME, FRAME, 3), base, dtype=np.uint8)
        return image

    if kind == "table":                      # warm wood with grain
        base = np.array([70, 105, 150], dtype=np.float32)   # BGR brown
        base += np.array([rng.randint(-18, 18) for _ in range(3)], dtype=np.float32)
        image = np.tile(base, (FRAME, FRAME, 1))
        for y in range(0, FRAME, rng.randint(14, 26)):
            shade = rng.randint(-14, 10)
            image[y : y + rng.randint(1, 4), :] += shade
        return np.clip(image, 0, 255).astype(np.uint8)

    if kind == "tile":                       # cool grey tiles
        base = rng.randint(150, 200)
        image = np.full((FRAME, FRAME, 3), base, dtype=np.uint8)
        step = rng.choice([60, 84, 105])
        cv2.line(image, (0, step), (FRAME, step), (base - 30, base - 30, base - 30), 2)
        cv2.line(image, (step, 0), (step, FRAME), (base - 30, base - 30, base - 30), 2)
        return image

    base = np.array([rng.randint(90, 150) for _ in range(3)], dtype=np.uint8)  # cloth
    image = np.tile(base, (FRAME, FRAME, 1))
    for y in range(0, FRAME, 3):
        image[y, :] = np.clip(image[y, :].astype(int) + rng.randint(-10, 10), 0, 255)
    return np.clip(image, 0, 255).astype(np.uint8)


def _shadow(image: np.ndarray, cx: int, cy: int, rx: int, ry: int) -> None:
    layer = np.zeros_like(image, dtype=np.float32)
    cv2.ellipse(layer, (cx, cy), (max(rx, 4), max(ry, 3)), 0, 0, 360, (40, 40, 40), -1)
    layer = cv2.GaussianBlur(layer, (31, 31), 0)
    darkened = np.clip(image.astype(np.float32) - layer, 0, 255).astype(np.uint8)
    image[:] = darkened


def _finish(image: np.ndarray, rng: random.Random) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0, 255)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(1.5, 7.0), image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.4, 1.1))
    return image


def _clutter(image: np.ndarray, rng: random.Random) -> None:
    """Distractors: a photo of one object on an empty table is the easy case."""
    for _ in range(rng.randint(1, 3)):
        w = rng.randint(24, 70)
        h = rng.randint(24, 70)
        x = rng.choice([rng.randint(0, 90), rng.randint(FRAME - 110, FRAME - w - 1)])
        y = rng.choice([rng.randint(0, 90), rng.randint(FRAME - 110, FRAME - h - 1)])
        colour = [rng.randint(0, 255) for _ in range(3)]
        shape = rng.choice(["rect", "circle", "poly"])
        if shape == "rect":
            cv2.rectangle(image, (x, y), (x + w, y + h), colour, -1)
        elif shape == "circle":
            cv2.circle(image, (x + w // 2, y + h // 2), min(w, h) // 2, colour, -1)
        else:
            points = np.array(
                [[x + rng.randint(0, w), y + rng.randint(0, h)] for _ in range(3)], dtype=np.int32
            )
            cv2.fillPoly(image, [points], colour)


# --------------------------------------------------------------------------
# per-class drawing
# --------------------------------------------------------------------------
def _pet_bottle(image: np.ndarray, rng: random.Random, cx: int, cy: int, h: int) -> None:
    w = int(h * rng.uniform(0.28, 0.44))
    tint = [rng.randint(215, 250), rng.randint(205, 245), rng.randint(170, 215)]
    body_top = cy - h // 2 + h // 4
    cv2.rectangle(image, (cx - w // 2, body_top), (cx + w // 2, cy + h // 2), tint, -1)
    cv2.ellipse(image, (cx, cy + h // 2), (w // 2, w // 6), 0, 0, 180, tint, -1)
    cv2.rectangle(image, (cx - w // 6, cy - h // 2), (cx + w // 6, body_top), tint, -1)
    cv2.rectangle(image, (cx - w // 7, cy - h // 2 - h // 16), (cx + w // 7, cy - h // 2), 
                  [rng.randint(30, 90)] * 3, -1)
    label_h = int(h * 0.34)
    label_y = cy - h // 10
    cv2.rectangle(image, (cx - w // 2, label_y), (cx + w // 2, label_y + label_h),
                  [rng.randint(225, 255)] * 3, -1)
    for i in range(rng.randint(2, 4)):
        y = label_y + 6 + i * (label_h // 4)
        cv2.line(image, (cx - w // 3, y), (cx + w // 4, y), (60, 70, 80), rng.randint(1, 2))
    cv2.line(image, (cx - w // 3, cy - h // 2 + 10), (cx - w // 3, cy + h // 2 - 10),
             (250, 250, 250), 2)


def _metal_can(image: np.ndarray, rng: random.Random, cx: int, cy: int, h: int) -> None:
    w = int(h * rng.uniform(0.55, 0.80))
    body = rng.randint(150, 215)
    cv2.rectangle(image, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), 
                  (body, body, body + rng.randint(-6, 6)), -1)
    cv2.ellipse(image, (cx, cy - h // 2), (w // 2, w // 7), 0, 0, 360, (body + 25,) * 3, -1)
    cv2.ellipse(image, (cx, cy + h // 2), (w // 2, w // 7), 0, 0, 360, (body - 30,) * 3, -1)
    band = rng.randint(0, h // 3)
    colour = [rng.randint(0, 230) for _ in range(3)]
    cv2.rectangle(image, (cx - w // 2, cy - band), (cx + w // 2, cy - band + h // 5), colour, -1)
    cv2.line(image, (cx - w // 5, cy - h // 2 + 6), (cx - w // 5, cy + h // 2 - 6),
             (252, 252, 252), rng.randint(2, 4))


def _glass_bottle(image: np.ndarray, rng: random.Random, cx: int, cy: int, h: int) -> None:
    w = int(h * rng.uniform(0.26, 0.42))
    tint = rng.choice(
        [[36, 84, 40], [40, 70, 96], [30, 46, 96], [58, 52, 30], [34, 96, 52]]
    )
    cv2.rectangle(image, (cx - w // 2, cy - h // 2 + h // 5), (cx + w // 2, cy + h // 2), tint, -1)
    cv2.rectangle(image, (cx - w // 7, cy - h // 2), (cx + w // 7, cy - h // 2 + h // 5), tint, -1)
    cv2.ellipse(image, (cx, cy - h // 2), (w // 7, w // 14), 0, 0, 360, (int(tint[0] * 1.4),) * 3, -1)
    for i in range(2):
        x = cx - w // 4 + i * (w // 2)
        cv2.line(image, (x, cy - h // 2 + h // 6), (x, cy + h // 2 - h // 12), (240, 240, 235), 
                 rng.randint(2, 5))


def _paper(image: np.ndarray, rng: random.Random, cx: int, cy: int, h: int) -> None:
    w = int(h * rng.uniform(0.68, 1.05))
    shift = rng.randint(6, 18)
    points = np.array(
        [
            [cx - w // 2 + shift, cy - h // 2],
            [cx + w // 2, cy - h // 2 + shift // 2],
            [cx + w // 2 - shift, cy + h // 2],
            [cx - w // 2, cy + h // 2 - shift // 2],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [points], [rng.randint(225, 252) for _ in range(3)])
    for i in range(rng.randint(4, 9)):
        y = cy - h // 2 + 14 + i * (h // 12)
        cv2.line(image, (cx - w // 3, y), (cx + w // 4 - rng.randint(0, 30), y), (70, 70, 75), 1)


def _cardboard(image: np.ndarray, rng: random.Random, cx: int, cy: int, h: int) -> None:
    w = int(h * rng.uniform(0.75, 1.25))
    brown = [rng.randint(55, 95), rng.randint(105, 135), rng.randint(150, 185)]
    cv2.rectangle(image, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), brown, -1)
    wobble = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 9, (h, w, 3))
    target = image[cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2]
    if target.shape[:2] == wobble.shape[:2]:
        image[cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2] = np.clip(
            target + wobble, 0, 255
        ).astype(np.uint8)
    cv2.line(image, (cx, cy - h // 2), (cx, cy + h // 2), (brown[0] - 20, brown[1] - 25, brown[2] - 30), 3)
    tape = rng.choice([True, False])
    if tape:
        cv2.line(image, (cx - w // 2, cy - h // 4), (cx + w // 2, cy - h // 4), (200, 195, 175), 7)


def _plastic_film(image: np.ndarray, rng: random.Random, cx: int, cy: int, h: int) -> None:
    w = int(h * rng.uniform(0.7, 1.2))
    base = [rng.randint(215, 250) for _ in range(3)]
    if rng.random() < 0.5:   # a tray
        cv2.rectangle(image, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), base, -1)
        cv2.rectangle(image, (cx - w // 2 + 12, cy - h // 2 + 12),
                      (cx + w // 2 - 12, cy + h // 2 - 12), (base[0] - 25, base[1] - 25, base[2] - 25), 3)
        cv2.line(image, (cx - w // 2 + 6, cy - h // 2 + 6), (cx + w // 2 - 6, cy - h // 2 + 6),
                 (255, 255, 255), 3)
    else:                    # crumpled film
        points = np.array(
            [
                [cx - w // 2, cy - h // 3],
                [cx - w // 4, cy - h // 2],
                [cx + w // 3, cy - h // 4],
                [cx + w // 2, cy + h // 5],
                [cx + w // 6, cy + h // 2],
                [cx - w // 3, cy + h // 3],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [points], base)
        for _ in range(rng.randint(2, 4)):
            x1 = cx + rng.randint(-w // 2, w // 2)
            y1 = cy + rng.randint(-h // 2, h // 2)
            cv2.line(image, (x1, y1), (x1 + rng.randint(-40, 40), y1 + rng.randint(-30, 30)),
                     (255, 255, 255), rng.randint(1, 3))


def _food_waste(image: np.ndarray, rng: random.Random, cx: int, cy: int, h: int) -> None:
    rng_np = np.random.default_rng(rng.randint(0, 2**31))
    points = []
    for i in range(14):
        angle = 2 * np.pi * i / 14.0
        radius = (h / 2.4) * float(rng.uniform(0.55, 1.15))
        points.append([int(cx + radius * np.cos(angle)), int(cy + radius * np.sin(angle))])
    cv2.fillPoly(image, [np.array(points, dtype=np.int32)],
                 [rng.randint(20, 120) for _ in range(3)])
    for _ in range(rng.randint(6, 14)):       # bits of peel, core, leftovers
        offset_x = cx + rng.randint(-h // 3, h // 3)
        offset_y = cy + rng.randint(-h // 3, h // 3)
        size = rng.randint(8, max(9, h // 5))
        colour = [rng.randint(0, 255) for _ in range(3)]
        cv2.circle(image, (offset_x, offset_y), size, colour, -1)
    image[:] = np.clip(image.astype(np.float32) + rng_np.normal(0, 3.0, image.shape), 0, 255).astype(np.uint8)


def _battery(image: np.ndarray, rng: random.Random, cx: int, cy: int, h: int) -> None:
    short = int(h * rng.uniform(0.24, 0.36))
    cv2.rectangle(image, (cx - short // 2, cy - h // 2), (cx + short // 2, cy + h // 2),
                  (rng.randint(25, 55),) * 3, -1)
    cv2.rectangle(image, (cx - short // 2, cy - h // 2), (cx + short // 2, cy - h // 2 + h // 5),
                  [rng.randint(0, 120) for _ in range(3)], -1)
    cv2.rectangle(image, (cx - short // 5, cy + h // 2), (cx + short // 5, cy + h // 2 + h // 18),
                  (190, 195, 200), -1)


def _textile(image: np.ndarray, rng: random.Random, cx: int, cy: int, h: int) -> None:
    w = int(h * rng.uniform(0.8, 1.3))
    colour = [rng.randint(40, 190) for _ in range(3)]
    points = np.array(
        [
            [cx - w // 2, cy - h // 2 + rng.randint(0, 15)],
            [cx + w // 2, cy - h // 2],
            [cx + w // 2 - rng.randint(0, 15), cy + h // 2],
            [cx - w // 2, cy + h // 2 - rng.randint(0, 12)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [points], colour)
    weave = tuple(min(255, c + 26) for c in colour)
    for y in range(cy - h // 2, cy + h // 2, rng.randint(3, 5)):
        cv2.line(image, (cx - w // 2, y), (cx + w // 2, y), weave, 1)
    for _ in range(rng.randint(1, 3)):        # folds
        x = cx + rng.randint(-w // 3, w // 3)
        cv2.line(image, (x, cy - h // 2), (x + rng.randint(-20, 20), cy + h // 2),
                 tuple(max(0, c - 30) for c in colour), 2)


def _ceramic(image: np.ndarray, rng: random.Random, cx: int, cy: int, h: int) -> None:
    white = [rng.randint(220, 252) for _ in range(3)]
    if rng.random() < 0.5:   # a bowl
        cv2.ellipse(image, (cx, cy + h // 6), (h // 2, h // 4), 0, 0, 180, white, -1)
        cv2.ellipse(image, (cx, cy - h // 8), (h // 2, h // 6), 0, 0, 360, white, -1)
        cv2.ellipse(image, (cx, cy - h // 8), (h // 3, h // 9), 0, 0, 360,
                    tuple(max(0, c - 40) for c in white), -1)
    else:                    # a plate
        w = int(h * rng.uniform(0.9, 1.15))
        cv2.ellipse(image, (cx, cy), (w // 2, h // 3), 0, 0, 360, white, -1)
        cv2.ellipse(image, (cx, cy), (w // 3, h // 5), 0, 0, 360,
                    tuple(max(0, c - 25) for c in white), 2)
    cv2.ellipse(image, (cx - h // 6, cy - h // 6), (h // 8, h // 14), 0, 0, 360, (255, 255, 255), -1)


_DRAW = {
    "pet_bottle": _pet_bottle,
    "metal_can": _metal_can,
    "glass_bottle": _glass_bottle,
    "paper": _paper,
    "cardboard": _cardboard,
    "plastic_film": _plastic_film,
    "food_waste": _food_waste,
    "battery": _battery,
    "textile": _textile,
    "ceramic": _ceramic,
}


def render(cls: str, seed: int, clutter: bool = False) -> np.ndarray:
    rng = random.Random(seed)
    image = _background(rng)
    height = int(FRAME * rng.uniform(0.34, 0.62))
    cx = FRAME // 2 + rng.randint(-FRAME // 12, FRAME // 12)
    cy = FRAME // 2 + rng.randint(-FRAME // 12, FRAME // 12)
    _shadow(image, cx, cy + height // 2, height // 3, height // 10)
    _DRAW[cls](image, rng, cx, cy, height)
    if clutter:
        _clutter(image, rng)
    return _finish(image, rng)


def build(out: Path, per_class: int, seed0: int = 20261020) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    labels: list[dict] = []
    for index, cls in enumerate(CLASSES):
        folder = out / cls
        folder.mkdir(exist_ok=True)
        for sample in range(per_class):
            seed = seed0 + index * 1000 + sample
            clutter = sample % 2 == 1        # half the set carries distractors
            image = render(cls, seed, clutter=clutter)
            path = folder / f"{cls}-{sample:02d}.png"
            cv2.imwrite(str(path), image)
            labels.append(
                {
                    "path": str(path.relative_to(out)),
                    "label": cls,
                    "difficulty": "cluttered" if clutter else "clean",
                    # The first half of each class is the development split
                    # (thresholds were set looking at it); the second half is
                    # the test split and is only ever read by the evaluator.
                    "split": "dev" if sample < per_class // 2 else "test",
                    "seed": seed,
                }
            )
    (out / "labels.json").write_text(
        json.dumps({"generated_by": "tools/synth.py", "items": labels}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="eval/synthetic")
    parser.add_argument("--per-class", type=int, default=12)
    args = parser.parse_args()
    labels = build(Path(args.out), args.per_class)
    print(f"wrote {len(labels)} images to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
