"""Write the four sample photos the UI offers, so a visitor can try the app
without first hunting for a piece of rubbish.

These are the same procedural renders the evaluation corpus uses - drawings of
an object class, not photographs of a real object.  They exist to show the
pipeline end to end, and every one of them is labelled 合成画像 in the UI.  The
accuracy numbers in docs/EVALUATION.md come from a held-out split of the same
generator, which is why they overstate real-world performance: that caveat is
part of the deliverable, not a footnote.

    python -m tools.samples --out static/samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from tools import synth

# (file stem, generator class, seed, human label, the item name a user would
# type for it - used only for the on-screen caption)
SAMPLES = [
    ("pet_bottle", "pet_bottle", 4101, "ペットボトル"),
    ("metal_can", "metal_can", 4102, "アルミ缶"),
    ("cardboard", "cardboard", 4103, "段ボール"),
    ("ceramic", "ceramic", 4104, "茶わん（陶磁器）"),
]


def build(out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, cls, seed, label in SAMPLES:
        image = synth.render(cls, seed)
        path = out / f"{stem}.png"
        cv2.imwrite(str(path), image)
        written.append({"file": path.name, "label": label, "class": cls, "seed": seed})

    manifest = {
        "note": (
            "合成画像（プログラムで描いた見本）。実物の写真ではありません。"
            "アプリの動作確認用で、精度の主張はこの画像ではなく docs/EVALUATION.md を参照。"
        ),
        "generator": "tools/synth.py",
        "samples": written,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("static/samples"))
    args = parser.parse_args(argv)
    for entry in build(args.out):
        print(f"{args.out / entry['file']}  <- {entry['label']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
