"""Measure the classifier and the pipeline on labelled images.

Two numbers, kept separate on purpose:

* **material top-1 accuracy** - did the rule table name the right material?
* **bin accuracy** - did the whole pipeline (photo -> material words ->
  municipality dictionary) land on the bin the dictionary itself gives for that
  material?  The expected bin is read *from the dictionary*, so this second
  number measures plumbing (did the keywords reach an entry at all, and did the
  ranking pick it), not whether a real city would agree.

A "always answer the most common class" baseline is printed next to both, since
a single accuracy number with no reference point is a decoration.

Usage:
    python -m tools.evaluate --corpus eval/synthetic
    python -m tools.evaluate --corpus eval/synthetic --json eval/report.json \\
        --markdown eval/CONFUSION.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from poimae.advisor import advise_photo
from poimae.classifier import MATERIAL_LABELS_JA, classify
from poimae.features import analyze
from poimae.municipalities import Registry

#: The word a person would look up for each material class.  Order matters: the
#: first word the dictionary answers confidently is the expected bin.
CANONICAL: dict[str, list[str]] = {
    "pet_bottle": ["ペットボトル"],
    "metal_can": ["缶", "かん"],
    "glass_bottle": ["びん"],
    "paper": ["紙くず", "紙"],
    "cardboard": ["段ボール"],
    "plastic_film": ["トレイ", "トレー"],
    "food_waste": ["生ごみ"],
    "battery": ["電池", "乾電池"],
    "textile": ["衣類", "衣服"],
    "ceramic": ["陶磁器"],
}


def load_corpus(root: Path) -> list[dict]:
    labels = root / "labels.json"
    if labels.exists():
        data = json.loads(labels.read_text(encoding="utf-8"))
        items = data["items"] if isinstance(data, dict) else data
        return [dict(item, abspath=root / item["path"]) for item in items]
    # fall back to one folder per class
    items = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for image in sorted(folder.glob("*.*")):
            items.append(
                {"path": str(image.relative_to(root)), "label": folder.name,
                 "difficulty": "unknown", "abspath": image}
            )
    return items


def expected_bins(registry: Registry) -> dict[tuple[str, str], str | None]:
    """(municipality, material) -> the dictionary's own answer, or None."""
    table: dict[tuple[str, str], str | None] = {}
    for municipality in registry.all():
        for material, words in CANONICAL.items():
            answer = None
            for word in words:
                lookup = municipality.lookup(word)
                if lookup["found"] and lookup["confident"]:
                    answer = lookup["category"]
                    break
            table[(municipality.id, material)] = answer
    return table


def _matrix(
    rows: list[tuple[str, str]], columns: list[str]
) -> dict[str, dict[str, int]]:
    matrix = {row: {col: 0 for col in columns} for row in sorted(set(r for r, _ in rows))}
    for actual, predicted in rows:
        matrix.setdefault(actual, {col: 0 for col in columns})
        matrix[actual][predicted] = matrix[actual].get(predicted, 0) + 1
    return matrix


def evaluate(
    corpus: Path, municipalities: list[str] | None = None, split: str = "all"
) -> dict:
    registry = Registry()
    items = load_corpus(corpus)
    if split != "all":
        items = [item for item in items if item.get("split", "dev") == split]
    if not items:
        raise SystemExit(f"no labelled images under {corpus}")

    materials = sorted({item["label"] for item in items})
    predicted_labels = materials + ["unknown"]

    top1: list[tuple[str, str]] = []
    top3_hits = 0
    per_class_total: Counter[str] = Counter()
    per_class_hit: Counter[str] = Counter()
    difficulty_total: Counter[str] = Counter()
    difficulty_hit: Counter[str] = Counter()

    for item in items:
        image = item["abspath"]
        payload = pathlib_read(image)
        features = analyze(payload)
        ranked = classify(features)
        predicted = ranked[0].material if ranked else "unknown"
        actual = item["label"]

        top1.append((actual, predicted))
        per_class_total[actual] += 1
        if predicted == actual:
            per_class_hit[actual] += 1
        if actual in [entry.material for entry in ranked[:3]]:
            top3_hits += 1
        difficulty = item.get("difficulty", "unknown")
        difficulty_total[difficulty] += 1
        if predicted == actual:
            difficulty_hit[difficulty] += 1

    total = len(items)
    matrix = _matrix(top1, predicted_labels)
    majority = Counter(actual for actual, _ in top1).most_common(1)[0]

    expected = expected_bins(registry)
    chosen_cities = municipalities or [m.id for m in registry.all()]

    pipelines: dict[str, dict] = {}
    for city in chosen_cities:
        bins_total = 0
        bins_hit = 0
        bin_rows: list[tuple[str, str]] = []
        skipped: list[str] = []
        for item in items:
            material = item["label"]
            want = expected.get((city, material))
            if want is None:
                if material not in skipped:
                    skipped.append(material)
                continue
            answer = advise_photo(registry, city, pathlib_read(item["abspath"]))
            got = answer["lookup"]["category"] or "（判定なし）"
            bins_total += 1
            bins_hit += int(got == want)
            bin_rows.append((want, got))
        # A no-information baseline: always answer the most common expected bin.
        common = Counter(want for want, _ in bin_rows).most_common(1)
        baseline = (common[0][1] / bins_total) if common and bins_total else 0.0
        pipelines[city] = {
            "scored_images": bins_total,
            "accuracy": round(bins_hit / bins_total, 4) if bins_total else None,
            "baseline_majority": round(baseline, 4),
            "skipped_materials": skipped,
            "matrix": _matrix(bin_rows, sorted({g for _, g in bin_rows})),
        }

    return {
        "corpus": str(corpus),
        "images": total,
        "material_top1_accuracy": round(sum(a == p for a, p in top1) / total, 4),
        "material_top3_recall": round(top3_hits / total, 4),
        "material_baseline_most_common_class": majority[0],
        "material_baseline_accuracy": round(majority[1] / total, 4),
        "per_class_recall": {
            cls: round(per_class_hit[cls] / per_class_total[cls], 4)
            for cls in sorted(per_class_total)
        },
        "by_difficulty": {
            key: {
                "images": difficulty_total[key],
                "accuracy": round(difficulty_hit[key] / difficulty_total[key], 4),
            }
            for key in sorted(difficulty_total)
        },
        "confusion": matrix,
        "labels": {key: MATERIAL_LABELS_JA.get(key, key) for key in predicted_labels},
        "pipelines": pipelines,
        "note": (
            "material_* は材質ルール表の実測。pipelines.* は『写真→材質語→その自治体の"
            "辞書』の配管を測ったもので、実際の自治体の合意正解ではない。期待値は辞書自身"
            "の応答（confident なもの）から取っている。"
        ),
    }


def pathlib_read(path: Path) -> bytes:
    return path.read_bytes()


def to_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"# 評価結果（実測）\n")
    lines.append(
        f"- コーパス: `{report['corpus']}`（{report['images']} 枚, "
        f"split={report.get('split', 'all')}）"
    )
    lines.append(
        f"- 材質の top-1 正解率: **{report['material_top1_accuracy'] * 100:.1f}%**"
        f"（top-3 再現率 {report['material_top3_recall'] * 100:.1f}%）"
    )
    lines.append(
        f"- 基準線（常に最頻クラス「{MATERIAL_LABELS_JA.get(report['material_baseline_most_common_class'], '')}」"
        f"と答える）: {report['material_baseline_accuracy'] * 100:.1f}%"
    )
    for key, value in report["by_difficulty"].items():
        label = {"clean": "背景がきれい", "cluttered": "他の物が写り込む"}.get(key, key)
        lines.append(f"- {label}: {value['accuracy'] * 100:.1f}%（{value['images']} 枚）")
    lines.append("")

    lines.append("## 材質ごとの再現率\n")
    lines.append("| 材質 | 再現率 |")
    lines.append("| --- | ---: |")
    for cls, recall in report["per_class_recall"].items():
        lines.append(f"| {MATERIAL_LABELS_JA.get(cls, cls)} | {recall * 100:.0f}% |")
    lines.append("")

    labels = report["labels"]
    columns = [key for key in labels if key != "unknown"] + ["unknown"]
    lines.append("## 混同行列（行=正解, 列=予測）\n")
    header = "| 正解 \\ 予測 | " + " | ".join(
        short(labels.get(col, col)) for col in columns
    ) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---:" for _ in columns) + " |")
    for row, counts in report["confusion"].items():
        cells = " | ".join(str(counts.get(col, 0)) for col in columns)
        lines.append(f"| **{short(labels.get(row, row))}** | {cells} |")
    lines.append("")

    lines.append("## 配管（写真→材質語→辞書）の正解率\n")
    lines.append("| 自治体 | 採点枚数 | 正解率 | 基準線（最頻の区だけ答える） | 除外した材質 |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for city, data in report["pipelines"].items():
        accuracy = "—" if data["accuracy"] is None else f"{data['accuracy'] * 100:.1f}%"
        skipped = "、".join(MATERIAL_LABELS_JA.get(s, s) for s in data["skipped_materials"]) or "なし"
        lines.append(
            f"| {city} | {data['scored_images']} | {accuracy} | "
            f"{data['baseline_majority'] * 100:.1f}% | {skipped} |"
        )
    lines.append("")
    lines.append(f"> {report['note']}")
    return "\n".join(lines) + "\n"


def short(label: str) -> str:
    return (
        label.replace("ペットボトル", "ペット")
        .replace("プラ容器・フィルム", "フィルム")
        .replace("金属缶", "缶")
        .replace("ガラスびん", "びん")
        .replace("段ボール", "段ボ")
        .replace("生ごみ・食品", "生ごみ")
        .replace("布・衣類", "布")
        .replace("判別できない", "不明")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="eval/synthetic")
    parser.add_argument("--json", default="")
    parser.add_argument("--markdown", default="")
    parser.add_argument("--city", nargs="*", default=None)
    parser.add_argument("--split", default="all", choices=["all", "dev", "test"])
    args = parser.parse_args()

    report = evaluate(Path(args.corpus), args.city, args.split)
    report["split"] = args.split
    print(json.dumps(
        {key: report[key] for key in (
            "images", "material_top1_accuracy", "material_top3_recall",
            "material_baseline_accuracy", "per_class_recall", "by_difficulty",
        )},
        ensure_ascii=False, indent=1,
    ))
    for city, data in report["pipelines"].items():
        print(city, "bin accuracy:", data["accuracy"], "baseline:", data["baseline_majority"],
              "scored:", data["scored_images"], "skipped:", data["skipped_materials"])

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", args.json)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(to_markdown(report), encoding="utf-8")
        print("wrote", args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
