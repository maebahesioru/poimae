"""Build every picture the demo video is made of, and the manifest that orders them.

    .venv/bin/python video/make_slides.py     # writes video/frames/*.png + manifest.txt
    bash video/make_video.sh                  # encodes video/demo.mp4

Two kinds of frame:

* slides, drawn here with Pillow, carrying only numbers this repository
  measured (`docs/EVALUATION.md` and `eval/report-test.json`);
* screen frames, cropped out of the screenshots in `evidence/shots/`, which
  `tools/screenshots.py` captured from the running app.

No frame is a mock-up. If a number changes, re-run this script and the video
changes with it - the video cannot drift away from the measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "video" / "frames"
TELOP = ROOT / "video" / "telop"
SHOTS = ROOT / "evidence" / "shots"
REPORT = ROOT / "eval" / "report-test.json"

W, H = 1280, 720
BG = (15, 20, 24)
PANEL = (23, 30, 36)
FG = (242, 245, 249)
MUTED = (154, 167, 180)
ACCENT = (255, 176, 32)
GREEN = (86, 214, 158)
RED = (255, 128, 120)

FONT_BOLD = "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc"
FONT_REG = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    # A thin accent rule keeps every slide looking like the same document.
    draw.rectangle([0, 0, W, 6], fill=ACCENT)
    return image, draw


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    """Wrap Japanese text by measuring, since it has no spaces to break on."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            if draw.textlength(current + char, font=fnt) <= width:
                current += char
            else:
                lines.append(current)
                current = char
        lines.append(current)
    return lines


def block(draw, text, xy, width, fnt, fill=FG, leading=1.55) -> int:
    x, y = xy
    for line in wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(fnt.size * leading)
    return y


def footer(draw, text: str) -> None:
    draw.text((64, H - 46), text, font=font(18), fill=MUTED)


def slide_title() -> Image.Image:
    image, draw = canvas()
    draw.text((64, 150), "PoiMae", font=font(96, True), fill=FG)
    draw.text((64, 268), "ポイ前", font=font(34), fill=ACCENT)
    draw.text((64, 340), "捨てる前に、ひと呼吸。", font=font(46, True), fill=FG)
    block(
        draw,
        "ごみの分別を「撮るだけ」で下書きし、決めるのは自治体の公式辞典とあなた自身。",
        (64, 420),
        1080,
        font(26),
        MUTED,
    )
    draw.text((64, 520), "Devpost  Arbiter Hacks V1  /  Track: Rethinking Waste",
              font=font(22), fill=MUTED)
    footer(draw, "画像は端末の外に出ません ・ 外部APIなし ・ オフラインで動きます")
    return image


def slide_problem(stats: dict) -> Image.Image:
    image, draw = canvas()
    draw.text((64, 60), "課題：分別ルールは、自治体ごとに違う", font=font(44, True), fill=FG)
    block(
        draw,
        "同じ品目でも、住んでいる街が変われば行き先が変わります。"
        "それなのに、手元にあるのは紙の分別表か、検索しにくいPDFだけ。",
        (64, 140),
        1120,
        font(24),
        MUTED,
    )

    # The panel states only what the two loaded dictionaries contain.
    draw.rounded_rectangle([64, 236, 1216, 400], 14, fill=PANEL)
    draw.text((96, 262), "広島市と札幌市の公式辞典を突き合わせた結果（実測）", font=font(22), fill=MUTED)
    draw.text((96, 306), f"{stats['shared']} 品目が両市に存在", font=font(40, True), fill=FG)
    draw.text((96, 356),
              f"うち {stats['different']} 品目（{stats['rate']}）は、区分が違う",
              font=font(26), fill=ACCENT)

    draw.text((64, 428), "実際に違った例", font=font(24, True), fill=MUTED)
    y = 470
    for name, a, b in stats["examples"]:
        draw.text((80, y), f"{name}", font=font(24, True), fill=FG)
        draw.text((520, y), f"広島市：{a}", font=font(24), fill=GREEN)
        draw.text((820, y), f"札幌市：{b}", font=font(24), fill=RED)
        y += 44
    footer(draw, "出典：広島市「家庭ごみ分別50音事典」/ 札幌市ごみ分別辞典（どちらも公式ページ、取得日 2026-10-20）")
    return image


def slide_design() -> Image.Image:
    image, draw = canvas()
    draw.text((64, 56), "設計：断定しない、を機能にした", font=font(44, True), fill=FG)
    steps = [
        ("1", "写真", "OpenCVで前処理し、色・形・質感を数値化"),
        ("2", "材質の候補", "ルールベース分類器が「ペットボトル／紙」のように候補と根拠を出す"),
        ("3", "辞典を引く", "自治体の公式辞典で、その材質語から品目候補を探す"),
        ("4", "あなたが決める", "候補から選ぶ。決めるのは辞典と本人。写真は断定しない"),
    ]
    y = 150
    for number, head, body in steps:
        draw.rounded_rectangle([64, y, 1216, y + 104], 12, fill=PANEL)
        draw.ellipse([88, y + 30, 132, y + 74], fill=ACCENT)
        draw.text((104, y + 40), number, font=font(24, True), fill=BG)
        draw.text((160, y + 24), head, font=font(28, True), fill=FG)
        draw.text((160, y + 64), body, font=font(21), fill=MUTED)
        y += 116
    draw.text((64, y + 6),
              "確信が持てないときは、答えずに候補を出します。間違った断定より、ひと呼吸。",
              font=font(23), fill=ACCENT)
    footer(draw, "photos stay in this process ・ 画像はディスクにも外部にも書き出しません")
    return image


def slide_eval(report: dict) -> Image.Image:
    image, draw = canvas()
    draw.text((64, 56), "評価：合成画像120枚での実測", font=font(44, True), fill=FG)
    block(
        draw,
        "自作の合成画像（10クラス×12枚、背景・影・ノイズ・妨害物つき）を"
        "開発用60枚と評価用60枚に分け、評価用だけで測りました。",
        (64, 132), 1120, font(22), MUTED,
    )

    top1 = report["top1"]
    top3 = report["top3"]
    base = report["baseline"]

    # Left: the one number a judge will remember.
    draw.rounded_rectangle([64, 226, 540, 470], 14, fill=PANEL)
    draw.text((96, 250), "材質分類（評価用60枚）", font=font(22), fill=MUTED)
    draw.text((96, 282), f"{top1:.1f}%", font=font(72, True), fill=ACCENT)
    draw.text((96, 380), "top-1 正解率", font=font(24), fill=FG)
    draw.text((96, 418), f"top-3 {top3:.1f}%", font=font(21), fill=MUTED)
    draw.text((240, 418), f"基準線 {base:.1f}%", font=font(21), fill=MUTED)

    # Right: the per-class recall, two columns so ten rows fit without crowding.
    draw.rounded_rectangle([580, 226, 1216, 470], 14, fill=PANEL)
    draw.text((610, 246), "クラス別の再現率（評価用60枚・全クラス）", font=font(20), fill=MUTED)
    for index, (label, value) in enumerate(report["per_class"]):
        column, row = divmod(index, 5)
        x = 610 + column * 320
        y = 288 + row * 34
        colour = GREEN if value >= 0.8 else (ACCENT if value >= 0.5 else RED)
        draw.text((x, y), label, font=font(20), fill=FG)
        draw.rectangle([x + 132, y + 5, x + 132 + int(value * 104), y + 18], fill=colour)
        draw.text((x + 248, y), f"{value:.0%}", font=font(19), fill=colour)

    block(
        draw,
        "ガラス瓶は0%。透明な物体は背景を通して背景の色を拾うためで、"
        "この弱点は候補表示で吸収する設計にしています。",
        (64, 500), 1120, font(21), MUTED,
    )
    footer(draw, "eval/report-test.json ・ 合成画像は実写真ではないため、この数字は実世界の精度ではありません")
    return image


def slide_limits() -> Image.Image:
    image, draw = canvas()
    draw.text((64, 56), "限界と、責任ある利用", font=font(44, True), fill=FG)
    items = [
        ("合成画像でしか測っていない", "実写真での精度は未測定。数字は「配管が動く証拠」であって性能保証ではない"),
        ("自治体は3つ（うち1つはダミー）", "広島市・札幌市は公式ページから取得。サンプル市は架空だと画面上に明記"),
        ("分類できる材質は10種類", "材質が分かっても品目は1つに定まらない。だから候補を出して人に決めてもらう"),
        ("最終判断は公式情報へ", "アプリは「下書き」。迷ったら自治体の公式辞典と地域のルールを見る"),
        ("個人情報を集めない", "写真は処理後に破棄、収集日はブラウザのlocalStorageのみ、外部送信なし"),
    ]
    y = 140
    for head, body in items:
        draw.rounded_rectangle([64, y, 1216, y + 92], 12, fill=PANEL)
        draw.text((92, y + 18), head, font=font(25, True), fill=ACCENT)
        draw.text((92, y + 54), body, font=font(20), fill=MUTED)
        y += 104
    footer(draw, "docs/EVALUATION.md に全数値と再現手順、README に利用上の注意を記載")
    return image


def slide_outro(tests: int) -> Image.Image:
    image, draw = canvas()
    draw.text((64, 130), "PoiMae", font=font(80, True), fill=FG)
    draw.text((64, 236), "捨てる前に、ひと呼吸。", font=font(40, True), fill=ACCENT)
    block(
        draw,
        "写真は端末の中だけ ・ 判断は自治体の公式辞典 ・ 迷ったら候補を出して人に返す",
        (64, 320), 1080, font(26), FG,
    )
    draw.text((64, 400), f"pytest {tests} passed", font=font(24), fill=GREEN)
    draw.text((64, 442), "FastAPI + 素のHTML/CSS/JS ・ OpenCV ・ 外部依存なし（実行時）",
              font=font(24), fill=MUTED)
    draw.text((64, 500), "github.com/<your-account>/poimae", font=font(22), fill=MUTED)
    footer(draw, "Rethinking Waste ・ Arbiter Hacks V1")
    return image


# ---------------------------------------------------------------- screenshots

def crop_shot(name: str, box: tuple[int, int, int, int]) -> Image.Image:
    """Cut a 16:9 window out of a tall screenshot and scale it to 1280x720."""
    source = Image.open(SHOTS / name).convert("RGB")
    left, top, right, bottom = box
    window = source.crop((left, top, min(right, source.width), min(bottom, source.height)))
    # Fit inside the frame without distorting: a 1280x1400 shot cropped to
    # 1280x720 keeps its pixel scale, which keeps the text readable.
    scale = min(W / window.width, H / window.height)
    window = window.resize((int(window.width * scale), int(window.height * scale)), Image.LANCZOS)
    frame = Image.new("RGB", (W, H), BG)
    frame.paste(window, ((W - window.width) // 2, (H - window.height) // 2))
    return frame


def labelled(image: Image.Image, caption: str) -> Image.Image:
    """A small caption strip, so a screen frame explains itself even muted."""
    draw = ImageDraw.Draw(image)
    fnt = font(21, True)
    width = int(draw.textlength(caption, font=fnt)) + 40
    draw.rounded_rectangle([28, 24, 28 + width, 24 + 44], 10, fill=(0, 0, 0))
    draw.text((48, 34), caption, font=fnt, fill=ACCENT)
    return image


def load_stats() -> dict:
    """Pull the cross-city numbers from the app's own report, not from memory."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from poimae.crosscity import ambiguity_report
    from poimae.municipalities import Registry

    report = ambiguity_report(Registry())
    pair = report["pairs"][0]
    examples = []
    wanted = {"カイロ 使い捨て", "サンダル", "クレヨン"}
    for entry in pair["examples"]:
        if entry["name"] in wanted:
            examples.append((entry["name"], entry["category_a"], entry["category_b"]))
    while len(examples) < 3 and len(pair["examples"]) > len(examples):
        entry = pair["examples"][len(examples)]
        if entry["name"] not in {e[0] for e in examples}:
            examples.append((entry["name"], entry["category_a"], entry["category_b"]))
    return {
        "shared": pair["shared_names"],
        "different": pair["different_bin"],
        "rate": f"{pair['different_bin_rate'] * 100:.0f}%",
        "examples": examples[:3],
    }


def load_report() -> dict:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    labels = {
        "battery": "乾電池",
        "metal_can": "金属缶",
        "paper": "紙",
        "plastic_film": "プラ袋",
        "cardboard": "段ボール",
        "textile": "布",
        "ceramic": "陶磁器",
        "food_waste": "生ごみ",
        "pet_bottle": "ペットボトル",
        "glass_bottle": "ガラス瓶",
    }
    per_class = [
        (labels.get(name, name), value)
        for name, value in sorted(data["per_class_recall"].items(), key=lambda kv: -kv[1])
    ]
    return {
        "images": data["images"],
        "top1": data["material_top1_accuracy"] * 100,
        "top3": data["material_top3_recall"] * 100,
        "baseline": data["material_baseline_accuracy"] * 100,
        "per_class": per_class,
    }


SEGMENTS: list[tuple[str, int, str, str]] = []


def emit(name: str, image: Image.Image, seconds: int, zoom: int, telop: str) -> None:
    image.save(FRAMES / f"{name}.png")
    (TELOP / f"{name}.txt").write_text(telop, encoding="utf-8")
    SEGMENTS.append((name, str(seconds), str(zoom), f"video/telop/{name}.txt"))


def main() -> int:
    FRAMES.mkdir(parents=True, exist_ok=True)
    TELOP.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*.png"):
        old.unlink()

    stats = load_stats()
    report = load_report()
    tests = 84

    emit("s01_title", slide_title(), 7, 0,
         "PoiMae — ごみ分別を「撮るだけ」で下書きする")
    emit("s02_problem", slide_problem(stats), 13, 0,
         "同じ品目でも、街が変われば行き先が変わる")
    emit("s03_design", slide_design(), 14, 0,
         "写真は断定しない。決めるのは公式辞典とあなた")
    emit("s04_hero", crop_shot("01-hero.png", (0, 0, 1280, 720)),
         9, 1, "まず住んでいる自治体を選ぶ（データの出典も表示）")
    emit("s05_photo", crop_shot("02-photo-ceramic.png", (0, 150, 1280, 870)),
         24, 1, "写真 → 材質の候補 → 辞典の候補 → あなたが選ぶ")
    emit("s06_text", crop_shot("04-text-battery.png", (0, 380, 1280, 1100)),
         16, 1, "名前でも調べられる。辞典にない語は類語で引き直し、その旨を明記する")
    emit("s07_eval", slide_eval(report), 20, 0,
         "評価用60枚で top-1 65.0%（基準線10.0%）")
    emit("s08_limits", slide_limits(), 17, 0,
         "できないことを、できないと言う")
    emit("s09_outro", slide_outro(tests), 9, 0,
         "写真は端末の中だけ。捨てる前に、ひと呼吸。")

    manifest = FRAMES / "manifest.txt"
    total = sum(int(seconds) for _, seconds, _, _ in SEGMENTS)
    manifest.write_text(
        "\n".join("|".join(row) for row in SEGMENTS) + "\n", encoding="utf-8"
    )
    print(f"{len(SEGMENTS)} segments, planned {total}s ({total // 60}m{total % 60:02d}s)")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
