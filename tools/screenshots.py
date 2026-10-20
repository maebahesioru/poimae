"""Capture the screenshots the README and the demo video use.

Headless Chromium (Brave) is driven from the command line, one page load per
shot.  The pages are served by the app itself, so every picture is a picture of
this repository running - nothing is mocked or retouched afterwards.

    python -m tools.screenshots --out evidence/shots --base http://127.0.0.1:8765

The app must already be running (`make run`).  Each shot names the URL it came
from in `evidence/shots/manifest.json`, so a reader can reproduce it by hand.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

#: (file stem, query string, viewport, what the shot is for)
SHOTS = [
    ("01-hero", "", "1280,900", "初期画面。自治体を選び、写真か名前で調べる。"),
    ("02-photo-ceramic", "?demo=ceramic", "1280,1500", "写真→材質→辞典の候補→区分（陶磁器）。"),
    ("03-photo-bottle", "?demo=pet_bottle", "1280,1500", "同じ写真でも候補が複数出る例（ペットボトル）。"),
    ("04-text-battery", "?q=%E4%B9%BE%E9%9B%BB%E6%B1%A0", "1280,1400", "名前で検索。辞典にない語は類語で引き直す。"),
    ("05-week", "?q=%E3%83%9A%E3%83%83%E3%83%88%E3%83%9C%E3%83%88%E3%83%AB", "1280,2000", "今週のごみの日（曜日は保存のみ・端末内）。"),
]

CANDIDATES = ("brave", "brave-browser", "chromium", "chromium-browser", "google-chrome")


def find_browser() -> str:
    for name in CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit(f"no headless browser found (looked for {', '.join(CANDIDATES)})")


def shoot(browser: str, url: str, size: str, out: Path) -> None:
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--window-size={size}",
            f"--screenshot={out}",
            "--virtual-time-budget=7000",
            url,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evidence/shots"))
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    args = parser.parse_args(argv)

    browser = find_browser()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for stem, query, size, caption in SHOTS:
        path = args.out / f"{stem}.png"
        url = f"{args.base}/{query}"
        shoot(browser, url, size, path)
        manifest.append({"file": path.name, "url": url, "viewport": size, "caption": caption})
        print(f"{path}  <- {url}")

    (args.out / "manifest.json").write_text(
        json.dumps({"note": "実際に動作中のアプリを headless で撮影したもの。", "shots": manifest},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
