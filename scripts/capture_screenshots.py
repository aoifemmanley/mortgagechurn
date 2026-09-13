"""Capture prototype screenshots for the deck.

Run *after* Streamlit is serving on localhost:8766.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots"
OUT.mkdir(exist_ok=True)

BASE = "http://localhost:8766"
VIEWPORT = {"width": 1600, "height": 1000}

HERO_CUSTOMERS = [
    ("C0105859", "customer_detail_hero.png"),   # Julie Johnson — top VaR, 52% risk
    ("C0103303", "customer_detail_alt.png"),    # Bryan Jones — 52% risk, 260 bps gap
]


def _shot(page, path: Path, full=True):
    time.sleep(2.5)  # let Streamlit finish rendering
    page.screenshot(path=str(path), full_page=full)
    print(f"  -> {path}")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = ctx.new_page()

        # 1. Command center (default view)
        print("Command center...")
        page.goto(BASE, wait_until="networkidle", timeout=45000)
        time.sleep(10)  # command center runs `apply(recommend)` on 200 rows
        _shot(page, OUT / "01_command_center.png", full=False)
        _shot(page, OUT / "01b_command_center_full.png", full=True)

        # 2. Customer detail views (wide)
        for cid, fname in HERO_CUSTOMERS:
            print(f"Customer {cid} (wide)...")
            page.goto(f"{BASE}?customer={cid}", wait_until="networkidle", timeout=45000)
            time.sleep(3)
            _shot(page, OUT / fname, full=True)

        # 3. Narrow / half-slide customer detail — for embedding on a half-slide layout
        print("Customer C0105859 (narrow / half-slide)...")
        narrow_ctx = browser.new_context(
            viewport={"width": 1200, "height": 1500},
            device_scale_factor=2,
        )
        narrow_page = narrow_ctx.new_page()
        narrow_page.goto(f"{BASE}?customer=C0105859", wait_until="networkidle", timeout=45000)
        time.sleep(3)
        narrow_page.screenshot(
            path=str(OUT / "customer_detail_narrow.png"),
            full_page=True,
        )
        print(f"  -> {OUT / 'customer_detail_narrow.png'}")
        narrow_ctx.close()

        browser.close()
    print("Done.")


if __name__ == "__main__":
    main()
