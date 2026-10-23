#!/usr/bin/env python3
"""
nerdsight - setup_assets.py
Downloads official Nerd Fonts assets directly from ryanoasis/nerd-fonts:
1. glyphnames.json (canonical mapping of icon names, codepoints, and characters)
2. SymbolsNerdFontMono-Regular.ttf (extracted from NerdFontsSymbolsOnly.zip)
"""

import io
import os
import sys
import zipfile
import requests
from tqdm import tqdm

import core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data")

GLYPHNAMES_URL = "https://raw.githubusercontent.com/ryanoasis/nerd-fonts/master/glyphnames.json"
SYMBOLS_ZIP_URL = "https://github.com/ryanoasis/nerd-fonts/releases/latest/download/NerdFontsSymbolsOnly.zip"
TARGET_FONT = "SymbolsNerdFontMono-Regular.ttf"
FALLBACK_FONT = "SymbolsNerdFont-Regular.ttf"


def download_file_with_progress(url: str, dest_path: str, description: str):
    """Downloads a file with streaming progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=description,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)


def download_and_extract_font():
    """Downloads NerdFontsSymbolsOnly.zip and extracts SymbolsNerdFontMono-Regular.ttf."""
    font_path = os.path.join(ASSETS_DIR, TARGET_FONT)
    if os.path.exists(font_path) and os.path.getsize(font_path) > 100000:
        print(f"✔️ Font already present: {font_path} ({os.path.getsize(font_path) // 1024} KB)")
        return

    print("⬇️ Downloading NerdFontsSymbolsOnly release archive...")
    resp = requests.get(SYMBOLS_ZIP_URL, stream=True, timeout=90)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    buffer = io.BytesIO()

    with tqdm(desc="Archive", total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
        for chunk in resp.iter_content(chunk_size=16384):
            bar.update(buffer.write(chunk))

    buffer.seek(0)
    print("📦 Extracting symbols font...")
    with zipfile.ZipFile(buffer) as z:
        names = z.namelist()
        selected = TARGET_FONT if TARGET_FONT in names else (FALLBACK_FONT if FALLBACK_FONT in names else None)
        if not selected:
            raise FileNotFoundError(f"Neither {TARGET_FONT} nor {FALLBACK_FONT} found in archive: {names}")

        with z.open(selected) as zf, open(font_path, "wb") as out_f:
            out_f.write(zf.read())

    print(f"✅ Font saved successfully to: {font_path}")


def download_glyphnames():
    """Downloads the official glyphnames.json mapping."""
    json_path = os.path.join(ASSETS_DIR, "glyphnames.json")
    if os.path.exists(json_path) and os.path.getsize(json_path) > 10000:
        print(f"✔️ glyphnames.json already present: {json_path}")
        return

    print("⬇️ Downloading official glyphnames.json...")
    download_file_with_progress(GLYPHNAMES_URL, json_path, "glyphnames.json")
    print(f"✅ glyphnames.json saved to: {json_path}")


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("🚀 Initializing nerdsight asset pipeline...")
    try:
        download_glyphnames()
        download_and_extract_font()
        core.ensure_static_font()
        print("\n✨ Assets setup complete! You can now run `python indexer.py` to build embeddings.")
    except Exception as exc:
        print(f"\n❌ Error during asset setup: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
