#!/usr/bin/env python3
"""
nerdsight - indexer.py
Renders Nerd Fonts glyphs into canonical 128x128 images, synthesizes rich
semantic descriptions, and extracts decoupled visual and textual CLIP embeddings.
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ASSETS_DIR = "assets"
DATA_DIR = "data"
FONT_PATH = os.path.join(ASSETS_DIR, "SymbolsNerdFontMono-Regular.ttf")
GLYPHNAMES_PATH = os.path.join(ASSETS_DIR, "glyphnames.json")

# Contextual prefix mapping to provide semantic richness to CLIP
PREFIX_MAP = {
    "fa": "font awesome icon",
    "fae": "font awesome extension symbol",
    "md": "material design icon",
    "mdi": "material design icon",
    "dev": "developer programming technology logo",
    "oct": "github octicon",
    "cod": "vscode codicon",
    "linux": "linux distro logo",
    "custom": "custom utility icon",
    "weather": "weather meteorological symbol",
    "seti": "file type programming icon",
    "pl": "powerline prompt symbol",
    "ple": "powerline extra glyph",
    "pom": "pomodoro productivity timer",
    "iec": "power electronic symbol",
}


def render_glyph(char: str, font: ImageFont.FreeTypeFont, size: int = 128) -> Image.Image:
    """Renders a glyph centered with tight bounding-box calculation."""
    img = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(img)

    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            return img

        x = (size - w) / 2 - bbox[0]
        y = (size - h) / 2 - bbox[1]
        draw.text((x, y), char, fill="black", font=font)
    except Exception:
        pass

    return img


def build_semantic_prompt(raw_name: str) -> str:
    """Transforms technical identifiers like 'dev-git_branch' into descriptive prompts."""
    parts = raw_name.split("-", 1)
    prefix = parts[0] if len(parts) > 1 else ""
    clean_name = parts[1] if len(parts) > 1 else raw_name
    clean_name = clean_name.replace("_", " ").replace("-", " ")
    category_context = PREFIX_MAP.get(prefix, "symbol")
    return f"a minimalist black and white icon symbol of {clean_name} ({category_context})"


def normalize_l2(vecs: np.ndarray) -> np.ndarray:
    """Normalizes vectors along axis 1 using L2 norm."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def main():
    parser = argparse.ArgumentParser(description="Build nerdsight hybrid CLIP index.")
    parser.add_argument("--model", default="clip-ViT-B-32", help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding generation")
    args = parser.parse_args()

    if not os.path.exists(GLYPHNAMES_PATH) or not os.path.exists(FONT_PATH):
        print("❌ Missing assets! Please run `python setup_assets.py` first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"📖 Loading glyph data from {GLYPHNAMES_PATH}...")
    with open(GLYPHNAMES_PATH, "r", encoding="utf-8") as f:
        glyphs_data = json.load(f)

    glyphs_data.pop("METADATA", None)

    print(f"🔤 Loading font {FONT_PATH}...")
    font = ImageFont.truetype(FONT_PATH, int(128 * 0.72))

    print(f"🤖 Loading multimodal model '{args.model}'...")
    model = SentenceTransformer(args.model)

    metadata = []
    images = []
    prompts = []

    print("🎨 Rendering glyphs and synthesizing semantic prompts...")
    for name, info in tqdm(glyphs_data.items(), desc="Glyphs"):
        # Skip obsolete v2 Material Design duplicates
        if name.startswith("mdi-"):
            continue

        char = info["char"]
        code = info["code"]

        img = render_glyph(char, font, size=128)
        prompt = build_semantic_prompt(name)

        images.append(img)
        prompts.append(prompt)
        metadata.append({
            "name": f"nf-{name}",
            "char": char,
            "code": code,
            "prompt": prompt,
        })

    print(f"\n🧠 Generating visual embeddings for {len(images)} icons...")
    img_emb = model.encode(images, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    img_emb = normalize_l2(img_emb).astype(np.float32)

    print(f"\n📝 Generating semantic text embeddings for {len(prompts)} descriptions...")
    txt_emb = model.encode(prompts, batch_size=args.batch_size * 2, show_progress_bar=True, convert_to_numpy=True)
    txt_emb = normalize_l2(txt_emb).astype(np.float32)

    img_out = os.path.join(DATA_DIR, "img_embeddings.npy")
    txt_out = os.path.join(DATA_DIR, "txt_embeddings.npy")
    meta_out = os.path.join(DATA_DIR, "metadata.json")

    print("\n💾 Saving index matrices and metadata to disk...")
    np.save(img_out, img_emb)
    np.save(txt_out, txt_emb)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Indexing completed successfully!")
    print(f"   - Visual embeddings: {img_out} ({img_emb.shape})")
    print(f"   - Textual embeddings: {txt_out} ({txt_emb.shape})")
    print(f"   - Metadata catalogue: {meta_out} ({len(metadata)} entries)")


if __name__ == "__main__":
    main()
