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
from tqdm import tqdm

import core

# Name tokens that describe drawing style rather than the depicted object. They are
# stripped from the noun phrase and re-expressed once as "outline"/"filled", so
# 'md-ghost_outline' and 'md-ghost' share the object noun and differ only in style.
STYLE_TOKENS = {"outline", "outlined", "line", "o", "solid", "fill", "filled", "alt", "variant", "box", "small", "large", "thin", "bold"}
OUTLINE_TOKENS = {"outline", "outlined", "line", "o"}


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
    """Transforms identifiers like 'md-ghost_outline' into 'a outline icon of a ghost'.

    Keeps the object noun phrase free of style words (which otherwise dominate short
    CLIP prompts) and states the style exactly once. Benchmarked against the previous
    verbose templates: same or better ranks on image queries, cleaner text search.
    """
    prefix, _, rest = raw_name.partition("-")
    words = [w for w in rest.replace("-", "_").split("_") if w]
    nouns = [w for w in words if w not in STYLE_TOKENS]
    style = "outline" if OUTLINE_TOKENS & set(words) else "filled"
    return f"a {style} icon of a {' '.join(nouns) or prefix}"


def normalize_l2(vecs: np.ndarray) -> np.ndarray:
    """Normalizes vectors along axis 1 using L2 norm."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def rebuild_text_embeddings(args):
    """Re-encodes prompts for the glyphs already in metadata.json; image embeddings are untouched."""
    if not core.has_embeddings_index():
        print("❌ No existing index found. Run `python indexer.py` first.", file=sys.stderr)
        sys.exit(1)

    with open(core.META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    for item in metadata:
        item["prompt"] = build_semantic_prompt(item["name"].removeprefix("nf-"))

    model = core.get_model(args.model)
    print(f"\n📝 Re-encoding {len(metadata)} semantic prompts...")
    txt_emb = model.encode([m["prompt"] for m in metadata], batch_size=args.batch_size * 2, show_progress_bar=True, convert_to_numpy=True)
    np.save(core.TXT_EMB_PATH, normalize_l2(txt_emb).astype(np.float32))
    with open(core.META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ Text embeddings rebuilt: {core.TXT_EMB_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Build nerdsight hybrid CLIP index.")
    parser.add_argument("--model", default="clip-ViT-B-32", help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of glyphs to index (for quick testing/demo)")
    parser.add_argument("--text-only", action="store_true", help="Only rebuild txt_embeddings.npy from the existing metadata (after prompt changes)")
    args = parser.parse_args()

    if args.text_only:
        rebuild_text_embeddings(args)
        return

    if not os.path.exists(core.GLYPHNAMES_PATH) or not os.path.exists(core.FONT_PATH):
        print("❌ Missing assets! Please run `python setup_assets.py` first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(core.DATA_DIR, exist_ok=True)

    print(f"📖 Loading glyph data from {core.GLYPHNAMES_PATH}...")
    with open(core.GLYPHNAMES_PATH, "r", encoding="utf-8") as f:
        glyphs_data = json.load(f)

    glyphs_data.pop("METADATA", None)

    print(f"🔤 Loading font {core.FONT_PATH}...")
    font = ImageFont.truetype(core.FONT_PATH, int(128 * 0.72))

    print(f"🤖 Loading multimodal model '{args.model}'...")
    model = core.get_model(args.model)

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

        if args.limit and len(images) >= args.limit:
            print(f"⏱️ Limit reached ({args.limit} glyphs). Proceeding to embedding generation.")
            break

    print(f"\n🧠 Generating visual embeddings for {len(images)} icons...")
    img_emb = model.encode(images, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    img_emb = normalize_l2(img_emb).astype(np.float32)

    print(f"\n📝 Generating semantic text embeddings for {len(prompts)} descriptions...")
    txt_emb = model.encode(prompts, batch_size=args.batch_size * 2, show_progress_bar=True, convert_to_numpy=True)
    txt_emb = normalize_l2(txt_emb).astype(np.float32)

    img_out, txt_out, meta_out = core.IMG_EMB_PATH, core.TXT_EMB_PATH, core.META_PATH

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
