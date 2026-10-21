#!/usr/bin/env python3
"""
nerdsight - cli.py
Command-line search tool for Nerd Fonts visual & semantic matching.
Usage:
    python cli.py path/to/icon.png [--alpha 0.65] [--top 10] [--boost "git"]
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
IMG_EMB_PATH = os.path.join(DATA_DIR, "img_embeddings.npy")
TXT_EMB_PATH = os.path.join(DATA_DIR, "txt_embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "metadata.json")


def preprocess_image(image_path: str, size: int = 128, invert: bool = False) -> Image.Image:
    """Preprocesses local image for CLIP embedding."""
    raw = Image.open(image_path).convert("RGBA")
    white_bg = Image.new("RGBA", raw.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white_bg, raw).convert("RGB")

    if invert:
        composited = ImageOps.invert(composited)

    gray = composited.convert("L")
    gray = ImageOps.autocontrast(gray)
    gray = gray.point(lambda p: 0 if p < 185 else 255, "L")
    processed = gray.convert("RGB")

    inv = ImageOps.invert(processed.convert("L"))
    inv = inv.point(lambda p: 0 if p < 30 else 255)
    bbox = inv.getbbox()
    if bbox:
        cropped = processed.crop(bbox)
        target_box = int(size * 0.72)
        cropped.thumbnail((target_box, target_box), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        ox = (size - cropped.width) // 2
        oy = (size - cropped.height) // 2
        canvas.paste(cropped, (ox, oy))
        return canvas

    return ImageOps.pad(processed, (size, size), color=(255, 255, 255))


def main():
    parser = argparse.ArgumentParser(description="nerdsight - Visual CLI search for Nerd Fonts.")
    parser.add_argument("image", help="Path to input image file (PNG, JPG, WEBP)")
    parser.add_argument("-a", "--alpha", type=float, default=0.65, help="Visual vs semantic balance (0.0 to 1.0, default: 0.65)")
    parser.add_argument("-k", "--top", type=int, default=10, help="Number of results to return (default: 10)")
    parser.add_argument("-b", "--boost", type=str, default="", help="Optional text hint concept")
    parser.add_argument("-i", "--invert", action="store_true", help="Invert dark-mode image colors")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)

    if not (os.path.exists(IMG_EMB_PATH) and os.path.exists(TXT_EMB_PATH) and os.path.exists(META_PATH)):
        print("Error: Index not built. Run `python setup_assets.py && python indexer.py` first.", file=sys.stderr)
        sys.exit(1)

    img_emb = np.load(IMG_EMB_PATH)
    txt_emb = np.load(TXT_EMB_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SentenceTransformer("clip-ViT-B-32")

    proc_img = preprocess_image(args.image, invert=args.invert)
    query_vec = model.encode([proc_img], convert_to_numpy=True)[0]
    query_vec = query_vec / np.linalg.norm(query_vec)

    if args.boost.strip():
        boost_vec = model.encode([f"icon symbol of {args.boost.strip()}"], convert_to_numpy=True)[0]
        boost_vec = boost_vec / np.linalg.norm(boost_vec)
        query_vec = (0.75 * query_vec) + (0.25 * boost_vec)
        query_vec = query_vec / np.linalg.norm(query_vec)

    # Hybrid blend
    hybrid_matrix = (args.alpha * img_emb) + ((1.0 - args.alpha) * txt_emb)
    norms = np.linalg.norm(hybrid_matrix, axis=1, keepdims=True)
    hybrid_matrix = hybrid_matrix / np.maximum(norms, 1e-12)

    scores = np.dot(hybrid_matrix, query_vec)
    top_indices = np.argsort(scores)[::-1][:args.top]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        item = metadata[idx]
        score = float(scores[idx])
        results.append({
            "rank": rank,
            "name": item["name"],
            "char": item["char"],
            "code": item["code"],
            "score": round(score * 100, 2),
        })

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print(f"\n🔍 nerdsight search results for: {args.image}")
    print(f"⚖️  Alpha: {args.alpha:.2f} (Visual: {args.alpha*100:.0f}% / Semantic: {(1-args.alpha)*100:.0f}%)")
    if args.boost.strip():
        print(f"💡 Boost Concept: '{args.boost.strip()}'")
    print("-" * 65)
    print(f"{'#':<3} {'Glyph':<6} {'Similarity':<12} {'Unicode':<10} {'Name'}")
    print("-" * 65)

    for res in results:
        print(f"{res['rank']:<3} {res['char']:<6} {res['score']:>5.1f}%      \\u{res['code']:<6}  {res['name']}")

    print("-" * 65)


if __name__ == "__main__":
    main()
