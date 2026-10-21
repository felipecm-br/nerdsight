#!/usr/bin/env python3
"""
nerdsight - cli.py
Command-line search tool for Nerd Fonts visual, name & semantic matching.
Usage:
    # 1. Visual Search (by image or screenshot)
    python cli.py image.png [--alpha 0.65] [--top 10] [--boost "git"]

    # 2. Name Search (instant search across 10,000+ icons)
    python cli.py --name "git" [--family dev]

    # 3. Semantic Concept Search (search by meaning via CLIP)
    python cli.py --text "database" [--top 10]

    # 4. Catalog / Show All Icons
    python cli.py --all [--family fa] [--limit 50]
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
IMG_EMB_PATH = os.path.join(DATA_DIR, "img_embeddings.npy")
TXT_EMB_PATH = os.path.join(DATA_DIR, "txt_embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "metadata.json")
GLYPHNAMES_PATH = os.path.join(ASSETS_DIR, "glyphnames.json")

PREFIX_MAP = {
    "md": "Material Design",
    "fa": "Font Awesome",
    "dev": "Devicons",
    "cod": "Codicons",
    "oct": "GitHub Octicons",
    "weather": "Weather",
    "fae": "Font Awesome Extension",
    "seti": "Seti UI",
    "linux": "Linux Distros",
    "custom": "Custom Tools",
    "pl": "Powerline",
    "ple": "Powerline Extra",
    "pom": "Pomodoro",
    "iec": "Power Symbols",
    "indent": "Indentation",
    "extra": "Extra Glyphs",
}


def load_master_catalog():
    """Loads all 10,000+ Nerd Fonts glyphs directly from official glyphnames.json."""
    if not os.path.exists(GLYPHNAMES_PATH):
        return []

    with open(GLYPHNAMES_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data.pop("METADATA", None)
    catalog = []

    for key, info in raw_data.items():
        parts = key.split("-", 1)
        prefix = parts[0] if len(parts) > 1 else "custom"
        clean_name = parts[1] if len(parts) > 1 else key
        family_name = PREFIX_MAP.get(prefix, prefix.capitalize())

        catalog.append({
            "name": f"nf-{key}",
            "key": key,
            "clean_name": clean_name,
            "char": info["char"],
            "code": info["code"],
            "prefix": prefix,
            "family": family_name,
        })

    return catalog


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


def search_by_name(catalog, query: str, family_filter: str | None, limit: int):
    """Searches catalog by exact or partial name match."""
    q = query.strip().lower()
    matches = []

    for item in catalog:
        if family_filter and family_filter.lower() not in item["prefix"].lower() and family_filter.lower() not in item["family"].lower():
            continue

        raw_k = item["key"].lower()
        clean_n = item["clean_name"].lower()
        code = item["code"].lower()

        if q == raw_k or q == clean_n:
            matches.append((100, item, "Exact Match"))
        elif f"_{q}_" in f"_{clean_n}_" or f"-{q}-" in f"-{raw_k}-":
            matches.append((95, item, "Word Match"))
        elif raw_k.startswith(q) or clean_n.startswith(q):
            matches.append((90, item, "Prefix Match"))
        elif q in raw_k or q in code:
            matches.append((85, item, "Substring Match"))

    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[:limit]


def search_by_semantic_text(query: str, family_filter: str | None, limit: int):
    """Searches icons semantically using CLIP text encoder."""
    if not (os.path.exists(TXT_EMB_PATH) and os.path.exists(META_PATH)):
        print("Error: Embeddings index not found. Run `python setup_assets.py && python indexer.py` first.", file=sys.stderr)
        sys.exit(1)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("clip-ViT-B-32")

    txt_emb = np.load(TXT_EMB_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    prompt = f"an icon symbol of {query.strip()}"
    q_vec = model.encode([prompt], convert_to_numpy=True)[0]
    q_vec = q_vec / np.linalg.norm(q_vec)

    scores = np.dot(txt_emb, q_vec)
    top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices:
        if len(results) >= limit:
            break
        item = meta[idx]
        parts = item["name"].replace("nf-", "").split("-", 1)
        pfx = parts[0] if len(parts) > 1 else ""
        fam = PREFIX_MAP.get(pfx, "Nerd Font")

        if family_filter and family_filter.lower() not in pfx.lower() and family_filter.lower() not in fam.lower():
            continue

        results.append({
            "name": item["name"],
            "char": item["char"],
            "code": item["code"],
            "family": fam,
            "score": round(float(scores[idx]) * 100, 1),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="nerdsight - Visual, Name & Semantic CLI for Nerd Fonts.")
    parser.add_argument("image", nargs="?", default=None, help="Path to input image for visual search")
    parser.add_argument("-n", "--name", type=str, default=None, help="Search icons by exact or partial name")
    parser.add_argument("-t", "--text", type=str, default=None, help="Search icons semantically by concept using CLIP")
    parser.add_argument("--all", action="store_true", help="List all cataloged icons")
    parser.add_argument("-f", "--family", type=str, default=None, help="Filter by collection/prefix (fa, dev, cod, md, etc.)")
    parser.add_argument("-a", "--alpha", type=float, default=0.65, help="Visual vs semantic balance for image search (default: 0.65)")
    parser.add_argument("-k", "--top", type=int, default=12, help="Number of results to return (default: 12)")
    parser.add_argument("-b", "--boost", type=str, default="", help="Optional text hint concept for visual search")
    parser.add_argument("-i", "--invert", action="store_true", help="Invert dark-mode image colors")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    catalog = load_master_catalog()

    # Mode 1: Search by Name
    if args.name is not None:
        matches = search_by_name(catalog, args.name, args.family, args.top)
        results = [
            {
                "rank": idx + 1,
                "name": item["name"],
                "char": item["char"],
                "code": item["code"],
                "family": item["family"],
                "match_type": label,
            }
            for idx, (score, item, label) in enumerate(matches)
        ]

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return

        print(f"\n🔍 nerdsight name search for: '{args.name}'")
        if args.family:
            print(f"📁 Family filter: {args.family}")
        print("-" * 75)
        print(f"{'#':<3} {'Glyph':<6} {'Unicode':<10} {'Collection':<18} {'Match Type':<16} {'Name'}")
        print("-" * 75)
        for r in results:
            print(f"{r['rank']:<3} {r['char']:<6} \\u{r['code']:<8} {r['family']:<18} {r['match_type']:<16} {r['name']}")
        print("-" * 75)
        print(f"Total matching: {len(results)}\n")
        return

    # Mode 2: Semantic Text Search (CLIP)
    if args.text is not None:
        sem_results = search_by_semantic_text(args.text, args.family, args.top)
        results = [
            {
                "rank": idx + 1,
                "name": r["name"],
                "char": r["char"],
                "code": r["code"],
                "family": r["family"],
                "score": r["score"],
            }
            for idx, r in enumerate(sem_results)
        ]

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return

        print(f"\n🧠 nerdsight semantic search for: '{args.text}'")
        if args.family:
            print(f"📁 Family filter: {args.family}")
        print("-" * 75)
        print(f"{'#':<3} {'Glyph':<6} {'Score':<10} {'Unicode':<10} {'Collection':<18} {'Name'}")
        print("-" * 75)
        for r in results:
            print(f"{r['rank']:<3} {r['char']:<6} {r['score']:>5.1f}%     \\u{r['code']:<8} {r['family']:<18} {r['name']}")
        print("-" * 75)
        print(f"Total returned: {len(results)}\n")
        return

    # Mode 3: Show All / Catalog List
    if args.all:
        filtered = catalog
        if args.family:
            filtered = [
                item for item in catalog
                if args.family.lower() in item["prefix"].lower() or args.family.lower() in item["family"].lower()
            ]

        display_items = filtered[:args.top] if args.top > 0 else filtered

        results = [
            {
                "rank": idx + 1,
                "name": item["name"],
                "char": item["char"],
                "code": item["code"],
                "family": item["family"],
            }
            for idx, item in enumerate(display_items)
        ]

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return

        print(f"\n📚 nerdsight catalog: {len(filtered):,} total icons")
        if args.family:
            print(f"📁 Collection filter: {args.family}")
        print(f"Showing first {len(display_items)} icons:")
        print("-" * 65)
        print(f"{'#':<5} {'Glyph':<6} {'Unicode':<10} {'Collection':<18} {'Name'}")
        print("-" * 65)
        for r in results:
            print(f"{r['rank']:<5} {r['char']:<6} \\u{r['code']:<8} {r['family']:<18} {r['name']}")
        print("-" * 65)
        return

    # Mode 4: Visual Search (by Image)
    if args.image is not None:
        if not os.path.exists(args.image):
            print(f"Error: Image '{args.image}' not found.", file=sys.stderr)
            sys.exit(1)

        if not (os.path.exists(IMG_EMB_PATH) and os.path.exists(TXT_EMB_PATH) and os.path.exists(META_PATH)):
            print("Error: Index not built. Run `python setup_assets.py && python indexer.py` first.", file=sys.stderr)
            sys.exit(1)

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("clip-ViT-B-32")

        img_emb = np.load(IMG_EMB_PATH)
        txt_emb = np.load(TXT_EMB_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        proc_img = preprocess_image(args.image, invert=args.invert)
        query_vec = model.encode([proc_img], convert_to_numpy=True)[0]
        query_vec = query_vec / np.linalg.norm(query_vec)

        if args.boost.strip():
            boost_vec = model.encode([f"icon symbol of {args.boost.strip()}"], convert_to_numpy=True)[0]
            boost_vec = boost_vec / np.linalg.norm(boost_vec)
            query_vec = (0.75 * query_vec) + (0.25 * boost_vec)
            query_vec = query_vec / np.linalg.norm(query_vec)

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

        print(f"\n🔍 nerdsight visual search results for: {args.image}")
        print(f"⚖️  Alpha: {args.alpha:.2f} (Visual: {args.alpha*100:.0f}% / Semantic: {(1-args.alpha)*100:.0f}%)")
        if args.boost.strip():
            print(f"💡 Boost Concept: '{args.boost.strip()}'")
        print("-" * 65)
        print(f"{'#':<3} {'Glyph':<6} {'Similarity':<12} {'Unicode':<10} {'Name'}")
        print("-" * 65)

        for res in results:
            print(f"{res['rank']:<3} {res['char']:<6} {res['score']:>5.1f}%      \\u{res['code']:<6}  {res['name']}")

        print("-" * 65)
        return

    # No arguments provided
    parser.print_help()


if __name__ == "__main__":
    main()
