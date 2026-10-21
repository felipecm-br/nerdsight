#!/usr/bin/env python3
"""
nerdsight - cli.py
Command-line search tool for Nerd Fonts visual, name & semantic matching.
Usage:
    # 1. Visual Search (by image or screenshot)
    python cli.py image.png [--alpha 0.7] [--top 10] [--boost "git"]

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

import core

PREFIX_MAP = core.PREFIX_MAP


def main():
    parser = argparse.ArgumentParser(description="nerdsight - Visual, Name & Semantic CLI for Nerd Fonts.")
    parser.add_argument("image", nargs="?", default=None, help="Path to input image for visual search")
    parser.add_argument("-n", "--name", type=str, default=None, help="Search icons by exact or partial name")
    parser.add_argument("-t", "--text", type=str, default=None, help="Search icons semantically by concept using CLIP")
    parser.add_argument("--all", action="store_true", help="List all cataloged icons")
    parser.add_argument("-f", "--family", type=str, default=None, help="Filter by collection/prefix (fa, dev, cod, md, etc.)")
    parser.add_argument("-a", "--alpha", type=float, default=core.DEFAULT_ALPHA, help=f"Visual vs semantic balance for image search (default: {core.DEFAULT_ALPHA})")
    parser.add_argument("-k", "--top", type=int, default=12, help="Number of results to return (default: 12)")
    parser.add_argument("-b", "--boost", type=str, default="", help="Optional text hint concept for visual search")
    parser.add_argument("-i", "--invert", action="store_true", help="Invert dark-mode image colors")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    catalog = core.load_master_catalog()

    # Mode 1: Search by Name
    if args.name is not None:
        matches = core.search_by_name(catalog, args.name, args.family, args.top)
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
        if not core.has_embeddings_index():
            print("Error: Embeddings index not found. Run `python setup_assets.py && python indexer.py` first.", file=sys.stderr)
            sys.exit(1)

        model = core.get_model()
        _, txt_emb, meta = core.load_embeddings()
        order, scores = core.search_by_semantic_text(model, txt_emb, args.text)

        sem_results = []
        for idx, score in zip(order, scores):
            if len(sem_results) >= args.top:
                break
            item = meta[idx]
            parts = item["name"].replace("nf-", "").split("-", 1)
            pfx = parts[0] if len(parts) > 1 else ""
            fam = PREFIX_MAP.get(pfx, "Nerd Font")

            if args.family and args.family.lower() not in pfx.lower() and args.family.lower() not in fam.lower():
                continue

            sem_results.append({
                "name": item["name"],
                "char": item["char"],
                "code": item["code"],
                "family": fam,
                "score": round(float(score) * 100, 1),
            })

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

        if not core.has_embeddings_index():
            print("Error: Index not built. Run `python setup_assets.py && python indexer.py` first.", file=sys.stderr)
            sys.exit(1)

        from PIL import Image

        model = core.get_model()
        img_emb, txt_emb, metadata = core.load_embeddings()

        raw_img = Image.open(args.image)
        proc_img = core.preprocess_image(raw_img, invert_colors=args.invert)
        query_vec = core.encode_image_query(model, proc_img, boost_text=args.boost)

        top_indices, top_scores = core.search_hybrid_visual(
            query_vec, img_emb=img_emb, txt_emb=txt_emb, alpha=args.alpha, top_k=args.top
        )
        results = []
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
            item = metadata[idx]
            results.append({
                "rank": rank,
                "name": item["name"],
                "char": item["char"],
                "code": item["code"],
                "score": round(float(score) * 100, 2),
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
