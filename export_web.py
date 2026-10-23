#!/usr/bin/env python3
"""
nerdsight - export_web.py
Exports the catalog, the CLIP index and the symbols font as static files for
the browser front end in web/ (Hugging Face Static Space). The browser encodes
queries itself with transformers.js, so nothing here runs at request time.

    python export_web.py            # writes web/dist/data and web/dist/fonts
    python export_web.py --golden   # also writes web/tests/golden/ for the parity tests
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tomllib
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import core
import i18n

WEB_DIR = os.path.join(core.BASE_DIR, "web")
DIST_DIR = os.path.join(WEB_DIR, "dist")
GOLDEN_DIR = os.path.join(WEB_DIR, "tests", "golden")
STREAMLIT_CONFIG = os.path.join(core.BASE_DIR, ".streamlit", "config.toml")
# Front-end sources copied verbatim into dist/ next to the generated data.
WEB_SOURCES = ["index.html", "style.css", "js"]
SHARED_UI = ["nerdsight.css", "nerdsight.js"]

SPACE_README = """---
title: nerdsight
colorFrom: yellow
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Find Nerd Fonts glyphs by name, concept or image
---

# nerdsight

Visual, name and semantic search over the {count:,} [Nerd Fonts](https://www.nerdfonts.com) glyphs.
CLIP ({model}) runs in the browser with transformers.js; nothing is sent to a server.

Source, CLI and the Streamlit app: https://github.com/felipecm-br/nerdsight
"""

# Browser checkpoint: the ONNX export of the same OpenAI CLIP ViT-B/32 weights
# that sentence-transformers' clip-ViT-B-32 wraps. Dtypes were chosen with the
# parity tests (web/tests/parity.test.mjs, NERDSIGHT_DTYPE=vision:text):
#   fp32  cosine 1.0000 vs PyTorch, identical rankings, ~580 MB first download
#   fp16  cosine >= 0.997, only near-tie reorderings,     ~290 MB
#   q8    text cosine 0.84-0.89, concept rankings broken  (rejected)
WEB_MODEL = "Xenova/clip-vit-base-patch32"
WEB_DTYPES = {"vision": "fp16", "text": "fp16"}

QUERY_TEMPLATE = "an icon symbol of {q}"   # core.encode_text_query
BOOST_TEMPLATE = "icon symbol of {q}"      # core.encode_image_query
BOOST_WEIGHT = 0.75

TEXT_QUERIES = ["ghost", "database", "git branch", "bell outline", "rocket", "battery low"]
NAME_QUERIES = [("python", None), ("git", None), ("folder", None), ("f1c0", None), ("arrow_left", None),
                ("git", "Devicons"), ("heart", "Font Awesome")]
GLYPH_SAMPLE = 300


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj, indent=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"))


def build_rows(catalog, metadata):
    """Maps every catalog entry to its embedding row, or -1 when it was not indexed."""
    row_of = {m["name"]: i for i, m in enumerate(metadata)}
    return [row_of.get(item["name"], -1) for item in catalog]


def export(catalog, img_emb, txt_emb, metadata) -> dict:
    data_dir = os.path.join(DIST_DIR, "data")
    fonts_dir = os.path.join(DIST_DIR, "fonts")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fonts_dir, exist_ok=True)

    rows = build_rows(catalog, metadata)
    indexed = [r for r in rows if r >= 0]
    # Embedding rows follow catalog order, so the browser maps row -> entry directly.
    order = np.array(indexed, dtype=np.int64)
    new_row = {old: new for new, old in enumerate(indexed)}

    families = sorted({item["family"] for item in catalog})
    family_id = {f: i for i, f in enumerate(families)}
    entries = [[item["key"], item["char"], item["code"], family_id[item["family"]], new_row.get(r, -1)]
               for item, r in zip(catalog, rows)]
    with open(core.GLYPHNAMES_PATH, encoding="utf-8") as f:
        nf_version = json.load(f).get("METADATA", {}).get("version", "")
    write_json(os.path.join(data_dir, "catalog.json"), {
        "fields": ["key", "char", "code", "family", "row"],
        "families": families,
        "prefixes": core.PREFIX_MAP,
        "nerd_fonts_version": nf_version,
        "glyphs": entries,
    })

    bins = {}
    for name, matrix in (("img_emb.f16.bin", img_emb), ("txt_emb.f16.bin", txt_emb)):
        path = os.path.join(data_dir, name)
        matrix[order].astype("<f2").tofile(path)
        bins[name] = sha256(path)

    proc = core.get_model()[0].processor.image_processor
    manifest = {
        "model": WEB_MODEL,
        "dtype": WEB_DTYPES,
        "dim": int(img_emb.shape[1]),
        "count": len(indexed),
        "alpha": core.DEFAULT_ALPHA,
        "query_template": QUERY_TEMPLATE,
        "boost_template": BOOST_TEMPLATE,
        "boost_weight": BOOST_WEIGHT,
        "preprocess": {"size": 128, "cropBox": 0.72, "threshold": 185, "bboxThreshold": 30, "darkBorder": 96},
        "clip_image": {"size": int(proc.crop_size["height"]), "mean": proc.image_mean, "std": proc.image_std},
        "files": {"catalog.json": sha256(os.path.join(data_dir, "catalog.json")), **bins},
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(os.path.join(data_dir, "manifest.json"), manifest, indent=2)

    shutil.copyfile(core.FONT_PATH, os.path.join(fonts_dir, os.path.basename(core.FONT_PATH)))
    write_json(os.path.join(data_dir, "i18n.json"),
               {"default": i18n.DEFAULT_LANG, "languages": i18n.LANGUAGES, "strings": i18n.STRINGS})
    write_theme()
    copy_sources()
    with open(os.path.join(DIST_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(SPACE_README.format(count=len(catalog), model=WEB_MODEL))
    return {"rows": rows, "order": order}


def write_theme():
    """theme.css: the colors app.py injects from .streamlit/config.toml, so both UIs share one theme."""
    with open(STREAMLIT_CONFIG, "rb") as f:
        theme = tomllib.load(f)["theme"]
    tokens = {
        "--ns-bg": theme["backgroundColor"],
        "--ns-surface": theme["secondaryBackgroundColor"],
        "--ns-border": theme["borderColor"],
        "--ns-text": theme["textColor"],
        "--ns-accent": theme["primaryColor"],
        "--ns-font": "'Geist', 'NerdSymbols', system-ui, sans-serif",
    }
    face = theme["fontFaces"][0]
    css = (
        "/* Generated by export_web.py from .streamlit/config.toml. Do not edit. */\n"
        f"@font-face {{ font-family: '{face['family']}'; src: url('fonts/{os.path.basename(face['url'])}') format('truetype'); "
        "font-display: block; }\n"
        ":root {\n" + "".join(f"  {k}: {v};\n" for k, v in tokens.items()) + "}\n"
    )
    with open(os.path.join(DIST_DIR, "theme.css"), "w", encoding="utf-8") as f:
        f.write(css)


def copy_sources():
    for name in WEB_SOURCES:
        src, dst = os.path.join(WEB_DIR, name), os.path.join(DIST_DIR, name)
        if os.path.isdir(src):
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        else:
            shutil.copyfile(src, dst)
    os.makedirs(os.path.join(DIST_DIR, "ui"), exist_ok=True)
    for name in SHARED_UI:
        shutil.copyfile(os.path.join(core.BASE_DIR, "ui", name), os.path.join(DIST_DIR, "ui", name))


# ------------------------------------------------------------------ golden


def _glyph(char: str, px: int, fg, bg, canvas, offset=(0, 0), mode="RGB") -> Image.Image:
    font = ImageFont.truetype(core.FONT_PATH, px)
    img = Image.new(mode, canvas, bg)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    x = (canvas[0] - (bbox[2] - bbox[0])) / 2 - bbox[0] + offset[0]
    y = (canvas[1] - (bbox[3] - bbox[1])) / 2 - bbox[1] + offset[1]
    draw.text((x, y), char, fill=fg, font=font)
    return img


def make_fixtures(by_key) -> list[dict]:
    """Synthetic query images covering each preprocessing branch (invert, alpha, reduce, no-upscale, pad)."""
    rng = np.random.default_rng(7)
    ch = lambda key: by_key[key]["char"]
    noise = Image.fromarray(rng.integers(215, 255, (480, 640), dtype=np.uint8), "L").convert("RGB")
    db = _glyph(ch("fa-database"), 300, (20, 20, 20, 255), (0, 0, 0, 0), (640, 480), (60, -20), "RGBA")
    noise.paste(db, (0, 0), db)

    return [
        {"file": "ghost_terminal.png", "expect": "md-ghost_outline", "auto_crop": True,
         "image": _glyph(ch("md-ghost_outline"), 96, (205, 214, 244), (30, 30, 46), (260, 180), (-40, 10))
            .filter(ImageFilter.GaussianBlur(0.8))},
        {"file": "git_transparent.png", "expect": "dev-git", "auto_crop": True,
         "image": _glyph(ch("dev-git"), 560, (240, 80, 50, 255), (0, 0, 0, 0), (800, 800), mode="RGBA")},
        {"file": "database_noise.png", "expect": "fa-database", "auto_crop": True, "image": noise},
        {"file": "bell_tiny.png", "expect": "md-bell_outline", "auto_crop": True,
         "image": _glyph(ch("md-bell_outline"), 24, (0, 0, 0), (255, 255, 255), (32, 32))},
        {"file": "heart_wide_pad.png", "expect": "fa-heart", "auto_crop": False,
         "image": _glyph(ch("fa-heart"), 100, (200, 30, 60), (255, 255, 255), (300, 120))},
    ]


def golden(catalog, img_emb, txt_emb, metadata, export_info):
    import indexer

    os.makedirs(os.path.join(GOLDEN_DIR, "glyphs"), exist_ok=True)
    model = core.get_model()
    names = [m["name"] for m in metadata]
    by_key = {item["key"]: item for item in catalog}
    names_of = lambda idx: [names[i] for i in idx]

    out = {"text": [], "name": [], "images": [], "glyphs": []}

    for q in TEXT_QUERIES:
        order, _ = core.search_by_semantic_text(model, txt_emb, q, top_k=10)
        out["text"].append({"q": q, "top": names_of(order), "vec": core.encode_text_query(model, q).round(6).tolist()})

    for q, family in NAME_QUERIES:
        pool = catalog if family is None else [i for i in catalog if i["family"] == family]
        matches = core.search_by_name(pool, q, None, len(pool))
        out["name"].append({"q": q, "family": family, "total": len(matches),
                            "top": [[m[1]["name"], m[2]] for m in matches[:24]]})

    for fx in make_fixtures(by_key):
        img = fx.pop("image")
        img.save(os.path.join(GOLDEN_DIR, fx["file"]))
        src = Image.open(os.path.join(GOLDEN_DIR, fx["file"]))
        invert = core.has_dark_border(src)
        processed = core.preprocess_image(src, invert_colors=invert, auto_crop=fx["auto_crop"])
        proc_file = fx["file"].replace(".png", ".proc.png")
        processed.save(os.path.join(GOLDEN_DIR, proc_file))
        vec = core.encode_image_query(model, processed)
        top = np.argsort(core.hybrid_scores(vec, img_emb, txt_emb))[::-1][:10]
        boosted = core.encode_image_query(model, processed, boost_text=fx["expect"].split("-", 1)[1].replace("_", " "))
        top_b = np.argsort(core.hybrid_scores(boosted, img_emb, txt_emb))[::-1][:10]
        out["images"].append({**fx, "invert": invert, "processed": proc_file, "top": names_of(top),
                              "boost": fx["expect"].split("-", 1)[1].replace("_", " "), "top_boost": names_of(top_b),
                              "vec": vec.round(6).tolist()})

    font = ImageFont.truetype(core.FONT_PATH, int(128 * 0.72))
    for i in np.linspace(0, len(metadata) - 1, GLYPH_SAMPLE).astype(int):
        file = f"glyphs/{i}.png"
        indexer.render_glyph(metadata[i]["char"], font).save(os.path.join(GOLDEN_DIR, file))
        out["glyphs"].append({"name": metadata[i]["name"], "file": file})

    write_json(os.path.join(GOLDEN_DIR, "golden.json"), out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the nerdsight index for the static web front end.")
    parser.add_argument("--golden", action="store_true", help="Also write web/tests/golden/ (needs the CLIP model)")
    args = parser.parse_args(argv)

    if not core.has_embeddings_index() or not os.path.exists(core.FONT_PATH):
        print("Missing index or font. Run `python setup_assets.py` and `python indexer.py` first.", file=sys.stderr)
        sys.exit(1)

    catalog = core.load_master_catalog()
    img_emb, txt_emb, metadata = core.load_embeddings()
    info = export(catalog, img_emb, txt_emb, metadata)
    print(f"Exported {len(catalog)} glyphs, {len(info['order'])} embeddings to {DIST_DIR}")

    if args.golden:
        golden(catalog, img_emb, txt_emb, metadata, info)
        print(f"Golden set written to {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
