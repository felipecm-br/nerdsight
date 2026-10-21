#!/usr/bin/env python3
"""
nerdsight - core.py
Shared catalog loading, image preprocessing, and hybrid CLIP search logic
used by app.py and cli.py. No Streamlit-specific code here.
"""

import json
import os
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONT_PATH = os.path.join(ASSETS_DIR, "SymbolsNerdFontMono-Regular.ttf")
GLYPHNAMES_PATH = os.path.join(ASSETS_DIR, "glyphnames.json")
IMG_EMB_PATH = os.path.join(DATA_DIR, "img_embeddings.npy")
TXT_EMB_PATH = os.path.join(DATA_DIR, "txt_embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "metadata.json")

# Mapping of prefix codes to friendly collection names
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


def load_master_catalog() -> list[dict]:
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


def has_embeddings_index() -> bool:
    return (
        os.path.exists(IMG_EMB_PATH)
        and os.path.exists(TXT_EMB_PATH)
        and os.path.exists(META_PATH)
    )


def load_embeddings():
    """Loads precomputed visual/textual embedding matrices and their shared metadata."""
    img_emb = np.load(IMG_EMB_PATH)
    txt_emb = np.load(TXT_EMB_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return img_emb, txt_emb, metadata


_model_singleton = None


def get_model(model_name: str = "clip-ViT-B-32"):
    """Lazily loads and caches the SentenceTransformer CLIP model (process-wide singleton)."""
    global _model_singleton
    if _model_singleton is None:
        from sentence_transformers import SentenceTransformer
        _model_singleton = SentenceTransformer(model_name)
    return _model_singleton


def flatten_alpha(image: Image.Image) -> Image.Image:
    """Composites any transparency onto a solid white background and returns RGB."""
    raw = image.convert("RGBA")
    white_bg = Image.new("RGBA", raw.size, (255, 255, 255, 255))
    return Image.alpha_composite(white_bg, raw).convert("RGB")


def preprocess_image(
    image: Image.Image,
    invert_colors: bool = False,
    binarize: bool = True,
    auto_crop: bool = True,
    size: int = 128,
) -> Image.Image:
    """Preprocesses a query image to align with canonical Nerd Font glyph embeddings."""
    # 1. Flatten alpha transparency against solid white background
    flattened = flatten_alpha(image)

    # 2. Inversion for dark-mode screenshots (light icon on dark background)
    if invert_colors:
        flattened = ImageOps.invert(flattened)

    # 3. Grayscale and high-contrast binarization
    gray = flattened.convert("L")
    if binarize:
        gray = ImageOps.autocontrast(gray)
        gray = gray.point(lambda p: 0 if p < 185 else 255, "L")
        processed = gray.convert("RGB")
    else:
        processed = flattened

    # 4. Auto-crop empty margins and center with canonical proportional padding
    if auto_crop:
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


def search_by_name(catalog: list[dict], query: str, family_filter: Optional[str], limit: int):
    """Searches catalog by exact or partial name match. Returns [(score, item, label), ...]."""
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


def encode_text_query(model, query: str) -> np.ndarray:
    """Encodes a free-text concept into a normalized CLIP embedding for semantic search."""
    prompt = f"an icon symbol of {query.strip()}"
    vec = model.encode([prompt], convert_to_numpy=True)[0]
    return vec / np.linalg.norm(vec)


def search_by_semantic_text(model, txt_emb: np.ndarray, query: str, top_k: Optional[int] = None):
    """Ranks all indexed icons by cosine similarity to a free-text concept. Returns sorted indices + scores."""
    q_vec = encode_text_query(model, query)
    scores = np.dot(txt_emb, q_vec)
    order = np.argsort(scores)[::-1]
    if top_k is not None:
        order = order[:top_k]
    return order, scores[order]


DEFAULT_ALPHA = 0.7


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def hybrid_scores(query_vec: np.ndarray, img_emb: np.ndarray, txt_emb: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Fuses visual (image->image) and semantic (image->text) cosine scores over the whole index.

    The two modalities live on very different scales (image->image cosines ~0.6-0.95,
    image->text ~0.15-0.35), so each is z-scored across the catalog before weighting:
      fused = alpha * z(vis) + (1 - alpha) * z(sem)
    Z-scores are used instead of min-max because a single outlier (the catalog
    max/min) dictates the min-max range and squashes everything else. The fused
    score is then min-max scaled to [0, 1] purely for display; 1.0 = best in catalog.
    """
    vis_scores = img_emb @ query_vec
    sem_scores = txt_emb @ query_vec
    fused = alpha * _zscore(vis_scores) + (1.0 - alpha) * _zscore(sem_scores)
    return (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)


def search_hybrid_visual(
    query_vec: np.ndarray,
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    top_k: int = 12,
):
    """Ranks the index for an image query by hybrid_scores(). Returns (top indices, their scores)."""
    final_scores = hybrid_scores(query_vec, img_emb, txt_emb, alpha)
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    return top_indices, final_scores[top_indices]


def encode_image_query(model, image: Image.Image, boost_text: str = "") -> np.ndarray:
    """Encodes a preprocessed query image, optionally blended with a semantic text boost."""
    vec = model.encode([image], convert_to_numpy=True)[0]
    vec = vec / np.linalg.norm(vec)

    if boost_text.strip():
        boost_vec = model.encode([f"icon symbol of {boost_text.strip()}"], convert_to_numpy=True)[0]
        boost_vec = boost_vec / np.linalg.norm(boost_vec)
        vec = (0.75 * vec) + (0.25 * boost_vec)
        vec = vec / np.linalg.norm(vec)

    return vec
