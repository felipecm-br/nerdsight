#!/usr/bin/env python3
"""
nerdsight - app.py
Interactive Streamlit Web Application for Visual & Semantic Nerd Fonts Search.
Powered by Hybrid CLIP Multimodal Embeddings.
"""

import base64
import json
import os
import sys
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from sentence_transformers import SentenceTransformer

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

# Page Configuration
st.set_page_config(
    page_title="nerdsight - Visual & Semantic Nerd Fonts Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"
ASSETS_DIR = "assets"
FONT_PATH = os.path.join(ASSETS_DIR, "SymbolsNerdFontMono-Regular.ttf")
IMG_EMB_PATH = os.path.join(DATA_DIR, "img_embeddings.npy")
TXT_EMB_PATH = os.path.join(DATA_DIR, "txt_embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "metadata.json")


@st.cache_resource
def load_resources():
    """Loads CLIP model, precomputed embeddings, metadata, and font file into memory cache."""
    if not (os.path.exists(IMG_EMB_PATH) and os.path.exists(TXT_EMB_PATH) and os.path.exists(META_PATH)):
        return None, None, None, None, None

    model = SentenceTransformer("clip-ViT-B-32")
    img_emb = np.load(IMG_EMB_PATH)
    txt_emb = np.load(TXT_EMB_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    font_b64 = None
    if os.path.exists(FONT_PATH):
        with open(FONT_PATH, "rb") as f:
            font_b64 = base64.b64encode(f.read()).decode("utf-8")

    return model, img_emb, txt_emb, metadata, font_b64


def inject_custom_css(font_b64: str | None):
    """Injects custom CSS styling and embedded Nerd Font for native client-side glyph rendering."""
    font_face_rule = ""
    if font_b64:
        font_face_rule = f"""
        @font-face {{
            font-family: 'NerdSymbolsEmbedded';
            src: url(data:font/ttf;base64,{font_b64}) format('truetype');
            font-weight: normal;
            font-style: normal;
        }}
        """

    st.markdown(f"""
    <style>
    {font_face_rule}
    .nf-glyph {{
        font-family: 'NerdSymbolsEmbedded', 'Symbols Nerd Font Mono', 'JetBrainsMono Nerd Font', monospace;
        font-size: 58px;
        line-height: 1.1;
        text-align: center;
        margin: 14px 0 8px 0;
        color: #58a6ff;
        display: block;
        transition: transform 0.15s ease-in-out;
    }}
    .nf-glyph:hover {{
        transform: scale(1.15);
    }}
    .icon-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }}
    .score-badge {{
        background: #1f6feb22;
        border: 1px solid #388bfd44;
        color: #58a6ff;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .icon-title {{
        font-size: 14px;
        font-weight: 600;
        color: #c9d1d9;
        margin-top: 8px;
        word-break: break-all;
    }}
    .unicode-tag {{
        font-family: monospace;
        color: #8b949e;
        font-size: 12px;
    }}
    </style>
    """, unsafe_allow_html=True)


def preprocess_image(
    image: Image.Image,
    invert_colors: bool = False,
    binarize: bool = True,
    auto_crop: bool = True,
    size: int = 128,
) -> Image.Image:
    """Preprocesses user images to align with canonical Nerd Font glyph embeddings."""
    raw = image.convert("RGBA")

    # 1. Flatten alpha transparency against solid white background
    white_bg = Image.new("RGBA", raw.size, (255, 255, 255, 255))
    flattened = Image.alpha_composite(white_bg, raw).convert("RGB")

    # 2. Inversion for dark-mode screenshots (light icon on dark background)
    if invert_colors:
        flattened = ImageOps.invert(flattened)

    # 3. Grayscale and high-contrast binarization
    gray = flattened.convert("L")
    if binarize:
        gray = ImageOps.autocontrast(gray)
        # Threshold: darker pixels become pure black, background pure white
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


def search_hybrid(
    query_vec: np.ndarray,
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    alpha: float,
    top_k: int = 12,
):
    """Executes linear combination of visual and semantic indices, returning Top K matches."""
    # V_hybrid = alpha * V_img + (1 - alpha) * V_txt
    hybrid_matrix = (alpha * img_emb) + ((1.0 - alpha) * txt_emb)
    norms = np.linalg.norm(hybrid_matrix, axis=1, keepdims=True)
    hybrid_matrix = hybrid_matrix / np.maximum(norms, 1e-12)

    # Cosine similarity via dot product of normalized vectors
    scores = np.dot(hybrid_matrix, query_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


def main():
    model, img_emb, txt_emb, metadata, font_b64 = load_resources()

    if model is None:
        st.error("⚠️ Embeddings or assets not found!")
        st.info("Please initialize the index by running the following commands in your terminal:")
        st.code("python setup_assets.py\npython indexer.py", language="bash")
        st.stop()

    inject_custom_css(font_b64)

    st.title("🔍 nerdsight")
    st.caption("Visual & Semantic search engine for **Nerd Fonts** icons powered by multimodal CLIP embeddings.")

    # Sidebar: Hyperparameters & Preprocessing Controls
    with st.sidebar:
        st.header("⚙️ Search Engine Tuning")
        alpha = st.slider(
            "Visual vs. Semantic Balance (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
            help="1.0 = Pure visual geometry/shape matching.\n0.0 = Pure semantic meaning/concept.\n0.65 = Recommended golden ratio.",
        )

        st.divider()
        st.header("🛠️ Image Normalization")
        auto_crop = st.checkbox("Auto-Crop & Center Glyph", value=True, help="Trims blank border padding.")
        binarize = st.checkbox("High-Contrast Silhouette", value=True, help="Converts colored graphics to clean black-and-white silhouettes.")
        invert_colors = st.checkbox("Invert Colors (Dark Terminal Screenshot)", value=False, help="Inverts white-on-black terminal prints to black-on-white.")
        top_k = st.selectbox("Number of Results", [8, 12, 16, 24, 32, 48], index=1)

        st.divider()
        text_boost = st.text_input(
            "Optional Concept Hint (Text Boost)",
            placeholder="e.g. git, folder, database, rust",
            help="Subtly blends an extra keyword concept into the image query vector.",
        )

    # Input modes: File Upload or Canvas Draw
    tabs = ["📤 Upload Image", "✏️ Draw Icon (Canvas)"] if HAS_CANVAS else ["📤 Upload Image"]
    active_tabs = st.tabs(tabs)

    query_pil = None

    with active_tabs[0]:
        uploaded_file = st.file_uploader(
            "Drop or upload any image (PNG, JPG, SVG export, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
        )
        if uploaded_file:
            query_pil = Image.open(uploaded_file)

    if HAS_CANVAS and len(active_tabs) > 1:
        with active_tabs[1]:
            st.write("Draw a sketch or rough silhouette below:")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=6,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=200,
                width=200,
                drawing_mode="freedraw",
                key="canvas",
            )
            if canvas_result.image_data is not None:
                # Check if user actually drew something
                arr = canvas_result.image_data
                if np.any(arr[:, :, :3] < 240):
                    query_pil = Image.fromarray(arr.astype("uint8"), "RGBA")

    if query_pil is None:
        st.info("👆 Upload an image or screenshot above to find visually and semantically matching Nerd Font glyphs.")
        return

    # Pipeline: Preprocessing
    processed_img = preprocess_image(
        query_pil,
        invert_colors=invert_colors,
        binarize=binarize,
        auto_crop=auto_crop,
    )

    col_orig, col_proc, col_stats = st.columns([1, 1, 2])
    with col_orig:
        st.markdown("**Original Query:**")
        st.image(query_pil, width=128)
    with col_proc:
        st.markdown("**Normalized Input (CLIP):**")
        st.image(processed_img, width=128)
    with col_stats:
        st.markdown("**Search Mode:**")
        st.markdown(f"- Visual Weight: **{alpha * 100:.0f}%** | Semantic Weight: **{(1 - alpha) * 100:.0f}%**")
        if text_boost.strip():
            st.markdown(f"- Text Concept Hint: *'{text_boost.strip()}'*")

    # Generate Query Embedding
    query_vec = model.encode([processed_img], convert_to_numpy=True)[0]
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Optional Concept Text Boost
    if text_boost.strip():
        boost_prompt = f"icon symbol of {text_boost.strip()}"
        boost_vec = model.encode([boost_prompt], convert_to_numpy=True)[0]
        boost_vec = boost_vec / np.linalg.norm(boost_vec)
        query_vec = (0.75 * query_vec) + (0.25 * boost_vec)
        query_vec = query_vec / np.linalg.norm(query_vec)

    # Perform Search
    top_indices, top_scores = search_hybrid(
        query_vec,
        img_emb=img_emb,
        txt_emb=txt_emb,
        alpha=alpha,
        top_k=top_k,
    )

    st.markdown("---")
    st.subheader(f"🎯 Top {top_k} Matching Icons")

    num_cols = 4
    cols = st.columns(num_cols)
    for idx_pos, (icon_idx, score) in enumerate(zip(top_indices, top_scores)):
        item = metadata[icon_idx]
        col = cols[idx_pos % num_cols]
        with col:
            st.markdown(f"""
            <div class="icon-card">
                <span class="score-badge">{score * 100:.1f}% match</span>
                <span class="nf-glyph">{item['char']}</span>
                <div class="icon-title">{item['name']}</div>
                <div class="unicode-tag">\\u{item['code']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.code(f"{item['char']} {item['name']}", language="text")


if __name__ == "__main__":
    main()
