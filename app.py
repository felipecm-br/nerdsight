#!/usr/bin/env python3
"""
nerdsight - app.py
Interactive Streamlit Web Application for Visual, Name & Semantic Nerd Fonts Search.
Powered by Hybrid CLIP Multimodal Embeddings and Official Glyph Catalog.
"""

import base64
import json
import math
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
    page_title="nerdsight - Nerd Fonts Visual & Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


@st.cache_data
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


@st.cache_resource
def load_model_and_embeddings():
    """Loads CLIP model, precomputed embeddings, metadata, and font file into memory cache."""
    has_embeddings = (
        os.path.exists(IMG_EMB_PATH)
        and os.path.exists(TXT_EMB_PATH)
        and os.path.exists(META_PATH)
    )

    model = None
    img_emb = None
    txt_emb = None
    emb_metadata = []

    if has_embeddings:
        try:
            model = SentenceTransformer("clip-ViT-B-32")
            img_emb = np.load(IMG_EMB_PATH)
            txt_emb = np.load(TXT_EMB_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                emb_metadata = json.load(f)
        except Exception as exc:
            st.warning(f"Não foi possível carregar os embeddings pré-calculados: {exc}")

    font_b64 = None
    if os.path.exists(FONT_PATH):
        try:
            with open(FONT_PATH, "rb") as f:
                font_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass

    return model, img_emb, txt_emb, emb_metadata, font_b64


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
        font-size: 56px;
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
    .badge-exact {{
        background: #23863622;
        border: 1px solid #2ea04366;
        color: #3fb950;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .badge-partial {{
        background: #8957e522;
        border: 1px solid #a371f766;
        color: #bc8cff;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .badge-semantic {{
        background: #1f6feb22;
        border: 1px solid #388bfd44;
        color: #58a6ff;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .badge-family {{
        background: #21262d;
        border: 1px solid #30363d;
        color: #8b949e;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
    }}
    .icon-title {{
        font-size: 13px;
        font-weight: 600;
        color: #c9d1d9;
        margin-top: 8px;
        word-break: break-all;
    }}
    .unicode-tag {{
        font-family: monospace;
        color: #8b949e;
        font-size: 11px;
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


def search_hybrid_visual(
    query_vec: np.ndarray,
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    alpha: float,
    top_k: int = 12,
):
    """Executes linear combination of visual and semantic indices for image search."""
    hybrid_matrix = (alpha * img_emb) + ((1.0 - alpha) * txt_emb)
    norms = np.linalg.norm(hybrid_matrix, axis=1, keepdims=True)
    hybrid_matrix = hybrid_matrix / np.maximum(norms, 1e-12)

    scores = np.dot(hybrid_matrix, query_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


def render_icon_card(col, item: dict, badge_html: str):
    """Renders a single standardized icon card inside a Streamlit column."""
    with col:
        st.markdown(f"""
        <div class="icon-card">
            {badge_html}
            <span class="nf-glyph">{item['char']}</span>
            <div class="icon-title">{item['name']}</div>
            <div class="unicode-tag">\\u{item['code']} &bull; {item.get('family', '')}</div>
        </div>
        """, unsafe_allow_html=True)
        st.code(f"{item['char']} {item['name']}", language="text")


def main():
    catalog = load_master_catalog()
    model, img_emb, txt_emb, emb_metadata, font_b64 = load_model_and_embeddings()

    inject_custom_css(font_b64)

    st.title("🔍 nerdsight")
    st.caption("Mecanismo inteligente de busca **Visual**, por **Nome** e **Semântica** para ícones do **Nerd Fonts**.")

    # Unique list of families with counts
    family_counts = {}
    for item in catalog:
        fam = item["family"]
        family_counts[fam] = family_counts.get(fam, 0) + 1

    sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)
    family_options = ["Todas as Famílias"] + [f"{fam} ({count})" for fam, count in sorted_families]

    # Main Tabs Navigation
    tab_text, tab_visual, tab_all = st.tabs([
        "🔤 Busca por Nome & Semântica",
        "🖼️ Busca Visual (Imagem / Desenho)",
        "📚 Catálogo Completo (Mostrar Todos)",
    ])

    # =========================================================================
    # TAB 1: BUSCA POR NOME & PALAVRAS SIMILARES
    # =========================================================================
    with tab_text:
        st.subheader("🔤 Encontrar Ícones por Nome ou Palavras Similares")
        st.markdown("Digite o nome exato (ex: `git`, `folder`, `python`), ou descreva o conceito (ex: `raposa`, `banco de dados`, `nuvem`, `velocímetro`).")

        col_input, col_mode, col_fam = st.columns([3, 2, 2])
        with col_input:
            text_query = st.text_input(
                "Digite sua busca:",
                placeholder="Ex: git, folder, database, python, warning, music...",
                key="text_search_query",
            )
        with col_mode:
            search_mode = st.selectbox(
                "Modo de Correspondência:",
                [
                    "🧠 Híbrido (Nome + Similaridade por IA)",
                    "🔤 Somente Nome (Substring exata/parcial)",
                    "✨ Somente Semântica (Conceito por IA)",
                ],
                index=0,
                help="Híbrido traz primeiro os ícones cujo nome bate com a palavra, complementados por ícones com significado similar encontrados pelo modelo CLIP.",
            )
        with col_fam:
            selected_family_raw = st.selectbox(
                "Filtrar por Família:",
                family_options,
                key="text_search_family",
            )
            filter_family = None if selected_family_raw == "Todas as Famílias" else selected_family_raw.split(" (")[0]

        col_top, col_spacer = st.columns([2, 5])
        with col_top:
            top_k_text = st.selectbox("Quantidade de resultados:", [12, 24, 48, 96, 120], index=1, key="text_top_k")

        if text_query.strip():
            query_clean = text_query.strip().lower()

            results = []
            seen_names = set()

            # 1. Name Match across ALL 10,000+ icons in master catalog
            name_matches = []
            if search_mode in ["🧠 Híbrido (Nome + Similaridade por IA)", "🔤 Somente Nome (Substring exata/parcial)"]:
                for item in catalog:
                    if filter_family and item["family"] != filter_family:
                        continue

                    raw_k = item["key"].lower()
                    clean_n = item["clean_name"].lower()
                    code = item["code"].lower()

                    if query_clean == raw_k or query_clean == clean_n:
                        name_matches.append((100, item, '<span class="badge-exact">Nome Exato (100%)</span>'))
                    elif f"_{query_clean}_" in f"_{clean_n}_" or f"-{query_clean}-" in f"-{raw_k}-":
                        name_matches.append((95, item, '<span class="badge-exact">Palavra Exata (95%)</span>'))
                    elif raw_k.startswith(query_clean) or clean_n.startswith(query_clean):
                        name_matches.append((90, item, '<span class="badge-partial">Inicia com termo</span>'))
                    elif query_clean in raw_k or query_clean in code:
                        name_matches.append((85, item, '<span class="badge-partial">Substring no nome</span>'))

                name_matches.sort(key=lambda x: x[0], reverse=True)
                for score, item, badge in name_matches:
                    if item["name"] not in seen_names:
                        results.append((item, badge))
                        seen_names.add(item["name"])

            # 2. Semantic Match via CLIP text embeddings
            if search_mode in ["🧠 Híbrido (Nome + Similaridade por IA)", "✨ Somente Semântica (Conceito por IA)"]:
                if model is not None and txt_emb is not None and len(emb_metadata) > 0:
                    prompt = f"an icon symbol of {query_clean}"
                    q_vec = model.encode([prompt], convert_to_numpy=True)[0]
                    q_vec = q_vec / np.linalg.norm(q_vec)

                    scores = np.dot(txt_emb, q_vec)
                    top_sem_indices = np.argsort(scores)[::-1]

                    for idx in top_sem_indices:
                        if len(results) >= top_k_text * 2:
                            break
                        item_meta = emb_metadata[idx]
                        name = item_meta["name"]
                        if name in seen_names:
                            continue

                        # Find matching entry in master catalog for full details
                        cat_item = next((c for c in catalog if c["name"] == name), None)
                        if not cat_item:
                            cat_item = {
                                "name": name,
                                "char": item_meta["char"],
                                "code": item_meta["code"],
                                "family": "Nerd Font",
                            }

                        if filter_family and cat_item.get("family") != filter_family:
                            continue

                        sim = float(scores[idx]) * 100
                        badge = f'<span class="badge-semantic">Similaridade: {sim:.1f}%</span>'
                        results.append((cat_item, badge))
                        seen_names.add(name)
                elif search_mode == "✨ Somente Semântica (Conceito por IA)":
                    st.info("💡 Embeddings do modelo CLIP não encontrados para busca puramente semântica. Execute `python indexer.py` para ativá-la.")

            # Limit to chosen Top K
            display_results = results[:top_k_text]

            st.markdown(f"**Encontrados {len(results)} ícones correspondentes** (exibindo os {len(display_results)} melhores):")

            if display_results:
                num_cols = 4
                cols = st.columns(num_cols)
                for idx, (item, badge) in enumerate(display_results):
                    render_icon_card(cols[idx % num_cols], item, badge)
            else:
                st.warning(f"Nenhum ícone encontrado para '{text_query}'. Tente outro termo ou limpe os filtros.")
        else:
            st.info("👆 Digite um termo de busca acima para começar (ex: `folder`, `git`, `python`, `heart`, `cloud`, `gear`).")

    # =========================================================================
    # TAB 2: BUSCA VISUAL (IMAGEM / DESENHO)
    # =========================================================================
    with tab_visual:
        st.subheader("🖼️ Encontrar Ícones Parecidos com uma Imagem")
        st.markdown("Envie um print de tela, logotipo ou rascunho. O motor multimodal CLIP compara a **silhueta geométrica** e o **conceito**.")

        # Sidebar parameters for visual search
        with st.sidebar:
            st.header("⚙️ Ajustes da Busca Visual")
            alpha = st.slider(
                "Equilíbrio Visual vs. Semântico (α)",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05,
                help="1.0 = Compara estritamente a silhueta/desenho.\n0.0 = Compara o significado da imagem.\n0.65 = Padrão recomendado.",
            )

            st.divider()
            st.header("🛠️ Normalização da Imagem")
            auto_crop = st.checkbox("Auto-Crop & Centralização", value=True, help="Remove bordas brancas e centraliza.")
            binarize = st.checkbox("Forçar Silhueta P&B (Alto Contraste)", value=True, help="Converte imagens coloridas em silhueta limpa.")
            invert_colors = st.checkbox("Inverter Cores (Print de Terminal Escuro)", value=False, help="Inverte fundo preto para fundo branco.")
            top_k_visual = st.selectbox("Resultados Visuais:", [8, 12, 16, 24, 32, 48], index=1)

            st.divider()
            text_boost = st.text_input(
                "Dica Semântica Extra (Boost Opcional):",
                placeholder="Ex: git, folder, rust, database",
                help="Mistura uma palavra-chave no vetor da imagem.",
            )

        # Input modes: File Upload or Canvas Draw
        vtabs = ["📤 Upload de Imagem", "✏️ Desenhar Ícone (Canvas)"] if HAS_CANVAS else ["📤 Upload de Imagem"]
        active_vtabs = st.tabs(vtabs)

        query_pil = None

        with active_vtabs[0]:
            uploaded_file = st.file_uploader(
                "Arraste ou selecione uma imagem (PNG, JPG, SVG export, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
            )
            if uploaded_file:
                query_pil = Image.open(uploaded_file)

        if HAS_CANVAS and len(active_vtabs) > 1:
            with active_vtabs[1]:
                st.write("Desenhe um esboço abaixo:")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=6,
                    stroke_color="#000000",
                    background_color="#FFFFFF",
                    height=200,
                    width=200,
                    drawing_mode="freedraw",
                    key="visual_canvas",
                )
                if canvas_result.image_data is not None:
                    arr = canvas_result.image_data
                    if np.any(arr[:, :, :3] < 240):
                        query_pil = Image.fromarray(arr.astype("uint8"), "RGBA")

        if query_pil is not None:
            if model is None or img_emb is None:
                st.error("⚠️ Embeddings visuais não encontrados! Execute `python setup_assets.py && python indexer.py`.")
            else:
                processed_img = preprocess_image(
                    query_pil,
                    invert_colors=invert_colors,
                    binarize=binarize,
                    auto_crop=auto_crop,
                )

                col_orig, col_proc, col_stats = st.columns([1, 1, 2])
                with col_orig:
                    st.markdown("**Imagem Original:**")
                    st.image(query_pil, width=128)
                with col_proc:
                    st.markdown("**Normalizada (CLIP):**")
                    st.image(processed_img, width=128)
                with col_stats:
                    st.markdown("**Modo de Busca:**")
                    st.markdown(f"- Peso Visual (Forma): **{alpha * 100:.0f}%**")
                    st.markdown(f"- Peso Semântico (Conceito): **{(1 - alpha) * 100:.0f}%**")
                    if text_boost.strip():
                        st.markdown(f"- Dica de Texto Extra: *'{text_boost.strip()}'*")

                query_vec = model.encode([processed_img], convert_to_numpy=True)[0]
                query_vec = query_vec / np.linalg.norm(query_vec)

                if text_boost.strip():
                    boost_prompt = f"icon symbol of {text_boost.strip()}"
                    boost_vec = model.encode([boost_prompt], convert_to_numpy=True)[0]
                    boost_vec = boost_vec / np.linalg.norm(boost_vec)
                    query_vec = (0.75 * query_vec) + (0.25 * boost_vec)
                    query_vec = query_vec / np.linalg.norm(query_vec)

                top_indices, top_scores = search_hybrid_visual(
                    query_vec,
                    img_emb=img_emb,
                    txt_emb=txt_emb,
                    alpha=alpha,
                    top_k=top_k_visual,
                )

                st.markdown("---")
                st.subheader(f"🎯 Top {top_k_visual} Ícones Visualmente Parecidos")

                num_cols = 4
                cols = st.columns(num_cols)
                for idx_pos, (icon_idx, score) in enumerate(zip(top_indices, top_scores)):
                    item_meta = emb_metadata[icon_idx]
                    cat_item = next((c for c in catalog if c["name"] == item_meta["name"]), item_meta)
                    badge = f'<span class="badge-semantic">{score * 100:.1f}% match</span>'
                    render_icon_card(cols[idx_pos % num_cols], cat_item, badge)
        else:
            st.info("👆 Envie uma imagem acima para encontrar os ícones mais parecidos.")

    # =========================================================================
    # TAB 3: CATÁLOGO COMPLETO (MOSTRAR TODOS)
    # =========================================================================
    with tab_all:
        st.subheader("📚 Catálogo Completo de Ícones Nerd Fonts")
        st.caption(f"Navegue por todos os **{len(catalog):,}** glifos oficiais catalogados da Nerd Fonts.")

        # Metrics overview
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total de Ícones Catalogados", f"{len(catalog):,}")
        with m2:
            st.metric("Coleções / Famílias", f"{len(family_counts)}")
        with m3:
            st.metric("Fonte Base de Símbolos", "SymbolsNerdFontMono")

        st.markdown("---")

        # Filters
        c_fam, c_search, c_per_page = st.columns([2, 3, 2])
        with c_fam:
            cat_family_raw = st.selectbox(
                "Filtrar Coleção:",
                family_options,
                key="catalog_family_filter",
            )
            cat_selected_family = None if cat_family_raw == "Todas as Famílias" else cat_family_raw.split(" (")[0]

        with c_search:
            cat_filter_query = st.text_input(
                "Filtrar dentro do catálogo:",
                placeholder="Ex: arrow, heart, check, wifi, battery...",
                key="catalog_filter_query",
            )

        with c_per_page:
            per_page_options = [48, 96, 192, 384, "Mostrar Todos nesta Categoria"]
            per_page_choice = st.selectbox("Ícones por Página:", per_page_options, index=1, key="cat_per_page")

        # Filter items
        filtered_catalog = catalog
        if cat_selected_family:
            filtered_catalog = [item for item in filtered_catalog if item["family"] == cat_selected_family]

        if cat_filter_query.strip():
            q_term = cat_filter_query.strip().lower()
            filtered_catalog = [
                item for item in filtered_catalog
                if q_term in item["name"].lower()
                or q_term in item["code"].lower()
                or q_term in item["clean_name"].lower()
            ]

        total_filtered = len(filtered_catalog)
        st.markdown(f"**Exibindo {total_filtered:,} ícones** correspondentes aos filtros.")

        if total_filtered == 0:
            st.warning("Nenhum ícone encontrado com os filtros atuais.")
        else:
            # Pagination Logic
            if per_page_choice == "Mostrar Todos nesta Categoria":
                page_items = filtered_catalog
                page = 1
                total_pages = 1
                if total_filtered > 1500:
                    st.warning(f"⚠️ Renderizar {total_filtered:,} ícones simultaneamente pode causar lentidão no navegador. Use a paginação para melhor fluidez.")
            else:
                items_per_page = int(per_page_choice)
                total_pages = max(1, math.ceil(total_filtered / items_per_page))

                col_prev, col_info, col_next, col_jump = st.columns([1, 2, 1, 1])

                if "catalog_page" not in st.session_state:
                    st.session_state.catalog_page = 1

                # Keep page in bounds
                if st.session_state.catalog_page > total_pages:
                    st.session_state.catalog_page = total_pages
                if st.session_state.catalog_page < 1:
                    st.session_state.catalog_page = 1

                with col_prev:
                    if st.button("◀ Anterior", disabled=(st.session_state.catalog_page <= 1), use_container_width=True):
                        st.session_state.catalog_page -= 1
                        st.rerun()

                with col_info:
                    st.markdown(
                        f"<div style='text-align: center; padding-top: 6px; font-weight: 600;'>"
                        f"Página {st.session_state.catalog_page} de {total_pages}</div>",
                        unsafe_allow_html=True,
                    )

                with col_next:
                    if st.button("Próxima ▶", disabled=(st.session_state.catalog_page >= total_pages), use_container_width=True):
                        st.session_state.catalog_page += 1
                        st.rerun()

                with col_jump:
                    jump_page = st.number_input(
                        "Ir para página:",
                        min_value=1,
                        max_value=total_pages,
                        value=st.session_state.catalog_page,
                        key="jump_page_input",
                        label_visibility="collapsed",
                    )
                    if jump_page != st.session_state.catalog_page:
                        st.session_state.catalog_page = jump_page
                        st.rerun()

                page = st.session_state.catalog_page
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_filtered)
                page_items = filtered_catalog[start_idx:end_idx]

            # Render Cards Grid
            num_cols = 4
            cols = st.columns(num_cols)
            for idx, item in enumerate(page_items):
                badge = f'<span class="badge-family">{item["family"]}</span>'
                render_icon_card(cols[idx % num_cols], item, badge)


if __name__ == "__main__":
    main()
