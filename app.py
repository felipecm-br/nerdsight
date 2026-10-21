#!/usr/bin/env python3
"""
nerdsight - app.py
Interactive Streamlit Web Application for Visual, Name & Semantic Nerd Fonts Search.
Powered by Hybrid CLIP Multimodal Embeddings, Official Glyph Catalog, and an optional
two-stage multimodal VLM reranker (see reranker.py).
"""

import base64
import math
import os
from typing import Optional

import numpy as np
from PIL import Image
import streamlit as st

import core

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

FONT_PATH = core.FONT_PATH
PREFIX_MAP = core.PREFIX_MAP


@st.cache_data
def load_master_catalog():
    """Loads all 10,000+ Nerd Fonts glyphs directly from official glyphnames.json."""
    return core.load_master_catalog()


@st.cache_resource
def load_model_and_embeddings():
    """Loads CLIP model, precomputed embeddings, metadata, and font file into memory cache."""
    model = None
    img_emb = None
    txt_emb = None
    emb_metadata = []

    if core.has_embeddings_index():
        try:
            model = core.get_model()
            img_emb, txt_emb, emb_metadata = core.load_embeddings()
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


def inject_custom_css(font_b64: Optional[str]):
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
    .badge-vlm {{
        background: #d2992222;
        border: 1px solid #d2992266;
        color: #e3b341;
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
        st.code(f"{item['char']}  {item['name']}  \\u{item['code']}", language="text")


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
                    with st.spinner("🧠 Calculando similaridade semântica (CLIP)..."):
                        top_sem_indices, sem_scores = core.search_by_semantic_text(model, txt_emb, query_clean)

                    for idx, sim_score in zip(top_sem_indices, sem_scores):
                        if len(results) >= top_k_text * 2:
                            break
                        item_meta = emb_metadata[idx]
                        name = item_meta["name"]
                        if name in seen_names:
                            continue

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

                        sim = float(sim_score) * 100
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
        st.markdown("Envie um print de tela, logotipo ou rascunho. O motor multimodal CLIP compara a **silhueta geométrica** e o **conceito**, com calibração z-score e opção de reranker VLM.")

        with st.expander("⚙️ Ajustes da Busca Visual", expanded=True):
            col_alpha, col_topk = st.columns([3, 1])
            with col_alpha:
                alpha = st.slider(
                    "Equilíbrio Visual vs. Semântico (α)",
                    min_value=0.0,
                    max_value=1.0,
                    value=core.DEFAULT_ALPHA,
                    step=0.05,
                    help=f"α = {core.DEFAULT_ALPHA} dá {core.DEFAULT_ALPHA:.0%} de peso para a geometria/silhueta e o restante para a correspondência semântica (ambas em z-score).",
                )
            with col_topk:
                top_k_visual = st.selectbox("Resultados:", [8, 12, 16, 24, 32, 48], index=1)

            col_crop, col_bin, col_inv = st.columns(3)
            with col_crop:
                auto_crop = st.checkbox("Auto-Crop & Centralização", value=True, help="Remove bordas brancas e centraliza o glifo.")
            with col_bin:
                binarize = st.checkbox("Forçar Silhueta P&B (Alto Contraste)", value=True, help="Converte imagens coloridas em silhueta limpa.")
            with col_inv:
                invert_colors = st.checkbox("Inverter Cores (Print de Terminal Escuro)", value=False, help="Inverte fundo preto para fundo branco.")

            text_boost = st.text_input(
                "Dica Semântica Extra (Boost Textual Opcional):",
                placeholder="Ex: ghost, git, folder, database, rust...",
                help="Mistura uma palavra-chave no vetor da imagem.",
                key="visual_text_boost",
            )

            # Two-stage VLM reranker (Stage A: análise da imagem, Stage B: contact sheet)
            st.markdown("---")
            use_vlm_reranker = st.toggle(
                "✨ Ativar Reranker VLM (análise da imagem + verificação por contact sheet)",
                value=False,
                help="Stage A: o VLM descreve a imagem e gera palavras-chave. Stage B: os ~24 melhores candidatos são renderizados da fonte real numa grade e o VLM os pontua lado a lado com a sua imagem.",
            )
            vlm_backend_name = "claude"
            vlm_api_key = ""
            if use_vlm_reranker:
                vlm_backend_name = st.radio(
                    "Backend VLM:",
                    ["claude", "gemini"],
                    format_func=lambda b: {"claude": "Claude CLI (`claude -p`)", "gemini": "Gemini API (google-genai)"}[b],
                    horizontal=True,
                )
                if vlm_backend_name == "gemini":
                    env_key_present = bool(os.environ.get("GEMINI_API_KEY"))
                    vlm_api_key = st.text_input(
                        "Chave da API Gemini:" + (" (Detectada em GEMINI_API_KEY)" if env_key_present else ""),
                        type="password",
                        value=os.environ.get("GEMINI_API_KEY", ""),
                        placeholder="AIzaSy...",
                    )

        if "visual_input_round" not in st.session_state:
            st.session_state.visual_input_round = 0

        # Input modes: File Upload or Canvas Draw
        vtabs = ["📤 Upload de Imagem", "✏️ Desenhar Ícone (Canvas)"] if HAS_CANVAS else ["📤 Upload de Imagem"]
        active_vtabs = st.tabs(vtabs)

        query_pil = None
        round_key = st.session_state.visual_input_round

        with active_vtabs[0]:
            uploaded_file = st.file_uploader(
                "Arraste ou selecione uma imagem (PNG, JPG, SVG export, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"visual_uploader_{round_key}",
            )
            if uploaded_file:
                query_pil = Image.open(uploaded_file)
                if st.button("🗑️ Limpar imagem"):
                    st.session_state.visual_input_round += 1
                    st.rerun()

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
                    key=f"visual_canvas_{round_key}",
                )
                if canvas_result.image_data is not None:
                    arr = canvas_result.image_data
                    if np.any(arr[:, :, :3] < 240):
                        query_pil = Image.fromarray(arr.astype("uint8"), "RGBA")
                        if st.button("🗑️ Limpar desenho"):
                            st.session_state.visual_input_round += 1
                            st.rerun()

        if query_pil is not None:
            if model is None or img_emb is None:
                st.error("⚠️ Embeddings visuais não encontrados! Execute `python setup_assets.py && python indexer.py`.")
            else:
                processed_img = core.preprocess_image(
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
                    st.markdown("**Configuração da Busca:**")
                    st.markdown(f"- Peso da Forma Visual ($S_{{vis}}$): **{alpha * 100:.0f}%**")
                    st.markdown(f"- Peso do Conceito Semântico ($S_{{sem}}$): **{(1 - alpha) * 100:.0f}%**")
                    st.markdown("- Calibração: **Z-Score independente por modalidade**")
                    if text_boost.strip():
                        st.markdown(f"- Dica de Texto Extra: *'{text_boost.strip()}'*")

                with st.spinner("🧠 Calculando similaridade híbrida calibrada (CLIP)..."):
                    query_vec = core.encode_image_query(model, processed_img, boost_text=text_boost)
                    final_scores = core.hybrid_scores(query_vec, img_emb, txt_emb, alpha)

                rerank_result = None
                if use_vlm_reranker:
                    import reranker

                    backend = reranker.get_backend(vlm_backend_name, api_key=vlm_api_key)
                    with st.spinner(f"✨ Reranker VLM ({backend.name}): analisando imagem e verificando contact sheet..."):
                        try:
                            rerank_result = reranker.rerank(
                                query_pil, catalog, reranker.local_ranking_keys(final_scores, emb_metadata), backend
                            )
                        except Exception as exc:
                            st.error(f"⚠️ Reranker VLM falhou ({backend.name}): {exc}. Exibindo ranking local.")

                st.markdown("---")
                cols = st.columns(4)

                if rerank_result:
                    a = rerank_result.analysis
                    st.success(
                        f"🧠 **Stage A:** conceito **{a['primary_concept']}** · estilo **{a['visual_style'] or 'N/A'}** · "
                        f"palavras-chave `{', '.join(a['candidate_keywords'])}`"
                    )
                    with st.expander("🖼️ Stage B: contact sheet enviado ao VLM"):
                        st.image(rerank_result.contact_sheet)
                    st.subheader(f"🎯 Top {top_k_visual} Ícones (reranqueados por VLM)")
                    by_name = {c["name"]: c for c in catalog}
                    for idx_pos, r in enumerate(rerank_result.ranked[:top_k_visual]):
                        badge = f'<span class="badge-vlm" title="{r["reason"]}">VLM: {r["score"]:.0f}%</span>'
                        render_icon_card(cols[idx_pos % 4], by_name[r["name"]], badge)
                else:
                    st.subheader(f"🎯 Top {top_k_visual} Ícones Mais Parecidos")
                    top_indices = np.argsort(final_scores)[::-1][:top_k_visual]
                    for idx_pos, icon_idx in enumerate(top_indices):
                        item_meta = emb_metadata[icon_idx]
                        cat_item = next((c for c in catalog if c["name"] == item_meta["name"]), item_meta)
                        badge = f'<span class="badge-semantic">Match Híbrido: {final_scores[icon_idx] * 100:.1f}%</span>'
                        render_icon_card(cols[idx_pos % 4], cat_item, badge)
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
