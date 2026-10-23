#!/usr/bin/env python3
"""
nerdsight - app.py
Interactive Streamlit Web Application for Visual, Name & Semantic Nerd Fonts Search.
Powered by Hybrid CLIP Multimodal Embeddings and the Official Glyph Catalog.
"""

import html
import json
import math
import os
from typing import Optional

import numpy as np
from PIL import Image
import streamlit as st

import core
from i18n import DEFAULT_LANG, LANGUAGES, Translator

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

st.set_page_config(
    page_title="nerdsight",
    page_icon=":material/search:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

UI_DIR = os.path.join(core.BASE_DIR, "ui")

VIEWS = {
    "search": ":material/search:",
    "visual": ":material/image_search:",
    "catalog": ":material/grid_view:",
}
MODES = ["hybrid", "name", "concept"]
MATCH_KEYS = {
    "Exact Match": "match_exact",
    "Word Match": "match_word",
    "Prefix Match": "match_prefix",
    "Substring Match": "match_substring",
}
SUGGESTIONS = ["git", "python", "folder", "database", "rocket", "cloud upload", "battery low", "warning"]
PAGE_SIZES = [96, 192, 384, "all"]
LARGE_RENDER = 1500
LOGO_GLYPH = "\U000f06d0"  # nf-md-eye_outline


# ---------------------------------------------------------------- data


@st.cache_resource
def load_catalog():
    """Loads the master catalog plus a name index. Shared read-only across sessions."""
    catalog = core.load_master_catalog()
    by_name = {item["name"]: item for item in catalog}
    counts: dict[str, int] = {}
    for item in catalog:
        counts[item["family"]] = counts.get(item["family"], 0) + 1
    families = sorted(counts, key=counts.get, reverse=True)
    return catalog, by_name, families, counts


@st.cache_resource
def load_index():
    """Loads precomputed embeddings. Returns (img_emb, txt_emb, metadata, error)."""
    if not core.has_embeddings_index():
        return None, None, [], None
    try:
        img_emb, txt_emb, metadata = core.load_embeddings()
        return img_emb, txt_emb, metadata, None
    except Exception as exc:
        return None, None, [], str(exc)


@st.cache_resource(show_spinner=False)
def _load_model():
    return core.get_model()


def load_model(t: Translator):
    """Loads CLIP once per process; only the first call shows a (localized) spinner."""
    if core._model_singleton is not None:
        return _load_model()
    with st.spinner(t("loading_model")):
        return _load_model()


@st.cache_resource
def load_ui_assets() -> tuple[str, str, bool]:
    with open(os.path.join(UI_DIR, "nerdsight.css"), encoding="utf-8") as f:
        css = f.read()
    with open(os.path.join(UI_DIR, "nerdsight.js"), encoding="utf-8") as f:
        js = f.read()
    return css, js, core.ensure_static_font()


def resolve_item(meta: dict, by_name: dict) -> dict:
    return by_name.get(meta["name"]) or {**meta, "family": "Nerd Font"}


def relevance(selected: np.ndarray, all_scores: np.ndarray) -> np.ndarray:
    """Maps scores to 0..1 bar widths: catalog median -> 0, catalog best -> 1."""
    floor = float(np.median(all_scores))
    top = float(all_scores.max())
    return np.clip((selected - floor) / (top - floor + 1e-8), 0.0, 1.0)


# ------------------------------------------------------------- url state
# Option widgets are synced to the URL by hand rather than with
# bind="query-params", which would store the translated label instead of the id.


def init_from_url(key: str, options, default):
    if key not in st.session_state:
        value = st.query_params.get(key)
        st.session_state[key] = value if value in options else default


def sync_to_url(key: str, value, default):
    if value == default:
        st.query_params.pop(key, None)
    else:
        st.query_params[key] = value


# ------------------------------------------------------------- rendering


def inject_theme(css: str, js: str, t: Translator):
    tokens = {
        "--ns-bg": st.get_option("theme.backgroundColor") or "#0f1217",
        "--ns-surface": st.get_option("theme.secondaryBackgroundColor") or "#161a21",
        "--ns-border": st.get_option("theme.borderColor") or "#2a313c",
        "--ns-text": st.get_option("theme.textColor") or "#e7eaf0",
        "--ns-accent": st.get_option("theme.primaryColor") or "#f2b84b",
    }
    root = ";".join(f"{k}:{v}" for k, v in tokens.items())
    st.html(f"<style>:root{{{root}}}\n{css}</style>")
    # The strings prefix changes with the language, so the script re-runs and
    # picks it up; the listeners themselves are bound only once.
    i18n = json.dumps(t.js_strings())
    st.html(f"<script>window.__nsI18n = {i18n};\n{js}</script>", unsafe_allow_javascript=True)


def render_header(t: Translator, catalog_size: int, family_count: int, index_size: int, index_error: Optional[str]):
    ok = index_size > 0
    if ok:
        status = t("index_vectors", n=t.int(index_size))
    else:
        status = t("index_error") if index_error else t("index_missing")
    st.html(f"""
    <header class="ns-header">
      <div class="ns-brand">
        <span class="ns-logo" aria-hidden="true">{LOGO_GLYPH}</span>
        <div>
          <h1 class="ns-title">nerdsight</h1>
          <p class="ns-tagline">{t("tagline")}</p>
        </div>
      </div>
      <dl class="ns-stats">
        <div><dt>{t("stat_glyphs")}</dt><dd>{t.int(catalog_size)}</dd></div>
        <div><dt>{t("stat_families")}</dt><dd>{family_count}</dd></div>
        <div><dt>{t("stat_index")}</dt><dd class="ns-status" data-ok="{str(ok).lower()}">{status}</dd></div>
      </dl>
    </header>
    """)


def display_name(name: str) -> str:
    """Card label: drops the nf- prefix, keeps the family prefix glued to the
    name (non-breaking hyphen) and lets long names wrap after underscores."""
    shown = html.escape(name.removeprefix("nf-")).replace("-", "‑", 1)
    return shown.replace("_", "_<wbr>")


def card_html(item: dict, label: Optional[str] = None, score: Optional[float] = None, strong: bool = False) -> str:
    name = html.escape(item["name"])
    code = html.escape(item["code"].lower())
    family = html.escape(item.get("family", ""))
    char = html.escape(item["char"])
    detail = html.escape(label) if label else family
    bar = ""
    if score is not None:
        bar = f'<span class="ns-bar" aria-hidden="true"><i style="width:{score:.0%}"></i></span>'
    strong_attr = " data-strong" if strong or (score is not None and score >= 0.75) else ""
    aria = f"{name}, U+{code.upper()}, {detail}"
    return (
        f'<button type="button" class="ns-card"{strong_attr} aria-label="{aria}" title="{name}" '
        f'data-char="{char}" data-name="{name}" data-code="{code}" data-family="{family}">'
        f'<span class="ns-glyph" aria-hidden="true">{char}</span>'
        f'<span class="ns-name">{display_name(item["name"])}</span>'
        f'<span class="ns-meta"><span class="ns-code">{code}</span>{bar}<span>{detail}</span></span>'
        f"</button>"
    )


def render_grid(cards: list[str], label: str):
    st.html(f'<div class="ns-grid" role="group" aria-label="{html.escape(label)}">{"".join(cards)}</div>')


def render_summary(t: Translator, text: str):
    st.html(f"""
    <div class="ns-summary">
      <span>{text}</span>
      <span class="ns-keys" aria-label="{t("keys_label")}">
        <span><kbd>/</kbd>{t("key_search")}</span>
        <span><kbd>Enter</kbd>{t("key_details")}</span>
        <span><kbd>C</kbd>{t("key_copy")}</span>
      </span>
    </div>
    """)


def render_empty(title: str, body: str):
    st.html(f'<div class="ns-empty"><h3>{title}</h3><p>{body}</p></div>')


# ----------------------------------------------------------------- views


def _apply_suggestion():
    st.session_state.q = st.session_state.suggestion
    st.session_state.suggestion = None


@st.fragment
def view_search(t: Translator, catalog, by_name, families, counts, index):
    _, txt_emb, metadata, _ = index
    has_index = txt_emb is not None and len(metadata) > 0

    query = st.text_input(
        t("search_label"),
        key="q",
        type="search",
        live="300ms",
        bind="query-params",
        icon=":material/search:",
        placeholder=t("search_placeholder"),
        label_visibility="collapsed",
    )

    init_from_url("mode", MODES, "hybrid")
    c_mode, c_family, c_top = st.columns([3, 3, 1.4], vertical_alignment="bottom")
    with c_mode:
        mode = st.segmented_control(
            t("match"),
            MODES,
            format_func=lambda m: t(f"mode_{m}"),
            required=True,
            key="mode",
            help=t("mode_help"),
        )
    sync_to_url("mode", mode, "hybrid")
    with c_family:
        family = st.selectbox(
            t("family"),
            [None] + families,
            format_func=lambda f: t("all_families") if f is None else f"{f} ({t.int(counts[f])})",
            key="family",
        )
    with c_top:
        top_k = st.selectbox(t("results"), [24, 48, 96, 192], index=1, key="top_k")

    q = (query or "").strip()
    if not q:
        st.pills(t("try"), SUGGESTIONS, key="suggestion", on_change=_apply_suggestion)
        st.html(f'<p class="ns-hint">{t("search_hint")}</p>')
        return

    pool = catalog if family is None else [item for item in catalog if item["family"] == family]
    cards: list[str] = []
    seen: set[str] = set()
    n_name = n_concept = name_total = 0

    if mode in ("hybrid", "name"):
        name_matches = core.search_by_name(pool, q, None, len(pool))
        name_total = len(name_matches)
        for _, item, label in name_matches[:top_k]:
            cards.append(card_html(item, t(MATCH_KEYS[label]), strong=label == "Exact Match"))
            seen.add(item["name"])
            n_name += 1

    if mode in ("hybrid", "concept") and len(cards) < top_k:
        if has_index:
            order, scores = core.search_by_semantic_text(load_model(t), txt_emb, q)
            widths = relevance(scores, scores)
            for idx, width in zip(order, widths):
                if len(cards) >= top_k:
                    break
                item = resolve_item(metadata[idx], by_name)
                if item["name"] in seen or (family and item.get("family") != family):
                    continue
                cards.append(card_html(item, t("match_concept"), float(width)))
                seen.add(item["name"])
                n_concept += 1
        elif mode == "concept":
            st.warning(t("concept_needs_index"), icon=":material/database_off:")
            return

    in_family = t("in_family", family=html.escape(family)) if family else ""
    if not cards:
        st.markdown(t("no_name_match", q=html.escape(q)) + in_family + ".")
        c1, c2, _ = st.columns([1.3, 1.3, 4])
        if mode == "name" and has_index:
            c1.button(t("btn_search_concept"), icon=":material/auto_awesome:",
                      on_click=lambda: st.session_state.update(mode="concept"))
        if family:
            c2.button(t("btn_clear_family"), icon=":material/filter_alt_off:",
                      on_click=lambda: st.session_state.update(family=None))
        return

    parts = []
    if n_name:
        parts.append(t("by_name", n=n_name) + (t("of_total", total=t.int(name_total)) if name_total > n_name else ""))
    if n_concept:
        parts.append(t("by_concept", n=n_concept))
    render_summary(t, t("search_summary", n=len(cards), q=html.escape(q)) + f" · {', '.join(parts)}")
    if mode == "hybrid" and not has_index:
        st.caption(t("concept_disabled"))
    render_grid(cards, t("results_for", q=q))


@st.fragment
def view_visual(t: Translator, by_name, index):
    img_emb, txt_emb, metadata, _ = index
    if img_emb is None or not metadata:
        render_empty(t("visual_needs_index_title"), t("visual_needs_index_body"))
        return

    left, right = st.columns([5, 7], gap="large")
    query_pil = None
    source_id = "none"

    with left:
        source = "upload"
        if HAS_CANVAS:
            source = st.segmented_control(
                t("source_label"),
                ["upload", "draw"],
                format_func=lambda s: f":material/upload: {t('source_upload')}" if s == "upload"
                else f":material/draw: {t('source_draw')}",
                default="upload",
                required=True,
                key="visual_source",
                label_visibility="collapsed",
            )

        if source == "upload":
            uploaded = st.file_uploader(t("upload_label"), type=["png", "jpg", "jpeg", "webp"], help=t("upload_help"))
            if uploaded:
                query_pil = Image.open(uploaded)
                source_id = uploaded.file_id
        else:
            canvas = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=6,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=240,
                width=240,
                drawing_mode="freedraw",
                key="visual_canvas",
            )
            if canvas.image_data is not None and np.any(canvas.image_data[:, :, :3] < 240):
                query_pil = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
                source_id = "canvas"

        boost = st.text_input(t("boost_label"), placeholder="ghost, git, database", key="visual_text_boost",
                              help=t("boost_help"))
        alpha = st.slider(
            t("alpha_label"),
            min_value=0.0,
            max_value=1.0,
            value=core.DEFAULT_ALPHA,
            step=0.05,
            key="alpha",
            help=t("alpha_help", alpha=t.decimal(core.DEFAULT_ALPHA)),
        )
        with st.expander(t("preprocess"), icon=":material/tune:"):
            auto_crop = st.toggle(t("crop"), value=True, key="auto_crop", help=t("crop_help"))
            binarize = st.toggle(t("binarize"), value=True, key="binarize", help=t("binarize_help"))
            invert = st.toggle(
                t("invert"),
                value=query_pil is not None and core.has_dark_border(query_pil),
                key=f"invert_{source_id}",
                help=t("invert_help"),
            )

    with right:
        if query_pil is None:
            render_empty(t("visual_empty_title"), t("visual_empty_body"))
            return

        processed = core.preprocess_image(query_pil, invert_colors=invert, binarize=binarize, auto_crop=auto_crop)
        c_orig, c_proc, c_info = st.columns([1, 1, 2])
        c_orig.image(query_pil, caption=t("original"), width=120)
        c_proc.image(processed, caption=t("model_view"), width=120)
        with c_info:
            st.markdown(t("weights", shape=f"{alpha:.0%}", concept=f"{1 - alpha:.0%}"))
            st.caption(t("zscore_note"))
            if boost.strip():
                st.caption(t("boost_note", hint=html.escape(boost.strip())))

    top_k = st.session_state.get("visual_top_k", 24)
    model = load_model(t)
    with st.spinner(t("comparing")):
        query_vec = core.encode_image_query(model, processed, boost_text=boost)
        scores = core.hybrid_scores(query_vec, img_emb, txt_emb, alpha)
    top = np.argsort(scores)[::-1][:top_k]
    widths = relevance(scores[top], scores)
    cards = [card_html(resolve_item(metadata[i], by_name), t("match_similar"), float(w)) for i, w in zip(top, widths)]

    st.space("small")
    c_sum, c_top = st.columns([5, 1], vertical_alignment="bottom")
    with c_sum:
        render_summary(t, t("visual_summary", n=len(cards)))
    with c_top:
        st.selectbox(t("results"), [12, 24, 48, 96], index=1, key="visual_top_k",
                     format_func=lambda n: t("n_results", n=n), label_visibility="collapsed")
    render_grid(cards, t("visual_grid"))


def _set_page(page: int):
    st.session_state.catalog_page = page


@st.fragment
def view_catalog(t: Translator, catalog, families, counts):
    family = st.pills(
        t("family"),
        families,
        format_func=lambda f: f"{f} {t.int(counts[f])}",
        key="catalog_family",
        help=t("catalog_family_help"),
    )
    c_filter, c_size = st.columns([5, 1.4], vertical_alignment="bottom")
    with c_filter:
        term = st.text_input(
            t("filter_label"),
            key="catalog_filter",
            type="search",
            live="200ms",
            icon=":material/filter_list:",
            placeholder=t("filter_placeholder"),
            label_visibility="collapsed",
        )
    with c_size:
        per_page = st.selectbox(t("per_page"), PAGE_SIZES, index=0, key="catalog_per_page",
                                label_visibility="collapsed",
                                format_func=lambda s: t("n_per_page", n=s) if isinstance(s, int) else t("all"))

    items = catalog if family is None else [item for item in catalog if item["family"] == family]
    term = (term or "").strip().lower()
    if term:
        items = [i for i in items
                 if term in i["name"].lower() or term in i["code"].lower() or term in i["clean_name"].lower()]
    total = len(items)

    # Changing any filter returns to the first page.
    signature = (family, term, per_page)
    if st.session_state.get("catalog_signature") != signature:
        st.session_state.catalog_signature = signature
        st.session_state.catalog_page = 1

    in_family = t("in_family", family=html.escape(family)) if family else ""
    if total == 0:
        render_empty(t("filter_empty_title"), t("filter_empty_body", term=html.escape(term), family=in_family))
        return

    size = total if per_page == "all" else int(per_page)
    pages = max(1, math.ceil(total / size))
    page = min(max(1, st.session_state.get("catalog_page", 1)), pages)
    start = (page - 1) * size
    page_items = items[start:start + size]

    render_summary(t, t("catalog_summary", a=t.int(start + 1), b=t.int(start + len(page_items)),
                        total=t.int(total), family=in_family))
    if per_page == "all" and total > LARGE_RENDER:
        st.caption(t("large_render", n=t.int(total)))
    render_grid([card_html(item) for item in page_items], t("catalog_grid"))

    if pages > 1:
        st.space("small")
        c_prev, c_info, c_next = st.columns([1, 2, 1], vertical_alignment="center")
        c_prev.button(t("prev"), icon=":material/chevron_left:", disabled=page <= 1,
                      on_click=_set_page, args=(page - 1,), width="stretch")
        c_info.markdown(f"<div style='text-align:center'>{t('page', page=page, pages=pages)}</div>",
                        unsafe_allow_html=True)
        c_next.button(t("next"), icon=":material/chevron_right:", icon_position="right", disabled=page >= pages,
                      on_click=_set_page, args=(page + 1,), width="stretch")


# ------------------------------------------------------------------ main


def main():
    init_from_url("lang", LANGUAGES, DEFAULT_LANG)
    t = Translator(st.session_state.lang)

    css, js, font_ok = load_ui_assets()
    inject_theme(css, js, t)

    catalog, by_name, families, counts = load_catalog()
    index = load_index()
    render_header(t, len(catalog), len(families), len(index[2]), index[3])

    if not catalog:
        st.error(t("catalog_missing"), icon=":material/error:")
        st.stop()
    if not font_ok:
        st.warning(t("font_missing"), icon=":material/font_download_off:")
    if index[3]:
        st.warning(t("index_load_error", err=index[3]), icon=":material/database_off:")

    init_from_url("view", VIEWS, "search")
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        view = st.segmented_control(
            t("view_label"),
            list(VIEWS),
            format_func=lambda v: f"{VIEWS[v]} {t(f'view_{v}')}",
            required=True,
            key="view",
            label_visibility="collapsed",
        )
        st.segmented_control(
            t("language"),
            list(LANGUAGES),
            format_func=LANGUAGES.get,
            required=True,
            key="lang",
            label_visibility="collapsed",
        )
    sync_to_url("view", view, "search")
    sync_to_url("lang", st.session_state.lang, DEFAULT_LANG)

    if view == "visual":
        view_visual(t, by_name, index)
    elif view == "catalog":
        view_catalog(t, catalog, families, counts)
    else:
        view_search(t, catalog, by_name, families, counts, index)


if __name__ == "__main__":
    main()
