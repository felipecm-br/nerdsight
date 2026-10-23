# CLAUDE.md — Claude Code Developer Guide

This repository contains **`nerdsight`**, an AI-powered visual, name, and semantic search engine for [Nerd Fonts](https://www.nerdfonts.com) glyphs using multimodal CLIP embeddings (`clip-ViT-B-32`).

See [AGENTS.md](AGENTS.md) for the full agent guidelines and architecture manual.

---

## 🚀 Quick Reference Commands

### Environment & Dependencies
```bash
# Activate virtual environment
source .venv/bin/activate

# Install / update dependencies (using uv or pip)
uv pip install -r requirements.txt
```

### Assets & Indexing
```bash
# 1. Download official Nerd Fonts glyphnames.json and symbol font
python setup_assets.py

# 2. Build or test the multimodal embeddings index
python indexer.py --limit 300     # Fast smoke test / partial index
python indexer.py                 # Full index (~10,000+ glyphs, ~5 min on CPU)
python indexer.py --text-only     # Re-encode prompts only, after changing build_semantic_prompt()
```

### Running Applications
```bash
# Web UI (Streamlit on port 8501)
streamlit run app.py

# Terminal CLI: Name Search
python cli.py --name "python"
python cli.py --name "git" --family dev

# Terminal CLI: Semantic Search (Concept via CLIP)
python cli.py --text "database" --top 5

# Terminal CLI: Catalog / Browse All
python cli.py --all --family fa --top 20

# Terminal CLI: Visual Image Search
python cli.py path/to/icon.png --alpha 0.7 --top 5
```

### Static Web Build (Hugging Face Space)
```bash
python export_web.py --golden     # web/dist + web/tests/golden (reference results)
cd web && npm install && npm test # parity: preprocessing, name search, CLIP rankings
npm run test:browser              # same checks in headless Chromium (needs network)
python deploy_space.py            # upload web/dist to <hf-user>/nerdsight (hf auth login first)
```

---

## 🧠 Core Architecture & Development Rules

1. **Path Safety**: Always resolve paths relative to `__file__` using `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`. Never rely on current working directory (`os.getcwd()`).
2. **Master Catalog vs Embeddings**:
   - `assets/glyphnames.json` is the canonical master catalog for all 10,000+ glyphs.
   - Exact/substring name search and catalog browsing must work across the full catalog with sub-millisecond response times without requiring complete CLIP matrices.
   - Semantic concept search and visual image search use `data/txt_embeddings.npy` and `data/img_embeddings.npy`.
3. **Decoupled Embeddings Storage**:
   - Store visual (`img_embeddings.npy`) and textual (`txt_embeddings.npy`) matrices separately in `data/`.
   - Image queries score each matrix separately, z-score each score vector across the catalog (the two modalities live on very different cosine scales), then weight them (`core.hybrid_scores`, default $\alpha = 0.7$):
     $$S = \alpha \cdot z(\mathbf{E}_{img}\,\vec{q}) + (1 - \alpha) \cdot z(\mathbf{E}_{txt}\,\vec{q})$$
   - This allows instant, real-time $\alpha$ slider adjustments in `app.py` without model re-computation.
4. **Font Independence**:
   - The Web UI serves `SymbolsNerdFontMono-Regular.ttf` from `static/` (Streamlit static serving, mirrored from `assets/` by `core.ensure_static_font()`) and registers it as the `NerdSymbols` font face in `.streamlit/config.toml`. Glyphs render on any client regardless of host OS fonts, and the browser caches the font instead of receiving a base64 copy on every rerun.
5. **Web UI Theme**:
   - `.streamlit/config.toml` is the single source of truth for theme colors; `app.py` reads them back with `st.get_option()` and exposes them as CSS custom properties.
   - Custom styles live in `ui/nerdsight.css` (every value is a `--ns-*` token) and client behavior in `ui/nerdsight.js`. Streamlit sanitizes `st.html` scripts with DOMPurify, so the JS must never contain tag-like `<x` sequences; build DOM with `createElement`.
   - No emoji in UI copy. Use Streamlit `:material/...:` icons or Nerd Font glyphs.
   - Every user-facing string lives in `i18n.py` (English default, plus `pt-BR` and `es`). Add a key to all three languages and run `python i18n.py` to check parity. Strings for `ui/nerdsight.js` use the `js_` prefix. Widget values are language-neutral ids (`hybrid`, `concept`, `all`) with translated `format_func` labels.
6. **Static Web Front End (`web/`)**:
   - Free Hugging Face accounts only get Static Spaces, so `web/` is a no-build browser port of `app.py` + `core.py`; `export_web.py` writes everything it needs into `web/dist/` (gitignored).
   - `web/js/preprocess.js` reimplements Pillow's resampling and must stay pixel-exact with `core.preprocess_image`; any change to preprocessing, prompts, fusion or `i18n.py` must be mirrored in `web/js/` and re-checked with `npm test`.
   - The theme (`theme.css`) and strings (`data/i18n.json`) are generated from `.streamlit/config.toml` and `i18n.py`; never edit them in `web/dist/`. `ui/nerdsight.css` and `ui/nerdsight.js` are shared by both UIs.
   - transformers.js is loaded from jsDelivr at the exact version pinned in `web/js/main.js` and `web/package.json`; bump both together and re-run the tests.
7. **Git Hygiene & Commit Format**:
   - Always write commit messages in English following Conventional Commits (`feat:`, `fix:`, `style:`, `docs:`, `refactor:`, `chore:`).
   - Never commit `.npy`, `.ttf`, or `.venv` files. Keep `.gitignore` strictly enforced.
