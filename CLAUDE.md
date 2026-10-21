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
   - The Web UI injects `SymbolsNerdFontMono-Regular.ttf` as a base64 `@font-face` stylesheet. This guarantees that glyphs render natively on client browsers regardless of host OS fonts.
5. **Git Hygiene & Commit Format**:
   - Always write commit messages in English following Conventional Commits (`feat:`, `fix:`, `style:`, `docs:`, `refactor:`, `chore:`).
   - Never commit `.npy`, `.ttf`, or `.venv` files. Keep `.gitignore` strictly enforced.
