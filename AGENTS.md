# AGENTS.md — Repository Guidelines & Operating Manual

This repository contains **`nerdsight`**, a visual, name, and semantic search engine for [Nerd Fonts](https://www.nerdfonts.com) icons powered by hybrid CLIP multimodal embeddings.

---

## 🤖 Rules for AI Coding Assistants (Agents)

### 1. Git & Commit Guidelines
- **Always commit in English** when implementing features, bug fixes, refactoring, or documentation updates.
- Use [Conventional Commits](https://www.conventionalcommits.org/) format:
  - `feat: <description>` for new capabilities (CLI flags, UI widgets, search algorithms)
  - `fix: <description>` for bug fixes or path corrections
  - `style: <description>` for UI styling, CSS tweaks, card layouts
  - `docs: <description>` for README, guides, or docstring updates
  - `refactor: <description>` for code restructuring without changing behavior
  - `chore: <description>` for asset scripts, dependencies, gitignore updates
- Make atomic, well-tested commits. Never amend existing commits unless explicitly requested.

### 2. Architecture & Design Principles
- **Multimodal Hybrid Embeddings Engine**:
  - `nerdsight` uses `clip-ViT-B-32` from OpenAI / OpenCLIP to map both glyph imagery and textual descriptions into a shared 512-dimensional vector space.
  - Score fusion (`core.hybrid_scores`): visual and semantic cosine scores are z-scored separately across the catalog, then weighted (default $\alpha = 0.7$):
    $$S = \alpha \cdot z(\mathbf{E}_{img}\,\vec{q}) + (1 - \alpha) \cdot z(\mathbf{E}_{txt}\,\vec{q})$$
  - Embeddings are stored decoupled in `data/img_embeddings.npy` and `data/txt_embeddings.npy` to allow real-time $\alpha$ adjustments with zero model re-computation.
- **Two-Tier Data Model (Catalog vs. Embeddings)**:
  - **Master Catalog (`assets/glyphnames.json`)**: Ground truth containing all 10,000+ Nerd Fonts glyphs (`name`, `char`, `code`, prefix). Exact/substring name search and the full catalog browser operate against this catalog with sub-millisecond latency.
  - **Embeddings Index (`data/*.npy`, `data/metadata.json`)**: Precomputed vector representations used for visual image search and semantic concept queries.
- **Client Font Independence**:
  - Never assume the client browser or machine has Nerd Fonts installed locally.
  - The Web UI injects `SymbolsNerdFontMono-Regular.ttf` as a base64-encoded `@font-face` rule in custom CSS to ensure native, crisp rendering across all browsers.
- **Query Image Preprocessing Pipeline**:
  - Alpha channel flattening against solid white canvas.
  - Dark-mode terminal print inversion (white-on-black $\rightarrow$ black-on-white).
  - High-contrast binarization for multi-color logos.
  - Bounding box auto-crop and proportional centering (~72% target canvas occupancy) to mirror official font rasterization metrics.
- **Storage & Binary Asset Hygiene**:
  - Never commit large generated binary files to Git (`assets/SymbolsNerdFontMono-Regular.ttf`, `data/*.npy`).
  - Keep `assets/.gitkeep` and `data/.gitkeep` in the repository so directory structures exist upon cloning.

### 3. File Map & Responsibilities
- `setup_assets.py`: Downloads `glyphnames.json` and extracts `SymbolsNerdFontMono-Regular.ttf` from official GitHub releases.
- `indexer.py`: Rasterizes font glyphs, synthesizes contextual semantic prompts, and computes visual + textual CLIP embeddings (supports `--limit`, `--batch-size`, `--model`, `--text-only`). Prompts are `a {outline|filled} icon of a {object nouns}` with style tokens stripped from the noun phrase.
- `core.py`: Shared catalog loading, image preprocessing, and hybrid CLIP search logic. `app.py` and `cli.py` import from here — add new search logic here, not in the entry points.
- `app.py`: Streamlit web application providing Visual Search, Name & Semantic Text Search, and Catalog Browser with pagination.
- `cli.py`: Fast terminal CLI supporting `--name`, `--text`, `--all`, and `<image>` input with live terminal glyph rendering.
- Coding agents should call `python cli.py ... --json` (name, `--text`, image path, `--all --family`) to look up icons.
- `requirements.txt`: Python package manifests.

---

## 🛠️ Testing & Local Verification

Before finalizing any changes:
1. **CLI Validation**:
   - Test name search: `python cli.py --name "git" --top 3`
   - Test catalog browsing: `python cli.py --all --family dev --top 5`
   - Test syntax & imports: `python -m py_compile app.py cli.py indexer.py setup_assets.py`
2. **Path Robustness**:
   - Ensure all scripts use directory-relative paths (`BASE_DIR = os.path.dirname(os.path.abspath(__file__))`) so execution does not fail when invoked from outside the project root.
3. **Streamlit Health**:
   - If running in tmux (`nerdsight/main`), verify port `8501` responds with HTTP 200:
     `curl -sI http://localhost:8501 | head -n 5`
