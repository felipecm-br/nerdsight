# 🔍 nerdsight

> **Visual & Semantic Search Engine for [Nerd Fonts](https://www.nerdfonts.com) Icons**  
> Find glyphs, CSS classes (`nf-*`), and unicode codepoints from screenshots, drawings, or logos using **Hybrid CLIP Multimodal Embeddings**.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-CLIP--ViT--B--32-orange)](https://www.sbert.net)

---

## 💡 The Problem

The official [Nerd Fonts Cheat Sheet](https://www.nerdfonts.com/cheat-sheet) catalogs over **10,000 glyphs**, but finding the right icon when you only have:
- A screenshot of a terminal prompt or status bar
- A corporate logo (e.g., GitLab fox, Docker whale, Rust crab)
- A rough hand-drawn sketch or mockup

... is nearly impossible by keyword search alone unless you already know the exact maintainer name (`nf-oct-*`, `nf-fa-*`, `nf-dev-*`, `nf-md-*`). Web-scraping the cheat sheet is also fragile because glyphs render dynamically on-demand via virtual DOM.

**`nerdsight`** solves this by downloading the official font and mapping directly from the source repository, rendering every glyph to canonical canonical bitmaps, and building a **dual multimodal index** combining visual geometry and contextual semantics.

---

## 🧠 Multimodal Hybrid Fusion Architecture

`nerdsight` uses `clip-ViT-B-32` from OpenAI / OpenCLIP to project both images and natural language text into a shared 512-dimensional vector space:

1. **Visual Embedding ($\vec{V}_{img}$):** Generated from the centered 128×128 rasterization of the glyph from `SymbolsNerdFontMono-Regular.ttf`.
2. **Semantic Description Embedding ($\vec{V}_{txt}$):** Generated from the glyph name, with style tokens split from the object nouns (e.g., `md-ghost_outline` $\rightarrow$ *"a outline icon of a ghost"*, `dev-git_branch` $\rightarrow$ *"a filled icon of a git branch"*).

```
 ┌──────────────────────┐         ┌─────────────────────────┐
 │ Query Image / Sketch │         │ Nerd Fonts Glyph Source │
 └──────────┬───────────┘         └────────────┬────────────┘
            │                                  │
    [ Preprocessing ]                 ┌────────┴────────┐
   (Auto-crop, Invert,                │                 │
    Binarize, Pad)               [Render 128x128]  [Prefix Prompt]
            │                         │                 │
      [CLIP Vision]              [CLIP Vision]    [CLIP Language]
            │                         │                 │
       Query Vector               V_img (512d)     V_txt (512d)
            │                         └────────┬────────┘
            │                                  │
            └───────────────┬──────────────────┘
                            ▼
            [ Cosine vs V_img ]  [ Cosine vs V_txt ]
                            ▼
          S = α · z(vis) + (1 - α) · z(sem)
                            ▼
              Top K Ranked Nerd Font Icons
```

### The Hybrid Formula

At query time, `nerdsight` scores the query against both matrices and fuses the scores. Image-to-image and image-to-text cosines live on very different scales (~0.6–0.95 vs ~0.15–0.35), so each score vector is z-scored across the catalog first:

$$S = \alpha \cdot z(\mathbf{E}_{img}\,\vec{q}) + (1 - \alpha) \cdot z(\mathbf{E}_{txt}\,\vec{q})$$

- **$\alpha = 1.0$ (Pure Visual):** Matches strictly by geometric silhouette and line shapes.
- **$\alpha = 0.0$ (Pure Semantic):** Matches conceptually by object meaning and category.
- **$\alpha = 0.7$ *(Default)*:** Chosen by benchmark (exact-shape recall vs. concept neighbours).

---

## 🛠️ Image Normalization Pipeline

To match user uploads against pristine font glyphs, `nerdsight` includes an automated preprocessing pipeline:

1. **Alpha Flattening:** Composites PNG alpha channels against an opaque solid white canvas.
2. **Dark-Mode Inversion:** Inverts terminal prints (white-on-black) to font-compatible black-on-white.
3. **Adaptive Binarization:** Otsu-inspired thresholding converts multi-color artwork into high-contrast silhouettes.
4. **Auto-Crop & Proportional Padding:** Detects bounding box boundaries and scales the glyph to occupy exactly 72% of the canvas with centered margins, mirroring the official font metrics.

---

## 📂 Project Structure

```text
nerdsight/
├── assets/
│   ├── glyphnames.json                # Official mapping (name, char, hex code)
│   └── SymbolsNerdFontMono-Regular.ttf# Official symbol font
├── data/
│   ├── img_embeddings.npy             # Visual matrix (N, 512) in float32
│   ├── txt_embeddings.npy             # Semantic matrix (N, 512) in float32
│   └── metadata.json                  # Icon names, characters, codepoints
├── app.py                             # Interactive Streamlit Web UI
├── cli.py                             # Fast command-line terminal search
├── indexer.py                         # Glyph renderer and CLIP embedding builder
├── setup_assets.py                    # Official asset fetcher (GitHub release)
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & Environment Setup

```bash
git clone https://github.com/felipecm-br/nerdsight.git
cd nerdsight/main  # or cd nerdsight

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Official Assets

Downloads `glyphnames.json` and extracts `SymbolsNerdFontMono-Regular.ttf` from the latest Nerd Fonts release archive (~30 seconds):

```bash
python setup_assets.py
```

### 3. Build Embeddings Index

Renders all glyphs and generates decoupled visual & semantic matrices (~2 to 4 minutes on CPU, faster on CUDA/Metal):

```bash
python indexer.py
```

### 4. Launch the Web Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. Drag and drop any image or sketch to find matching icons!

---

## 💻 Terminal CLI Usage

`nerdsight` includes a fast, versatile command-line tool for developers:

```bash
# 1. Search by Icon Name (instant search across 10,000+ icons)
python cli.py --name "python"
python cli.py --name "git" --family dev

# 2. Semantic Search by Concept / Meaning (powered by CLIP)
python cli.py --text "database" --top 5
python cli.py --text "speedometer"

# 3. Browse / Show All Icons
python cli.py --all --family fa --top 20

# 4. Visual Search by Image / Screenshot
python cli.py path/to/icon.png --alpha 0.7 --top 5

# Terminal dark-mode print inversion + semantic boost
python cli.py screenshot.png --invert --boost "git"

# Output as JSON for scripts and pipelines
python cli.py --name "git" --json
```

Example Name Search Output:
```text
🔍 nerdsight name search for: 'python'
---------------------------------------------------------------------------
#   Glyph  Unicode    Collection         Match Type       Name
---------------------------------------------------------------------------
1         \uec39     Codicons           Exact Match      nf-cod-python
2         \ue73c     Devicons           Exact Match      nf-dev-python
3         \ued1b     Font Awesome       Exact Match      nf-fa-python
4         \ue235     Font Awesome Ext   Exact Match      nf-fae-python
5         \ue606     Seti UI            Exact Match      nf-seti-python
6   󰌠      \uf0320    Material Design    Word Match       nf-md-language_python
---------------------------------------------------------------------------
```
⚖️  Alpha: 0.70 (Visual: 70% / Semantic: 30%)
-----------------------------------------------------------------
#   Glyph  Similarity   Unicode     Name
-----------------------------------------------------------------
1         94.2%        \ue725      nf-dev-git_branch
2         91.1%        \uf418      nf-oct-git_branch
3         86.4%        \uf1d3      nf-fa-git
-----------------------------------------------------------------
```

---

## ⚡ Future Roadmap & Browser-Only Deployment

- **Zero-Backend Deployment (Transformers.js + WebGPU):** Quantize the precomputed matrices to `float16` (~10MB per matrix) and run `Xenova/clip-vit-base-patch32` entirely inside the client's browser using WebAssembly / WebGPU on GitHub Pages or Cloudflare Pages.
- **Multi-Scale Augmentation:** Average embeddings across multiple stroke widths during indexing for enhanced sketch recognition.

---

## 📄 License

MIT © [Felipe Miranda](https://github.com/felipecm-br)
