# Plan: Deploy nerdsight as a free Hugging Face Static Space

**Status:** implemented (2026-10-22) · **Option:** A (browser-side inference) · **Date:** 2026-10-21

## Outcome

- **Phase 0 gate passed.** The JS preprocessing (`web/js/preprocess.js`, a port of Pillow's `Resample.c`
  and `Reduce.c`) is pixel-exact with `core.preprocess_image` on all five fixtures. The CLIP processor
  resize is PIL bicubic (torch bicubic differs by up to 0.4 in normalized units), so canvas scaling is
  never used.
- **Dtype decision (measured, `NERDSIGHT_DTYPE=vision:text npm test`):**

  | Variant | Text cosine | Glyph cosine (300) | Rankings | First download |
  |---|---|---|---|---|
  | fp32 | 1.0000 | 1.0000 | identical | ~580 MB |
  | **fp16 (chosen)** | 1.0000 | 0.9994 mean | near-tie swaps; Python #1 always in top 3 | ~290 MB |
  | q8 | 0.84–0.89 | 0.92 mean | concept rankings broken | ~150 MB |

  q8 is rejected outright. Switching to fp32 is one line in `export_web.py` (`WEB_DTYPES`).
- `cru.png` is not in the repo, so `export_web.py --golden` generates synthetic fixtures covering each
  preprocessing branch (dark terminal capture, transparent PNG, large noisy image via `reduce()`, tiny
  image, wide image through `pad()`).
- **Open decisions resolved:** all three UI languages ship (strings exported from `i18n.py`); Space id
  defaults to `<hf-user>/nerdsight` (`--repo` overrides); the Streamlit app stays for local use; deploys
  are manual (`python deploy_space.py`). The sketch canvas shipped in v1. The Space card has no
  `emoji:` (no-emoji rule); Hugging Face shows its default.

## Goal

Publish nerdsight online at zero cost. Hugging Face only lets free personal accounts create **Static** Spaces
(Gradio/Docker/Streamlit Spaces on CPU now require PRO), so the server cannot run Python or PyTorch.
Everything that happens per query moves into the visitor's browser; everything expensive stays offline on
the developer machine.

## Architecture

```
 Developer machine (offline, Python)                 Static Space (files only)          Visitor's browser
 ──────────────────────────────────                  ─────────────────────────          ─────────────────
 setup_assets.py ─► glyphnames.json, .ttf ─┐
 indexer.py ──────► img/txt_embeddings.npy ├─► export_web.py ─► web/dist/ ──upload──►  index.html, app.js
                                           │      (f16 bins, catalog.json,             data/*.bin, *.json
                                           │       manifest.json, font)                 fonts/*.ttf
                                           │                                                │
                                           └─ golden.json (expected results) ──► parity test │
                                                                                            ▼
                                                     Hugging Face Hub (model CDN) ─► transformers.js loads
                                                     Xenova/clip-vit-base-patch32       CLIP ONNX (WASM),
                                                                                        cached by the browser
```

| Concern | Today (Streamlit) | Static Space |
|---|---|---|
| Glyph rendering + index build | `indexer.py` (PyTorch) | unchanged, offline |
| Name search, catalog browser | `core.search_by_name` | JS port over `catalog.json` |
| Query image preprocessing | `core.preprocess_image` (PIL) | JS port on `<canvas>` |
| Query encoding | sentence-transformers CLIP | transformers.js, same OpenAI CLIP weights in ONNX |
| Score fusion | `core.hybrid_scores` (numpy) | JS port, z-score fusion, alpha 0.7 |
| Font in UI | base64 `@font-face` | `@font-face` from `fonts/SymbolsNerdFontMono-Regular.ttf` |

The Python CLI keeps working exactly as it does now; `web/` is an additional front end over the same index.

## Constraints carried over from CLAUDE.md

- `.npy`, `.ttf` and generated binaries are never committed to the GitHub repo. `web/dist/` is gitignored and
  only ever pushed to the Space repo.
- Paths in new Python scripts resolve from `__file__` through `core.BASE_DIR`.
- Name search and catalog browsing must stay instant and must not wait for any model download.

---

## Phase 0 — Parity spike (go/no-go gate, ~2 h)

The browser encodes queries with an ONNX export of CLIP while the index was built with PyTorch. Prove they
agree before writing any UI.

1. Add `scripts/parity_onnx.py`: load `Xenova/clip-vit-base-patch32` ONNX encoders with `onnxruntime`
   (vision + text, each in fp32, fp16 and q8 variants) and encode:
   - 300 rendered glyphs (reuse `indexer.render_glyph`),
   - `cru.png` after `core.preprocess_image`,
   - the text queries `ghost`, `database`, `git branch`, `bell outline`.
2. Compare with the PyTorch vectors in `data/*.npy`.

**Pass criteria (per variant):**
- mean cosine(ONNX, PyTorch) ≥ 0.99 on glyph image embeddings;
- `cru.png` top 3 is still `nf-md-ghost_outline`, `nf-fa-ghost`, `nf-md-ghost`;
- text-query top 5 identical to `cli.py --text`.

**Decision:** pick the smallest variant that passes (target: q8, about 90 MB vision download). If no browser
variant passes, rebuild the index *with the chosen ONNX variant* (onnxruntime in Python) so index and query
share numerics exactly. Write the choice into `manifest.json` (Phase 1).

## Phase 1 — Web index export (~3 h)

New script `export_web.py` (repo root, imports `core`) writes `web/dist/`:

| File | Content | Size (approx.) |
|---|---|---|
| `data/img_emb.f16.bin` | 10,995 × 512 float16, row order = `catalog.json` | 11 MB |
| `data/txt_emb.f16.bin` | same shape, semantic prompt embeddings | 11 MB |
| `data/catalog.json` | `[{key, name, char, code, prefix, family, indexed}]` for every glyph | 1 MB |
| `data/manifest.json` | model id + dtype per encoder, embedding dim, count, default alpha, prompt template, query template (`an icon symbol of {q}`), preprocessing constants (threshold 185, crop box 0.72, size 128), sha256 of each bin, build date | <1 KB |
| `data/golden.json` | expected top-10 from Python for `cru.png` and the four text queries | <10 KB |
| `fonts/SymbolsNerdFontMono-Regular.ttf` | copied from `assets/` | 2.5 MB |

Notes:
- Float16 halves the download; the browser converts once to `Float32Array` (22 MB each in memory).
  Verify the f16 round-trip does not change the golden top-10 before shipping.
- Rows are emitted in one canonical order so JS can index embeddings by catalog position. Catalog
  entries without an embedding (none today) get `indexed: false`.
- Add `web/dist/` to `.gitignore`.

## Phase 2 — Static front end (~1–1.5 days)

No build step: plain ES modules plus `@huggingface/transformers` loaded from jsDelivr at an **exact pinned
version** (choose the current stable release at implementation time; never a floating tag). If CDN
availability becomes a concern, vendor the library file into `web/dist/vendor/`.

```
web/
  index.html          # layout, tabs, @font-face
  style.css           # light/dark tokens, responsive grid
  js/
    main.js           # tab routing, state, lazy model loading
    catalog.js        # load catalog.json, name search (port of core.search_by_name), family filter, pagination
    preprocess.js     # port of core.preprocess_image on <canvas>
    model.js          # transformers.js: vision + text encoders, progress events
    search.js         # dot products, z-score fusion, top-k
    ui.js             # icon cards, copy buttons (glyph, nf-* class, \uXXXX)
```

Feature parity with `app.py`:

1. **Name search + catalog browser** — available on first paint, before any model loads.
   Port the ranking tiers exactly: Exact 100 → Word 95 → Prefix 90 → Substring 85.
2. **Concept search** — loads the text encoder on first use (progress bar), encodes
   `an icon symbol of {query}`, ranks `txt_emb` by cosine.
3. **Image search** — upload, drag-and-drop and clipboard paste; options for invert, binarize, auto-crop,
   alpha slider (default from manifest), optional text boost (0.75 image / 0.25 text blend, as in
   `core.encode_image_query`). Show the preprocessed 128×128 preview like the current app.
4. *(Optional, later)* sketch canvas, replacing `streamlit_drawable_canvas`.

`preprocess.js` must mirror PIL step for step, because it is the most likely source of silent drift:
- composite alpha onto white → optional invert;
- grayscale with PIL's `L` weights (0.299 R + 0.587 G + 0.114 B);
- autocontrast (stretch min..max to 0..255) → threshold `< 185 → 0 else 255`;
- bounding box of pixels darker than 226 (PIL: inverted ≥ 30) → crop;
- fit inside 92 × 92 (`0.72 × 128`) keeping aspect ratio → paste centered on a 128 × 128 white canvas.

`search.js` ports `core.hybrid_scores`: two dot-product passes (10,995 × 512, ~11M multiply-adds, a few ms),
z-score each vector, `alpha·z(vis) + (1−alpha)·z(sem)`, min-max to [0, 1] for display.

Performance budget:
- first paint with name search ready < 2 s on broadband (catalog + font ≈ 3.5 MB);
- vision model first download ≈ 90 MB (q8), then served from browser cache;
- image query after the model is warm < 1 s on a mid-range laptop.

Language: keep the current pt-BR UI copy unless decided otherwise (see Open decisions).

## Phase 3 — Verification (~0.5 day)

1. **Automated parity test** `web/tests/parity.mjs` (Node + `@huggingface/transformers` + a headless
   browser, or Playwright alone): serve `web/dist`, upload `cru.png`, run the four text queries, and
   assert the top-k matches `golden.json` (top 3 exact, top 10 overlap ≥ 8).
2. **Preprocessing check:** export the JS 128×128 preview of `cru.png` and diff it against
   `core.preprocess_image` output; mean absolute pixel difference ≤ 2/255.
3. **Manual pass:** Chrome, Firefox and Safari; one phone (Android or iOS) for memory and layout;
   dark mode; a slow-3G throttle run to check that the loading states read well.

## Phase 4 — Deploy (~2 h)

1. Create the Space (once): `huggingface-cli repo create nerdsight --type space --space_sdk static`
   (or through the web UI), owned by the personal account.
2. Space front matter (`web/dist/README.md`, generated by `export_web.py`):
   ```yaml
   ---
   title: nerdsight
   emoji: 🔍
   sdk: static
   app_file: index.html
   pinned: false
   license: mit
   ---
   ```
3. New script `deploy_space.py`: runs `export_web.py`, copies `web/` sources into `web/dist/`, then
   `huggingface_hub.HfApi().upload_folder(repo_id="<user>/nerdsight", repo_type="space", folder_path="web/dist")`.
   The token comes from `HF_TOKEN` or `huggingface-cli login`; it is never written to disk by the script.
4. Optional: a GitHub Actions workflow on tag push. It needs the generated index, so either
   (a) keep `data/*.npy` in a private HF dataset repo that CI downloads, or (b) deploy manually from the
   developer machine. Start with (b).
5. Smoke-test the live URL with the Phase 3 checklist.

## Phase 5 — Follow-ups (not in scope)

- **SigLIP upgrade:** `Xenova/siglip-base-patch16-224` exists for transformers.js. Re-index offline with the same
  checkpoint, re-run Phase 0 parity, re-export. The UI does not change; `manifest.json` selects the model.
- WebGPU backend when available (fallback stays WASM).
- Sketch canvas input.
- Attribution page for the bundled icon sets.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| ONNX/q8 embeddings drift from the PyTorch index | wrong rankings | Phase 0 gate; fall back to fp16/fp32 or re-index with the ONNX model |
| JS preprocessing differs from PIL | silent ranking drift on uploads | step-by-step port + pixel-diff test in Phase 3 |
| 90–180 MB first model download | slow first image search, mobile data use | lazy load only on first image/concept query; progress bar; name search never waits |
| Mobile memory pressure | tab crash on low-end phones | q8 model; release the text encoder when idle if needed; test on a real phone |
| Static Space size or bandwidth limits (not documented on the pages read) | deploy rejected or throttled | total payload ≈ 26 MB plus models served by the Hub CDN; re-check limits at deploy time |
| Hugging Face changes free-tier rules again | hosting lost | the build is plain static files: GitHub Pages, Cloudflare Pages or Netlify work unchanged |
| Icon-set licences | compliance | Nerd Fonts bundles sets under their own licences (e.g. MIT, Apache-2.0, CC BY 4.0, OFL); add an attribution section linking the upstream licence list |

## Effort

| Phase | Estimate |
|---|---|
| 0 Parity spike | 2 h |
| 1 Export | 3 h |
| 2 Front end | 1–1.5 days |
| 3 Verification | 0.5 day |
| 4 Deploy | 2 h |
| **Total** | **≈ 2.5–3 days** |

## Open decisions

1. **UI language:** keep pt-BR (current app), switch to English, or ship both?
2. **Space name and owner:** `<user>/nerdsight`?
3. **Keep the Streamlit app** for local use, or retire it once the static site reaches parity?
4. **CI deploy** (needs the index in a private HF dataset) vs manual deploy from the developer machine.

## References

- Spaces overview (SDKs, free vs PRO, hardware): https://huggingface.co/docs/hub/en/spaces-overview
- Static HTML Spaces: https://huggingface.co/docs/hub/en/spaces-sdks-static
- transformers.js: https://huggingface.co/docs/transformers.js
- Browser CLIP checkpoint: https://huggingface.co/Xenova/clip-vit-base-patch32
