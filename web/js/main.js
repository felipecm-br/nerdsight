// nerdsight static front end: the three views of app.py over files exported by
// export_web.py. Name search and the catalog work as soon as catalog.json
// arrives; the CLIP encoders and embedding matrices load on first use.

import { expandCatalog, filterCatalog, searchByName } from "./catalog.js";
import { argsortDesc, decodeF16, dot, hybridScores, relevance } from "./search.js";
import { hasDarkBorder, preprocessImage } from "./preprocess.js";
import { createEncoders } from "./model.js";
import * as ui from "./ui.js";

const { t, esc, md } = ui;

// Exact pinned version: the browser must load the same library the parity tests ran.
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";

const VIEWS = ["search", "visual", "catalog"];
const MODES = ["hybrid", "name", "concept"];
const MATCH_KEYS = { exact: "match_exact", word: "match_word", prefix: "match_prefix", substring: "match_substring" };
const SUGGESTIONS = ["git", "python", "folder", "database", "rocket", "cloud upload", "battery low", "warning"];
const TOP_K = [24, 48, 96, 192];
const VISUAL_TOP_K = [12, 24, 48, 96];
const PAGE_SIZES = [96, 192, 384, "all"];
const LARGE_RENDER = 1500;

const $ = (sel) => document.querySelector(sel);
const debounce = (fn, ms) => {
  let id;
  return (...args) => {
    clearTimeout(id);
    id = setTimeout(() => fn(...args), ms);
  };
};

// ------------------------------------------------------------------ state

const params = new URLSearchParams(location.search);
const pick = (key, options, fallback) => (options.includes(params.get(key)) ? params.get(key) : fallback);

const state = {
  lang: "en",
  view: pick("view", VIEWS, "search"),
  q: params.get("q") ?? "",
  mode: pick("mode", MODES, "hybrid"),
  family: null,
  topK: 48,
  catalog: { family: null, filter: "", perPage: 96, page: 1 },
  visual: { source: "upload", image: null, imageId: 0, topK: 24 },
};

let I18N;
let manifest;
let items = [];
let byRow = [];
let families = [];
let counts = {};

function syncUrl() {
  const p = new URLSearchParams();
  if (state.q) p.set("q", state.q);
  if (state.mode !== "hybrid") p.set("mode", state.mode);
  if (state.view !== "search") p.set("view", state.view);
  if (state.lang !== I18N.default) p.set("lang", state.lang);
  const qs = p.toString();
  history.replaceState(null, "", qs ? `?${qs}` : location.pathname);
}

// ---------------------------------------------------------- lazy loading

const fetchJson = async (path) => {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
};

const matrices = {};
const loadMatrix = (name) =>
  (matrices[name] ??= fetch(`data/${name}`)
    .then((res) => {
      if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
      return res.arrayBuffer();
    })
    .then(decodeF16)
    .catch((err) => {
      delete matrices[name];
      throw err;
    }));

let encoders;
const progress = { text: {}, vision: {} };

async function getEncoders() {
  if (!encoders) {
    const tf = await import(TRANSFORMERS_URL);
    encoders = createEncoders(tf, manifest, { onProgress: onModelProgress });
  }
  return encoders;
}

function onModelProgress({ part, status, file, loaded, total }) {
  const files = progress[part];
  if (status === "progress" && total) files[file] = [loaded, total];
  if (status === "done" && files[file]) files[file][0] = files[file][1];
  let sum = 0;
  let size = 0;
  for (const [l, tt] of Object.values(files)) {
    sum += l;
    size += tt;
  }
  const pct = size ? Math.floor((sum / size) * 100) : 0;
  for (const node of document.querySelectorAll(`[data-model-status="${part}"]`)) {
    const label = t(`web_part_${part}`);
    node.querySelector("span").textContent = pct > 0 && pct < 100 ? t("web_model_downloading", { part: label, pct }) : t("web_model_starting", { part: label });
    const bar = node.querySelector("progress");
    if (bar) bar.value = pct;
  }
}

const statusHtml = (part) =>
  `<div class="ns-status-line" role="status" data-model-status="${part}"><span>${esc(t("web_model_starting", { part: t(`web_part_${part}`) }))}</span><progress max="100" value="0"></progress></div>`;

const errorHtml = (part, err) =>
  `<div class="ns-alert" role="alert"><span>${esc(t("web_model_error", { part: t(`web_part_${part}`), err: err.message || err }))}</span>
   <button type="button" class="ns-btn" data-retry>${esc(t("web_retry"))}</button></div>`;

// ----------------------------------------------------------------- search

const conceptCache = new Map();
let searchToken = 0;

async function conceptOrder(q) {
  if (conceptCache.has(q)) return conceptCache.get(q);
  const [enc, txt] = await Promise.all([getEncoders(), loadMatrix("txt_emb.f16.bin")]);
  const vec = await enc.encodeText(q);
  const scores = dot(txt, manifest.dim, vec);
  const order = argsortDesc(scores);
  const result = { order, widths: relevance(order.map((i) => scores[i]), scores) };
  conceptCache.set(q, result);
  return result;
}

async function runSearch() {
  const token = ++searchToken;
  const q = state.q.trim();
  const out = $("#search-results");
  $("#search-idle").hidden = Boolean(q);
  out.hidden = !q;
  if (!q) return;

  const pool = state.family === null ? items : items.filter((i) => i.family === state.family);
  const cards = [];
  const seen = new Set();
  let nName = 0;
  let nConcept = 0;
  let nameTotal = 0;

  if (state.mode !== "concept") {
    const matches = searchByName(pool, q);
    nameTotal = matches.length;
    for (const { item, match } of matches.slice(0, state.topK)) {
      cards.push(ui.cardHtml(item, t(MATCH_KEYS[match]), undefined, match === "exact"));
      seen.add(item.name);
      nName++;
    }
  }

  const wantConcept = state.mode !== "name" && cards.length < state.topK;
  const render = (extra = "") => {
    const inFamily = state.family ? t("in_family", { family: esc(state.family) }) : "";
    if (!cards.length && !extra) {
      out.innerHTML = `<p>${md(t("no_name_match", { q: esc(q) }))}${inFamily}.</p><div class="ns-actions">
        ${state.mode === "name" ? `<button type="button" class="ns-btn" data-action="concept"><span class="ns-ic" aria-hidden="true">\u{f0674}</span>${esc(t("btn_search_concept"))}</button>` : ""}
        ${state.family ? `<button type="button" class="ns-btn" data-action="clear-family"><span class="ns-ic" aria-hidden="true">\u{f14ec}</span>${esc(t("btn_clear_family"))}</button>` : ""}
      </div>`;
      return;
    }
    const parts = [];
    if (nName) parts.push(t("by_name", { n: nName }) + (nameTotal > nName ? t("of_total", { total: t.int(nameTotal) }) : ""));
    if (nConcept) parts.push(t("by_concept", { n: nConcept }));
    const summary = cards.length
      ? ui.summaryHtml(t("search_summary", { n: cards.length, q: esc(q) }) + (parts.length ? ` · ${parts.join(", ")}` : ""))
      : "";
    out.innerHTML = summary + extra + (cards.length ? ui.gridHtml(cards, t("results_for", { q })) : "");
  };

  if (!wantConcept) return render();
  const cached = conceptCache.get(q);
  if (!cached) render(statusHtml("text"));
  try {
    const { order, widths } = cached ?? (await conceptOrder(q));
    if (token !== searchToken) return;
    for (let k = 0; k < order.length && cards.length < state.topK; k++) {
      const item = byRow[order[k]];
      if (!item || seen.has(item.name) || (state.family && item.family !== state.family)) continue;
      cards.push(ui.cardHtml(item, t("match_concept"), widths[k]));
      seen.add(item.name);
      nConcept++;
    }
    render();
  } catch (err) {
    if (token === searchToken) render(errorHtml("text", err));
  }
}

const runSearchSoon = debounce(runSearch, 300);

function initSearch() {
  const input = $("#q");
  input.value = state.q;
  input.addEventListener("input", () => {
    state.q = input.value;
    syncUrl();
    runSearchSoon();
  });
  const mode = ui.segmented($("#mode"), (m) => {
    state.mode = m;
    syncUrl();
    runSearch();
  });
  const family = ui.picker($("#family"), (f) => {
    state.family = f;
    runSearch();
  });
  const topK = ui.picker($("#top-k"), (k) => {
    state.topK = k;
    runSearch();
  });
  const suggestions = $("#suggestions");
  for (const s of SUGGESTIONS) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "ns-pill";
    b.textContent = s;
    b.addEventListener("click", () => {
      input.value = state.q = s;
      syncUrl();
      runSearch();
    });
    suggestions.append(b);
  }
  $("#search-results").addEventListener("click", (event) => {
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (action === "concept") {
      state.mode = "concept";
      mode.set("concept");
      syncUrl();
    } else if (action === "clear-family") {
      state.family = null;
      family.set(null);
    } else if (!event.target.closest("[data-retry]")) return;
    runSearch();
  });
  return {
    relabel() {
      ui.segmentedOptions($("#mode"), MODES.map((m) => ({ value: m, label: t(`mode_${m}`) })));
      mode.set(state.mode);
      family.setOptions(
        [{ value: null, label: t("all_families") }, ...families.map((f) => ({ value: f, label: `${f} (${t.int(counts[f])})` }))],
        state.family,
      );
      topK.setOptions(TOP_K.map((n) => ({ value: n, label: String(n) })), state.topK);
    },
  };
}

// ----------------------------------------------------------------- visual

let visualToken = 0;
let lastVec = null; // {key, vec}: alpha changes only re-score, never re-encode

async function setImage(img) {
  state.visual.image = img;
  state.visual.imageId++;
  lastVec = null;
  $("#opt-invert").checked = hasDarkBorder(img, manifest.preprocess.darkBorder);
  ui.drawImage($("#preview-original"), img);
  runVisual();
}

async function decodeFile(file) {
  try {
    const bitmap = await createImageBitmap(file, { premultiplyAlpha: "none", colorSpaceConversion: "none" });
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);
    const { data, width, height } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return { width, height, channels: 4, data };
  } catch {
    showBanner(t("web_bad_image"));
    return null;
  }
}

async function runVisual() {
  const token = ++visualToken;
  const img = state.visual.image;
  const has = Boolean(img);
  $("#visual-empty").hidden = has;
  $("#visual-query").hidden = !has;
  const out = $("#visual-results");
  out.hidden = !has;
  if (!has) return;

  const opts = {
    invertColors: $("#opt-invert").checked,
    binarize: $("#opt-binarize").checked,
    autoCrop: $("#opt-crop").checked,
    ...manifest.preprocess,
  };
  const processed = preprocessImage(img, opts);
  ui.drawImage($("#preview-processed"), processed);

  const alpha = Number($("#alpha").value);
  const boost = $("#boost").value.trim();
  $("#weights").innerHTML = md(t("weights", { shape: `${Math.round(alpha * 100)}%`, concept: `${Math.round((1 - alpha) * 100)}%` }));
  $("#boost-note").hidden = !boost;
  $("#boost-note").innerHTML = boost ? md(t("boost_note", { hint: esc(boost) })) : "";

  const key = `${state.visual.imageId}|${opts.invertColors}|${opts.binarize}|${opts.autoCrop}|${boost}`;
  try {
    if (lastVec?.key !== key) {
      if (!out.querySelector(".ns-grid")) out.innerHTML = statusHtml("vision");
      const [enc] = await Promise.all([getEncoders(), loadMatrix("img_emb.f16.bin"), loadMatrix("txt_emb.f16.bin")]);
      const vec = await enc.encodeImage(processed, boost);
      if (token !== visualToken) return;
      lastVec = { key, vec };
    }
    const [img_, txt] = await Promise.all([loadMatrix("img_emb.f16.bin"), loadMatrix("txt_emb.f16.bin")]);
    if (token !== visualToken) return;
    const scores = hybridScores(lastVec.vec, img_, txt, manifest.dim, alpha);
    const top = argsortDesc(scores, state.visual.topK);
    const widths = relevance(top.map((i) => scores[i]), scores);
    const cards = top.map((r, k) => ui.cardHtml(byRow[r], t("match_similar"), widths[k]));
    out.innerHTML =
      `<div class="ns-row-between">${ui.summaryHtml(t("visual_summary", { n: cards.length }))}<div class="ns-picker" id="visual-top-k"></div></div>` +
      ui.gridHtml(cards, t("visual_grid"));
    out.dataset.imageId = state.visual.imageId;
    const p = ui.picker(out.querySelector("#visual-top-k"), (n) => {
      state.visual.topK = n;
      runVisual();
    });
    p.setOptions(VISUAL_TOP_K.map((n) => ({ value: n, label: t("n_results", { n }) })), state.visual.topK);
    out.querySelector(".ns-pick-btn").setAttribute("aria-label", t("results"));
  } catch (err) {
    if (token === visualToken) out.innerHTML = errorHtml("vision", err);
  }
}

function initSketch() {
  const canvas = $("#sketch");
  const ctx = canvas.getContext("2d");
  const clear = () => {
    ctx.fillStyle = "#fff"; // ds-allow-hardcode: model input (black on white like the rendered glyphs), not theme
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };
  clear();
  let drawing = false;
  let dirty = false;
  const pos = (e) => {
    const r = canvas.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * canvas.width, ((e.clientY - r.top) / r.height) * canvas.height];
  };
  canvas.addEventListener("pointerdown", (e) => {
    drawing = true;
    canvas.setPointerCapture(e.pointerId);
    ctx.beginPath();
    ctx.moveTo(...pos(e));
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    ctx.lineWidth = 6;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.strokeStyle = "#000"; // ds-allow-hardcode: model input ink
    ctx.lineTo(...pos(e));
    ctx.stroke();
    dirty = true;
  });
  const end = () => {
    if (!drawing) return;
    drawing = false;
    if (!dirty) return;
    const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setImage({ width, height, channels: 4, data });
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  $("#sketch-clear").addEventListener("click", () => {
    clear();
    dirty = false;
    clearImage();
  });
}

function clearImage() {
  state.visual.image = null;
  lastVec = null;
  $("#file").value = "";
  runVisual();
}

function initVisual() {
  const alpha = $("#alpha");
  alpha.value = manifest.alpha;
  const showAlpha = () => {
    $("#alpha-out").textContent = t.decimal(Number(alpha.value), 2);
  };
  showAlpha();
  alpha.addEventListener("input", () => {
    showAlpha();
    runVisual();
  });
  $("#boost").addEventListener("input", debounce(runVisual, 400));
  for (const id of ["#opt-crop", "#opt-binarize", "#opt-invert"]) $(id).addEventListener("change", runVisual);

  const drop = $("#drop");
  const file = $("#file");
  const takeFile = async (f) => {
    if (!f) return;
    const img = await decodeFile(f);
    if (img) setImage(img);
  };
  drop.addEventListener("click", () => file.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      file.click();
    }
  });
  file.addEventListener("change", () => takeFile(file.files[0]));
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.setAttribute("data-over", "");
  });
  drop.addEventListener("dragleave", () => drop.removeAttribute("data-over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.removeAttribute("data-over");
    takeFile(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", (e) => {
    const f = [...(e.clipboardData?.files ?? [])].find((x) => x.type.startsWith("image/"));
    if (!f) return;
    e.preventDefault();
    if (state.view !== "visual") setView("visual");
    if (state.visual.source !== "upload") source.set((state.visual.source = "upload")) || showSource();
    takeFile(f);
  });
  $("#remove-image").addEventListener("click", clearImage);
  $("#visual-results").addEventListener("click", (e) => {
    if (e.target.closest("[data-retry]")) runVisual();
  });

  const showSource = () => {
    $("#drop").hidden = state.visual.source !== "upload";
    $("#draw").hidden = state.visual.source !== "draw";
  };
  const source = ui.segmented($("#source"), (s) => {
    state.visual.source = s;
    showSource();
    clearImage();
  });
  source.set(state.visual.source);
  showSource();
  initSketch();
  return {
    relabel() {
      showAlpha();
      $("#alpha").setAttribute("aria-label", t("alpha_label"));
      $("#alpha").title = t("alpha_help", { alpha: t.decimal(manifest.alpha) });
      if (state.visual.image) runVisual();
    },
  };
}

// ---------------------------------------------------------------- catalog

function runCatalog() {
  const c = state.catalog;
  const out = $("#catalog-results");
  let list = c.family === null ? items : items.filter((i) => i.family === c.family);
  list = filterCatalog(list, c.filter);
  const total = list.length;
  const inFamily = c.family ? t("in_family", { family: esc(c.family) }) : "";
  if (!total) {
    out.innerHTML = ui.emptyHtml(esc(t("filter_empty_title")), t("filter_empty_body", { term: esc(c.filter.trim()), family: inFamily }));
    return;
  }
  const size = c.perPage === "all" ? total : c.perPage;
  const pages = Math.max(1, Math.ceil(total / size));
  c.page = Math.min(Math.max(1, c.page), pages);
  const start = (c.page - 1) * size;
  const pageItems = list.slice(start, start + size);
  let html = ui.summaryHtml(t("catalog_summary", { a: t.int(start + 1), b: t.int(start + pageItems.length), total: t.int(total), family: inFamily }));
  if (c.perPage === "all" && total > LARGE_RENDER) html += `<p class="ns-caption">${esc(t("large_render", { n: t.int(total) }))}</p>`;
  html += ui.gridHtml(pageItems.map((i) => ui.cardHtml(i)), t("catalog_grid"));
  if (pages > 1) {
    html += `<nav class="ns-pager" aria-label="${esc(t("catalog_grid"))}">
      <button type="button" class="ns-btn" data-page="${c.page - 1}" ${c.page <= 1 ? "disabled" : ""}><span class="ns-ic" aria-hidden="true">\u{f0141}</span>${esc(t("prev"))}</button>
      <span aria-live="polite">${t("page", { page: c.page, pages })}</span>
      <button type="button" class="ns-btn" data-page="${c.page + 1}" ${c.page >= pages ? "disabled" : ""}>${esc(t("next"))}<span class="ns-ic" aria-hidden="true">\u{f0142}</span></button>
    </nav>`;
  }
  out.innerHTML = html;
}

function initCatalog() {
  const c = state.catalog;
  const filter = $("#catalog-filter");
  filter.addEventListener(
    "input",
    debounce(() => {
      c.filter = filter.value;
      c.page = 1;
      runCatalog();
    }, 200),
  );
  const perPage = ui.picker($("#per-page"), (n) => {
    c.perPage = n;
    c.page = 1;
    runCatalog();
  });
  $("#catalog-results").addEventListener("click", (e) => {
    const b = e.target.closest("[data-page]");
    if (!b) return;
    c.page = Number(b.dataset.page);
    runCatalog();
    $("#catalog-results").scrollIntoView({ block: "start" });
  });
  return {
    relabel() {
      ui.pills($("#catalog-families"), families.map((f) => ({ value: f, label: `${f} ${t.int(counts[f])}` })), c.family, (f) => {
        c.family = f;
        c.page = 1;
        runCatalog();
      });
      perPage.setOptions(PAGE_SIZES.map((s) => ({ value: s, label: s === "all" ? t("all") : t("n_per_page", { n: s }) })), c.perPage);
      $("#per-page .ns-pick-btn").setAttribute("aria-label", t("per_page"));
    },
  };
}

// ------------------------------------------------------------------- shell

function showBanner(msg) {
  const b = $("#banner");
  b.textContent = msg;
  b.hidden = !msg;
}

const views = {};
let viewSeg;

function setView(v) {
  state.view = v;
  viewSeg?.set(v);
  for (const name of VIEWS) $(`#view-${name}`).hidden = name !== v;
  syncUrl();
  if (v === "catalog") runCatalog();
}

function setLang(lang) {
  state.lang = lang;
  ui.setStrings(I18N.strings[lang], I18N.strings[I18N.default]);
  document.documentElement.lang = lang;
  // ui/nerdsight.js reads its strings from here at use time.
  const js = Object.fromEntries(Object.entries(I18N.strings[lang]).filter(([k]) => k.startsWith("js_")).map(([k, v]) => [k.slice(3), v]));
  window.__nsI18n = { ...js, search_label: t("search_label"), filter_label: t("filter_label"), lang };
  ui.applyStrings();
  $("#stat-glyphs").textContent = t.int(items.length);
  $("#stat-families").textContent = String(families.length);
  $("#stat-index").textContent = t("index_vectors", { n: t.int(manifest.count) });
  $("#stat-index").dataset.ok = "true";
  $("#footer-icons").textContent = t("web_footer_icons", { version: I18N.nerdFontsVersion });
  for (const v of Object.values(views)) v.relabel();
  syncUrl();
  runSearch();
  if (state.view === "catalog") runCatalog();
}

async function boot() {
  try {
    const [strings, man, cat] = await Promise.all([fetchJson("data/i18n.json"), fetchJson("data/manifest.json"), fetchJson("data/catalog.json")]);
    I18N = { ...strings, nerdFontsVersion: cat.nerd_fonts_version };
    manifest = man;
    items = expandCatalog(cat);
    for (const item of items) {
      if (item.row >= 0) byRow[item.row] = item;
      counts[item.family] = (counts[item.family] ?? 0) + 1;
    }
    families = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  } catch (err) {
    showBanner(`Could not load the catalog: ${err.message}`);
    return;
  }

  const langs = Object.keys(I18N.languages);
  const requested = params.get("lang");
  const browser = navigator.languages?.map((l) => langs.find((x) => x === l || x.split("-")[0] === l.split("-")[0])).find(Boolean);
  state.lang = langs.includes(requested) ? requested : browser ?? I18N.default;

  ui.segmentedOptions($("#lang"), langs.map((l) => ({ value: l, label: I18N.languages[l] })));
  ui.segmented($("#lang"), setLang).set(state.lang);
  viewSeg = ui.segmented($("#view"), setView);

  views.search = initSearch();
  views.visual = initVisual();
  views.catalog = initCatalog();
  setLang(state.lang);
  setView(state.view);
  document.body.dataset.ready = "";
}

boot();
