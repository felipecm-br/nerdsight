// DOM helpers for the static front end: i18n, cards (port of app.card_html),
// a segmented radio group, a listbox picker and toggle pills.

let strings = {};
let fallback = {};

export function setStrings(table, english) {
  strings = table;
  fallback = english;
}

export const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/** Translator.__call__: callers escape any values that end up in HTML. */
export function t(key, vars = {}) {
  const text = strings[key] ?? fallback[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? "");
}
t.int = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, t("thousands_sep"));
t.decimal = (x, digits = 1) => x.toFixed(digits).replace(".", t("decimal_sep"));

/** The few Markdown marks used in i18n.py strings (Streamlit renders them natively). */
export const md = (s) =>
  s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>");

export function applyStrings(root = document) {
  for (const node of root.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of root.querySelectorAll("[data-i18n-html]")) node.innerHTML = t(node.dataset.i18nHtml);
  for (const node of root.querySelectorAll("[data-i18n-label]")) node.setAttribute("aria-label", t(node.dataset.i18nLabel));
  for (const node of root.querySelectorAll("[data-i18n-title]")) node.title = t(node.dataset.i18nTitle);
  for (const node of root.querySelectorAll("[data-i18n-placeholder]")) node.placeholder = t(node.dataset.i18nPlaceholder);
}

// ------------------------------------------------------------------ cards

/** app.display_name: drops nf-, glues the family prefix, lets long names wrap after underscores. */
const displayName = (name) => esc(name.replace(/^nf-/, "")).replace("-", "‑").replaceAll("_", "_<wbr>");

/** app.card_html. score is a 0..1 bar width; strong highlights exact or close matches. */
export function cardHtml(item, label, score, strong = false) {
  const name = esc(item.name);
  const code = esc(item.code.toLowerCase());
  const family = esc(item.family);
  const detail = label ? esc(label) : family;
  const bar = score === undefined ? "" : `<span class="ns-bar" aria-hidden="true"><i style="width:${Math.round(score * 100)}%"></i></span>`;
  const strongAttr = strong || (score !== undefined && score >= 0.75) ? " data-strong" : "";
  return (
    `<button type="button" class="ns-card"${strongAttr} aria-label="${name}, U+${code.toUpperCase()}, ${detail}" title="${name}" ` +
    `data-char="${esc(item.char)}" data-name="${name}" data-code="${code}" data-family="${family}">` +
    `<span class="ns-glyph" aria-hidden="true">${esc(item.char)}</span>` +
    `<span class="ns-name">${displayName(item.name)}</span>` +
    `<span class="ns-meta"><span class="ns-code">${code}</span>${bar}<span>${detail}</span></span></button>`
  );
}

export const gridHtml = (cards, label) => `<div class="ns-grid" role="group" aria-label="${esc(label)}">${cards.join("")}</div>`;

export function summaryHtml(text) {
  return `<div class="ns-summary"><span>${text}</span>
    <span class="ns-keys" aria-label="${esc(t("keys_label"))}">
      <span><kbd>/</kbd>${esc(t("key_search"))}</span>
      <span><kbd>Enter</kbd>${esc(t("key_details"))}</span>
      <span><kbd>C</kbd>${esc(t("key_copy"))}</span>
    </span></div>`;
}

export const emptyHtml = (title, body) => `<div class="ns-empty"><h3>${title}</h3><p>${body}</p></div>`;

// ------------------------------------------------------------- segmented

/** Radio group with roving focus. Buttons carry data-value; returns {set}. */
export function segmented(host, onChange) {
  const buttons = () => [...host.querySelectorAll("[role=radio]")];
  const set = (value) => {
    for (const b of buttons()) {
      const on = b.dataset.value === value;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    }
  };
  host.addEventListener("click", (event) => {
    const b = event.target.closest("[role=radio]");
    if (!b || b.getAttribute("aria-checked") === "true") return;
    set(b.dataset.value);
    onChange(b.dataset.value);
  });
  host.addEventListener("keydown", (event) => {
    const list = buttons();
    const i = list.indexOf(document.activeElement);
    if (i < 0) return;
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const next = list[(i + step + list.length) % list.length];
    next.focus();
    next.click();
  });
  return { set };
}

export function segmentedOptions(host, options) {
  host.replaceChildren(
    ...options.map(({ value, label }) => {
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "radio");
      b.dataset.value = value;
      b.textContent = label;
      return b;
    }),
  );
}

// ---------------------------------------------------------------- picker

let pickerSeq = 0;

/** Listbox picker (button + popup) replacing a native select. */
export function picker(host, onChange) {
  const id = `ns-pick-${++pickerSeq}`;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ns-pick-btn";
  button.setAttribute("aria-haspopup", "listbox");
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-controls", id);
  if (host.dataset.labelledby) button.setAttribute("aria-labelledby", `${host.dataset.labelledby} ${id}-value`);
  const valueEl = document.createElement("span");
  valueEl.id = `${id}-value`;
  const caret = document.createElement("span");
  caret.className = "ns-ic";
  caret.setAttribute("aria-hidden", "true");
  caret.textContent = "\u{f0140}"; // nf-md-chevron_down
  button.append(valueEl, caret);

  const list = document.createElement("ul");
  list.id = id;
  list.className = "ns-pick-list";
  list.setAttribute("role", "listbox");
  list.tabIndex = -1;
  list.hidden = true;
  host.replaceChildren(button, list);

  let options = [];
  let current;
  let active = 0;

  const render = () => {
    list.replaceChildren(
      ...options.map((o, i) => {
        const li = document.createElement("li");
        li.id = `${id}-${i}`;
        li.setAttribute("role", "option");
        li.setAttribute("aria-selected", String(o.value === current));
        li.textContent = o.label;
        return li;
      }),
    );
    const sel = options.find((o) => o.value === current);
    valueEl.textContent = sel ? sel.label : "";
  };
  const highlight = (i) => {
    active = Math.max(0, Math.min(options.length - 1, i));
    for (const [j, li] of [...list.children].entries()) li.toggleAttribute("data-active", j === active);
    const li = list.children[active];
    if (li) {
      list.setAttribute("aria-activedescendant", li.id);
      li.scrollIntoView({ block: "nearest" });
    }
  };
  const open = () => {
    list.hidden = false;
    button.setAttribute("aria-expanded", "true");
    highlight(Math.max(0, options.findIndex((o) => o.value === current)));
    list.focus();
  };
  const close = (refocus = true) => {
    if (list.hidden) return;
    list.hidden = true;
    button.setAttribute("aria-expanded", "false");
    if (refocus) button.focus();
  };
  const choose = (i) => {
    const o = options[i];
    close();
    if (!o || o.value === current) return;
    current = o.value;
    render();
    onChange(current);
  };

  button.addEventListener("click", () => (list.hidden ? open() : close()));
  button.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      open();
    }
  });
  list.addEventListener("click", (event) => {
    const li = event.target.closest("[role=option]");
    if (li) choose([...list.children].indexOf(li));
  });
  list.addEventListener("keydown", (event) => {
    const keys = {
      ArrowDown: () => highlight(active + 1),
      ArrowUp: () => highlight(active - 1),
      Home: () => highlight(0),
      End: () => highlight(options.length - 1),
      Enter: () => choose(active),
      " ": () => choose(active),
      Escape: () => close(),
      Tab: () => close(false),
    };
    if (keys[event.key]) {
      if (event.key !== "Tab") event.preventDefault();
      keys[event.key]();
      return;
    }
    if (event.key.length === 1) {
      const k = event.key.toLowerCase();
      const i = options.findIndex((o, j) => j > active && o.label.toLowerCase().startsWith(k));
      highlight(i >= 0 ? i : options.findIndex((o) => o.label.toLowerCase().startsWith(k)));
    }
  });
  document.addEventListener("pointerdown", (event) => {
    if (!host.contains(event.target)) close(false);
  });

  return {
    setOptions(next, value = current) {
      options = next;
      current = value;
      render();
    },
    set(value) {
      current = value;
      render();
    },
  };
}

// ----------------------------------------------------------------- pills

/** Toggle pills: clicking the active one clears it (st.pills behaviour). */
export function pills(host, options, current, onChange) {
  host.querySelectorAll(".ns-pill").forEach((n) => n.remove());
  for (const { value, label } of options) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "ns-pill";
    b.dataset.value = value;
    b.setAttribute("aria-pressed", String(value === current));
    b.textContent = label;
    host.append(b);
  }
  host.onclick = (event) => {
    const b = event.target.closest(".ns-pill");
    if (!b) return;
    const next = b.getAttribute("aria-pressed") === "true" ? null : b.dataset.value;
    for (const p of host.querySelectorAll(".ns-pill")) p.setAttribute("aria-pressed", String(p.dataset.value === next));
    onChange(next);
  };
}

/** Draws an RGB(A) image object onto a canvas, resizing the canvas to fit. */
export function drawImage(canvas, img) {
  canvas.width = img.width;
  canvas.height = img.height;
  const rgba = new ImageData(img.width, img.height);
  for (let i = 0, n = img.width * img.height; i < n; i++) {
    for (let c = 0; c < 3; c++) rgba.data[i * 4 + c] = img.data[i * img.channels + c];
    rgba.data[i * 4 + 3] = img.channels === 4 ? img.data[i * 4 + 3] : 255;
  }
  canvas.getContext("2d").putImageData(rgba, 0, 0);
}
