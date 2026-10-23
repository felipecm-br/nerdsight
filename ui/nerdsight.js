// nerdsight client behaviour: glyph inspector dialog, copy-to-clipboard and
// keyboard shortcuts. st.html may re-run this script on every rerun, so all
// listeners are bound once on document and the dialog lives on document.body.
// app.py prepends window.__nsI18n (UI strings for the active language), which
// is read at use time so a language switch takes effect without rebinding.
(() => {
  const i18n = window.__nsI18n || {};
  if (i18n.lang) document.documentElement.lang = i18n.lang;
  if (window.__nerdsight) return;
  window.__nerdsight = true;

  const ICON_COPY = "";
  const ICON_CHECK = "";
  const ICON_CLOSE = "";
  const t = (key, vars = {}) =>
    String((window.__nsI18n || {})[key] ?? key).replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? "");

  const el = (tag, attrs = {}, text) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  };

  // Copy formats shown in the inspector, derived from the hex codepoint.
  const formats = (char, name, hex) => {
    const big = hex.length > 4;
    const esc = big ? "\\U" + hex.padStart(8, "0") : "\\u" + hex;
    return [
      [t("glyph"), char],
      [t("name"), name],
      ["Python", esc],
      ["Bash", "$'" + esc + "'"],
      ["HTML", "&#x" + hex + ";"],
      ["CSS", '"\\' + hex + '"'],
    ];
  };

  let toast;
  let toastTimer;
  const announce = (msg) => {
    if (!toast || !document.body.contains(toast)) {
      toast = el("div", { class: "ns-toast", role: "status", "aria-live": "polite" });
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.setAttribute("data-show", "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.removeAttribute("data-show"), 1600);
  };

  const copy = async (text, host) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Plain-HTTP origins (LAN access) have no async clipboard API.
      const area = el("textarea", { readonly: "" });
      area.value = text;
      area.style.cssText = "position:fixed;opacity:0;pointer-events:none";
      (host || document.body).appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
  };

  let dialog;
  const ensureDialog = () => {
    if (dialog && document.body.contains(dialog)) return dialog;
    // Built with DOM calls, not an HTML string: Streamlit's DOMPurify pass drops
    // any script whose text contains tag-like sequences.
    dialog = el("dialog", { class: "ns-dialog", "aria-labelledby": "ns-dlg-title" });
    const head = el("div", { class: "ns-dlg-head" });
    const info = el("div");
    info.append(
      el("h2", { class: "ns-dlg-title", id: "ns-dlg-title" }),
      el("p", { class: "ns-dlg-sub" }),
      el("div", { class: "ns-sizes", "aria-hidden": "true" }),
    );
    head.append(
      el("div", { class: "ns-dlg-preview", "aria-hidden": "true" }),
      info,
      el("button", { type: "button", class: "ns-icon-btn", "data-close": "", "aria-label": t("close") }, ICON_CLOSE),
    );
    dialog.append(head, el("ul", { class: "ns-rows" }));
    dialog.addEventListener("click", async (event) => {
      if (event.target === dialog || event.target.closest("[data-close]")) {
        dialog.close();
        return;
      }
      const btn = event.target.closest("[data-copy]");
      if (!btn) return;
      await copy(btn.dataset.copy, dialog);
      const label = btn.querySelector(".ns-label");
      const icon = btn.querySelector(".ns-ic");
      btn.setAttribute("data-done", "");
      label.textContent = t("copied");
      icon.textContent = ICON_CHECK;
      announce(t("copied_msg", { x: btn.dataset.what }));
      setTimeout(() => {
        btn.removeAttribute("data-done");
        label.textContent = t("copy");
        icon.textContent = ICON_COPY;
      }, 1400);
    });
    document.body.appendChild(dialog);
    return dialog;
  };

  const openInspector = (card) => {
    const { char, name, code, family } = card.dataset;
    const hex = code.toLowerCase();
    const dlg = ensureDialog();
    dlg.querySelector("[data-close]").setAttribute("aria-label", t("close"));
    dlg.querySelector(".ns-dlg-preview").textContent = char;
    dlg.querySelector(".ns-dlg-title").textContent = name;
    dlg.querySelector(".ns-dlg-sub").textContent = `${family} · U+${hex.toUpperCase()}`;

    const sizes = dlg.querySelector(".ns-sizes");
    sizes.replaceChildren(
      ...[14, 16, 20, 24, 32].map((px) => {
        const cell = el("span");
        cell.append(el("b", { style: `font-size:${px}px;font-weight:400` }, char), el("small", {}, px));
        return cell;
      }),
    );

    const rows = dlg.querySelector(".ns-rows");
    rows.replaceChildren(
      ...formats(char, name, hex).map(([label, value]) => {
        const row = el("li", { class: "ns-row" });
        const btn = el("button", {
          type: "button",
          class: "ns-copy",
          "data-copy": value,
          "data-what": label,
          "aria-label": t("copy_aria", { label, value }),
        });
        btn.append(el("span", { class: "ns-ic", "aria-hidden": "true" }, ICON_COPY), el("span", { class: "ns-label" }, t("copy")));
        row.append(el("span", { class: "ns-row-label" }, label), el("span", { class: "ns-row-value" }, value), btn);
        return row;
      }),
    );

    if (!dlg.open) dlg.showModal();
    dlg.querySelector("[data-copy]").focus();
  };

  document.addEventListener("click", (event) => {
    const card = event.target.closest(".ns-card");
    if (card) openInspector(card);
  });

  const isTyping = (node) =>
    node instanceof HTMLElement && (node.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(node.tagName));

  document.addEventListener("keydown", async (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.key === "/" && !isTyping(event.target)) {
      // Search view has the search box, catalog view has the filter box. The static
      // front end keeps hidden views in the DOM, so only a visible box counts.
      const input = ["search_label", "filter_label"]
        .map((key) => document.querySelector(`input[aria-label="${CSS.escape(t(key))}"]`))
        .find((node) => node && node.offsetParent !== null);
      if (input) {
        event.preventDefault();
        input.focus();
        input.select();
      }
      return;
    }
    const card = event.target instanceof HTMLElement && event.target.closest(".ns-card");
    if (card && (event.key === "c" || event.key === "C")) {
      event.preventDefault();
      await copy(card.dataset.char);
      announce(t("copied_msg", { x: card.dataset.name }));
    }
  });
})();
