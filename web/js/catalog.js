// Catalog loading and name search: a port of core.load_master_catalog and
// core.search_by_name over data/catalog.json. Needs no model, so it is ready on
// first paint.

/** Expands the compact catalog.json rows into the same fields core.load_master_catalog builds. */
export function expandCatalog({ fields, families, glyphs }) {
  const at = Object.fromEntries(fields.map((f, i) => [f, i]));
  return glyphs.map((g) => {
    const key = g[at.key];
    const dash = key.indexOf("-");
    return {
      key,
      name: `nf-${key}`,
      clean: dash < 0 ? key : key.slice(dash + 1),
      prefix: dash < 0 ? "custom" : key.slice(0, dash),
      char: g[at.char],
      code: g[at.code],
      family: families[g[at.family]],
      row: g[at.row],
    };
  });
}

export const MATCH_SCORE = { exact: 100, word: 95, prefix: 90, substring: 85 };

/** core.search_by_name: returns [{item, match}] ranked exact > word > prefix > substring. */
export function searchByName(items, query) {
  const q = query.trim().toLowerCase();
  const out = [];
  for (const item of items) {
    const key = item.key.toLowerCase();
    const clean = item.clean.toLowerCase();
    const code = item.code.toLowerCase();
    let match = null;
    if (q === key || q === clean) match = "exact";
    else if (`_${clean}_`.includes(`_${q}_`) || `-${key}-`.includes(`-${q}-`)) match = "word";
    else if (key.startsWith(q) || clean.startsWith(q)) match = "prefix";
    else if (key.includes(q) || code.includes(q)) match = "substring";
    if (match) out.push({ item, match });
  }
  // Array.prototype.sort is stable, like Python's sort.
  return out.sort((a, b) => MATCH_SCORE[b.match] - MATCH_SCORE[a.match]);
}

/** Catalog view filter (app.view_catalog): substring of name, codepoint or clean name. */
export function filterCatalog(items, term) {
  const t = term.trim().toLowerCase();
  if (!t) return items;
  return items.filter((i) => i.name.toLowerCase().includes(t) || i.code.toLowerCase().includes(t) || i.clean.toLowerCase().includes(t));
}
