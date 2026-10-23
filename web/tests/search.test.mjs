// core.search_by_name and the exported index, without any model.
import test from "node:test";
import assert from "node:assert/strict";
import { golden, items, manifest, matrix } from "./helpers.mjs";
import { searchByName } from "../js/catalog.js";

const LABEL = { exact: "Exact Match", word: "Word Match", prefix: "Prefix Match", substring: "Substring Match" };

for (const g of golden.name) {
  test(`name search "${g.q}"${g.family ? ` in ${g.family}` : ""}`, () => {
    const pool = g.family ? items.filter((i) => i.family === g.family) : items;
    const res = searchByName(pool, g.q);
    assert.equal(res.length, g.total);
    assert.deepEqual(res.slice(0, g.top.length).map((r) => [r.item.name, LABEL[r.match]]), g.top);
  });
}

test("embedding matrices match the manifest and are unit length", () => {
  for (const name of ["img_emb.f16.bin", "txt_emb.f16.bin"]) {
    const m = matrix(name);
    assert.equal(m.length, manifest.count * manifest.dim, name);
    for (const r of [0, Math.floor(manifest.count / 2), manifest.count - 1]) {
      let n = 0;
      for (let d = 0; d < manifest.dim; d++) n += m[r * manifest.dim + d] ** 2;
      assert.ok(Math.abs(Math.sqrt(n) - 1) < 2e-3, `${name} row ${r} norm ${Math.sqrt(n)}`);
    }
  }
});
