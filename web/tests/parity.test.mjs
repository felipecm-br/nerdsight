// CLIP parity: transformers.js (ONNX, dtypes from manifest.json) against the
// PyTorch index. Downloads the model on first run. Override the dtypes with
// NERDSIGHT_DTYPE=vision:text, e.g. NERDSIGHT_DTYPE=q8:q8, to evaluate others.
import test from "node:test";
import assert from "node:assert/strict";
import * as tf from "@huggingface/transformers";
import { byRow, cosine, golden, loadImage, manifest, matrix, overlap, rowOf } from "./helpers.mjs";
import { createEncoders } from "../js/model.js";
import { hasDarkBorder, preprocessImage } from "../js/preprocess.js";
import { argsortDesc, dot, hybridScores } from "../js/search.js";

const [vision, text] = (process.env.NERDSIGHT_DTYPE ?? `${manifest.dtype.vision}:${manifest.dtype.text}`).split(":");
const enc = createEncoders(tf, { ...manifest, dtype: { vision, text } }, { device: "cpu" });
const img = matrix("img_emb.f16.bin");
const txt = matrix("txt_emb.f16.bin");
const dim = manifest.dim;
const names = (rows) => rows.map((r) => byRow[r]);
// fp32 reproduces PyTorch exactly; half precision may reorder near-ties.
const exact = vision === "fp32" && text === "fp32";

test(`text encoder (${text})`, async (t) => {
  for (const g of golden.text) {
    await t.test(g.q, async () => {
      const vec = await enc.encodeText(g.q);
      assert.ok(cosine(vec, g.vec) >= 0.99, `cosine ${cosine(vec, g.vec).toFixed(4)}`);
      const top = names(argsortDesc(dot(txt, dim, vec), 10));
      // Glyphs with identical prompts tie exactly, so compare sets, not order.
      assert.ok(overlap(top.slice(0, 5), g.top.slice(0, 5)) >= 4, `top-5 ${top.slice(0, 5)} vs ${g.top.slice(0, 5)}`);
      assert.ok(overlap(top, g.top) >= 8, `top-10 overlap ${overlap(top, g.top)}`);
    });
  }
});

test(`vision encoder on rendered glyphs (${vision})`, async () => {
  const sims = [];
  for (const g of golden.glyphs) {
    const vec = await enc.encodeImage(await loadImage(g.file));
    const r = rowOf[g.name];
    sims.push(cosine(vec, img.subarray(r * dim, (r + 1) * dim)));
  }
  const mean = sims.reduce((a, b) => a + b) / sims.length;
  assert.ok(mean >= 0.99, `mean cosine ${mean.toFixed(4)} over ${sims.length} glyphs`);
});

test(`image queries end to end (${vision}/${text})`, async (t) => {
  for (const fx of golden.images) {
    await t.test(fx.file, async () => {
      const src = await loadImage(fx.file);
      const p = preprocessImage(src, { invertColors: hasDarkBorder(src), autoCrop: fx.auto_crop, ...manifest.preprocess });
      for (const [boost, expected] of [["", fx.top], [fx.boost, fx.top_boost]]) {
        const vec = await enc.encodeImage(p, boost);
        const top = names(argsortDesc(hybridScores(vec, img, txt, dim, manifest.alpha), 10));
        if (exact) assert.deepEqual(top.slice(0, 3), expected.slice(0, 3), `top-3 (boost "${boost}")`);
        else assert.ok(top.slice(0, 3).includes(expected[0]), `Python #1 ${expected[0]} not in top-3 ${top.slice(0, 3)} (boost "${boost}")`);
        assert.ok(overlap(top, expected) >= 8, `top-10 overlap ${overlap(top, expected)} (boost "${boost}")`);
      }
    });
  }
});
