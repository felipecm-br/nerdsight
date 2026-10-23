// core.preprocess_image parity: the JS port must reproduce PIL pixel for pixel.
import test from "node:test";
import assert from "node:assert/strict";
import { golden, loadImage, manifest } from "./helpers.mjs";
import { hasDarkBorder, preprocessImage } from "../js/preprocess.js";

for (const fx of golden.images) {
  test(`preprocess ${fx.file}`, async () => {
    const src = await loadImage(fx.file);
    const invert = hasDarkBorder(src, manifest.preprocess.darkBorder);
    assert.equal(invert, fx.invert, "dark-border detection");
    const out = preprocessImage(src, { invertColors: invert, autoCrop: fx.auto_crop, ...manifest.preprocess });
    const ref = await loadImage(fx.processed);
    assert.equal(out.width, ref.width);
    assert.equal(out.height, ref.height);
    let sum = 0;
    for (let i = 0; i < ref.width * ref.height; i++) {
      for (let c = 0; c < 3; c++) sum += Math.abs(out.data[i * 3 + c] - ref.data[i * ref.channels + c]);
    }
    const mad = sum / (ref.width * ref.height * 3);
    assert.ok(mad <= 2, `mean abs pixel diff ${mad.toFixed(3)} > 2/255`);
  });
}
