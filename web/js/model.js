// CLIP query encoders on transformers.js. The library module is passed in, so
// the browser can load it from the CDN and the Node tests from node_modules.
// Each encoder downloads on first use only; the browser caches the weights.

import { normalize } from "./search.js";
import { toPixelValues } from "./preprocess.js";

export function createEncoders(tf, manifest, { device = "wasm", onProgress = () => {} } = {}) {
  const fill = (template, q) => template.replace("{q}", q.trim());
  const report = (part) => (event) => onProgress({ part, ...event });

  let text;
  let vision;
  const loadText = () =>
    (text ??= Promise.all([
      tf.AutoTokenizer.from_pretrained(manifest.model),
      tf.CLIPTextModelWithProjection.from_pretrained(manifest.model, {
        dtype: manifest.dtype.text,
        device,
        progress_callback: report("text"),
      }),
    ]).catch((err) => {
      text = undefined;
      throw err;
    }));
  const loadVision = () =>
    (vision ??= tf.CLIPVisionModelWithProjection.from_pretrained(manifest.model, {
      dtype: manifest.dtype.vision,
      device,
      progress_callback: report("vision"),
    }).catch((err) => {
      vision = undefined;
      throw err;
    }));

  async function encodeRaw(prompt) {
    const [tokenizer, model] = await loadText();
    const inputs = tokenizer([prompt], { padding: true, truncation: true });
    const { text_embeds } = await model(inputs);
    return normalize(text_embeds.data);
  }

  return {
    loadText,
    loadVision,
    /** core.encode_text_query */
    encodeText: (query) => encodeRaw(fill(manifest.query_template, query)),
    /** core.encode_image_query, on an already preprocessed 128x128 image. */
    async encodeImage(img, boost = "") {
      const model = await loadVision();
      const pixels = toPixelValues(img, manifest.clip_image);
      const size = manifest.clip_image.size;
      const { image_embeds } = await model({ pixel_values: new tf.Tensor("float32", pixels, [1, 3, size, size]) });
      let vec = normalize(image_embeds.data);
      if (boost.trim()) {
        const b = await encodeRaw(fill(manifest.boost_template, boost));
        const w = manifest.boost_weight;
        vec = normalize(vec.map((v, i) => w * v + (1 - w) * b[i]));
      }
      return vec;
    },
  };
}
