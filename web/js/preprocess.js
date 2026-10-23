// Port of core.preprocess_image and of the CLIP image processor, step for step
// with Pillow. Browser canvas scaling is not used anywhere: the index was built
// from PIL-resampled pixels, so resizing reimplements Pillow's Resample.c
// (fixed-point separable filters, horizontal pass first) and Reduce.c.
//
// Images are plain objects {width, height, channels, data: Uint8ClampedArray},
// channels 3 (RGB) or 4 (RGBA), so the same code runs in the browser and in the
// Node parity tests.

const PRECISION_BITS = 32 - 8 - 2;

export const DEFAULTS = {
  size: 128,
  cropBox: 0.72,
  threshold: 185,
  bboxThreshold: 30,
  darkBorder: 96,
};

const image = (width, height, channels = 3, fill) => {
  const data = new Uint8ClampedArray(width * height * channels);
  if (fill !== undefined) data.fill(fill);
  return { width, height, channels, data };
};

// Python's round(): half to even.
const pyRound = (x) => {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
};

// ------------------------------------------------------------- resampling

const FILTERS = {
  bicubic: {
    support: 2,
    fn(x) {
      const a = -0.5;
      x = Math.abs(x);
      if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
      if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
      return 0;
    },
  },
  lanczos: {
    support: 3,
    fn(x) {
      const sinc = (v) => (v === 0 ? 1 : Math.sin(Math.PI * v) / (Math.PI * v));
      return x > -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0;
    },
  },
};

function precomputeCoeffs(inSize, in0, in1, outSize, filter) {
  const scale = (in1 - in0) / outSize;
  const filterscale = Math.max(scale, 1);
  const support = filter.support * filterscale;
  const ss = 1 / filterscale;
  const bounds = new Int32Array(outSize * 2);
  const coeffs = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = in0 + (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    const k = new Float64Array(xmax);
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      k[x] = filter.fn((x + xmin - center + 0.5) * ss);
      ww += k[x];
    }
    const fixed = new Int32Array(xmax);
    for (let x = 0; x < xmax; x++) {
      const w = ww !== 0 ? k[x] / ww : k[x];
      fixed[x] = Math.trunc(w < 0 ? -0.5 + w * (1 << PRECISION_BITS) : 0.5 + w * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
    coeffs.push(fixed);
  }
  return { bounds, coeffs };
}

// Fixed-point accumulation as in Resample.c. Products reach ~2^30, so plain
// doubles are used instead of 32-bit integer ops.
const clip8 = (acc) => {
  const v = Math.floor(acc / (1 << PRECISION_BITS));
  return v < 0 ? 0 : v > 255 ? 255 : v;
};
const HALF = 1 << (PRECISION_BITS - 1);

function resampleHorizontal(src, outWidth, rowFirst, rowCount, { bounds, coeffs }) {
  const c = src.channels;
  const out = image(outWidth, rowCount, c);
  for (let y = 0; y < rowCount; y++) {
    const rowIn = (y + rowFirst) * src.width * c;
    const rowOut = y * outWidth * c;
    for (let xx = 0; xx < outWidth; xx++) {
      const xmin = bounds[xx * 2];
      const k = coeffs[xx];
      for (let ch = 0; ch < c; ch++) {
        let acc = HALF;
        for (let x = 0; x < k.length; x++) acc += src.data[rowIn + (x + xmin) * c + ch] * k[x];
        out.data[rowOut + xx * c + ch] = clip8(acc);
      }
    }
  }
  return out;
}

function resampleVertical(src, outHeight, { bounds, coeffs }) {
  const c = src.channels;
  const out = image(src.width, outHeight, c);
  for (let yy = 0; yy < outHeight; yy++) {
    const ymin = bounds[yy * 2];
    const k = coeffs[yy];
    for (let x = 0; x < src.width; x++) {
      for (let ch = 0; ch < c; ch++) {
        let acc = HALF;
        for (let y = 0; y < k.length; y++) acc += src.data[((y + ymin) * src.width + x) * c + ch] * k[y];
        out.data[(yy * src.width + x) * c + ch] = clip8(acc);
      }
    }
  }
  return out;
}

// Image.im.resize(size, resample, box): ImagingResampleInner.
function resampleBox(src, width, height, filterName, box) {
  const filter = FILTERS[filterName];
  const [x0, y0, x1, y1] = box;
  const needH = width !== src.width || x0 !== 0 || x1 !== width;
  const needV = height !== src.height || y0 !== 0 || y1 !== height;
  const horiz = precomputeCoeffs(src.width, x0, x1, width, filter);
  const vert = precomputeCoeffs(src.height, y0, y1, height, filter);
  const yFirst = vert.bounds[0];
  const yLast = vert.bounds[height * 2 - 2] + vert.bounds[height * 2 - 1];
  let cur = src;
  if (needH) {
    for (let i = 0; i < height; i++) vert.bounds[i * 2] -= yFirst;
    cur = resampleHorizontal(src, width, yFirst, yLast - yFirst, horiz);
  }
  if (needV) cur = resampleVertical(cur, height, vert);
  if (!needH && !needV) cur = { ...src, data: src.data.slice() };
  return cur;
}

// Image.reduce(factor, box): block average, partial blocks at the edges.
function reduce(src, fx, fy, box) {
  const [bx0, by0, bx1, by1] = box;
  const width = Math.ceil((bx1 - bx0) / fx);
  const height = Math.ceil((by1 - by0) / fy);
  const c = src.channels;
  const out = image(width, height, c);
  for (let y = 0; y < height; y++) {
    const ys = by0 + y * fy;
    const ye = Math.min(ys + fy, by1);
    for (let x = 0; x < width; x++) {
      const xs = bx0 + x * fx;
      const xe = Math.min(xs + fx, bx1);
      const n = (ye - ys) * (xe - xs);
      for (let ch = 0; ch < c; ch++) {
        let sum = 0;
        for (let yy = ys; yy < ye; yy++) for (let xx = xs; xx < xe; xx++) sum += src.data[(yy * src.width + xx) * c + ch];
        out.data[(y * width + x) * c + ch] = Math.floor((sum + Math.floor(n / 2)) / n);
      }
    }
  }
  return out;
}

// Image.resize(size, resample, reducing_gap=...).
export function resize(src, width, height, filterName = "bicubic", reducingGap = null) {
  let box = [0, 0, src.width, src.height];
  if (width === src.width && height === src.height) return { ...src, data: src.data.slice() };
  if (reducingGap !== null) {
    const fx = Math.trunc(src.width / width / reducingGap) || 1;
    const fy = Math.trunc(src.height / height / reducingGap) || 1;
    if (fx > 1 || fy > 1) {
      const support = FILTERS[filterName].support - 0.5;
      const sx = support * (src.width / width);
      const sy = support * (src.height / height);
      const rb = [
        Math.max(0, Math.trunc(-sx)),
        Math.max(0, Math.trunc(-sy)),
        Math.min(src.width, Math.ceil(src.width + sx)),
        Math.min(src.height, Math.ceil(src.height + sy)),
      ];
      src = reduce(src, fx, fy, rb);
      box = [(0 - rb[0]) / fx, (0 - rb[1]) / fy, (box[2] - rb[0]) / fx, (box[3] - rb[1]) / fy];
    }
  }
  return resampleBox(src, width, height, filterName, box);
}

// Image.thumbnail(size, resample): only ever shrinks, keeps the aspect ratio.
function thumbnail(src, maxSize, filterName) {
  if (maxSize >= src.width && maxSize >= src.height) return src;
  const aspect = src.width / src.height;
  const roundAspect = (n, key) => Math.max(key(Math.floor(n)) <= key(Math.ceil(n)) ? Math.floor(n) : Math.ceil(n), 1);
  let x = maxSize;
  let y = maxSize;
  if (x / y >= aspect) x = roundAspect(y * aspect, (n) => Math.abs(aspect - n / y));
  else y = roundAspect(x / aspect, (n) => (n === 0 ? 0 : Math.abs(aspect - x / n)));
  if (x === src.width && y === src.height) return src;
  return resize(src, x, y, filterName, 2.0);
}

// --------------------------------------------------------------- pixel ops

// Image.alpha_composite(white, rgba).convert("RGB"), AlphaComposite.c math.
export function flattenAlpha(src) {
  if (src.channels === 3) return src;
  const out = image(src.width, src.height, 3);
  const n = src.width * src.height;
  for (let i = 0; i < n; i++) {
    const a = src.data[i * 4 + 3];
    for (let ch = 0; ch < 3; ch++) {
      const v = src.data[i * 4 + ch];
      if (a === 0) {
        out.data[i * 3 + ch] = 255;
        continue;
      }
      const coef1 = Math.floor((a * 255 * 255 * 128) / (a * 255 + 255 * (255 - a)));
      const coef2 = 255 * 128 - coef1;
      const tmp = v * coef1 + 255 * coef2 + (0x80 << 7);
      out.data[i * 3 + ch] = (((tmp >> 8) + tmp) >> 8) >> 7;
    }
  }
  return out;
}

const invert = (src) => ({ ...src, data: src.data.map((v) => 255 - v) });

// convert("L") weights, kept as a single channel.
function toGray(src) {
  const n = src.width * src.height;
  const out = { width: src.width, height: src.height, channels: 1, data: new Uint8ClampedArray(n) };
  for (let i = 0; i < n; i++) {
    const o = i * src.channels;
    out.data[i] = (src.data[o] * 19595 + src.data[o + 1] * 38470 + src.data[o + 2] * 7471 + 0x8000) >>> 16;
  }
  return out;
}

const grayToRgb = (g) => {
  const out = image(g.width, g.height, 3);
  for (let i = 0; i < g.data.length; i++) out.data[i * 3] = out.data[i * 3 + 1] = out.data[i * 3 + 2] = g.data[i];
  return out;
};

// ImageOps.autocontrast(cutoff=0).
function autocontrast(g) {
  let lo = 0;
  let hi = 255;
  const hist = new Uint32Array(256);
  for (const v of g.data) hist[v]++;
  while (lo < 256 && !hist[lo]) lo++;
  while (hi >= 0 && !hist[hi]) hi--;
  if (hi <= lo) return g;
  const scale = 255 / (hi - lo);
  const offset = -lo * scale;
  const lut = new Uint8ClampedArray(256);
  for (let ix = 0; ix < 256; ix++) lut[ix] = Math.min(255, Math.max(0, Math.trunc(ix * scale + offset)));
  return { ...g, data: g.data.map((v) => lut[v]) };
}

function getBBox(g, threshold) {
  let x0 = g.width;
  let y0 = g.height;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < g.height; y++) {
    for (let x = 0; x < g.width; x++) {
      if (255 - g.data[y * g.width + x] >= threshold) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : [x0, y0, x1 + 1, y1 + 1];
}

function crop(src, [x0, y0, x1, y1]) {
  const c = src.channels;
  const out = image(x1 - x0, y1 - y0, c);
  for (let y = y0; y < y1; y++) {
    out.data.set(src.data.subarray((y * src.width + x0) * c, (y * src.width + x1) * c), (y - y0) * out.width * c);
  }
  return out;
}

function paste(dst, src, ox, oy) {
  const c = dst.channels;
  for (let y = 0; y < src.height; y++) {
    dst.data.set(src.data.subarray(y * src.width * c, (y + 1) * src.width * c), ((y + oy) * dst.width + ox) * c);
  }
  return dst;
}

// ImageOps.pad(image, (size, size), color=white): contain() then center.
function pad(src, size) {
  let w = size;
  let h = size;
  const ratio = src.width / src.height;
  if (ratio > 1) h = pyRound((src.height / src.width) * size);
  else if (ratio < 1) w = pyRound((src.width / src.height) * size);
  const resized = resize(src, w, h, "bicubic");
  if (w === size && h === size) return resized;
  const out = image(size, size, 3, 255);
  return w !== size ? paste(out, resized, pyRound((size - w) * 0.5), 0) : paste(out, resized, 0, pyRound((size - h) * 0.5));
}

// ------------------------------------------------------------ public API

/** True when the image frame is mostly dark, i.e. a light glyph on a terminal background. */
export function hasDarkBorder(src, limit = DEFAULTS.darkBorder) {
  const g = toGray(flattenAlpha(src));
  let sum = 0;
  let n = 0;
  for (let x = 0; x < g.width; x++) {
    sum += g.data[x] + g.data[(g.height - 1) * g.width + x];
    n += 2;
  }
  for (let y = 0; y < g.height; y++) {
    sum += g.data[y * g.width] + g.data[y * g.width + g.width - 1];
    n += 2;
  }
  return sum / n < limit;
}

/** core.preprocess_image: returns a size×size RGB image. */
export function preprocessImage(src, { invertColors = false, binarize = true, autoCrop = true, ...opts } = {}) {
  const p = { ...DEFAULTS, ...opts };
  let flattened = flattenAlpha(src);
  if (invertColors) flattened = invert(flattened);

  let processed = flattened;
  if (binarize) {
    const g = autocontrast(toGray(flattened));
    processed = grayToRgb({ ...g, data: g.data.map((v) => (v < p.threshold ? 0 : 255)) });
  }

  if (autoCrop) {
    const bbox = getBBox(toGray(processed), p.bboxThreshold);
    if (bbox) {
      const cropped = thumbnail(crop(processed, bbox), Math.trunc(p.size * p.cropBox), "lanczos");
      const canvas = image(p.size, p.size, 3, 255);
      return paste(canvas, cropped, Math.floor((p.size - cropped.width) / 2), Math.floor((p.size - cropped.height) / 2));
    }
  }
  return pad(processed, p.size);
}

/** CLIPImageProcessor: shortest-edge bicubic resize, center crop, rescale, normalize. Returns CHW floats. */
export function toPixelValues(src, { size = 224, mean, std }) {
  const scale = size / Math.min(src.width, src.height);
  const w = src.width <= src.height ? size : Math.trunc(src.width * scale);
  const h = src.height <= src.width ? size : Math.trunc(src.height * scale);
  const resized = resize(src, w, h, "bicubic");
  const left = Math.floor((w - size) / 2);
  const top = Math.floor((h - size) / 2);
  const out = new Float32Array(3 * size * size);
  const c = resized.channels;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const o = ((y + top) * w + x + left) * c;
      for (let ch = 0; ch < 3; ch++) out[ch * size * size + y * size + x] = (resized.data[o + ch] / 255 - mean[ch]) / std[ch];
    }
  }
  return out;
}
