// Port of core.hybrid_scores / search_by_semantic_text over Float32 matrices
// stored row-major (count x dim), plus the helpers app.py uses for display.

/** Decodes little-endian IEEE half floats into a Float32Array. */
export function decodeF16(buffer) {
  if (typeof Float16Array === "function") return Float32Array.from(new Float16Array(buffer));
  const src = new Uint16Array(buffer);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const h = src[i];
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) out[i] = sign * frac * 2 ** -24;
    else if (exp === 31) out[i] = frac ? NaN : sign * Infinity;
    else out[i] = sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return out;
}

export function normalize(vec) {
  let n = 0;
  for (const v of vec) n += v * v;
  n = Math.sqrt(n) || 1;
  return Float32Array.from(vec, (v) => v / n);
}

/** matrix @ q for a row-major (count x dim) matrix. */
export function dot(matrix, dim, q) {
  const count = matrix.length / dim;
  const out = new Float32Array(count);
  for (let r = 0; r < count; r++) {
    let s = 0;
    const o = r * dim;
    for (let d = 0; d < dim; d++) s += matrix[o + d] * q[d];
    out[r] = s;
  }
  return out;
}

function zscore(x) {
  let mean = 0;
  for (const v of x) mean += v;
  mean /= x.length;
  let varSum = 0;
  for (const v of x) varSum += (v - mean) ** 2;
  const std = Math.sqrt(varSum / x.length) + 1e-8;
  return Float32Array.from(x, (v) => (v - mean) / std);
}

/** core.hybrid_scores: alpha * z(vis) + (1 - alpha) * z(sem), min-max scaled to [0, 1]. */
export function hybridScores(q, img, txt, dim, alpha) {
  const vis = zscore(dot(img, dim, q));
  const sem = zscore(dot(txt, dim, q));
  const fused = vis.map((v, i) => alpha * v + (1 - alpha) * sem[i]);
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of fused) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return fused.map((v) => (v - lo) / (hi - lo + 1e-8));
}

/** Indices of scores sorted best first. */
export function argsortDesc(scores, k = scores.length) {
  const idx = Array.from(scores.keys());
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, k);
}

/** app.relevance: bar widths where the catalog median maps to 0 and the best score to 1. */
export function relevance(selected, all) {
  const sorted = Float32Array.from(all).sort();
  const mid = sorted.length >> 1;
  const floor = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  const top = sorted[sorted.length - 1];
  return selected.map((s) => Math.min(1, Math.max(0, (s - floor) / (top - floor + 1e-8))));
}
