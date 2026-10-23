// End-to-end check of web/dist in a real browser: the page, transformers.js
// from the CDN and the WASM backend, against the Python golden set.
//
//   npm run test:browser            # needs network for the CDN and the model
//   CHROMIUM=/path/to/chrome npm run test:browser
//   BASE_URL=https://user-nerdsight.static.hf.space npm run test:browser   # live Space
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { GOLDEN, golden, overlap } from "./helpers.mjs";

const DIST = fileURLToPath(new URL("../dist/", import.meta.url));
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".ttf": "font/ttf", ".bin": "application/octet-stream" };
const MODEL_TIMEOUT = 15 * 60_000;

let server;
let base = process.env.BASE_URL;
if (!base) {
  server = createServer(async (req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^(\.\.[/\\])+/, "");
    try {
      const body = await readFile(join(DIST, path.endsWith("/") ? `${path}index.html` : path));
      res.writeHead(200, { "content-type": TYPES[extname(path)] ?? "text/html" }).end(body);
    } catch {
      res.writeHead(404).end();
    }
  }).listen(0);
  base = `http://localhost:${server.address().port}`;
}

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium" });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
let failed = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
  if (!ok) failed++;
};
const cardNames = (scope) => page.locator(`${scope} .ns-card`).evaluateAll((cards) => cards.map((c) => c.dataset.name));

try {
  // 1. Name search is ready before any model.
  let t0 = Date.now();
  await page.goto(`${base}/?q=python&mode=name`);
  await page.waitForSelector("#search-results .ns-card");
  const python = await cardNames("#search-results");
  const ref = golden.name.find((g) => g.q === "python" && !g.family);
  check("name search", python.join() === ref.top.map(([n]) => n).join(), `${Date.now() - t0} ms`);

  // 2. Concept search downloads the text encoder, then ranks like Python.
  t0 = Date.now();
  await page.goto(`${base}/?q=rocket&mode=concept`);
  await page.waitForSelector("#search-results .ns-card", { timeout: MODEL_TIMEOUT });
  const rocket = await cardNames("#search-results");
  const refRocket = golden.text.find((g) => g.q === "rocket").top;
  check("concept search", overlap(rocket.slice(0, 10), refRocket) >= 8, `${Date.now() - t0} ms, ${rocket.slice(0, 3)}`);

  // 3. Image search: preprocessing preview and ranking for each fixture.
  await page.click("#view [data-value=visual]");
  for (const fx of golden.images) {
    t0 = Date.now();
    await page.evaluate((on) => {
      const box = document.querySelector("#opt-crop");
      if (box.checked !== on) {
        box.checked = on;
        box.dispatchEvent(new Event("change"));
      }
    }, fx.auto_crop);
    const before = Number((await page.getAttribute("#visual-results", "data-image-id")) ?? 0);
    await page.setInputFiles("#file", join(GOLDEN, fx.file));
    await page.waitForFunction((id) => document.querySelector("#visual-results").dataset.imageId === String(id), before + 1, {
      timeout: MODEL_TIMEOUT,
    });
    const top = (await cardNames("#visual-results")).slice(0, 10);
    const procRef = await readFile(join(GOLDEN, fx.processed), "base64");
    const mad = await page.evaluate(async (png) => {
      const img = new Image();
      img.src = `data:image/png;base64,${png}`;
      await img.decode();
      const c = new OffscreenCanvas(128, 128).getContext("2d");
      c.drawImage(img, 0, 0);
      const a = c.getImageData(0, 0, 128, 128).data;
      const b = document.querySelector("#preview-processed").getContext("2d").getImageData(0, 0, 128, 128).data;
      let s = 0;
      for (let i = 0; i < a.length; i += 4) for (let k = 0; k < 3; k++) s += Math.abs(a[i + k] - b[i + k]);
      return s / (128 * 128 * 3);
    }, procRef);
    check(`image ${fx.file} preview`, mad <= 2, `mean abs diff ${mad.toFixed(3)}`);
    check(
      `image ${fx.file} ranking`,
      top.slice(0, 3).includes(fx.top[0]) && overlap(top, fx.top) >= 8,
      `${Date.now() - t0} ms, ${top.slice(0, 3)}`,
    );
  }
  check("no page errors", errors.length === 0, errors.join(" | "));
} finally {
  await browser.close();
  server?.close();
}
process.exit(failed ? 1 : 0);
