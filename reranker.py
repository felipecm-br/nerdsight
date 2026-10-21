#!/usr/bin/env python3
"""
nerdsight - reranker.py
Two-stage multimodal (VLM) reranker layered on top of the local CLIP search:

  Stage A  query image -> VLM -> structured concept/style/keyword JSON
  Stage B  candidate pool (keyword name hits + local CLIP top-N) is rendered
           into a numbered contact sheet from SymbolsNerdFontMono-Regular.ttf;
           query image + contact sheet -> VLM -> ranked scores per candidate

Only ~24 rendered glyphs ever reach the VLM, never the 10,000+ catalog.
Backends are pluggable: the `claude` CLI (subprocess, default) or the Gemini SDK.
"""

import base64
import io
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import core

STAGE_A_PROMPT = """You are an icon analyst. The image is a user's query icon; they want the closest glyph in the Nerd Fonts catalog.

Describe the image's visual anatomy, then name what it depicts. Nerd Fonts glyph names are lowercase snake_case nouns with optional
style/modifier suffixes, e.g. `bell_outline`, `folder_open`, `account_circle`, `file_document_edit`.

Return ONLY a JSON object with exactly these keys:
{
  "anatomy": "<one sentence: outer shape, edges, inner details>",
  "primary_concept": "<single lowercase noun>",
  "synonyms": ["<2-6 alternative nouns a glyph designer might have used>"],
  "visual_style": "outline" | "solid",
  "candidate_keywords": ["<4-10 snake_case strings likely to appear in glyph names>"]
}"""

STAGE_B_PROMPT = """You are verifying icon search results. Image 1 is the user's QUERY icon. Image 2 is a CONTACT SHEET of {n}
candidate glyphs rendered from the real font, each cell labelled with an index number and its nf-* name.

The query was analysed as: {analysis}

Score EVERY candidate 0-100 for how well it could replace the query icon in a UI:
- depicted concept (most important), then
- silhouette/geometry, then
- style (outline vs solid fill).
A glyph of the same concept but the other fill style should still score high (roughly 80-90); unrelated concepts below 40.

Return ONLY a JSON array, best first: [{{"index": <int>, "score": <0-100>, "reason": "<max 12 words>"}}, ...]"""


class VLMBackend(Protocol):
    name: str

    def complete(self, prompt: str, images: list[Image.Image]) -> str: ...


def _png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class ClaudeCLIBackend:
    """Runs `claude -p` headless, streaming the images in as base64 content blocks over stream-json stdin."""

    def __init__(self, model: str = "sonnet", timeout: int = 180, executable: str = "claude"):
        self.model = model
        self.timeout = timeout
        self.executable = executable
        self.name = f"claude-cli:{model}"

    def complete(self, prompt: str, images: list[Image.Image]) -> str:
        if shutil.which(self.executable) is None:
            raise RuntimeError(f"`{self.executable}` not found on PATH")

        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(img)}}
            for img in images
        ]
        content.append({"type": "text", "text": prompt})
        message = json.dumps({"type": "user", "message": {"role": "user", "content": content}})

        proc = subprocess.run(
            [
                self.executable, "-p",
                "--input-format", "stream-json",
                "--output-format", "stream-json",
                "--verbose",
                "--tools", "",
                "--no-session-persistence",
                "--model", self.model,
            ],
            input=message,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=core.BASE_DIR,
        )
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                if event.get("is_error"):
                    raise RuntimeError(f"claude CLI error: {event.get('result')}")
                return event.get("result", "")
        raise RuntimeError(f"claude CLI returned no result (exit {proc.returncode}): {proc.stderr.strip()[:300]}")


class GeminiBackend:
    """Uses the google-genai SDK; requires GEMINI_API_KEY (or an explicit api_key)."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY")
        self.name = f"gemini:{model}"

    def complete(self, prompt: str, images: list[Image.Image]) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=[*images, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
        )
        return response.text


def get_backend(name: str = "claude", model: Optional[str] = None, api_key: Optional[str] = None) -> VLMBackend:
    if name == "claude":
        return ClaudeCLIBackend(model=model or "sonnet")
    if name == "gemini":
        return GeminiBackend(model=model or "gemini-2.5-flash", api_key=api_key)
    raise ValueError(f"Unknown VLM backend: {name!r} (expected 'claude' or 'gemini')")


def parse_json_payload(text: str):
    """Extracts the first JSON object/array from a model reply, tolerating ```json fences and prose."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"No JSON found in VLM reply: {text[:200]!r}")
    return json.JSONDecoder().raw_decode(text[min(starts):])[0]


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------

def analyze_query(backend: VLMBackend, image: Image.Image) -> dict:
    data = parse_json_payload(backend.complete(STAGE_A_PROMPT, [image]))
    if not isinstance(data, dict):
        raise ValueError("Stage A reply is not a JSON object")
    style = str(data.get("visual_style", "")).lower()
    return {
        "anatomy": str(data.get("anatomy", "")),
        "primary_concept": str(data.get("primary_concept", "")).lower().strip(),
        "synonyms": [str(s).lower().strip() for s in data.get("synonyms", []) if str(s).strip()],
        "visual_style": style if style in ("outline", "solid") else "",
        "candidate_keywords": [
            str(k).lower().strip().replace(" ", "_") for k in data.get("candidate_keywords", []) if str(k).strip()
        ],
    }


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

def build_candidate_pool(
    analysis: dict,
    catalog: list[dict],
    local_ranking: list[str],
    max_keyword_hits: int = 16,
    max_total: int = 24,
) -> list[str]:
    """Keyword name hits from the full master catalog first (they are what CLIP misses), then local CLIP top-N.

    Returns catalog keys (e.g. 'md-ghost_outline'), deduplicated, at most `max_total`.
    """
    terms = [analysis.get("primary_concept", "")] + analysis.get("candidate_keywords", []) + analysis.get("synonyms", [])
    terms = [t for t in dict.fromkeys(t.replace(" ", "_") for t in terms) if len(t) >= 3]

    by_key = {item["key"]: item for item in catalog}
    local_pos = {key: i for i, key in enumerate(local_ranking)}
    keyword_hits: dict[str, float] = {}

    for t_idx, term in enumerate(terms):
        for score, item, _label in core.search_by_name(catalog, term, None, limit=64):
            if item["key"].startswith("mdi-"):  # obsolete v2 duplicates
                continue
            # Rank by match strength, then by how early the term came (concept > keywords > synonyms), then local CLIP rank
            rank_key = (100 - score) * 1000 + t_idx * 10 + min(local_pos.get(item["key"], 9999), 9999) / 1000
            keyword_hits[item["key"]] = min(keyword_hits.get(item["key"], float("inf")), rank_key)

    pool = [k for k, _ in sorted(keyword_hits.items(), key=lambda kv: kv[1])][:max_keyword_hits]
    for key in local_ranking:
        if len(pool) >= max_total:
            break
        if key not in pool and key in by_key:
            pool.append(key)
    return pool


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------

def render_contact_sheet(items: list[dict], cols: int = 6, cell: int = 168, glyph_px: int = 96) -> Image.Image:
    """Renders candidate glyphs from the real symbol font into a numbered, labelled grid."""
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    draw = ImageDraw.Draw(sheet)
    glyph_font = ImageFont.truetype(core.FONT_PATH, glyph_px)
    label_font = ImageFont.load_default(size=13)
    index_font = ImageFont.load_default(size=18)

    for i, item in enumerate(items):
        x0, y0 = (i % cols) * cell, (i // cols) * cell
        draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=(190, 190, 190))
        draw.text((x0 + 6, y0 + 4), f"#{i}", fill=(200, 0, 0), font=index_font)

        bbox = draw.textbbox((0, 0), item["char"], font=glyph_font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        gx = x0 + (cell - w) / 2 - bbox[0]
        gy = y0 + 26 + (cell - 52 - h) / 2 - bbox[1]
        draw.text((gx, gy), item["char"], fill="black", font=glyph_font)

        label = item["name"]
        while draw.textlength(label, font=label_font) > cell - 8 and len(label) > 4:
            label = label[:-2] + "…"
        draw.text((x0 + 4, y0 + cell - 20), label, fill=(40, 40, 40), font=label_font)

    return sheet


def verify_with_contact_sheet(backend: VLMBackend, query: Image.Image, sheet: Image.Image, n: int, analysis: dict) -> dict[int, dict]:
    summary = json.dumps({k: analysis.get(k) for k in ("primary_concept", "visual_style", "anatomy")})
    data = parse_json_payload(backend.complete(STAGE_B_PROMPT.format(n=n, analysis=summary), [query, sheet]))
    if isinstance(data, dict):
        data = data.get("results") or data.get("ranking") or []
    scored = {}
    for row in data:
        try:
            idx = int(row["index"])
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < n and idx not in scored:
            scored[idx] = {"score": max(0.0, min(score, 100.0)), "reason": str(row.get("reason", ""))}
    return scored


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    backend: str
    analysis: dict
    candidates: list[str]
    ranked: list[dict] = field(default_factory=list)  # {key, name, char, code, score, reason, local_rank}
    contact_sheet: Optional[Image.Image] = None


def rerank(
    query: Image.Image,
    catalog: list[dict],
    local_ranking: list[str],
    backend: VLMBackend,
    max_total: int = 24,
) -> RerankResult:
    """Runs Stage A + Stage B. `local_ranking` is the local hybrid ranking as catalog keys, best first."""
    query_rgb = core.flatten_alpha(query)
    analysis = analyze_query(backend, query_rgb)
    pool = build_candidate_pool(analysis, catalog, local_ranking, max_total=max_total)

    by_key = {item["key"]: item for item in catalog}
    items = [by_key[k] for k in pool]
    sheet = render_contact_sheet(items)
    scored = verify_with_contact_sheet(backend, query_rgb, sheet, len(items), analysis)

    local_pos = {key: i + 1 for i, key in enumerate(local_ranking)}
    ranked = []
    for i, item in enumerate(items):
        s = scored.get(i, {"score": 0.0, "reason": "not scored by VLM"})
        ranked.append({
            "key": item["key"], "name": item["name"], "char": item["char"], "code": item["code"],
            "score": s["score"], "reason": s["reason"], "local_rank": local_pos.get(item["key"]),
        })
    # VLM score first; local CLIP rank breaks ties
    ranked.sort(key=lambda r: (-r["score"], r["local_rank"] or 10**6))
    return RerankResult(backend=backend.name, analysis=analysis, candidates=pool, ranked=ranked, contact_sheet=sheet)


def local_ranking_keys(scores: np.ndarray, metadata: list[dict], top_n: int = 200) -> list[str]:
    order = np.argsort(scores)[::-1][:top_n]
    return [metadata[i]["name"].removeprefix("nf-") for i in order]
