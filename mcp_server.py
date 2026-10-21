#!/usr/bin/env python3
"""
nerdsight - mcp_server.py
MCP (Model Context Protocol) server exposing nerdsight's name, semantic, and
visual icon search as tools for any MCP-capable coding-agent CLI (Claude Code,
Codex CLI, etc.) to call directly, instead of shelling out to cli.py and
parsing text output.

Run standalone for local testing:
    python mcp_server.py

Point an MCP-capable client at it over stdio, e.g. for Claude Code:
    claude mcp add nerdsight -- python /path/to/nerdsight/mcp_server.py
"""

import os
from typing import Optional

from mcp.server.mcpserver import MCPServer

import core

server = MCPServer(
    name="nerdsight",
    instructions=(
        "Search 10,000+ Nerd Fonts glyphs by exact/partial name, by semantic "
        "concept (CLIP text embeddings), or by visual similarity to an image "
        "file (CLIP hybrid image+text embeddings). Use this to pick the best "
        "Nerd Font icon for a file type, status indicator, or UI element "
        "instead of guessing a codepoint from memory."
    ),
)

_catalog_cache: Optional[list[dict]] = None


def _catalog() -> list[dict]:
    global _catalog_cache
    if _catalog_cache is None:
        _catalog_cache = core.load_master_catalog()
    return _catalog_cache


def _name_result(item: dict, match_type: str, score: Optional[float] = None) -> dict:
    out = {
        "name": item["name"],
        "char": item["char"],
        "codepoint": f"\\u{item['code']}",
        "family": item.get("family", ""),
        "match_type": match_type,
    }
    if score is not None:
        out["score_pct"] = score
    return out


@server.tool()
def find_icon_by_name(query: str, family: Optional[str] = None, limit: int = 12) -> dict:
    """Find Nerd Font icons by exact or partial name match (e.g. 'git', 'folder', 'python').

    Use this first when you already know roughly what the icon is called.
    Returns the glyph character, its \\uXXXX codepoint, and collection family.
    """
    catalog = _catalog()
    matches = core.search_by_name(catalog, query, family, limit)
    results = [_name_result(item, label) for _, item, label in matches]
    return {"query": query, "family": family, "count": len(results), "results": results}


@server.tool()
def find_icon_by_concept(query: str, family: Optional[str] = None, limit: int = 12) -> dict:
    """Find Nerd Font icons by semantic concept using CLIP text embeddings
    (e.g. 'database', 'warning', 'cloud storage', 'speedometer').

    Use this when you need an icon for an idea/meaning rather than a known name.
    """
    if not core.has_embeddings_index():
        return {
            "error": "Embeddings index not found. Run `python setup_assets.py && python indexer.py` first.",
            "results": [],
        }

    model = core.get_model()
    _, txt_emb, meta = core.load_embeddings()
    order, scores = core.search_by_semantic_text(model, txt_emb, query)

    results = []
    for idx, score in zip(order, scores):
        if len(results) >= limit:
            break
        item = meta[idx]
        parts = item["name"].replace("nf-", "").split("-", 1)
        pfx = parts[0] if len(parts) > 1 else ""
        fam = core.PREFIX_MAP.get(pfx, "Nerd Font")

        if family and family.lower() not in pfx.lower() and family.lower() not in fam.lower():
            continue

        results.append({
            "name": item["name"],
            "char": item["char"],
            "codepoint": f"\\u{item['code']}",
            "family": fam,
            "score_pct": round(float(score) * 100, 1),
        })

    return {"query": query, "family": family, "count": len(results), "results": results}


@server.tool()
def find_icon_by_image(
    image_path: str,
    alpha: float = core.DEFAULT_ALPHA,
    boost: str = "",
    limit: int = 12,
) -> dict:
    """Find Nerd Font icons visually similar to a local image file (logo, screenshot,
    sketch export). Uses a hybrid CLIP visual+semantic score with per-modality z-score normalization.

    Args:
        image_path: Absolute or relative path to a local PNG/JPG/WEBP image.
        alpha: 1.0 = match pure shape/silhouette, 0.0 = match meaning only. Default 0.7.
        boost: Optional extra text hint blended into the query (e.g. "git", "database").
        limit: Max results to return.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}", "results": []}

    if not core.has_embeddings_index():
        return {
            "error": "Embeddings index not found. Run `python setup_assets.py && python indexer.py` first.",
            "results": [],
        }

    from PIL import Image

    model = core.get_model()
    img_emb, txt_emb, metadata = core.load_embeddings()

    raw_img = Image.open(image_path)
    processed = core.preprocess_image(raw_img)
    query_vec = core.encode_image_query(model, processed, boost_text=boost)

    top_indices, top_scores = core.search_hybrid_visual(
        query_vec, img_emb=img_emb, txt_emb=txt_emb, alpha=alpha, top_k=limit
    )

    results = []
    for idx, score in zip(top_indices, top_scores):
        item = metadata[idx]
        results.append({
            "name": item["name"],
            "char": item["char"],
            "codepoint": f"\\u{item['code']}",
            "score_pct": round(float(score) * 100, 2),
        })

    return {"image_path": image_path, "alpha": alpha, "boost": boost, "count": len(results), "results": results}


@server.tool()
def list_icon_families() -> dict:
    """List all Nerd Font icon collection families (Font Awesome, Devicons, Material Design, etc.)
    with their icon counts. Use the returned family name as a filter in the other search tools.
    """
    catalog = _catalog()
    counts: dict[str, int] = {}
    for item in catalog:
        counts[item["family"]] = counts.get(item["family"], 0) + 1
    families = sorted(
        [{"family": fam, "count": n} for fam, n in counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )
    return {"families": families}


@server.tool()
def catalog_stats() -> dict:
    """Return overall nerdsight catalog/index stats: total icon count and whether the
    CLIP embeddings index (needed for find_icon_by_concept / find_icon_by_image) is built.
    """
    catalog = _catalog()
    return {
        "total_icons": len(catalog),
        "embeddings_index_ready": core.has_embeddings_index(),
    }


if __name__ == "__main__":
    server.run(transport="stdio")
