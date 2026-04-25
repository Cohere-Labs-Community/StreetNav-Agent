"""Final formatting tool — converts findings.json into a compact summary."""
from __future__ import annotations

import json
from typing import Any, Dict

from .. import _progress, config


def format_final_results() -> Dict[str, Any]:
    """Produce the final, user-facing summary from output/findings.json.

    This tool does NOT call any external API. It just collates already-
    persisted artifacts into a small, easy-to-display object.

    Returns:
        ``{"query_center": {...}, "count": int, "items": [...]}`` where each
        item has ``name``, ``address``, ``pano_id``, ``heading``, ``lat``,
        ``lng``, ``score``, and ``image_path`` (relative to project root).
    """
    _progress.start("format_final_results")
    try:
        if not config.FINDINGS_PATH.exists():
            _progress.fail("findings.json missing")
            return {"error": "findings.json missing — run enrich_findings first."}

        findings = json.loads(config.FINDINGS_PATH.read_text())
        geocode = (
            json.loads(config.GEOCODE_PATH.read_text())
            if config.GEOCODE_PATH.exists()
            else {}
        )

        items = []
        for item in findings:
            rel_image = config.RELEVANT_DIR / item["image_file"]
            sv_image = config.STREET_VIEWS_DIR / item["image_file"]
            image_path = str(rel_image if rel_image.exists() else sv_image)
            items.append(
                {
                    "name": item.get("name"),
                    "address": item.get("address"),
                    "pano_id": item.get("pano_id"),
                    "heading": item.get("heading"),
                    "lat": item.get("lat"),
                    "lng": item.get("lng"),
                    "score": item.get("score"),
                    "image_path": image_path,
                }
            )

        items.sort(key=lambda x: (x["score"] or 0), reverse=True)

        out = {
            "query_center": {
                "latitude": geocode.get("latitude"),
                "longitude": geocode.get("longitude"),
                "city": geocode.get("city"),
                "country": geocode.get("country"),
            },
            "count": len(items),
            "items": items,
        }
        _progress.done(f"{len(items)} items ready")
        return out
    except Exception as e:  # noqa: BLE001
        _progress.fail(str(e))
        return {"error": str(e)}
