"""Enrich YES-scored images with reverse-geocoded address + visible name."""
from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from .. import _progress, config
from ..clients import cohere_client, gmaps, openrouter_client


def _build_findings(
    yes_hits: List[Dict[str, Any]],
    saved_metadata: List[Dict[str, Any]],
    panos: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: list[dict] = []
    for r in yes_hits:
        image_file = os.path.basename(r["image_path"])
        pano_id = image_file.rsplit("_", 1)[0]
        heading = next(
            (m["heading"] for m in saved_metadata if m["image"] == image_file), None
        )
        pano = next((p for p in panos if p["pano_id"] == pano_id), None)
        out.append(
            {
                "pano_id": pano_id,
                "image_file": image_file,
                "heading": heading,
                "lat": pano["lat"] if pano else None,
                "lng": pano["lng"] if pano else None,
                "score": r.get("SCORE"),
                "address": None,
                "name": None,
                "source": "street_view",
            }
        )
    return out


def _reverse_geocode(item: Dict[str, Any]) -> None:
    if item["lat"] is None or item["lng"] is None:
        return
    try:
        results = gmaps().reverse_geocode((item["lat"], item["lng"]))
        if results:
            addr = results[0].get("formatted_address", "")
            item["address"] = " ".join(p for p in addr.split(", ") if "+" not in p)
    except Exception:
        pass


def _name_prompt(user_query_text: str) -> str:
    return f"""You are analyzing a street view image to extract a visible name related to a user query.
User query: "{user_query_text}"
Respond ONLY with a single valid JSON object — no markdown, no explanation outside JSON:
{{"NAME": "exact visible name or null", "CONFIDENCE": 0}}
NAME must be the exact text visible in the image relevant to the query, make sure its the right building/object based on the query that the name belongs to.
If the confidence is low, it can also be left null, but avoid too many nulls.
CONFIDENCE is 0 to 100 as an integer."""


def _extract_name(pano_id: str, image_file: str, prompt: str):
    img_path = os.path.join(config.RELEVANT_DIR, image_file)
    if not os.path.exists(img_path):
        img_path = os.path.join(config.STREET_VIEWS_DIR, image_file)
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    if config.IMAGE_PROVIDER == "COHERE":
        response = cohere_client().chat(
            model=config.COHERE_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMP,
        )
        raw = response.message.content[0].text.strip()
    else:
        response = openrouter_client().chat.completions.create(
            model=config.OPENROUTER_MODEL_ID,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMP,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        raw = response.choices[0].message.content.strip()

    parsed = json.loads(raw)
    name = parsed.get("NAME")
    confidence = int(parsed.get("CONFIDENCE", 0) or 0)
    return pano_id, (name if confidence >= 70 else None)


def enrich_findings(user_query_text: str) -> Dict[str, Any]:
    """Build the final findings list from YES-scored images:

    1. Join each YES image with its panorama metadata (lat/lng/heading).
    2. Reverse-geocode each pano's coordinates for a human-readable address.
    3. For the highest-scoring image per panorama, extract a visible name
       relevant to the user query using a vision LLM (only kept when the
       model's CONFIDENCE >= 70).

    Reads ``relevant_images/results.json``, ``street_views/metadata.json`` and
    ``panos.json``; writes the combined output to ``output/findings.json``.

    Args:
        user_query_text: The user's original natural-language query.

    Returns:
        ``{"count": int}`` ready for final formatting.
    """
    _progress.start("enrich_findings", f"provider={config.IMAGE_PROVIDER}")
    try:
        for path, label in [
            (config.RELEVANCY_PATH, "relevant_images/results.json"),
            (config.SV_METADATA_PATH, "street_views/metadata.json"),
            (config.PANOS_PATH, "panos.json"),
        ]:
            if not path.exists():
                _progress.fail(f"missing {label}")
                return {"error": f"Missing {label} — run earlier pipeline steps first."}

        relevancy = json.loads(config.RELEVANCY_PATH.read_text())
        yes_hits = [r for r in relevancy if r["CATEGORY"] == "YES"]
        saved_metadata = json.loads(config.SV_METADATA_PATH.read_text())
        panos = json.loads(config.PANOS_PATH.read_text())

        findings = _build_findings(yes_hits, saved_metadata, panos)
        if not findings:
            config.FINDINGS_PATH.write_text(json.dumps([], indent=2))
            _progress.done("0 findings")
            return {"count": 0}

        _progress.info(f"reverse-geocoding {len(findings)} findings")
        with ThreadPoolExecutor(max_workers=config.GCP_WORKERS) as executor:
            list(executor.map(_reverse_geocode, findings))

        best_per_pano: Dict[str, Dict[str, Any]] = {}
        for item in findings:
            pid = item["pano_id"]
            score = item.get("score") or 0
            if pid not in best_per_pano or score > best_per_pano[pid]["score"]:
                best_per_pano[pid] = {"item": item, "score": score}

        prompt = _name_prompt(user_query_text)
        total = len(best_per_pano)
        report_every = max(total // 5, 1)
        completed = 0
        _progress.info(f"extracting names for {total} top-scoring images")

        with ThreadPoolExecutor(max_workers=config.LLM_WORKERS) as executor:
            futures = {
                executor.submit(
                    _extract_name,
                    v["item"]["pano_id"],
                    v["item"]["image_file"],
                    prompt,
                ): v["item"]["pano_id"]
                for v in best_per_pano.values()
            }
            for future in as_completed(futures):
                try:
                    pano_id, name = future.result()
                    if name is not None:
                        for item in findings:
                            if item["pano_id"] == pano_id:
                                item["name"] = name
                except Exception:
                    pass
                completed += 1
                if completed % report_every == 0 or completed == total:
                    _progress.info(f"named {completed}/{total}")

        config.FINDINGS_PATH.write_text(json.dumps(findings, indent=2))
        named = sum(1 for f in findings if f.get("name"))
        _progress.done(f"{len(findings)} findings, {named} named")
        return {"count": len(findings), "named": named}
    except Exception as e:  # noqa: BLE001
        _progress.fail(str(e))
        return {"error": str(e)}
