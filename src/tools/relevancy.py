"""Vision-based image relevance scoring (YES/NO) using Cohere or OpenRouter."""
from __future__ import annotations

import base64
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from .. import _progress, config
from ..clients import cohere_client, openrouter_client


def _build_prompt(user_query_text: str) -> str:
    return f"""You are a visual relevance scorer for street view images.
User query: "{user_query_text}"
Analyze the image and respond ONLY with a single valid JSON object — no markdown, no explanation outside JSON:
{{"CATEGORY": "YES/NO", "SCORE": 00, "REASONING": "example reasoning here"}}
CATEGORY must be exactly YES or NO.
SCORE is 0 to 100 indicating likelihood the view in the image is relevant to the query when using street view for navigation.
REASONING is 1 to 3 sentences describing exactly what in the image informed the decision."""


def _score_one(filename: str, prompt: str) -> Dict[str, Any]:
    img_path = os.path.join(config.STREET_VIEWS_DIR, filename)
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
    return {
        "image_path": img_path,
        "CATEGORY": parsed["CATEGORY"],
        "SCORE": parsed["SCORE"],
        "REASONING": parsed["REASONING"],
    }


def filter_relevant_images(user_query_text: str) -> Dict[str, Any]:
    """Score every PNG in output/street_views/ for relevance to the user's
    query using a vision LLM. Images marked YES are copied into
    output/relevant_images/ and the full per-image result list is written to
    output/relevant_images/results.json.

    Args:
        user_query_text: The user's original natural-language query, e.g.
            "vegetarian restaurants near NTR stadium guntur".

    Returns:
        ``{"yes": int, "no": int, "errors": int, "total": int}``.
    """
    _progress.start(
        "filter_relevant_images", f"provider={config.IMAGE_PROVIDER}"
    )
    try:
        if not config.STREET_VIEWS_DIR.exists():
            _progress.fail("street_views/ missing")
            return {"error": "street_views/ missing — run save_pano_images first."}

        image_files = [
            f for f in os.listdir(config.STREET_VIEWS_DIR) if f.endswith(".png")
        ]
        if not image_files:
            _progress.done("0 images")
            return {"yes": 0, "no": 0, "errors": 0, "total": 0}

        prompt = _build_prompt(user_query_text)
        results: list[dict] = []
        errors = 0
        total = len(image_files)
        report_every = max(total // 5, 1)
        completed = 0

        with ThreadPoolExecutor(max_workers=config.LLM_WORKERS) as executor:
            futures = {executor.submit(_score_one, f, prompt): f for f in image_files}
            for future in as_completed(futures):
                fname = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result["CATEGORY"] == "YES":
                        shutil.copy(
                            result["image_path"],
                            os.path.join(
                                config.RELEVANT_DIR,
                                os.path.basename(result["image_path"]),
                            ),
                        )
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    results.append(
                        {
                            "image_path": os.path.join(
                                config.STREET_VIEWS_DIR, fname
                            ),
                            "CATEGORY": "ERROR",
                            "SCORE": 0,
                            "REASONING": str(e),
                        }
                    )
                completed += 1
                if completed % report_every == 0 or completed == total:
                    _progress.info(f"scored {completed}/{total}")

        config.RELEVANCY_PATH.write_text(json.dumps(results, indent=2))
        yes = sum(1 for r in results if r["CATEGORY"] == "YES")
        no = sum(1 for r in results if r["CATEGORY"] == "NO")
        _progress.done(f"YES={yes} NO={no} ERR={errors}")
        return {"yes": yes, "no": no, "errors": errors, "total": len(results)}
    except Exception as e:  # noqa: BLE001
        _progress.fail(str(e))
        return {"error": str(e)}
