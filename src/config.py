"""Central configuration. Loads .env.local and exposes constants & paths."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env.local"
load_dotenv(ENV_PATH)


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in {ENV_PATH}."
        )
    return val


GCP_GMAP_KEY = _require("GCP_GMAP_KEY")
COHERE_API_KEY = (os.environ.get("COHERE_API_KEY") or "").strip()
OPENROUTER_API_KEY = (os.environ.get("OPENROUTER_API_KEY") or "").strip()

OPENROUTER_MODEL_ID = os.environ.get(
    "OPENROUTER_MODEL_ID", "google/gemini-3-flash-preview"
)
COHERE_MODEL_ID = os.environ.get("COHERE_MODEL_ID", "command-a-vision-07-2025")

IMAGE_PROVIDER_PREF = os.environ.get("IMAGE_PROVIDER", "OPENROUTER").upper()
AGENT_PROVIDER_PREF = os.environ.get("AGENT_PROVIDER", "OPENROUTER").upper()

from .provider_resolve import resolve_llm_provider  # noqa: E402

IMAGE_PROVIDER = resolve_llm_provider(
    IMAGE_PROVIDER_PREF,
    COHERE_API_KEY,
    OPENROUTER_API_KEY,
    COHERE_MODEL_ID,
    OPENROUTER_MODEL_ID,
    role="vision (IMAGE_PROVIDER)",
)
AGENT_PROVIDER = resolve_llm_provider(
    AGENT_PROVIDER_PREF,
    COHERE_API_KEY,
    OPENROUTER_API_KEY,
    COHERE_MODEL_ID,
    OPENROUTER_MODEL_ID,
    role="agent (AGENT_PROVIDER)",
)

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1000"))
TEMP = float(os.environ.get("TEMP", "0.2"))

DEFAULT_RADIUS_M = 500
DEFAULT_STEP_M = 20
DEFAULT_DEDUP_M = 20
DEFAULT_HEADINGS = 3
DEFAULT_FOV = 135
DEFAULT_PITCH = 0
GCP_WORKERS = 24
LLM_WORKERS = 12

OUTPUT_DIR = PROJECT_ROOT / "output"
STREET_VIEWS_DIR = OUTPUT_DIR / "street_views"
RELEVANT_DIR = OUTPUT_DIR / "relevant_images"

GEOCODE_PATH = OUTPUT_DIR / "geocode.json"
PANOS_PATH = OUTPUT_DIR / "panos.json"
SV_METADATA_PATH = STREET_VIEWS_DIR / "metadata.json"
RELEVANCY_PATH = RELEVANT_DIR / "results.json"
FINDINGS_PATH = OUTPUT_DIR / "findings.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STREET_VIEWS_DIR.mkdir(parents=True, exist_ok=True)
RELEVANT_DIR.mkdir(parents=True, exist_ok=True)
