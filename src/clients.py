"""Lazy-initialized API clients shared across tools."""
from __future__ import annotations

from functools import lru_cache

import cohere
import googlemaps
from openai import OpenAI

from . import config


@lru_cache(maxsize=1)
def gmaps() -> googlemaps.Client:
    return googlemaps.Client(key=config.GCP_GMAP_KEY)


@lru_cache(maxsize=1)
def cohere_client() -> cohere.ClientV2:
    if not config.COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is empty")
    return cohere.ClientV2(api_key=config.COHERE_API_KEY)


@lru_cache(maxsize=1)
def openrouter_client() -> OpenAI:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is empty")
    return OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
