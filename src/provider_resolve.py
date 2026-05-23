"""Pick Cohere vs OpenRouter once per process: one probe per backend max (cached)."""
from __future__ import annotations

from typing import Literal

import cohere
from openai import OpenAI

Provider = Literal["OPENROUTER", "COHERE"]

_or_ok: bool | None = None
_co_ok: bool | None = None


def _probe_openrouter(api_key: str, model_id: str) -> bool:
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=20,
        )
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        return True
    except Exception:
        return False


def _probe_cohere(api_key: str, model_id: str) -> bool:
    try:
        client = cohere.ClientV2(api_key=api_key)
        client.chat(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "ping"}],
                }
            ],
            max_tokens=1,
            temperature=0,
        )
        return True
    except Exception:
        return False


def _openrouter_works(api_key: str, model_id: str) -> bool:
    global _or_ok
    if _or_ok is not None:
        return _or_ok
    if not api_key:
        _or_ok = False
        return False
    _or_ok = _probe_openrouter(api_key, model_id)
    return _or_ok


def _cohere_works(api_key: str, model_id: str) -> bool:
    global _co_ok
    if _co_ok is not None:
        return _co_ok
    if not api_key:
        _co_ok = False
        return False
    _co_ok = _probe_cohere(api_key, model_id)
    return _co_ok


def resolve_llm_provider(
    preference: str,
    cohere_api_key: str,
    openrouter_api_key: str,
    cohere_model_id: str,
    openrouter_model_id: str,
    *,
    role: str,
) -> Provider:
    """Try preferred provider first (one cached probe per backend), then the other."""
    pref = preference.strip().upper()
    if pref not in ("OPENROUTER", "COHERE"):
        raise RuntimeError(
            f"Invalid {role} provider {preference!r}: use OPENROUTER or COHERE."
        )
    other: Provider = "COHERE" if pref == "OPENROUTER" else "OPENROUTER"
    order: list[Provider] = [pref, other]

    for p in order:
        if p == "OPENROUTER" and _openrouter_works(
            openrouter_api_key, openrouter_model_id
        ):
            return "OPENROUTER"
        if p == "COHERE" and _cohere_works(cohere_api_key, cohere_model_id):
            return "COHERE"

    raise RuntimeError(
        f"No working LLM for {role}. Set OPENROUTER_API_KEY or COHERE_API_KEY "
        f"in .env.local (only one is required). Tried {pref} first, then {other}. "
        "If you ran `cp .env.example .env.local`, replace the your_* placeholders with real keys."
    )
