"""StreetNav Strands agent: orchestrates the fixed 6-step pipeline.

The same plain Python functions used by ``--no-agent`` mode (in
``src/tools/``) are wrapped here with ``strands.tool`` and handed to a Strands
``Agent``. A no-op callback handler is used so the terminal only shows the
per-tool progress lines emitted by the tools themselves, and the agent's final
answer at the end — never a stream of intermediate model tokens.
"""
from __future__ import annotations

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from . import config
from .tools import (
    enrich_findings,
    filter_relevant_images,
    find_street_view_panos,
    format_final_results,
    geocode_query,
    save_pano_images,
)


SYSTEM_PROMPT = """You are StreetNav, a navigation research agent.

Given a user's natural-language query about places near a location (for
example: "vegetarian restaurants near NTR stadium guntur"), you orchestrate a
FIXED pipeline of six tool calls — in this exact order — and then summarize.

PIPELINE
1. geocode_query(query)
   Pass the user's full original query as `query`. Read `latitude` and
   `longitude` from the result; you'll need them for step 2.

2. find_street_view_panos(center_lat, center_lng, ...)
   Use the latitude and longitude from step 1.

3. save_pano_images(...)

4. filter_relevant_images(user_query_text)
   Pass the user's full original query as `user_query_text`.

5. enrich_findings(user_query_text)
   Pass the user's full original query as `user_query_text`.

6. format_final_results()

PARAMETER POLICY
- Every tool above has sensible defaults for any optional parameter.
- DO NOT pass optional parameters unless the user's query gives a clear,
  high-confidence reason to deviate (for example, the user explicitly asks
  for a different search radius, more headings per panorama, etc.).
- When in doubt, omit optional parameters and let the defaults apply.

HARD RULES
- Call each of the six tools AT MOST ONCE. Never retry on success.
- If any tool returns a dict containing the key "error", STOP the pipeline
  and return that error to the user verbatim. Do NOT call later tools, do NOT
  attempt fallbacks, and do NOT re-run the failing tool.
- Do not call any tool not listed above.
- Do not invent arguments — only use values returned by previous tool calls
  or the user's original query.
- Image scoring and name extraction happen INSIDE the tools using a vision
  model; you must not try to inspect images yourself.
- After step 6 succeeds, write a short, plain-text summary (3-6 lines):
  the resolved location, how many relevant findings were produced, and the
  top 3 named results (name + address). Do NOT dump the full JSON.

Be terse. Be deterministic. Do not chit-chat. Do not loop."""


def _silent_callback(**_kwargs) -> None:
    """No-op callback: suppress the default token-streaming output so the
    terminal only shows the per-tool progress lines and the final answer."""


def _build_model() -> OpenAIModel:
    if config.AGENT_PROVIDER == "COHERE":
        return OpenAIModel(
            client_args={
                "api_key": config.COHERE_API_KEY,
                "base_url": "https://api.cohere.ai/compatibility/v1",
            },
            model_id=config.COHERE_MODEL_ID,
            params={"stream_options": None, "temperature": 0.1},
        )
    return OpenAIModel(
        client_args={
            "api_key": config.OPENROUTER_API_KEY,
            "base_url": "https://openrouter.ai/api/v1",
        },
        model_id=config.OPENROUTER_MODEL_ID,
        params={"temperature": 0.1},
    )


def build_agent() -> Agent:
    return Agent(
        model=_build_model(),
        tools=[
            tool(geocode_query),
            tool(find_street_view_panos),
            tool(save_pano_images),
            tool(filter_relevant_images),
            tool(enrich_findings),
            tool(format_final_results),
        ],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=_silent_callback,
    )
