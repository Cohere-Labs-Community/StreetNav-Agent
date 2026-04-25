"""CLI entry point for the StreetNav agent (and a direct no-agent pipeline)."""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict


def _print_summary(query: str, final: Dict[str, Any]) -> None:
    if not isinstance(final, dict) or "error" in final:
        print(final)
        return
    center = final.get("query_center") or {}
    items = final.get("items") or []
    named = [i for i in items if i.get("name")]
    print()
    print("=" * 60)
    print(f"Query     : {query}")
    print(
        f"Center    : ({center.get('latitude')}, {center.get('longitude')})"
        f"  {center.get('city') or ''}, {center.get('country') or ''}"
    )
    print(f"Findings  : {len(items)} relevant images, {len(named)} named")
    if named:
        print("Top named :")
        for it in named[:3]:
            print(f"  - {it['name']}  ({it.get('address') or 'no address'})")
    print("=" * 60)


def run_pipeline(user_query: str) -> Dict[str, Any]:
    """Run the 6-step pipeline directly without the Strands agent."""
    from src.tools import (
        enrich_findings,
        filter_relevant_images,
        find_street_view_panos,
        format_final_results,
        geocode_query,
        save_pano_images,
    )

    geo = geocode_query(user_query)
    if isinstance(geo, dict) and "error" in geo:
        return geo

    panos = find_street_view_panos(
        center_lat=geo["latitude"], center_lng=geo["longitude"]
    )
    if isinstance(panos, dict) and "error" in panos:
        return panos

    saved = save_pano_images()
    if isinstance(saved, dict) and "error" in saved:
        return saved

    rel = filter_relevant_images(user_query)
    if isinstance(rel, dict) and "error" in rel:
        return rel

    enr = enrich_findings(user_query)
    if isinstance(enr, dict) and "error" in enr:
        return enr

    return format_final_results()


def run_agent(user_query: str) -> Any:
    from src.agent import build_agent

    agent = build_agent()
    return agent(user_query)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the StreetNav agent on a free-text location query."
    )
    parser.add_argument(
        "query",
        nargs="+",
        help='User query, e.g. "vegetarian restaurants near NTR stadium guntur"',
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Run the same six-step pipeline without the Strands agent (sequential non-agentic workflow).",
    )
    args = parser.parse_args()

    user_query = " ".join(args.query).strip()
    if not user_query:
        print("Empty query.", file=sys.stderr)
        return 2

    if args.no_agent:
        final = run_pipeline(user_query)
        _print_summary(user_query, final)
    else:
        result = run_agent(user_query)
        print()
        print("=" * 60)
        print(result)
        print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
