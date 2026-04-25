"""Step 2 (baseline ground truth): query Google Places `nearby` directly.

NOT part of the agent's runtime path. Used to compare what the Places API
returns vs. what the Street-View pipeline rediscovers.

Usage:
    python scripts/baseline_nearby.py "vegetarian restaurants near NTR stadium guntur"
    python scripts/baseline_nearby.py 16.3157194 80.422113 --type restaurant --keyword vegetarian
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clients import gmaps  # noqa: E402
from src.tools.geocode import geocode_query  # noqa: E402


def search_nearby_places(lat, lng, radius=500, search_type=None, keyword=None):
    try:
        params = {"location": (lat, lng), "radius": radius, "keyword": keyword}
        if search_type:
            params["type"] = search_type
        api_response = gmaps().places_nearby(**params)
        results = api_response.get("results", [])
        cleaned = []
        for p in results:
            photo_list = p.get("photos", [])
            first_photo = photo_list[0].get("photo_reference") if photo_list else None
            cleaned.append(
                {
                    "name": p.get("name"),
                    "place_id": p.get("place_id"),
                    "address": p.get("vicinity"),
                    "rating": p.get("rating"),
                    "user_ratings_total": p.get("user_ratings_total"),
                    "types": p.get("types"),
                    "price_level": p.get("price_level"),
                    "business_status": p.get("business_status"),
                    "phone_number": p.get("international_phone_number"),
                    "open_now": p.get("opening_hours", {}).get("open_now"),
                    "photo_reference": first_photo,
                    "location": p.get("geometry", {}).get("location"),
                }
            )
        return cleaned
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("args", nargs="+", help="Either a free-text query, or `lat lng`.")
    p.add_argument("--radius", type=int, default=500)
    p.add_argument("--type", dest="search_type", default="restaurant")
    p.add_argument("--keyword", default="vegetarian")
    p.add_argument("--out", default="output/baseline_nearby.json")
    ns = p.parse_args()

    if len(ns.args) == 2:
        try:
            lat, lng = float(ns.args[0]), float(ns.args[1])
        except ValueError:
            lat = lng = None
    else:
        lat = lng = None

    if lat is None or lng is None:
        query = " ".join(ns.args)
        geo = geocode_query(query)
        if "error" in geo:
            print(json.dumps(geo, indent=2))
            return 1
        lat, lng = geo["latitude"], geo["longitude"]
        print(f"Geocoded {query!r} -> ({lat}, {lng})")

    found = search_nearby_places(
        lat=lat,
        lng=lng,
        radius=ns.radius,
        search_type=ns.search_type,
        keyword=ns.keyword,
    )
    print(f"Found {len(found) if isinstance(found, list) else 0} matches")
    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.out).write_text(json.dumps(found, indent=2))
    print(f"Wrote {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
