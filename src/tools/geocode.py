"""Geocoding tool: free-text query -> lat/lng + address components."""
from __future__ import annotations

import json
from typing import Any, Dict

from .. import _progress, config
from ..clients import gmaps


def geocode_query(query: str) -> Dict[str, Any]:
    """Resolve a free-text location query (place name, address) to a single
    geocoded result.

    Use this once at the start of the pipeline to anchor the search to a
    latitude/longitude. The result is also persisted to output/geocode.json.

    Args:
        query: Free text such as "NTR stadium guntur" or a postal address.

    Returns:
        A dict with at minimum ``latitude`` and ``longitude``, plus address
        components (building, street, area, city, state, country, postal_code,
        place_id, available_travel_modes). On failure returns ``{"error": ...}``.
    """
    _progress.start("geocode_query", f"query={query[:60]}")
    try:
        results = gmaps().geocode(query)
        if not results:
            _progress.fail("no results")
            return {"error": "No results found."}
        res = results[0]
        components = res.get("address_components", [])

        def get_comp(target_types, use_short=False):
            for c in components:
                if any(t in c["types"] for t in target_types):
                    return c["short_name"] if use_short else c["long_name"]
            return None

        nav_points = res.get("navigation_points", [])
        travel_modes: list[str] = []
        for point in nav_points:
            travel_modes.extend(point.get("restricted_travel_modes", []))

        out = {
            "latitude": res["geometry"]["location"]["lat"],
            "longitude": res["geometry"]["location"]["lng"],
            "place_id": res.get("place_id"),
            "building": get_comp(["premise", "street_number"]),
            "street": get_comp(["route"]),
            "area": get_comp(
                ["sublocality", "neighborhood", "administrative_area_level_2"]
            ),
            "city": get_comp(["locality"]),
            "state": get_comp(["administrative_area_level_1"], use_short=True),
            "country": get_comp(["country"]),
            "postal_code": get_comp(["postal_code"]),
            "available_travel_modes": list(set(travel_modes)),
        }

        config.GEOCODE_PATH.write_text(json.dumps(out, indent=2))
        _progress.done(f"lat={out['latitude']:.5f} lng={out['longitude']:.5f}")
        return out
    except Exception as e:  # noqa: BLE001
        _progress.fail(str(e))
        return {"error": str(e)}
