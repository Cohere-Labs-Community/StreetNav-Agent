"""Step 8: side-by-side comparison of Places-API baseline vs Street-View findings.

Reads:
  - output/geocode.json
  - output/baseline_nearby.json   (from scripts/baseline_nearby.py)
  - output/findings.json          (from the agent)

Prints, for each baseline place, whether it is inside the search radius and
what the Street-View pipeline found (matched by pano-derived address/name).
NOT part of the agent runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def is_in_search_radius(lat, lng, center_lat, center_lng, radius_m=500):
    lng_factor = np.cos(np.radians(center_lat))
    dlat = (lat - center_lat) * 111320
    dlng = (lng - center_lng) * 111320 * lng_factor
    return float(np.sqrt(dlat ** 2 + dlng ** 2)) <= radius_m


def main() -> int:
    if not config.GEOCODE_PATH.exists():
        print(f"{config.GEOCODE_PATH} missing — run the agent first.")
        return 1

    geocode = json.loads(config.GEOCODE_PATH.read_text())
    center_lat = geocode["latitude"]
    center_lng = geocode["longitude"]

    baseline_path = config.OUTPUT_DIR / "baseline_nearby.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        print(f"\n=== BASELINE places (Places API) — coverage check ===")
        print(f"Center: ({center_lat:.6f}, {center_lng:.6f})\n")
        for place in baseline:
            loc = place.get("location") or {}
            lat, lng = loc.get("lat"), loc.get("lng")
            if lat is None or lng is None:
                continue
            covered = is_in_search_radius(lat, lng, center_lat, center_lng)
            print(
                f"{(place.get('name') or '')[:40]:<40} "
                f"{lat:.6f}  {lng:.6f}  {'YES' if covered else 'NO'}"
            )
    else:
        print(f"(no {baseline_path}; run scripts/baseline_nearby.py first)")

    if config.FINDINGS_PATH.exists():
        findings = json.loads(config.FINDINGS_PATH.read_text())
        print(f"\n=== STREET-VIEW findings ({len(findings)}) ===")
        for item in findings:
            print(
                f"pano: {item.get('pano_id')}  "
                f"heading: {item.get('heading')}°  "
                f"name: {item.get('name')}  "
                f"address: {item.get('address')}"
            )
    else:
        print(f"(no {config.FINDINGS_PATH}; run the agent first)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
