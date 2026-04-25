"""Debug: download + save the headings of a single panorama for inspection.

Also includes a `closest 5 panos to a coordinate` helper. NOT part of the
agent runtime.

Usage:
    python scripts/debug_pano_display.py 9xFAU9D-8ES3C2ZOn5RUeg
    python scripts/debug_pano_display.py --closest 16.3159081 80.4220971
"""
from __future__ import annotations

import argparse
import json
import sys
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


SV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"


def haversine(lat1, lng1, lat2, lng2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(
        dlng / 2
    ) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def get_5_closest(lat, lng, pano_list):
    scored = [(haversine(lat, lng, p["lat"], p["lng"]), p) for p in pano_list]
    scored.sort(key=lambda x: x[0])
    return [{"distance_m": round(d, 1), **p} for d, p in scored[:5]]


def save_pano(pano_id: str, num: int, fov: int, pitch: int, step: int):
    out_dir = config.OUTPUT_DIR / "debug_pano" / pano_id
    out_dir.mkdir(parents=True, exist_ok=True)
    interval = 360 // max(num, 1)
    headings = [(step + i * interval) % 360 for i in range(num)]
    saved = []
    for h in headings:
        params = {
            "size": "640x640",
            "pano": pano_id,
            "heading": h,
            "pitch": pitch,
            "fov": fov,
            "key": config.GCP_GMAP_KEY,
        }
        r = requests.get(SV_IMAGE_URL, params=params, timeout=15)
        if r.status_code == 200:
            path = out_dir / f"{pano_id}_{h}.png"
            path.write_bytes(r.content)
            saved.append(str(path))
            print(f"Pano: {pano_id} | heading: {h}° | pitch: {pitch}° -> {path}")
    return saved


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("pano_id", nargs="?")
    p.add_argument("--closest", nargs=2, metavar=("LAT", "LNG"))
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--fov", type=int, default=100)
    p.add_argument("--pitch", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    ns = p.parse_args()

    if ns.closest:
        if not config.PANOS_PATH.exists():
            print(f"{config.PANOS_PATH} not found; run the agent first.")
            return 1
        panos = json.loads(config.PANOS_PATH.read_text())
        result = get_5_closest(float(ns.closest[0]), float(ns.closest[1]), panos)
        print(json.dumps(result, indent=2))
        return 0

    if not ns.pano_id:
        p.print_usage()
        return 2

    save_pano(ns.pano_id, ns.num, ns.fov, ns.pitch, ns.step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
