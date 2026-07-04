"""Annotator step 1: geocode query, fetch Street View panos, write metadata skeleton.

Requires QUERY and SAMPLE_NAME env vars and GCP_GMAP_KEY in .env.local. See docs/ANNOTATORS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

try:
    import googlemaps
except ImportError as e:  # pragma: no cover
    raise SystemExit("googlemaps not installed — pip install -r requirements.txt") from e


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SAMPLES_DIR = SCRIPT_DIR / "samples"

load_dotenv(PROJECT_ROOT / ".env.local")

SV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"

DEFAULT_RADIUS_M = 500
DEFAULT_STEP_M = 30
DEFAULT_DEDUP_M = 30
DEFAULT_IMAGES_PER_PANO = 3
DEFAULT_PITCH = 0
GCP_WORKERS = 24


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip().strip('"').strip("'")


GCP_GMAP_KEY = _env("GCP_GMAP_KEY")
if not GCP_GMAP_KEY or GCP_GMAP_KEY.startswith("your_"):
    raise SystemExit("Missing GCP_GMAP_KEY in .env.local")

_gmaps = googlemaps.Client(key=GCP_GMAP_KEY)


def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def geocode_query(query: str) -> Dict[str, Any]:
    results = _gmaps.geocode(query)
    if not results:
        raise SystemExit(f"No geocode results for: {query!r}")
    res = results[0]
    components = res.get("address_components", [])

    def get_comp(target_types, use_short=False):
        for c in components:
            if any(t in c["types"] for t in target_types):
                return c["short_name"] if use_short else c["long_name"]
        return None

    return {
        "latitude": res["geometry"]["location"]["lat"],
        "longitude": res["geometry"]["location"]["lng"],
        "place_id": res.get("place_id"),
        "city": get_comp(["locality"]),
        "state": get_comp(["administrative_area_level_1"], use_short=True),
        "country": get_comp(["country"]),
    }


def _snap_grid(center_lat, center_lng, total_radius, step_meters):
    lng_factor = np.cos(np.radians(center_lat))
    grid_step = step_meters / 111320.0
    n = int(total_radius / step_meters) + 1
    raw = []
    for r in range(-n, n + 1):
        for c in range(-n, n + 1):
            lat = center_lat + r * grid_step
            lng = center_lng + (c * grid_step / lng_factor)
            dlat = (lat - center_lat) * 111320
            dlng = (lng - center_lng) * 111320 * lng_factor
            if np.sqrt(dlat ** 2 + dlng ** 2) <= total_radius:
                raw.append((lat, lng))
    return raw, lng_factor


def _dedup(points, lng_factor, min_dist):
    kept: List[Tuple[float, float]] = []
    for lat, lng in points:
        too_close = False
        for klat, klng in kept:
            dlat = (lat - klat) * 111320
            dlng = (lng - klng) * 111320 * lng_factor
            if np.sqrt(dlat ** 2 + dlng ** 2) < min_dist:
                too_close = True
                break
        if not too_close:
            kept.append((lat, lng))
    return kept


def _tight_dedup(panos: List[dict]) -> List[dict]:
    if len(panos) < 2:
        return panos
    coords = np.array([[p["lat"], p["lng"]] for p in panos])
    lng_f = np.cos(np.radians(coords[:, 0].mean()))
    lat_m = coords[:, 0:1] - coords[:, 0]
    lng_m = coords[:, 1:2] - coords[:, 1]
    dist_matrix = np.sqrt((lat_m * 111320) ** 2 + (lng_m * 111320 * lng_f) ** 2)
    np.fill_diagonal(dist_matrix, np.inf)
    neighbor_counts = (dist_matrix < 50).sum(axis=1)
    drop: set = set()
    for i in range(len(panos)):
        if i in drop:
            continue
        for j in range(i + 1, len(panos)):
            if j in drop:
                continue
            if dist_matrix[i, j] < 8:
                drop.add(i if neighbor_counts[i] < neighbor_counts[j] else j)
    return [p for idx, p in enumerate(panos) if idx not in drop]


def find_street_view_panos(center_lat, center_lng, total_radius=DEFAULT_RADIUS_M,
                           step_meters=DEFAULT_STEP_M, dedup_min_dist=DEFAULT_DEDUP_M):
    raw_grid, lng_factor = _snap_grid(center_lat, center_lng, total_radius, step_meters)
    chunk_size = 100
    chunks = [raw_grid[i:i + chunk_size] for i in range(0, len(raw_grid), chunk_size)]

    def _snap_chunk(chunk):
        try:
            return _gmaps.snap_to_roads(chunk, interpolate=True)
        except Exception:
            return []

    road_points: List[Tuple[float, float]] = []
    with ThreadPoolExecutor(max_workers=GCP_WORKERS) as ex:
        for snapped in ex.map(_snap_chunk, chunks):
            for p in snapped:
                road_points.append((p["location"]["latitude"], p["location"]["longitude"]))

    kept_points = _dedup(road_points, lng_factor, dedup_min_dist)
    _log(f"snapped={len(road_points)} kept={len(kept_points)}; querying SV metadata")

    def _fetch_meta(latlng):
        lat, lng = latlng
        try:
            return requests.get(
                SV_METADATA_URL,
                params={"location": f"{lat},{lng}", "radius": 20,
                        "key": GCP_GMAP_KEY, "source": "outdoor"},
                timeout=10,
            ).json()
        except Exception:
            return None

    available: List[dict] = []
    seen: set = set()
    with ThreadPoolExecutor(max_workers=GCP_WORKERS) as ex:
        futures = [ex.submit(_fetch_meta, pt) for pt in kept_points]
        for fut in as_completed(futures):
            meta = fut.result()
            if not meta or meta.get("status") != "OK":
                continue
            pano_id = meta.get("pano_id")
            if not pano_id or pano_id in seen:
                continue
            seen.add(pano_id)
            available.append({
                "pano_id": pano_id,
                "lat": meta["location"]["lat"],
                "lng": meta["location"]["lng"],
                "date": meta.get("date"),
            })
    return _tight_dedup(available)


def render_images(panos, images_dir: Path, images_per_pano: int, pitch: int):
    """Download evenly-spaced headings per pano with a 5-deg overlap on each side.

    fov = (360 / images_per_pano) + 10 so neighbouring images overlap by 5 deg.
    """
    interval = 360.0 / images_per_pano
    fov = int(round(interval)) + 10
    headings = [int(round(i * interval)) % 360 for i in range(images_per_pano)]

    tasks = [(p["pano_id"], h) for p in panos for h in headings]

    def _download(pano_id, heading):
        params = {"size": "640x640", "pano": pano_id, "heading": heading,
                  "pitch": pitch, "fov": fov, "key": GCP_GMAP_KEY}
        try:
            resp = requests.get(SV_IMAGE_URL, params=params, timeout=15)
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        filename = f"{pano_id}_{heading}.png"
        (images_dir / filename).write_bytes(resp.content)
        return {"pano_id": pano_id, "heading": heading, "image": filename}

    rendered: List[dict] = []
    with ThreadPoolExecutor(max_workers=GCP_WORKERS) as ex:
        futures = [ex.submit(_download, pid, h) for pid, h in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                rendered.append(r)
    return rendered, fov, headings


def reverse_geocode_place_ids(pano_ids_latlng: Dict[str, Tuple[float, float]]):
    """Return {pano_id: place_id} by reverse-geocoding each pano coordinate once."""
    out: Dict[str, str] = {}

    def _one(item):
        pid, (lat, lng) = item
        try:
            res = _gmaps.reverse_geocode((lat, lng))
            return pid, (res[0].get("place_id") if res else None)
        except Exception:
            return pid, None

    with ThreadPoolExecutor(max_workers=GCP_WORKERS) as ex:
        for pid, place_id in ex.map(_one, pano_ids_latlng.items()):
            out[pid] = place_id
    return out


def _require_sample_name() -> str:
    sample_name = os.environ.get("SAMPLE_NAME", "").strip()
    if not sample_name:
        raise SystemExit('Missing SAMPLE_NAME — run: export SAMPLE_NAME="yourname_001"')
    return sample_name


def _require_query() -> str:
    query = os.environ.get("QUERY", "").strip()
    if not query:
        raise SystemExit('Missing QUERY — run: export QUERY="cafes near NTR stadium guntur"')
    return query


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotator step 1: build a sample for manual annotation.")
    parser.add_argument("--images-per-pano", type=int, default=DEFAULT_IMAGES_PER_PANO,
                        help=f"headings per pano (default {DEFAULT_IMAGES_PER_PANO}); fov auto = 360/N + 10")
    args = parser.parse_args()

    sample_name = _require_sample_name()
    query = _require_query()

    sample_dir = SAMPLES_DIR / sample_name
    images_dir = sample_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\u25b6 geocode_query  ({query[:60]})", flush=True)
    geo = geocode_query(query)
    _log(f"lat={geo['latitude']:.5f} lng={geo['longitude']:.5f} "
         f"{geo.get('city') or ''}, {geo.get('country') or ''}")

    print("\u25b6 find_street_view_panos", flush=True)
    panos = find_street_view_panos(geo["latitude"], geo["longitude"])
    _log(f"{len(panos)} unique panos")
    if not panos:
        print("No panoramas found — nothing to annotate.", file=sys.stderr)

    print(f"\u25b6 render_images  (images_per_pano={args.images_per_pano})", flush=True)
    rendered, fov, headings = render_images(panos, images_dir, args.images_per_pano, DEFAULT_PITCH)
    _log(f"{len(rendered)} images @ fov={fov} headings={headings}")

    pano_latlng = {p["pano_id"]: (p["lat"], p["lng"]) for p in panos}
    pano_lookup = {p["pano_id"]: p for p in panos}
    print("\u25b6 reverse_geocode (place_id per pano)", flush=True)
    place_ids = reverse_geocode_place_ids(pano_latlng)

    images_meta: List[dict] = []
    for r in sorted(rendered, key=lambda x: (x["pano_id"], x["heading"])):
        pid = r["pano_id"]
        pano = pano_lookup.get(pid, {})
        images_meta.append({
            "image": r["image"],
            "lat": pano.get("lat"),
            "lng": pano.get("lng"),
            "place_id": place_ids.get(pid),
            "relevancy": "NO",
            "ambiguous": "NO",
        })

    metadata = {
        "sample_name": sample_name,
        "query_used": query,
        "country": geo.get("country"),
        "state": geo.get("state"),
        "city": geo.get("city"),
        "query_center": {
            "lat": geo["latitude"],
            "lng": geo["longitude"],
            "place_id": geo.get("place_id"),
        },
        "images_per_pano": args.images_per_pano,
        "images": images_meta,
    }

    out_path = sample_dir / f"{sample_name}_annotated_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2))
    print(f"\u2713 wrote {out_path}", flush=True)
    print(f"  mark relevant images 'relevancy' NO -> YES, then run:", flush=True)
    print(f"  python3 src/benchmarking/post_annotation.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
