"""Street-view discovery + image download tools."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
import requests

from .. import _progress, cache, config
from ..clients import gmaps


SV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"


def _snap_grid(center_lat: float, center_lng: float, total_radius: int, step_meters: int):
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


def _dedup(points, lng_factor: float, min_dist: float):
    kept: list[tuple[float, float]] = []
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


def _tight_dedup(panos: list[dict]) -> list[dict]:
    if len(panos) < 2:
        return panos
    coords = np.array([[p["lat"], p["lng"]] for p in panos])
    lng_f = np.cos(np.radians(coords[:, 0].mean()))
    lat_m = coords[:, 0:1] - coords[:, 0]
    lng_m = coords[:, 1:2] - coords[:, 1]
    dist_matrix = np.sqrt((lat_m * 111320) ** 2 + (lng_m * 111320 * lng_f) ** 2)
    np.fill_diagonal(dist_matrix, np.inf)
    neighbor_counts = (dist_matrix < 50).sum(axis=1)
    drop: set[int] = set()
    for i in range(len(panos)):
        if i in drop:
            continue
        for j in range(i + 1, len(panos)):
            if j in drop:
                continue
            if dist_matrix[i, j] < 5:
                drop.add(i if neighbor_counts[i] < neighbor_counts[j] else j)
    return [p for idx, p in enumerate(panos) if idx not in drop]


def find_street_view_panos(
    center_lat: float,
    center_lng: float,
    total_radius: int = config.DEFAULT_RADIUS_M,
    step_meters: int = config.DEFAULT_STEP_M,
    dedup_min_dist: int = config.DEFAULT_DEDUP_M,
) -> Dict[str, Any]:
    """Discover unique outdoor Street View panoramas on roads inside a radius
    around (center_lat, center_lng).

    The function snaps a regular grid of candidate points to roads, queries
    Street View metadata for each, and dedupes by panorama id and proximity.
    The resulting list is persisted to output/panos.json.

    Default values for ``total_radius``, ``step_meters`` and ``dedup_min_dist``
    are tuned for typical urban queries; only override them when the user
    explicitly asks for something different.

    Args:
        center_lat: Center latitude in decimal degrees.
        center_lng: Center longitude in decimal degrees.
        total_radius: Search radius in meters (default 500).
        step_meters: Grid spacing in meters (default 20).
        dedup_min_dist: Minimum spacing between snapped points before metadata
            lookup, in meters (default 20).

    Returns:
        ``{"count": int, "panos": [...]}`` where each pano has
        ``pano_id, lat, lng, date, source_found``.
    """
    _progress.start(
        "find_street_view_panos",
        f"center=({center_lat:.5f},{center_lng:.5f}) r={total_radius}m",
    )
    try:
        raw_grid, lng_factor = _snap_grid(
            center_lat, center_lng, total_radius, step_meters
        )

        chunk_size = 100
        chunks = [
            raw_grid[i : i + chunk_size]
            for i in range(0, len(raw_grid), chunk_size)
        ]

        def _snap_chunk(chunk):
            try:
                return gmaps().snap_to_roads(chunk, interpolate=True)
            except Exception:
                return []

        road_points: list[tuple[float, float]] = []
        with ThreadPoolExecutor(max_workers=config.GCP_WORKERS) as ex:
            for snapped in ex.map(_snap_chunk, chunks):
                for p in snapped:
                    road_points.append(
                        (p["location"]["latitude"], p["location"]["longitude"])
                    )

        kept_points = _dedup(road_points, lng_factor, dedup_min_dist)
        _progress.info(
            f"snapped={len(road_points)} kept={len(kept_points)}; querying SV metadata"
        )

        def _fetch_meta(latlng):
            lat, lng = latlng
            try:
                return requests.get(
                    SV_METADATA_URL,
                    params={
                        "location": f"{lat},{lng}",
                        "radius": 20,
                        "key": config.GCP_GMAP_KEY,
                        "source": "outdoor",
                    },
                    timeout=10,
                ).json()
            except Exception:
                return None

        available: list[dict] = []
        seen: set[str] = set()
        total_pts = len(kept_points)
        report_every = max(total_pts // 5, 1)
        completed = 0
        with ThreadPoolExecutor(max_workers=config.GCP_WORKERS) as ex:
            futures = [ex.submit(_fetch_meta, pt) for pt in kept_points]
            for fut in as_completed(futures):
                completed += 1
                if completed % report_every == 0 or completed == total_pts:
                    _progress.info(f"sv metadata {completed}/{total_pts}")
                meta = fut.result()
                if not meta or meta.get("status") != "OK":
                    continue
                pano_id = meta.get("pano_id")
                if not pano_id or pano_id in seen:
                    continue
                seen.add(pano_id)
                available.append(
                    {
                        "pano_id": pano_id,
                        "lat": meta["location"]["lat"],
                        "lng": meta["location"]["lng"],
                        "date": meta.get("date"),
                        "source_found": "outdoor",
                    }
                )

        available = _tight_dedup(available)
        config.PANOS_PATH.write_text(json.dumps(available, indent=2))
        _progress.done(f"{len(available)} unique panos")
        return {"count": len(available), "panos": available}
    except Exception as e:  # noqa: BLE001
        _progress.fail(str(e))
        return {"error": str(e)}


def save_pano_images(
    num_per_pano: int = config.DEFAULT_HEADINGS,
    fov: int = config.DEFAULT_FOV,
    pitch: int = config.DEFAULT_PITCH,
    step: int = 0,
) -> Dict[str, Any]:
    """Download Street View images for every panorama listed in
    output/panos.json. Generates ``num_per_pano`` evenly spaced headings per
    pano and saves PNGs into output/street_views/.

    Defaults are tuned for typical queries; only override them when the
    user's intent gives strong reason to.

    Args:
        num_per_pano: Number of headings per panorama (default 3).
        fov: Field of view in degrees (default 135).
        pitch: Camera pitch in degrees (default 0).
        step: Optional heading offset in degrees (default 0).

    Returns:
        ``{"images": int, "panos": int}``. Per-image metadata is written to
        output/street_views/metadata.json.
    """
    _progress.start("save_pano_images", f"headings/pano={num_per_pano} fov={fov}")
    try:
        if not config.PANOS_PATH.exists():
            _progress.fail("panos.json missing")
            return {"error": "panos.json missing — run find_street_view_panos first."}

        pano_list: List[Dict[str, Any]] = json.loads(config.PANOS_PATH.read_text())
        if not pano_list:
            _progress.done("0 panos")
            return {"images": 0, "panos": 0}

        angle_interval = 360 // max(num_per_pano, 1)
        headings_list = [
            (step + (i * angle_interval)) % 360 for i in range(num_per_pano)
        ]
        all_pano_ids = [p["pano_id"] for p in pano_list]
        pano_lookup = {p["pano_id"]: p for p in pano_list}

        catalogue = cache.sync_catalogue_from_hf()
        hits, misses = cache.plan_cache_usage(
            all_pano_ids, catalogue, headings_list, fov, pitch
        )
        if cache.hf_enabled():
            _progress.info(
                f"cache: {len(hits)} hits, {len(misses)} misses "
                f"(repo={config.HF_DATASET_REPO})"
            )
        else:
            _progress.info(
                f"cache: disabled (no HF token) — downloading all {len(misses)} panos"
            )

        metadata: list[dict] = []

        if hits:
            cached_md = cache.fetch_cached_images(
                hits, headings_list, config.STREET_VIEWS_DIR
            )
            for m in cached_md:
                m["pitch"] = pitch
                m["fov"] = fov
            metadata.extend(cached_md)
            recovered_panos = {m["panoid"] for m in cached_md}
            actually_cached = recovered_panos & set(hits)
            stale_hits = [pid for pid in hits if pid not in actually_cached]
            if stale_hits:
                _progress.info(
                    f"cache: {len(stale_hits)} hits missing on HF — falling back to streetview_api"
                )
                misses.extend(stale_hits)

        tasks: list[tuple[str, int]] = []
        for pid in misses:
            for h in headings_list:
                tasks.append((pid, h))

        def _download(pano_id: str, heading: int):
            params = {
                "size": "640x640",
                "pano": pano_id,
                "heading": heading,
                "pitch": pitch,
                "fov": fov,
                "key": config.GCP_GMAP_KEY,
            }
            try:
                resp = requests.get(SV_IMAGE_URL, params=params, timeout=15)
            except Exception:
                return None
            if resp.status_code != 200:
                return None
            filename = f"{pano_id}_{heading}.png"
            with open(os.path.join(config.STREET_VIEWS_DIR, filename), "wb") as f:
                f.write(resp.content)
            return {
                "panoid": pano_id,
                "pitch": pitch,
                "fov": fov,
                "heading": heading,
                "image": filename,
                "source": "streetview_api",
            }

        downloaded_panos: set[str] = set()
        if tasks:
            total_imgs = len(tasks)
            report_every = max(total_imgs // 5, 1)
            completed = 0
            with ThreadPoolExecutor(max_workers=config.GCP_WORKERS) as ex:
                futures = [ex.submit(_download, pid, h) for pid, h in tasks]
                for fut in as_completed(futures):
                    completed += 1
                    try:
                        result = fut.result()
                    except Exception:
                        result = None
                    if result:
                        metadata.append(result)
                        downloaded_panos.add(result["panoid"])
                    if completed % report_every == 0 or completed == total_imgs:
                        _progress.info(
                            f"downloaded {completed}/{total_imgs} images (streetview_api)"
                        )

        config.SV_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

        uploaded = 0
        if downloaded_panos and cache.hf_enabled():
            _progress.info(
                f"cache: publishing {len(downloaded_panos)} panos to HF"
            )
            uploaded = cache.publish_to_hf(
                new_or_refreshed_pano_ids=downloaded_panos,
                pano_metadata=pano_lookup,
                headings=headings_list,
                fov=fov,
                pitch=pitch,
                source_dir=config.STREET_VIEWS_DIR,
            )

        summary = (
            f"{len(metadata)} images / {len(pano_list)} panos "
            f"(cache hits={len(hits) - len([p for p in hits if p in downloaded_panos])}, "
            f"gcp={len(downloaded_panos)}, pushed={uploaded})"
        )
        _progress.done(summary)
        return {
            "images": len(metadata),
            "panos": len(pano_list),
            "cache_hits": len(hits),
            "gcp_downloads": len(downloaded_panos),
            "uploaded_to_hf": uploaded,
        }
    except Exception as e:  # noqa: BLE001
        _progress.fail(str(e))
        return {"error": str(e)}
