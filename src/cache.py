"""HuggingFace-backed cache for Street View imagery.

Catalogue lives at ``<HF_DATASET_REPO>/catalogue.json``; PNGs live under
``<HF_DATASET_REPO>/images/<pano_id>/<pano_id>_<heading>.png``.

The cache is opportunistic: every public function degrades to a no-op when
the Hub is unreachable or no token is set, so the pipeline never breaks
because of caching. Worst case = redundant GCP fetches.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from . import _progress, config

try:
    from huggingface_hub import snapshot_download, upload_file, upload_folder
    from huggingface_hub.utils import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    _HF_AVAILABLE = True
except ImportError:
    GatedRepoError = HfHubHTTPError = RepositoryNotFoundError = None  # type: ignore[misc, assignment]
    _HF_AVAILABLE = False


_EMPTY_CATALOGUE: Dict[str, Any] = {"version": 1, "panos": {}}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hf_disabled_reason() -> str | None:
    """Return why the HF cache is off, or ``None`` when enabled."""
    if not _HF_AVAILABLE:
        return "huggingface_hub package not installed (pip install huggingface_hub)"
    if not config.HF_TOKEN:
        return (
            "no HUGGINGFACE_API_KEY in .env.local "
            "(set it to enable shared cache; pipeline continues without it)"
        )
    return None


def hf_enabled() -> bool:
    return hf_disabled_reason() is None


def _http_status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _format_hf_error(operation: str, exc: Exception) -> str:
    """Turn a Hub exception into an actionable debug message."""
    repo = config.HF_DATASET_REPO
    err_type = type(exc).__name__
    msg = str(exc).strip()
    lower = msg.lower()

    if not _HF_AVAILABLE:
        return (
            f"hf {operation} skipped: huggingface_hub is not installed "
            f"(pip install huggingface_hub)"
        )

    if GatedRepoError is not None and isinstance(exc, GatedRepoError):
        return (
            f"hf {operation} failed: access denied to gated dataset {repo!r}. "
            f"Request access at https://huggingface.co/datasets/{repo} and "
            f"ensure HUGGINGFACE_API_KEY in .env.local belongs to an approved account."
        )

    if RepositoryNotFoundError is not None and isinstance(exc, RepositoryNotFoundError):
        return (
            f"hf {operation} failed: dataset {repo!r} not found or not visible "
            f"with your token. Verify the repo exists and your token has "
            f"read access (downloads) or write access (uploads)."
        )

    status = _http_status(exc) if HfHubHTTPError is not None and isinstance(exc, HfHubHTTPError) else None
    if status is None:
        if "401" in lower or "unauthorized" in lower:
            status = 401
        elif "403" in lower or "forbidden" in lower:
            status = 403
        elif "404" in lower:
            status = 404

    if status == 401:
        return (
            f"hf {operation} failed: authentication error (HTTP 401). "
            f"Set a valid HUGGINGFACE_API_KEY in .env.local "
            f"(https://huggingface.co/settings/tokens)."
        )
    if status == 403:
        hint = (
            "write access is required for uploads"
            if operation == "upload"
            else "read access is required for downloads"
        )
        return (
            f"hf {operation} failed: permission denied (HTTP 403) for {repo!r}. "
            f"Your token may lack access to this private dataset ({hint})."
        )
    if status == 404:
        return f"hf {operation} failed: dataset {repo!r} not found (HTTP 404)."

    if HfHubHTTPError is not None and isinstance(exc, HfHubHTTPError):
        return f"hf {operation} failed: HTTP {status or '?'} from Hugging Face Hub — {msg}"

    if any(token in lower for token in ("connection", "timeout", "timed out", "network")):
        return (
            f"hf {operation} failed: network error reaching Hugging Face Hub "
            f"({err_type}: {msg}). Check your connection and retry."
        )

    return f"hf {operation} failed ({err_type}): {msg}"


def _log_hf_error(operation: str, exc: Exception) -> None:
    _progress.info(_format_hf_error(operation, exc))


def _load_local_catalogue() -> Dict[str, Any]:
    if not config.CATALOGUE_PATH.exists():
        return json.loads(json.dumps(_EMPTY_CATALOGUE))
    try:
        data = json.loads(config.CATALOGUE_PATH.read_text())
        data.setdefault("panos", {})
        data.setdefault("version", 1)
        return data
    except Exception:
        return json.loads(json.dumps(_EMPTY_CATALOGUE))


def _save_local_catalogue(catalogue: Dict[str, Any]) -> None:
    config.CATALOGUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CATALOGUE_PATH.write_text(
        json.dumps(catalogue, indent=2, sort_keys=True)
    )


def sync_catalogue_from_hf() -> Dict[str, Any]:
    """Pull ``catalogue.json`` from the HF dataset. Returns local copy on failure."""
    if not hf_enabled():
        return _load_local_catalogue()
    try:
        local_root = snapshot_download(
            repo_id=config.HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=["catalogue.json"],
            token=config.HF_TOKEN,
            local_dir=str(config.CACHE_DIR),
        )
        remote = Path(local_root) / "catalogue.json"
        if remote.exists():
            data = json.loads(remote.read_text())
            data.setdefault("panos", {})
            data.setdefault("version", 1)
            _save_local_catalogue(data)
            return data
    except Exception as e:  # noqa: BLE001
        _log_hf_error("catalogue sync", e)
    return _load_local_catalogue()


def _is_fresh(entry: Dict[str, Any]) -> bool:
    ts = entry.get("last_downloaded")
    if not ts:
        return False
    try:
        last = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc
        )
    except Exception:
        return False
    return (dt.datetime.now(dt.timezone.utc) - last).days < config.CACHE_MAX_AGE_DAYS


def _has_all_headings(entry: Dict[str, Any], headings: Iterable[int]) -> bool:
    stored = set(entry.get("headings") or [])
    return set(headings).issubset(stored)


def _params_compatible(entry: Dict[str, Any], fov: int, pitch: int) -> bool:
    return entry.get("fov") == fov and entry.get("pitch") == pitch


def plan_cache_usage(
    pano_ids: List[str],
    catalogue: Dict[str, Any],
    headings: List[int],
    fov: int,
    pitch: int,
) -> Tuple[List[str], List[str]]:
    """Split ``pano_ids`` into (hits, misses).

    A hit means the catalogue has the pano, all requested headings are stored
    under the same (fov, pitch), and ``last_downloaded`` is within
    ``CACHE_MAX_AGE_DAYS``.
    """
    panos_map = catalogue.get("panos") or {}
    hits: list[str] = []
    misses: list[str] = []
    for pid in pano_ids:
        entry = panos_map.get(pid)
        if (
            entry
            and _is_fresh(entry)
            and _params_compatible(entry, fov, pitch)
            and _has_all_headings(entry, headings)
        ):
            hits.append(pid)
        else:
            misses.append(pid)
    return hits, misses


def fetch_cached_images(
    hits: List[str], headings: List[int], target_dir: Path
) -> List[Dict[str, Any]]:
    """Download cached PNGs from HF and copy the requested headings into ``target_dir``."""
    if not hits or not hf_enabled():
        return []
    try:
        local_root = snapshot_download(
            repo_id=config.HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=[f"images/{pid}/*" for pid in hits],
            token=config.HF_TOKEN,
            local_dir=str(config.CACHE_DIR),
        )
    except Exception as e:  # noqa: BLE001
        _log_hf_error("image fetch", e)
        return []

    target_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(headings)
    metadata: list[dict] = []
    for pid in hits:
        pano_dir = Path(local_root) / "images" / pid
        if not pano_dir.exists():
            continue
        for png in sorted(pano_dir.glob("*.png")):
            try:
                heading = int(png.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if heading not in wanted:
                continue
            shutil.copy(png, target_dir / png.name)
            metadata.append(
                {
                    "panoid": pid,
                    "heading": heading,
                    "image": png.name,
                    "source": "hf_cache",
                }
            )
    return metadata


def publish_to_hf(
    new_or_refreshed_pano_ids: Iterable[str],
    pano_metadata: Dict[str, Dict[str, Any]],
    headings: List[int],
    fov: int,
    pitch: int,
    source_dir: Path,
) -> int:
    """Stage PNGs for the given panos, merge catalogue, push folder + catalogue to HF.

    Only files that exist on disk in ``source_dir`` are staged, so partial GCP
    failures don't pollute the dataset. Returns the number of panos uploaded
    (zero on any failure / no-op).
    """
    pano_ids = list(set(new_or_refreshed_pano_ids))
    if not pano_ids or not hf_enabled():
        return 0

    with tempfile.TemporaryDirectory(prefix="streetnav_upload_") as tmp:
        staging = Path(tmp)
        uploaded_per_pano: Dict[str, List[int]] = {}
        for pid in pano_ids:
            pano_staged: list[int] = []
            pano_dir = staging / pid
            pano_dir.mkdir(parents=True, exist_ok=True)
            for h in headings:
                src = source_dir / f"{pid}_{h}.png"
                if src.exists():
                    shutil.copy(src, pano_dir / src.name)
                    pano_staged.append(h)
            if pano_staged:
                uploaded_per_pano[pid] = pano_staged
            else:
                shutil.rmtree(pano_dir, ignore_errors=True)
        if not uploaded_per_pano:
            return 0

        latest = sync_catalogue_from_hf()
        panos_map = latest.setdefault("panos", {})
        now = _now_iso()
        for pid, staged_headings in uploaded_per_pano.items():
            base = pano_metadata.get(pid, {})
            existing = panos_map.get(pid, {})
            merged_headings = sorted(
                set(existing.get("headings") or []) | set(staged_headings)
            )
            panos_map[pid] = {
                "lat": base.get("lat", existing.get("lat")),
                "lng": base.get("lng", existing.get("lng")),
                "sv_date": base.get("date", existing.get("sv_date")),
                "last_downloaded": now,
                "fov": fov,
                "pitch": pitch,
                "headings": merged_headings,
            }
        latest["version"] = latest.get("version", 1)
        _save_local_catalogue(latest)

        try:
            upload_folder(
                repo_id=config.HF_DATASET_REPO,
                repo_type="dataset",
                folder_path=str(staging),
                path_in_repo="images",
                token=config.HF_TOKEN,
                commit_message=f"streetnav: +{len(uploaded_per_pano)} panos",
            )
            upload_file(
                repo_id=config.HF_DATASET_REPO,
                repo_type="dataset",
                path_or_fileobj=str(config.CATALOGUE_PATH),
                path_in_repo="catalogue.json",
                token=config.HF_TOKEN,
                commit_message=(
                    f"streetnav: catalogue +{len(uploaded_per_pano)}"
                ),
            )
            return len(uploaded_per_pano)
        except Exception as e:  # noqa: BLE001
            _log_hf_error("upload", e)
            return 0
