"""Annotator step 2: build response skeleton from YES images; --push uploads to HF.

Requires SAMPLE_NAME env var; HF_ANNOTATION_REPO and HUGGINGFACE_API_KEY in .env.local.
See docs/ANNOTATORS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from huggingface_hub import HfApi, upload_folder
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError as e:  # pragma: no cover
    raise SystemExit("huggingface_hub not installed — pip install -r requirements.txt") from e


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SAMPLES_DIR = SCRIPT_DIR / "samples"

load_dotenv(PROJECT_ROOT / ".env.local")


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip().strip('"').strip("'")


HF_TOKEN = _env("HUGGINGFACE_API_KEY") or _env("HF_TOKEN")
HF_ANNOTATION_REPO = _env("HF_ANNOTATION_REPO")


def _require_sample_name() -> str:
    sample_name = os.environ.get("SAMPLE_NAME", "").strip()
    if not sample_name:
        raise SystemExit('Missing SAMPLE_NAME — run: export SAMPLE_NAME="yourname_001"')
    return sample_name


def _ensure_repo(api: HfApi, repo_id: str) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN)
    except RepositoryNotFoundError:
        print(f"  repo {repo_id!r} not found — creating", flush=True)
        api.create_repo(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN, private=True)


def build_response_skeleton(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for img in metadata.get("images", []):
        if str(img.get("relevancy", "")).strip().upper() != "YES":
            continue
        out.append({
            "image": img.get("image"),
            "lat": img.get("lat"),
            "lng": img.get("lng"),
            "place_id": img.get("place_id"),
            "ambiguous": str(img.get("ambiguous", "NO")).strip().upper() or "NO",
            "name": "XXX",
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotator step 2: generate response skeleton or push to HF.")
    parser.add_argument("--push", action="store_true", help="validate and push to HF (phase 2)")
    args = parser.parse_args()

    if not HF_TOKEN or HF_TOKEN.startswith("your_"):
        raise SystemExit("Missing HUGGINGFACE_API_KEY / HF_TOKEN in .env.local")

    sample_name = _require_sample_name()
    sample_dir = SAMPLES_DIR / sample_name
    meta_path = sample_dir / f"{sample_name}_annotated_metadata.json"
    resp_path = sample_dir / f"{sample_name}_annotated_response.json"

    if not meta_path.exists():
        raise SystemExit(f"Not found: {meta_path} — run pre_annotation.py first.")

    metadata = json.loads(meta_path.read_text())

    pending_meta = [i for i in metadata.get("images", [])
                    if str(i.get("relevancy", "")).strip().upper() not in {"YES", "NO"}]
    if pending_meta:
        print(f"\u2717 {len(pending_meta)} images still have invalid relevancy — set each to YES or NO.",
              file=sys.stderr)
        return 2

    pending_ambiguous = [i for i in metadata.get("images", [])
                         if str(i.get("ambiguous", "NO")).strip().upper() not in {"YES", "NO"}]
    if pending_ambiguous:
        print(f"\u2717 {len(pending_ambiguous)} images still have invalid ambiguous — set each to YES or NO.",
              file=sys.stderr)
        return 2

    if not args.push:
        skeleton = build_response_skeleton(metadata)
        if resp_path.exists():
            existing = json.loads(resp_path.read_text())
            existing_map = {e["image"]: e for e in existing}
            for entry in skeleton:
                prev = existing_map.get(entry["image"])
                if not prev:
                    continue
                if prev.get("name", "XXX") != "XXX":
                    entry["name"] = prev["name"]
                if str(prev.get("ambiguous", "")).strip().upper() in {"YES", "NO"}:
                    entry["ambiguous"] = prev["ambiguous"]
        resp_path.write_text(json.dumps(skeleton, indent=2))
        print(f"\u2713 {len(skeleton)} YES images \u2192 {resp_path}", flush=True)
        print(f"  fill each \"name\" field (set \"ambiguous\" YES if unsure), then run:", flush=True)
        print(f"  python3 src/benchmarking/post_annotation.py --push", flush=True)
        return 0

    if not resp_path.exists():
        raise SystemExit(f"Not found: {resp_path} — run without --push first.")

    response = json.loads(resp_path.read_text())
    pending_names = [r["image"] for r in response if str(r.get("name", "XXX")).strip().upper() == "XXX"]
    if pending_names:
        print(f"\u2717 {len(pending_names)} entries still have name=XXX — fill them first.",
              file=sys.stderr)
        return 2

    if not HF_ANNOTATION_REPO or HF_ANNOTATION_REPO.startswith("your_"):
        raise SystemExit("Missing HF_ANNOTATION_REPO in .env.local")

    api = HfApi()
    print(f"\u25b6 ensuring repo {HF_ANNOTATION_REPO!r} exists", flush=True)
    _ensure_repo(api, HF_ANNOTATION_REPO)

    print(f"\u25b6 uploading {sample_dir.name}/ to {HF_ANNOTATION_REPO}", flush=True)
    upload_folder(
        repo_id=HF_ANNOTATION_REPO,
        repo_type="dataset",
        folder_path=str(sample_dir),
        path_in_repo=sample_name,
        token=HF_TOKEN,
        commit_message=f"annotation: {sample_name} ({len(response)} YES images)",
    )
    print(f"\u2713 pushed to https://huggingface.co/datasets/{HF_ANNOTATION_REPO}/tree/main/{sample_name}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
