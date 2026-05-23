# StreetNav-Agent

## 1 Overview

Turns a free-text place query into a short list of evidence-backed findings using Google Maps (geocode, roads, Street View) plus a vision LLM for relevance and on-image names.

### 1.1 Pipeline

1. Geocode the query.
2. Find Street View panoramas on roads in a radius.
3. Download a few headings per panorama as PNGs.
4. Score each image against the query with a **vision LLM** (provider is configurable; see 2.2 and 5).
5. For high-scoring images: reverse-geocode coordinates and ask the vision LLM for any visible metadata.
6. Format the final JSON answer.

### 1.2 Repository layout

- `src/` — agent, tools, config.
- `main.py` — CLI.
- `output/` — run artifacts (JSON + images); a sample run may be committed for demos.
- `scripts/` — optional debugging utilities (not used by the CLI pipeline).

## 2 Requirements

### 2.1 Python

Python **3.10 or newer** (required by Strands and dependencies). Use a venv from the quick start below.

### 2.2 API keys

- **Google Maps / GCP:** `GCP_GMAP_KEY` is always required (geocoding, roads, Street View).
- **Vision / text LLM:** set at least **one** of `OPENROUTER_API_KEY` or `COHERE_API_KEY` (both is fine; only one working key is required). On import, each backend is probed at most once; your `IMAGE_PROVIDER` / `AGENT_PROVIDER` preference is tried first, then the other if that side is unavailable or invalid. The chosen provider stays fixed for all LLM calls in that process.

## 3 Quick start

Create `.env.local` once (see `.env.example` for variable names). Do not overwrite an existing `.env.local`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py "your query here"
```

## 4 Run modes

### 4.1 Default (Strands agent)

The agent calls the same six tools in order; progress prints once per stage.

```bash
python main.py "your query here"
```

### 4.2 Direct pipeline (no agent)

Same steps and same `output/` files, implemented as plain function calls — no Strands orchestration.

```bash
python main.py --no-agent "your query here"
```

## 5 Environment overrides

Set variables on the same line **before** `python` (Unix). Examples:

```bash
IMAGE_PROVIDER=COHERE python main.py "…"
AGENT_PROVIDER=OPENROUTER python main.py "…"
OPENROUTER_MODEL_ID=your/vendor-model-id python main.py "…"
COHERE_MODEL_ID=your-cohere-vision-model python main.py "…"
GCP_WORKERS=16 LLM_WORKERS=8 python main.py "…"
MAX_TOKENS=500 TEMP=0.1 python main.py "…"
```

`IMAGE_PROVIDER` / `AGENT_PROVIDER` may be `OPENROUTER` or `COHERE`; each is resolved with a single cached probe per backend (see 2.2).

## 6 Shared image cache (HuggingFace)

To avoid re-burning the Google Street View Static quota for places someone else has already crawled, every run consults a shared private dataset before falling back to the GCP API.

### 6.1 Repo layout

`HF_DATASET_REPO` (default `c4ai-ml-agents/StreetView-Agents`, private, type `dataset`):

- `catalogue.json` — one entry per `pano_id` with `lat`, `lng`, `sv_date`, `last_downloaded` (UTC), `fov`, `pitch`, `headings`.
- `images/<pano_id>/<pano_id>_<heading>.png` — cumulative across runs; one subfolder per pano.

### 6.2 What happens inside `save_pano_images`

1. Pull the latest `catalogue.json`.
2. For each pano discovered in step 2 of the pipeline, classify as **hit** (catalogue has it, fresh, same `fov`/`pitch`, has every requested heading) or **miss**.
3. **Hits** — `snapshot_download` with `allow_patterns=["images/<pid>/*", ...]` (single HF call, no per-file rate limit).
4. **Misses** — existing 24-wide GCP downloader.
5. Stage newly-downloaded panos into a temp folder and `upload_folder(path_in_repo="images")`, then `upload_file("catalogue.json")` with the merged catalogue. The temp folder is removed.

Freshness threshold: `CACHE_MAX_AGE_DAYS` (default **730d / 2 years**).

### 6.3 Concurrency & race safety

- `upload_folder` is **additive** — it never deletes existing files. PNGs from concurrent runs cannot collide because each pano sits in its own subfolder.
- The catalogue uses pull-merge-push and is **last-writer-wins**. Worst case after a race: a recent entry is missed and that pano is re-downloaded from GCP on the next query (which then re-adds it). No image data is lost.

### 6.4 Disabling the cache

Leave `HUGGINGFACE_API_KEY` blank in `.env.local`. The pipeline will skip every Hub call and behave exactly like the pre-cache version.

## 7 Output files

Under `output/`:

- `geocode.json`, `panos.json`, `findings.json`
- `street_views/` — PNGs + `metadata.json`
- `relevant_images/` — copies of high-scoring PNGs + `results.json`

## 8 Optional scripts

All live under `scripts/`. Use the same venv as in section 3. See each file’s module docstring for full options.

### 8.1 `baseline_nearby.py`

Places API baseline vs the pipeline (writes `output/baseline_nearby.json` when used with the geocode path).

```bash
python scripts/baseline_nearby.py "vegetarian restaurants near NTR stadium guntur"
```

### 8.2 `compare_results.py`

Compares `output/baseline_nearby.json` (if present), `output/geocode.json`, and `output/findings.json`. No CLI arguments.

```bash
python scripts/compare_results.py
```

### 8.3 `debug_neighbor_dist.py`

Plots nearest-neighbor distances for panos in `output/panos.json`. No CLI arguments.

```bash
python scripts/debug_neighbor_dist.py
```

### 8.4 `debug_pano_display.py`

Download headings for one pano id, or list the five closest panos to a coordinate.

```bash
python scripts/debug_pano_display.py 9xFAU9D-8ES3C2ZOn5RUeg
```

```bash
python scripts/debug_pano_display.py --closest 16.3159081 80.4220971
```
