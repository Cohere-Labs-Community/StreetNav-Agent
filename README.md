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

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env.local
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

## 6 Output files

Under `output/`:

- `geocode.json`, `panos.json`, `findings.json`
- `street_views/` — PNGs + `metadata.json`
- `relevant_images/` — copies of high-scoring PNGs + `results.json`

## 7 Optional scripts

All live under `scripts/`. Use the same venv as in section 3. See each file’s module docstring for full options.

### 7.1 `baseline_nearby.py`

Places API baseline vs the pipeline (writes `output/baseline_nearby.json` when used with the geocode path).

```bash
python scripts/baseline_nearby.py "vegetarian restaurants near NTR stadium guntur"
```

### 7.2 `compare_results.py`

Compares `output/baseline_nearby.json` (if present), `output/geocode.json`, and `output/findings.json`. No CLI arguments.

```bash
python scripts/compare_results.py
```

### 7.3 `debug_neighbor_dist.py`

Plots nearest-neighbor distances for panos in `output/panos.json`. No CLI arguments.

```bash
python scripts/debug_neighbor_dist.py
```

### 7.4 `debug_pano_display.py`

Download headings for one pano id, or list the five closest panos to a coordinate.

```bash
python scripts/debug_pano_display.py 9xFAU9D-8ES3C2ZOn5RUeg
```

```bash
python scripts/debug_pano_display.py --closest 16.3159081 80.4220971
```
