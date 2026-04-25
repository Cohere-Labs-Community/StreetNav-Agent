# StreetNav-Agent

A Strands-based agent that turns a free-form location query (e.g. *"vegetarian restaurants near NTR stadium guntur"*) into a curated set of nearby places, evidenced by Google Street View imagery.

The agent runs a fixed pipeline:

1. **Geocode** the query (Google Maps).
2. **Find Street View panoramas** on roads within a radius around that point.
3. **Download** a few headings per panorama as images.
4. **Score image relevance** to the user's query with a vision LLM (Cohere Aya Vision or OpenRouter Gemini).
5. **Enrich** the YES-marked images: reverse-geocode and extract any visible name on the building/object.
6. **Format** the final answer.

A `scripts/` directory contains the notebook's debug/baseline cells (Places-API ground truth, neighbor-distance plot, pano viewer, coverage check) — they are NOT part of the agent's runtime path.

## Setup

Requires Python **≥ 3.10** (Strands SDK requirement). macOS ships 3.9, so install a newer one first.

### Quick start — paste into a fresh terminal

Run this from the project root. It installs `uv` if missing, drops any existing `.venv`, creates a clean Python 3.12 environment, and installs all dependencies.

```bash
cd path/to/StreetNav-Agent
deactivate 2>/dev/null || true
rm -rf .venv
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Alternative: official python.org installer

Download and install Python 3.12 from <https://www.python.org/downloads/macos/>, then:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install --upgrade pip
python3.12 -m pip install -r requirements.txt
```

Create `.env.local` (already gitignored):

```bash
cat > .env.local <<'EOF'
GCP_GMAP_KEY=your_google_maps_api_key
COHERE_API_KEY=your_cohere_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
EOF
chmod 600 .env.local
```

## Run

Strands agent (LLM-orchestrated):

```bash
python3 main.py "vegetarian restaurants near NTR stadium guntur"
```

Direct pipeline (deterministic, no LLM orchestrator — same six steps in fixed order):

```bash
python3 main.py --no-agent "vegetarian restaurants near NTR stadium guntur"
```

Both modes share the same per-tool progress output (one line per stage) and write the same artifacts under `output/` (gitignored):

```
output/
  geocode.json
  panos.json
  street_views/
  relevant_images/
  findings.json
```

## Provider switching

Both providers default to **OpenRouter** (Gemini). Cohere is opt-in.

Set in `src/config.py` (or via env):

- `IMAGE_PROVIDER`: `OPENROUTER` (default) or `COHERE` — used for vision scoring and on-image name extraction.
- `AGENT_PROVIDER`: `OPENROUTER` (default) or `COHERE` — used by the Strands orchestrator agent.

Both providers go through Strands' `OpenAIModel` (Cohere via its OpenAI-compatibility endpoint).

To use Cohere instead, just prefix the run command:

```bash
AGENT_PROVIDER=COHERE IMAGE_PROVIDER=COHERE python main.py "..."
```

## Scripts (not part of the agent)

```bash
python3 scripts/baseline_nearby.py "vegetarian restaurants near NTR stadium guntur"
python3 scripts/debug_neighbor_dist.py
python3 scripts/debug_pano_display.py 9xFAU9D-8ES3C2ZOn5RUeg
python3 scripts/compare_results.py
```
