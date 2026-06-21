# StreetNav-Agent — Annotators

Run a query to fetch street view images for a location, annotate which images are relevant and what languages appear, then name the relevant ones and push everything to HuggingFace. Three steps total — one script run, one JSON edit, one script run with `--push`.

- `GCP_GMAP_KEY` — Google Maps API key (needed for step 1)
- `HUGGINGFACE_API_KEY` — HuggingFace token (needed for step 3)
- `HF_ANNOTATION_REPO` — HF dataset repo to push to (set in `.env.local`, default `c4ai-ml-agents/annotations_example`)

| Step | Action | What to annotate |
|------|--------|-----------------|
| 1 | run `pre_annotation.py` | — |
| 2 | edit `_annotated_metadata.json` | `relevancy` YES/NO per image |
| 3 | run `post_annotation.py`, edit `_annotated_response.json`, run again with `--push` | `name` per YES image |

## 1 Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2 Keys

```text
# create .env.local (see .env.example for variable names)
```

## 3 Step 1 — pre_annotation.py

```bash
python3 src/benchmarking/pre_annotation.py "cafes near NTR stadium guntur" ram_001  # query, sample_name (yourname_samplenumber)
```

## 4 Produced files

```text
src/benchmarking/samples/<sample_name>/
  images/<pano_id>_<heading>.png
  <sample_name>_annotated_metadata.json
```

## 5 Skeleton metadata

```json
{
  "sample_name": "ram_001",                                   // annotator_key : yourname_samplenumber
  "query_used": "cafes near NTR stadium guntur",              // input query
  "country": "India",                                         // country (query)
  "state": "AP",                                              // state/province (query)
  "city": "Guntur",                                           // city (query)
  "query_center": {"lat": 0.0, "lng": 0.0, "place_id": "..."},// query anchor
  "images_per_pano": 4,                                       // headings/pano
  "images": [                                                 // one per image
    {
      "image": "<pano_id>_0.png",                             // pano_heading.png
      "lat": 0.0,                                             // image lat
      "lng": 0.0,                                             // image lng
      "place_id": "...",                                      // nearest place
      "relevancy": "XXX"                                      // edit: YES/NO
    }
  ]
}
```

## 6 Manual edits per image

```text
relevancy : "XXX" -> "YES" or "NO"
```

## 7 Step 2 phase 1 — generate response skeleton

```bash
python3 src/benchmarking/post_annotation.py ram_001   # sample_name
```

## 8 Response skeleton (YES images only)

```json
[
  {
    "image": "<pano_id>_0.png",                               // file name
    "lat": 0.0,                                               // image lat
    "lng": 0.0,                                               // image lng
    "place_id": "...",                                        // nearest place
    "name": "XXX"                                             // edit: visible name in image
  }
]
```

## 9 Manual edit per YES image

```text
name : "XXX" -> "Cafe Coffee Day"
```

## 10 Step 2 phase 2 — push

```bash
python3 src/benchmarking/post_annotation.py ram_001 --push   # validates name, creates repo if needed, uploads
```

## 11 Pushed layout

```text
HF_ANNOTATION_REPO (dataset)
  <sample_name>/images/<pano_id>_<heading>.png
  <sample_name>/<sample_name>_annotated_metadata.json
  <sample_name>/<sample_name>_annotated_response.json
```