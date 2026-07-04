# StreetNav-Agent — Annotators

Run a query to fetch street view images for a location, annotate which images are relevant, then name the relevant ones and push everything to HuggingFace.

- `GCP_GMAP_KEY` — Google Maps API key (needed for step 1)
- `HUGGINGFACE_API_KEY` — HuggingFace token (needed for step 3)
- `HF_ANNOTATION_REPO` — HF dataset repo to push to (set in `.env.local`, default `c4ai-ml-agents/annotations_example`)

| Step | Action | What to annotate |
|------|--------|-----------------|
| 1 | run `pre_annotation.py` | — |
| 2 | edit `_annotated_metadata.json` | `relevancy` NO → YES; `ambiguous` NO → YES if unclear |
| 3 | run `post_annotation.py`, edit `_annotated_response.json`, run again with `--push` | `name` per YES image |

## 1 - Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2 - Keys, query, and sample name

```text
# create .env.local (see .env.example for variable names)
```

yourname_XXX — set once per query, unique, do not reuse across queries

```bash
export QUERY="cafes near NTR stadium guntur"
export SAMPLE_NAME="ram_001"
export HEADING_SEED="42"
```

## 3 — pre_annotation.py

```bash
python3 src/benchmarking/pre_annotation.py
```

## 3.1 - Produced files

Verify they exist.

```text
src/benchmarking/samples/$SAMPLE_NAME/
  images/<pano_id>_<heading>.png
  $SAMPLE_NAME_annotated_metadata.json
```

## 3.2 - Skeleton metadata

View each image in `src/benchmarking/samples/$SAMPLE_NAME/images/` (~1–3 sec/image, ~5–10 min/query). Update relevancy in `$SAMPLE_NAME_annotated_metadata.json` for images relevant to the query. Only edit `relevancy` and `ambiguous`; leave everything else unchanged.

```json
{
  "sample_name": "ram_001",                                   // annotator_key : yourname_samplenumber
  "query_used": "cafes near NTR stadium guntur",              // input query
  "country": "India",                                         // country (query)
  "state": "AP",                                              // state/province (query)
  "city": "Guntur",                                           // city (query)
  "query_center": {"lat": 0.0, "lng": 0.0, "place_id": "..."},// query anchor
  "images_per_pano": 3,                                       // headings/pano
  "images": [                                                 // one per image
    {
      "image": "<pano_id>_0.png",                             // pano_heading.png
      "lat": 0.0,                                             // image lat
      "lng": 0.0,                                             // image lng
      "place_id": "...",                                      // nearest place
      "relevancy": "NO",                                      // EDIT: NO -> YES if relevant
      "ambiguous": "NO"                                       // EDIT: NO -> YES if match is unclear
    }
  ]
}
```

## 4 — generate response skeleton

```bash
python3 src/benchmarking/post_annotation.py
```

## 5 - Response skeleton (YES images only)

For YES images above, `image` / `lat` / `lng` / `place_id` / `ambiguous` are copied automatically — only `name` needs filling in.

Edit `src/benchmarking/samples/$SAMPLE_NAME/$SAMPLE_NAME_annotated_response.json`:

```json
[
  {
    "image": "<pano_id>_0.png",                               // auto from metadata
    "lat": 0.0,                                               // auto from metadata
    "lng": 0.0,                                               // auto from metadata
    "place_id": "...",                                        // auto from metadata
    "ambiguous": "NO",                                        // EDIT: change to YES if ambiguous case
    "name": "XXX"                                             // EDIT: visible name in image
  }
]
```

## 6 - push to hf

```bash
python3 src/benchmarking/post_annotation.py --push
```

## 6.1 Pushed layout

```text
HF_ANNOTATION_REPO (dataset)
  $SAMPLE_NAME/images/<pano_id>_<heading>.png
  $SAMPLE_NAME/$SAMPLE_NAME_annotated_metadata.json
  $SAMPLE_NAME/$SAMPLE_NAME_annotated_response.json
```
