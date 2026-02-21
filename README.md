# Hacklytics: Actian VectorAI + RAG + Scenario Forecasting

This project builds a movie-opinion intelligence pipeline on top of **Actian VectorAI DB**.

It supports:
- ingesting text evidence into Actian Vector DB
- semantic search over that evidence
- Gemini-powered RAG responses
- rating/market prediction from structured question/answer input

---

## 1) Repository Scripts

- `scrape_data_reddit.py`
  - Main ingestion/search utility.
  - Supports:
    - `ingest` (Reddit + Quora scrape path; may be blocked depending on environment)
    - `ingest-file --path ...` (JSONL file ingest; recommended reliable path)
    - `search --query ...` (semantic retrieval from Actian)

- `ingest_market_signals.py`
  - Ingests:
    - OMDb market/ratings data (IMDb, Rotten Tomatoes, Metacritic, BoxOffice fields)
    - YouTube comments (via YouTube Data API)
  - Writes all data to Actian through shared ingestion pipeline.

- `ingest_mcu_market_pack.py`
  - Bulk ingest for MCU/comparable titles.
  - Uses OMDb + optional YouTube comments.
  - Designed for larger baseline coverage.

- `app.py`
  - Basic Gemini script for generating controversial "what-if" scenarios from title only.

- `rag_app.py`
  - Interactive Gemini app that retrieves relevant context from Actian before scenario generation.

- `gemini_predict_from_vectordb.py`
  - Gemini-based prediction engine.
  - Retrieves related evidence from Actian and returns strict prediction JSON including:
    - IMDb change
    - Rotten Tomatoes change
    - fan sentiment metrics
    - box office change (% and predicted USD)

---

## 2) Architecture Overview

1. **Collect data**
   - Scrape/API load raw text + market metadata.
2. **Normalize to documents**
   - Unified schema (`RawDocument`) with text + metadata.
3. **Chunk**
   - Text split into retrieval chunks.
4. **Embed**
   - Local hasher embedder (no embedding API required).
5. **Store in Actian**
   - `CortexClient` collection + payload per chunk.
6. **Retrieve + Predict**
   - Semantic search for evidence.
   - Rule-based or Gemini-based forecasting.

---

## 3) Prerequisites

## Python
- Python 3.10+ (3.12 recommended)

## Actian VectorAI DB
- Run Actian server (commonly in Codespaces/Linux x86_64 for beta stability)
- Endpoint typically `localhost:50051`

## Keys
- `GEMINI_API_KEY` (for Gemini scripts)
- `OMDB_API_KEY` (for OMDb market data)
- `YOUTUBE_API_KEY` (for YouTube comments)

---

## 4) Setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requests beautifulsoup4 praw python-dotenv google-generativeai
pip install ./actian-vectorAI-db-beta/actiancortex-0.1.0b1-py3-none-any.whl
```

If wheel path differs, point to your actual `actiancortex-0.1.0b1-py3-none-any.whl`.

---

## 5) Environment Variables

Set these before running scripts:

```bash
export CORTEX_ADDRESS=localhost:50051
export ACTIAN_COLLECTION_NAME=endgame_opinions
export ACTIAN_RECREATE_COLLECTION=false
```

### For Gemini scripts
```bash
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
export GOOGLE_API_KEY="$GEMINI_API_KEY"
```

### For market ingestion
```bash
export OMDB_API_KEY="YOUR_OMDB_KEY"
export YOUTUBE_API_KEY="YOUR_YOUTUBE_KEY"
```

### Optional tuning
```bash
export LOCAL_EMBED_DIM=1024
export CHUNK_SIZE_WORDS=120
export CHUNK_OVERLAP_WORDS=20
```

Notes:
- Keep `ACTIAN_RECREATE_COLLECTION=false` for normal runs.
- Set `ACTIAN_RECREATE_COLLECTION=true` only when intentionally resetting collection.

---

## 6) Start Actian DB (example)

From bundled beta repo:

```bash
cd actian-vectorAI-db-beta
docker compose up -d
docker compose ps
```

---

## 7) Ingestion Workflows

## A) Reliable dataset ingest (recommended)

If you already have JSONL dataset:

```bash
python3 scrape_data_reddit.py ingest-file --path data/movie_opinions.jsonl
```

JSONL format:
- required: `text`
- optional: `id`, `source`, `url`, `title`, `author`, `created_at`, `score`, `metadata`

## B) Market + audience signals ingest

OMDb + YouTube for one movie:

```bash
python3 ingest_market_signals.py --movie-title "Avengers: Endgame"
```

OMDb only:

```bash
python3 ingest_market_signals.py --movie-title "Avengers: Endgame" --skip-youtube
```

YouTube only:

```bash
python3 ingest_market_signals.py --movie-title "Avengers: Endgame" --skip-omdb
```

## C) Bulk MCU pack ingest

Safe starter:

```bash
python3 ingest_mcu_market_pack.py \
  --max-titles 4 \
  --youtube-max-videos 1 \
  --youtube-comments-per-video 20
```

Scale up:

```bash
python3 ingest_mcu_market_pack.py \
  --max-titles 8 \
  --youtube-max-videos 1 \
  --youtube-comments-per-video 30
```

OMDb-only bulk:

```bash
python3 ingest_mcu_market_pack.py --max-titles 12 --skip-youtube
```

---

## 8) Validate Ingest

```bash
python3 scrape_data_reddit.py search --query "Avengers Endgame IMDb Rotten Tomatoes box office" --top-k 10
python3 scrape_data_reddit.py search --query "Infinity War audience reaction ending" --top-k 10
```

Look for `source` values like:
- `omdb_market`
- `youtube_comment`
- `dataset` (if file-ingested)

---

## 9) Run Apps / Predictors

## A) Basic scenario generator (Gemini only)
```bash
python3 app.py
```

## B) RAG scenario generator (Gemini + Actian retrieval)
```bash
python3 rag_app.py
```

## C) Gemini predictor with vector evidence
```bash
python3 gemini_predict_from_vectordb.py --input input.json --output prediction.json
```

With more retrieval per question:
```bash
python3 gemini_predict_from_vectordb.py --input input.json --output prediction.json --top-k-per-query 10
```

---

## 10) Input / Output Contracts

## Input format (`input.json`)
```json
{
  "questions": [
    {"id": "q1", "text": "Was the pacing of the climax well-executed?"},
    {"id": "q2", "text": "Did multiple viewings reveal new details and depth in the ending?"}
  ],
  "answers": {
    "q1": "yes",
    "q2": "no"
  }
}
```

## Typical output (`gemini_predict_from_vectordb.py`)
```json
{
  "id": "avengers-endgame",
  "title": "Avengers: Endgame",
  "year": 2019,
  "predictions": {
    "imdb": {"current": 84, "predicted": 86, "delta": 2},
    "rt": {"current": 90, "predicted": 92, "delta": 2},
    "fanRating": {"positivePercent": 61.2, "negativePercent": 38.8, "netSentiment": 22.4},
    "boxOffice": {"currentUsd": 858373000, "predictedUsd": 901291650, "deltaPercent": 5.0}
  },
  "assumptions": ["...", "..."]
}
```

---

## 11) Common Issues

## `No documents scraped`
- Source blocked (common in cloud dev envs).
- Use `ingest-file` or API-backed sources (`ingest_market_signals.py`).

## `CortexError: Failed to batch upsert`
- Reduce ingestion volume per run.
- Use smaller settings:
  - fewer titles/videos/comments
  - smaller `CHUNK_SIZE_WORDS`

## `No API_KEY or ADC found`
- Export both:
  - `GEMINI_API_KEY`
  - `GOOGLE_API_KEY=$GEMINI_API_KEY`

## Script/file not found
- Ensure you are in repo root:
  - `/workspaces/Hacklytics` (Codespaces)

---

## 12) Security Notes

- Never commit `.env`.
- Add to `.gitignore`:
  - `.env`
- Rotate keys if exposed in logs/chat.

---

## 13) Suggested Demo Flow

1. Start Actian DB.
2. Ingest OMDb + YouTube for Endgame:
   - `ingest_market_signals.py`
3. Run semantic search to show grounded evidence.
4. Run `gemini_predict_from_vectordb.py` on Q/A input.
5. Show JSON prediction output and explain evidence traceability.
