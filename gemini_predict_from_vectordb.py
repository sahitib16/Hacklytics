#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
import google.generativeai as genai

from scrape_data_reddit import ActianCortexStore, LocalHasherEmbedder, semantic_search, env_int


POS_WORDS = {
    "good", "great", "amazing", "excellent", "love", "loved", "fantastic",
    "emotional", "impactful", "satisfying", "strong", "favorite", "best",
    "perfect", "memorable", "awesome",
}
NEG_WORDS = {
    "bad", "boring", "weak", "terrible", "awful", "hate", "hated",
    "disappointing", "rushed", "messy", "predictable", "worse", "worst",
    "unnecessary", "flat",
}


def clean_text(v: str) -> str:
    return re.sub(r"\s+", " ", (v or "")).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", (text or "").lower())


def parse_money_to_int(v: str) -> int | None:
    if not v:
        return None
    digits = re.sub(r"[^\d]", "", v)
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def fan_sentiment_from_hits(hits: list[dict[str, Any]]) -> dict[str, float]:
    pos = 0
    neg = 0
    for h in hits:
        for t in tokens(str(h.get("text", ""))):
            if t in POS_WORDS:
                pos += 1
            elif t in NEG_WORDS:
                neg += 1
    total = max(1, pos + neg)
    pos_pct = (pos / total) * 100.0
    neg_pct = (neg / total) * 100.0
    return {
        "positivePercent": round(pos_pct, 2),
        "negativePercent": round(neg_pct, 2),
        "netSentiment": round(pos_pct - neg_pct, 2),
    }


def extract_market_baseline(hits: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {
        "imdb": 84.0,
        "rt": 90.0,
        "boxOfficeUsd": 858_373_000,
    }
    for h in hits:
        if h.get("source") != "omdb_market":
            continue
        meta_raw = h.get("metadata_json")
        if not meta_raw:
            continue
        try:
            meta = json.loads(meta_raw)
        except Exception:
            continue
        imdb = meta.get("imdb_rating")
        rt = meta.get("rt_rating")
        bo = meta.get("box_office")
        try:
            if imdb and str(imdb) != "N/A":
                baseline["imdb"] = float(str(imdb))
        except Exception:
            pass
        if rt and str(rt) != "N/A":
            rt_num = re.sub(r"[^\d.]", "", str(rt))
            if rt_num:
                baseline["rt"] = float(rt_num)
        bo_val = parse_money_to_int(str(bo))
        if bo_val:
            baseline["boxOfficeUsd"] = bo_val
        break
    return baseline


def retrieve_related_hits(payload: dict[str, Any], top_k_per_query: int) -> list[dict[str, Any]]:
    local_embed_dim = env_int("LOCAL_EMBED_DIM", 1024)
    cortex_address = os.getenv("CORTEX_ADDRESS", "localhost:50051")
    collection_name = os.getenv("ACTIAN_COLLECTION_NAME", "endgame_opinions")

    store = ActianCortexStore(
        address=cortex_address,
        collection_name=collection_name,
        dimension=local_embed_dim,
        recreate=False,
    )
    embedder = LocalHasherEmbedder(dim=local_embed_dim)

    queries = [
        "Avengers Endgame IMDb Rotten Tomatoes Metacritic box office",
        "Avengers Endgame audience fan reactions comments ending",
    ]
    for q in payload.get("questions", []):
        qid = str(q.get("id", ""))
        qtext = str(q.get("text", ""))
        ans = str(payload.get("answers", {}).get(qid, ""))
        queries.append(f"{qtext} answer={ans} Avengers Endgame")

    raw_hits: list[dict[str, Any]] = []
    for q in queries:
        raw_hits.extend(
            semantic_search(
                store=store,
                embedder=embedder,
                query=q,
                top_k=top_k_per_query,
                source_filter=None,
            )
        )

    # De-dup by chunk_id.
    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for h in sorted(raw_hits, key=lambda x: x.get("score", 0.0), reverse=True):
        cid = str(h.get("chunk_id", ""))
        if cid in seen:
            continue
        seen.add(cid)
        dedup.append(h)
    return dedup


def build_context(hits: list[dict[str, Any]], max_items: int = 40) -> str:
    blocks = []
    for i, h in enumerate(hits[:max_items], start=1):
        blocks.append(
            f"[{i}] source={h.get('source')} title={h.get('title')} url={h.get('url')} score={h.get('score')}\n"
            f"{clean_text(str(h.get('text', '')))}"
        )
    return "\n\n".join(blocks)


def safe_json_load(text: str) -> Any:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return json.loads(t)


def predict_with_gemini(payload: dict[str, Any], top_k_per_query: int = 8) -> Any:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (and/or GOOGLE_API_KEY).")

    # Set both to avoid env-name mismatch across SDK versions.
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    hits = retrieve_related_hits(payload, top_k_per_query=top_k_per_query)
    fan = fan_sentiment_from_hits(hits)
    base = extract_market_baseline(hits)
    context = build_context(hits)

    prompt = f"""
You are a movie market forecaster.
Use only the provided vector-db evidence and user answers.
Return ONLY valid JSON (no markdown).

Input QA payload:
{json.dumps(payload, ensure_ascii=True)}

Derived baseline:
{json.dumps(base, ensure_ascii=True)}

Derived fan sentiment from retrieved comments:
{json.dumps(fan, ensure_ascii=True)}

Retrieved evidence:
{context}

Return exactly this JSON shape:
{{
  "id": "avengers-endgame",
  "title": "Avengers: Endgame",
  "year": 2019,
  "predictions": {{
    "imdb": {{
      "current": <number>,
      "predicted": <number>,
      "delta": <number>
    }},
    "rt": {{
      "current": <number>,
      "predicted": <number>,
      "delta": <number>
    }},
    "fanRating": {{
      "positivePercent": <number>,
      "negativePercent": <number>,
      "netSentiment": <number>
    }},
    "boxOffice": {{
      "currentUsd": <integer>,
      "predictedUsd": <integer>,
      "deltaPercent": <number>
    }}
  }},
  "assumptions": [<string>, <string>]
}}

Rules:
- Keep ratings in range 0..100.
- boxOffice deltaPercent should usually be within -30..30 unless evidence is extreme.
- fanRating must be consistent with evidence sentiment.
- imdb/rt delta should align with user yes/no answers and retrieved sentiment.
"""

    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    return safe_json_load(response.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini prediction JSON using Actian vector evidence.")
    parser.add_argument("--input", required=True, help="Input JSON path (questions + answers).")
    parser.add_argument("--output", default=None, help="Optional output JSON file.")
    parser.add_argument("--top-k-per-query", type=int, default=8)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result = predict_with_gemini(payload, top_k_per_query=args.top_k_per_query)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
