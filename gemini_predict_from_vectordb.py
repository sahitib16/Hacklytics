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
                imdb_val = float(str(imdb))
                # OMDb IMDb is often on a 0..10 scale; convert to percentage.
                baseline["imdb"] = imdb_val * 10.0 if imdb_val <= 10.0 else imdb_val
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


def get_gemini_model():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (and/or GOOGLE_API_KEY).")
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if isinstance(v, str):
            v = v.replace("%", "").replace(",", "").strip()
        return float(v)
    except Exception:
        return default


def objective_score(pred: dict[str, Any]) -> float:
    p = pred.get("predictions", {})
    imdb_delta = to_float(p.get("imdb", {}).get("delta"), 0.0)
    rt_delta = to_float(p.get("rt", {}).get("delta"), 0.0)
    fan_net = to_float(p.get("fanRating", {}).get("netSentiment"), 0.0)
    box_delta_pct = to_float(p.get("boxOffice", {}).get("deltaPercent"), 0.0)
    # Weighted objective to rank "best" overall market/reception outcome.
    return imdb_delta + rt_delta + (0.2 * fan_net) + (0.1 * box_delta_pct)


def llm_question_impacts(payload: dict[str, Any], prepared: dict[str, Any], model: Any) -> list[dict[str, Any]]:
    prompt = f"""
Return ONLY valid JSON (no markdown).

Given this QA payload:
{json.dumps(payload, ensure_ascii=True)}

Retrieved evidence:
{prepared.get("context", "")}

Return exactly this JSON array shape:
[
  {{
    "id": "<question id>",
    "text": "<question text>",
    "answer": "yes|no|other",
    "impact": {{
      "imdbDelta": <number>,
      "rtDelta": <number>,
      "fanDelta": <number>,
      "boxOfficeDeltaPct": <number>
    }}
  }}
]

Rules:
- Include one object per question in input order.
- Deltas should be signed (positive or negative) based on the provided answer.
- Keep values realistic: imdb/rt in roughly -10..10, fanDelta in -25..25, boxOfficeDeltaPct in -30..30.
- Base impacts on evidence and answers, not random values.
"""
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    raw = safe_json_load(response.text)
    if not isinstance(raw, list):
        raise ValueError("LLM question impacts response was not a JSON array")
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        impact = item.get("impact", {})
        if not isinstance(impact, dict):
            impact = {}
        cleaned.append(
            {
                "id": str(item.get("id", "")),
                "text": str(item.get("text", "")),
                "answer": str(item.get("answer", "")),
                "impact": {
                    "imdbDelta": round(to_float(impact.get("imdbDelta"), 0.0), 3),
                    "rtDelta": round(to_float(impact.get("rtDelta"), 0.0), 3),
                    "fanDelta": round(to_float(impact.get("fanDelta"), 0.0), 3),
                    "boxOfficeDeltaPct": round(to_float(impact.get("boxOfficeDeltaPct"), 0.0), 3),
                },
            }
        )
    return cleaned


def prepare_retrieval_context(payload: dict[str, Any], top_k_per_query: int = 8) -> dict[str, Any]:
    hits = retrieve_related_hits(payload, top_k_per_query=top_k_per_query)
    fan = fan_sentiment_from_hits(hits)
    base = extract_market_baseline(hits)
    context = build_context(hits)
    return {
        "hits": hits,
        "fan": fan,
        "base": base,
        "context": context,
    }


def predict_with_gemini(
    payload: dict[str, Any],
    top_k_per_query: int = 8,
    prepared: dict[str, Any] | None = None,
    model: Any | None = None,
) -> Any:
    if model is None:
        model = get_gemini_model()
    if prepared is None:
        prepared = prepare_retrieval_context(payload, top_k_per_query=top_k_per_query)

    fan = prepared["fan"]
    base = prepared["base"]
    context = prepared["context"]

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


def compute_optimal_from_question_impacts(
    question_impacts: list[dict[str, Any]],
    max_combos: int = 256,
) -> dict[str, Any]:
    qids = [str(q.get("id", "")) for q in question_impacts if str(q.get("id", ""))]
    questions_by_id = {str(q.get("id", "")): q for q in question_impacts}

    if not qids:
        return {
            "evaluatedCombos": 0,
            "bestAnswers": {},
            "objectiveScore": 0.0,
            "note": "No question impacts provided.",
        }

    total_combos = 2 ** len(qids)
    if total_combos > max_combos:
        return {
            "evaluatedCombos": 0,
            "bestAnswers": {},
            "objectiveScore": 0.0,
            "note": f"Skipped optimal search: total combinations {total_combos} exceeds cap {max_combos}.",
        }

    # Use the impact for "yes" as baseline direction; "no" flips sign.
    yes_impacts: dict[str, dict[str, float]] = {}
    for qid in qids:
        q = questions_by_id.get(qid, {})
        imp = q.get("impact", {}) if isinstance(q.get("impact", {}), dict) else {}
        yes_impacts[qid] = {
            "imdbDelta": to_float(imp.get("imdbDelta"), 0.0),
            "rtDelta": to_float(imp.get("rtDelta"), 0.0),
            "fanDelta": to_float(imp.get("fanDelta"), 0.0),
            "boxOfficeDeltaPct": to_float(imp.get("boxOfficeDeltaPct"), 0.0),
        }

    best_score = None
    best_answers: dict[str, str] = {}
    best_delta = {"imdbDelta": 0.0, "rtDelta": 0.0, "fanDelta": 0.0, "boxOfficeDeltaPct": 0.0}

    for mask in range(total_combos):
        answers: dict[str, str] = {}
        delta = {"imdbDelta": 0.0, "rtDelta": 0.0, "fanDelta": 0.0, "boxOfficeDeltaPct": 0.0}
        for i, qid in enumerate(qids):
            ans = "yes" if ((mask >> i) & 1) else "no"
            answers[qid] = ans
            sign = 1.0 if ans == "yes" else -1.0
            yi = yes_impacts[qid]
            for k in delta:
                delta[k] += sign * yi[k]

        # Local objective aligned with existing ranking intent.
        score = (
            delta["imdbDelta"]
            + delta["rtDelta"]
            + (0.2 * delta["fanDelta"])
            + (0.1 * delta["boxOfficeDeltaPct"])
        )
        if best_score is None or score > best_score:
            best_score = score
            best_answers = answers
            best_delta = delta

    return {
        "evaluatedCombos": total_combos,
        "bestAnswers": best_answers,
        "objectiveScore": round(float(best_score or 0.0), 4),
        "bestDelta": {
            "imdbDelta": round(best_delta["imdbDelta"], 3),
            "rtDelta": round(best_delta["rtDelta"], 3),
            "fanDelta": round(best_delta["fanDelta"], 3),
            "boxOfficeDeltaPct": round(best_delta["boxOfficeDeltaPct"], 3),
        },
    }


def compute_optimal_combination(
    payload: dict[str, Any],
    top_k_per_query: int,
    max_combos: int = 256,
    prepared: dict[str, Any] | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    # Deprecated heavy path kept for compatibility; not used in main flow.
    questions = payload.get("questions", [])
    qids = [str(q.get("id", "")) for q in questions if str(q.get("id", ""))]
    if not qids:
        return {
            "evaluatedCombos": 0,
            "bestAnswers": {},
            "objectiveScore": 0.0,
            "note": "No question ids provided.",
        }

    total_combos = 2 ** len(qids)
    if total_combos > max_combos:
        return {
            "evaluatedCombos": 0,
            "bestAnswers": {},
            "objectiveScore": 0.0,
            "note": f"Skipped optimal search: total combinations {total_combos} exceeds cap {max_combos}.",
        }

    best_result = None
    best_answers = None
    best_score = None
    attempted = 0
    evaluated = 0
    failed = 0
    failure_reasons: dict[str, int] = {}

    for combo in itertools.product(["yes", "no"], repeat=len(qids)):
        attempted += 1
        combo_answers = {qid: ans for qid, ans in zip(qids, combo)}
        combo_payload = {
            "questions": questions,
            "answers": combo_answers,
        }
        pred = None
        err_reason = None
        # Retry a couple of times for transient LLM/API issues.
        for _ in range(3):
            try:
                pred = predict_with_gemini(
                    combo_payload,
                    top_k_per_query=top_k_per_query,
                    prepared=prepared,
                    model=model,
                )
                break
            except Exception as e:
                err_reason = type(e).__name__
        if pred is None:
            failed += 1
            key = err_reason or "UnknownError"
            failure_reasons[key] = failure_reasons.get(key, 0) + 1
            continue
        score = objective_score(pred if isinstance(pred, dict) else {})
        evaluated += 1
        if best_score is None or score > best_score:
            best_score = score
            best_result = pred
            best_answers = combo_answers

    if best_result is None:
        return {
            "attemptedCombos": attempted,
            "evaluatedCombos": evaluated,
            "failedCombos": failed,
            "failureReasons": failure_reasons,
            "bestAnswers": {},
            "objectiveScore": 0.0,
            "note": "No valid predictions from combination search.",
        }

    return {
        "attemptedCombos": attempted,
        "evaluatedCombos": evaluated,
        "failedCombos": failed,
        "failureReasons": failure_reasons,
        "bestAnswers": best_answers,
        "objectiveScore": round(float(best_score), 4),
        "bestPrediction": best_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini prediction JSON using Actian vector evidence.")
    parser.add_argument("--input", required=True, help="Input JSON path (questions + answers).")
    parser.add_argument("--output", default=None, help="Optional output JSON file.")
    parser.add_argument("--top-k-per-query", type=int, default=8)
    parser.add_argument("--skip-optimal", action="store_true", help="Skip exhaustive yes/no combination search.")
    parser.add_argument("--max-combos", type=int, default=256, help="Max combinations allowed for optimal search.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model = get_gemini_model()
    prepared = prepare_retrieval_context(payload, top_k_per_query=args.top_k_per_query)
    result = predict_with_gemini(
        payload,
        top_k_per_query=args.top_k_per_query,
        prepared=prepared,
        model=model,
    )
    if isinstance(result, dict):
        try:
            result["questionImpacts"] = llm_question_impacts(payload=payload, prepared=prepared, model=model)
        except Exception:
            result["questionImpacts"] = []
        if not args.skip_optimal:
            result["optimal"] = compute_optimal_from_question_impacts(
                question_impacts=result.get("questionImpacts", []),
                max_combos=args.max_combos,
            )

    print(json.dumps(result, indent=2, ensure_ascii=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
