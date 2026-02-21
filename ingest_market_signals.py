#!/usr/bin/env python3
"""
Ingest market + audience signals into Actian Vector DB.

Data sources:
1) OMDb API (aggregates IMDb, Rotten Tomatoes, Metacritic, BoxOffice)
2) YouTube Data API v3 (review/reaction comments)

Why this route:
- Direct scraping of IMDb/RT pages is often bot-blocked and policy-sensitive.
- OMDb gives a practical free API path for ratings/box-office fields.

Env vars:
    CORTEX_ADDRESS=localhost:50051
    ACTIAN_COLLECTION_NAME=endgame_opinions
    ACTIAN_RECREATE_COLLECTION=false
    OMDB_API_KEY=...
    YOUTUBE_API_KEY=...

Usage:
    python3 ingest_market_signals.py --movie-title "Avengers: Endgame"
    python3 ingest_market_signals.py --movie-title "Avengers: Endgame" --skip-youtube
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from typing import Any

import requests

from scrape_data_reddit import RawDocument, ingest_documents, stable_id


def clean_text(v: str) -> str:
    return re.sub(r"\s+", " ", (v or "")).strip()


def parse_int_from_text(v: str) -> int | None:
    if not v:
        return None
    digits = re.sub(r"[^\d]", "", v)
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def omdb_fetch_title(title: str, api_key: str) -> dict[str, Any] | None:
    try:
        r = requests.get(
            "https://www.omdbapi.com/",
            params={"apikey": api_key, "t": title},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    if data.get("Response") == "False":
        return None
    return data


def omdb_to_doc(data: dict[str, Any], query_title: str) -> RawDocument:
    imdb_id = str(data.get("imdbID", "")) or stable_id("omdb", query_title)
    title = clean_text(str(data.get("Title", query_title)))
    year = clean_text(str(data.get("Year", "")))
    imdb_rating = clean_text(str(data.get("imdbRating", "")))
    imdb_votes = clean_text(str(data.get("imdbVotes", "")))
    box_office = clean_text(str(data.get("BoxOffice", "")))
    metascore = clean_text(str(data.get("Metascore", "")))
    ratings = data.get("Ratings") or []

    rt_rating = ""
    mc_rating = ""
    for r in ratings:
        src = (r.get("Source") or "").lower()
        val = r.get("Value") or ""
        if "rotten tomatoes" in src:
            rt_rating = clean_text(str(val))
        if "metacritic" in src:
            mc_rating = clean_text(str(val))

    text = clean_text(
        f"Movie: {title} ({year}). "
        f"IMDb rating: {imdb_rating} from {imdb_votes} votes. "
        f"Rotten Tomatoes: {rt_rating or 'N/A'}. "
        f"Metacritic: {mc_rating or metascore or 'N/A'}. "
        f"Box office: {box_office or 'N/A'}. "
        f"Plot: {clean_text(str(data.get('Plot', '')))}"
    )

    return RawDocument(
        doc_id=stable_id("omdb_movie", imdb_id),
        source="omdb_market",
        url=f"https://www.imdb.com/title/{imdb_id}" if imdb_id.startswith("tt") else "https://www.omdbapi.com/",
        title=title,
        author="omdb",
        created_at=now_utc(),
        score=0,
        text=text,
        metadata={
            "query_title": query_title,
            "year": year,
            "imdb_id": imdb_id,
            "imdb_rating": imdb_rating,
            "imdb_votes": parse_int_from_text(imdb_votes),
            "rt_rating": rt_rating,
            "metacritic_rating": mc_rating or metascore,
            "box_office": box_office,
            "genre": data.get("Genre", ""),
            "director": data.get("Director", ""),
            "actors": data.get("Actors", ""),
            "runtime": data.get("Runtime", ""),
        },
    )


def youtube_search_videos(query: str, api_key: str, max_results: int = 5) -> list[dict[str, Any]]:
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "key": api_key,
                "part": "snippet",
                "type": "video",
                "q": query,
                "maxResults": str(max_results),
                "order": "relevance",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    return data.get("items", []) or []


def youtube_fetch_comments(video_id: str, api_key: str, max_comments: int = 80) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    next_page = None

    while len(comments) < max_comments:
        try:
            r = requests.get(
                "https://www.googleapis.com/youtube/v3/commentThreads",
                params={
                    "key": api_key,
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": "100",
                    "textFormat": "plainText",
                    "pageToken": next_page or "",
                    "order": "relevance",
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            break

        for item in data.get("items", []):
            top = (
                item.get("snippet", {})
                .get("topLevelComment", {})
                .get("snippet", {})
            )
            txt = clean_text(str(top.get("textDisplay", "")))
            if len(txt) < 20:
                continue
            comments.append(
                {
                    "comment_id": top.get("id", ""),
                    "author": top.get("authorDisplayName", "unknown"),
                    "likes": int(top.get("likeCount", 0) or 0),
                    "published_at": top.get("publishedAt", ""),
                    "text": txt,
                }
            )
            if len(comments) >= max_comments:
                break

        next_page = data.get("nextPageToken")
        if not next_page:
            break
        time.sleep(0.2)

    return comments[:max_comments]


def youtube_docs(movie_title: str, api_key: str, max_videos: int, comments_per_video: int) -> list[RawDocument]:
    queries = [
        f"{movie_title} review",
        f"{movie_title} ending reaction",
        f"{movie_title} explained",
    ]
    seen_video_ids: set[str] = set()
    docs: list[RawDocument] = []

    for q in queries:
        items = youtube_search_videos(query=q, api_key=api_key, max_results=max_videos)
        for item in items:
            vid = item.get("id", {}).get("videoId")
            if not vid or vid in seen_video_ids:
                continue
            seen_video_ids.add(vid)

            title = clean_text(str(item.get("snippet", {}).get("title", "YouTube video")))
            channel = clean_text(str(item.get("snippet", {}).get("channelTitle", "unknown")))
            video_url = f"https://www.youtube.com/watch?v={vid}"

            comments = youtube_fetch_comments(vid, api_key=api_key, max_comments=comments_per_video)
            for i, c in enumerate(comments):
                doc_id = stable_id("youtube_comment", vid, str(c.get("comment_id", "")), str(i))
                created_at = now_utc()
                pub = c.get("published_at")
                if isinstance(pub, str) and pub:
                    try:
                        created_at = dt.datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    except Exception:
                        created_at = now_utc()

                docs.append(
                    RawDocument(
                        doc_id=doc_id,
                        source="youtube_comment",
                        url=video_url,
                        title=title,
                        author=str(c.get("author", "unknown")),
                        created_at=created_at,
                        score=int(c.get("likes", 0) or 0),
                        text=str(c.get("text", "")),
                        metadata={
                            "video_id": vid,
                            "channel": channel,
                            "query": q,
                            "likes": int(c.get("likes", 0) or 0),
                        },
                    )
                )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OMDb + YouTube market signals into Actian.")
    parser.add_argument("--movie-title", default="Avengers: Endgame")
    parser.add_argument("--related-titles", default="Avengers: Infinity War,The Avengers,Avengers: Age of Ultron")
    parser.add_argument("--youtube-max-videos", type=int, default=4)
    parser.add_argument("--youtube-comments-per-video", type=int, default=60)
    parser.add_argument("--skip-youtube", action="store_true")
    parser.add_argument("--skip-omdb", action="store_true")
    args = parser.parse_args()

    docs: list[RawDocument] = []

    if not args.skip_omdb:
        omdb_key = os.getenv("OMDB_API_KEY", "").strip()
        if not omdb_key:
            print("OMDB_API_KEY not set; skipping OMDb ingest.")
        else:
            titles = [args.movie_title] + [t.strip() for t in args.related_titles.split(",") if t.strip()]
            for t in titles:
                data = omdb_fetch_title(title=t, api_key=omdb_key)
                if not data:
                    print(f"OMDb missing: {t}")
                    continue
                docs.append(omdb_to_doc(data, query_title=t))
            print(f"OMDb docs: {len([d for d in docs if d.source == 'omdb_market'])}")

    if not args.skip_youtube:
        yt_key = os.getenv("YOUTUBE_API_KEY", "").strip()
        if not yt_key:
            print("YOUTUBE_API_KEY not set; skipping YouTube ingest.")
        else:
            yt_docs = youtube_docs(
                movie_title=args.movie_title,
                api_key=yt_key,
                max_videos=args.youtube_max_videos,
                comments_per_video=args.youtube_comments_per_video,
            )
            docs.extend(yt_docs)
            print(f"YouTube comment docs: {len(yt_docs)}")

    if not docs:
        raise RuntimeError(
            "No docs collected. Set OMDB_API_KEY and/or YOUTUBE_API_KEY, "
            "or disable skipped sources."
        )

    ingest_documents(docs)


if __name__ == "__main__":
    main()
