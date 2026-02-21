#!/usr/bin/env python3
"""
Bulk-ingest MCU/comparable movie market signals + audience comments into Actian.

Uses:
- OMDb API for ratings + box office metadata
- YouTube Data API for review/reaction comments

Env vars:
    CORTEX_ADDRESS=localhost:50051
    ACTIAN_COLLECTION_NAME=endgame_opinions
    ACTIAN_RECREATE_COLLECTION=false
    OMDB_API_KEY=...
    YOUTUBE_API_KEY=...
"""

from __future__ import annotations

import argparse
from typing import Any
import os

from scrape_data_reddit import ingest_documents, RawDocument
from ingest_market_signals import omdb_fetch_title, omdb_to_doc, youtube_docs


DEFAULT_TITLES = [
    "Avengers: Endgame",
    "Avengers: Infinity War",
    "The Avengers",
    "Avengers: Age of Ultron",
    "Captain America: Civil War",
    "Captain America: The Winter Soldier",
    "Iron Man",
    "Iron Man 3",
    "Thor: Ragnarok",
    "Guardians of the Galaxy",
    "Black Panther",
    "Spider-Man: No Way Home",
]


def parse_titles(arg: str | None) -> list[str]:
    if not arg:
        return list(DEFAULT_TITLES)
    parts = [p.strip() for p in arg.split(",")]
    return [p for p in parts if p]


def omdb_docs_for_titles(titles: list[str], omdb_key: str) -> list[RawDocument]:
    docs: list[RawDocument] = []
    for t in titles:
        data = omdb_fetch_title(t, omdb_key)
        if not data:
            print(f"OMDb missing: {t}")
            continue
        docs.append(omdb_to_doc(data, query_title=t))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk ingest MCU market/audience signals.")
    parser.add_argument(
        "--titles",
        default=None,
        help="Comma-separated movie titles. Defaults to MCU pack.",
    )
    parser.add_argument("--youtube-max-videos", type=int, default=1)
    parser.add_argument("--youtube-comments-per-video", type=int, default=25)
    parser.add_argument("--skip-youtube", action="store_true")
    parser.add_argument("--skip-omdb", action="store_true")
    parser.add_argument(
        "--max-titles",
        type=int,
        default=8,
        help="Safety cap for one run to avoid oversized upserts.",
    )
    args = parser.parse_args()

    titles = parse_titles(args.titles)[: max(1, args.max_titles)]
    docs: list[RawDocument] = []

    if not args.skip_omdb:
        omdb_key = os.getenv("OMDB_API_KEY", "").strip()
        if not omdb_key:
            print("OMDB_API_KEY not set; skipping OMDb.")
        else:
            o_docs = omdb_docs_for_titles(titles, omdb_key)
            docs.extend(o_docs)
            print(f"OMDb docs: {len(o_docs)}")

    if not args.skip_youtube:
        yt_key = os.getenv("YOUTUBE_API_KEY", "").strip()
        if not yt_key:
            print("YOUTUBE_API_KEY not set; skipping YouTube.")
        else:
            y_docs: list[RawDocument] = []
            for title in titles:
                part = youtube_docs(
                    movie_title=title,
                    api_key=yt_key,
                    max_videos=args.youtube_max_videos,
                    comments_per_video=args.youtube_comments_per_video,
                )
                y_docs.extend(part)
                print(f"YouTube docs for '{title}': {len(part)}")
            docs.extend(y_docs)
            print(f"YouTube docs total: {len(y_docs)}")

    if not docs:
        raise RuntimeError("No documents collected. Check keys/options.")

    print(f"Total docs to ingest: {len(docs)}")
    ingest_documents(docs)


if __name__ == "__main__":
    main()
