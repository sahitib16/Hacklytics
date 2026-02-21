#!/usr/bin/env python3
"""
RAG data pipeline for Avengers Endgame opinions from Reddit + Quora into Actian.

Install deps (example):
    pip install praw requests beautifulsoup4
    pip install /path/to/actiancortex-0.1.0b1-py3-none-any.whl

Required env vars:
    CORTEX_ADDRESS=localhost:50051
    ACTIAN_COLLECTION_NAME=endgame_opinions
    ACTIAN_RECREATE_COLLECTION=false
    REDDIT_SCRAPE_MODE=unofficial
Optional env vars:
    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT
    REDDIT_SUBREDDITS=marvelstudios,movies,marvel
    REDDIT_POST_LIMIT=120
    REDDIT_COMMENT_LIMIT=500
    QUORA_TOPIC_URLS=https://www.quora.com/topic/Avengers-Endgame
    QUORA_MAX_QUESTIONS=60
    QUORA_REQUEST_DELAY_SECONDS=1.0
    LOCAL_EMBED_DIM=1024
    CHUNK_SIZE_WORDS=220
    CHUNK_OVERLAP_WORDS=40

JSONL ingest format for `ingest-file`:
    One JSON object per line, with at least:
      - text (required)
    Optional fields:
      - id, source, url, title, author, created_at, score, metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup

try:
    import praw
except Exception:  # pragma: no cover
    praw = None

try:
    from cortex import CortexClient, DistanceMetric
except Exception:  # pragma: no cover
    CortexClient = None
    DistanceMetric = None

try:
    from cortex import Field, Filter
except Exception:  # pragma: no cover
    try:
        from cortex.filters import Field, Filter
    except Exception:  # pragma: no cover
        Field = None
        Filter = None


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


@dataclass
class RawDocument:
    doc_id: str
    source: str
    url: str
    title: str
    author: str
    created_at: datetime
    score: int
    text: str
    metadata: dict[str, Any]


@dataclass
class ChunkRecord:
    chunk_id: str
    vector_id: int
    doc_id: str
    source: str
    url: str
    title: str
    text: str
    metadata: dict[str, Any]


def env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if not val:
        return default
    return int(val)


def env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if not val:
        return default
    return float(val)


def utc_dt_from_timestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def stable_id(*parts: str) -> str:
    m = hashlib.sha256()
    for part in parts:
        m.update(part.encode("utf-8", errors="ignore"))
        m.update(b"|")
    return m.hexdigest()[:32]


def stable_int_id(*parts: str) -> int:
    # 63-bit positive integer for VectorAI point IDs.
    hex_id = stable_id(*parts)[:15]
    return int(hex_id, 16)


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "")).strip()
    return value


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size_words - overlap_words)
    for i in range(0, len(words), step):
        piece = words[i : i + chunk_size_words]
        if not piece:
            continue
        chunks.append(" ".join(piece))
        if i + chunk_size_words >= len(words):
            break
    return chunks


class RedditScraper:
    def __init__(self) -> None:
        if praw is None:
            raise RuntimeError("praw is not installed. Run: pip install praw")

        self.client = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
        )

    def scrape(self, query: str, subreddits: list[str], post_limit: int, comment_limit: int) -> list[RawDocument]:
        docs: list[RawDocument] = []
        for subreddit_name in subreddits:
            subreddit_name = subreddit_name.strip()
            if not subreddit_name:
                continue
            subreddit = self.client.subreddit(subreddit_name)
            for submission in subreddit.search(query, sort="relevance", limit=post_limit):
                post_text = clean_text(f"{submission.title}\n\n{submission.selftext or ''}")
                if len(post_text) < 20:
                    continue

                post_doc = RawDocument(
                    doc_id=stable_id("reddit_post", submission.id),
                    source="reddit",
                    url=f"https://www.reddit.com{submission.permalink}",
                    title=clean_text(submission.title),
                    author=str(submission.author) if submission.author else "[deleted]",
                    created_at=utc_dt_from_timestamp(float(submission.created_utc)),
                    score=int(getattr(submission, "score", 0) or 0),
                    text=post_text,
                    metadata={
                        "kind": "post",
                        "subreddit": subreddit_name,
                        "num_comments": int(getattr(submission, "num_comments", 0) or 0),
                    },
                )
                docs.append(post_doc)

                submission.comment_sort = "top"
                submission.comments.replace_more(limit=0)
                for idx, comment in enumerate(submission.comments.list()):
                    if idx >= comment_limit:
                        break
                    comment_text = clean_text(getattr(comment, "body", ""))
                    if len(comment_text) < 20:
                        continue
                    comment_doc = RawDocument(
                        doc_id=stable_id("reddit_comment", comment.id),
                        source="reddit",
                        url=f"https://www.reddit.com{comment.permalink}",
                        title=clean_text(submission.title),
                        author=str(comment.author) if comment.author else "[deleted]",
                        created_at=utc_dt_from_timestamp(float(comment.created_utc)),
                        score=int(getattr(comment, "score", 0) or 0),
                        text=comment_text,
                        metadata={
                            "kind": "comment",
                            "subreddit": subreddit_name,
                            "post_id": submission.id,
                            "parent_id": getattr(comment, "parent_id", ""),
                        },
                    )
                    docs.append(comment_doc)
        return docs


class UnofficialRedditScraper:
    def __init__(self, delay_s: float = 0.6) -> None:
        self.delay_s = delay_s
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": os.getenv(
                    "REDDIT_USER_AGENT",
                    "mac:endgame-opinion-scraper:v1.0 (non-official-mode)",
                )
            }
        )

    def _sleep(self) -> None:
        time.sleep(self.delay_s)

    def _search_posts(self, subreddit: str, query: str, limit: int) -> list[dict[str, Any]]:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q": query,
            "restrict_sr": "on",
            "sort": "relevance",
            "limit": str(limit),
            "t": "all",
        }
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        self._sleep()
        children = data.get("data", {}).get("children", [])
        return [c.get("data", {}) for c in children if isinstance(c, dict)]

    def _extract_comments(self, node: dict[str, Any], out: list[dict[str, Any]]) -> None:
        kind = node.get("kind")
        if kind == "t1":
            data = node.get("data", {})
            out.append(data)
            replies = data.get("replies")
            if isinstance(replies, dict):
                for child in replies.get("data", {}).get("children", []):
                    if isinstance(child, dict):
                        self._extract_comments(child, out)

    def _fetch_comments(self, permalink: str, comment_limit: int) -> list[dict[str, Any]]:
        url = f"https://www.reddit.com{permalink}.json"
        params = {"limit": str(comment_limit), "sort": "top", "depth": "5"}
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        self._sleep()

        if not isinstance(data, list) or len(data) < 2:
            return []
        comments_listing = data[1]
        children = comments_listing.get("data", {}).get("children", [])
        comments: list[dict[str, Any]] = []
        for child in children:
            if isinstance(child, dict):
                self._extract_comments(child, comments)
            if len(comments) >= comment_limit:
                break
        return comments[:comment_limit]

    def scrape(self, query: str, subreddits: list[str], post_limit: int, comment_limit: int) -> list[RawDocument]:
        docs: list[RawDocument] = []
        per_sub_limit = max(1, post_limit)
        for subreddit_name in subreddits:
            subreddit_name = subreddit_name.strip()
            if not subreddit_name:
                continue

            posts = self._search_posts(subreddit_name, query=query, limit=per_sub_limit)
            for post in posts:
                post_title = clean_text(post.get("title", ""))
                post_body = clean_text(post.get("selftext", ""))
                post_text = clean_text(f"{post_title}\n\n{post_body}")
                if len(post_text) < 20:
                    continue

                permalink = post.get("permalink", "")
                if not permalink:
                    continue
                post_id = str(post.get("id", ""))
                created_utc = float(post.get("created_utc", time.time()))
                post_doc = RawDocument(
                    doc_id=stable_id("reddit_post_unofficial", post_id or permalink),
                    source="reddit",
                    url=f"https://www.reddit.com{permalink}",
                    title=post_title or "reddit post",
                    author=post.get("author", "[deleted]") or "[deleted]",
                    created_at=utc_dt_from_timestamp(created_utc),
                    score=int(post.get("score", 0) or 0),
                    text=post_text,
                    metadata={
                        "kind": "post",
                        "subreddit": subreddit_name,
                        "num_comments": int(post.get("num_comments", 0) or 0),
                        "scrape_mode": "unofficial",
                    },
                )
                docs.append(post_doc)

                comments = self._fetch_comments(permalink=permalink, comment_limit=comment_limit)
                for comment in comments:
                    body = clean_text(comment.get("body", ""))
                    if len(body) < 20:
                        continue
                    cid = str(comment.get("id", ""))
                    created_utc = float(comment.get("created_utc", time.time()))
                    comment_doc = RawDocument(
                        doc_id=stable_id("reddit_comment_unofficial", cid or permalink, body[:80]),
                        source="reddit",
                        url=f"https://www.reddit.com{permalink}",
                        title=post_title or "reddit post",
                        author=comment.get("author", "[deleted]") or "[deleted]",
                        created_at=utc_dt_from_timestamp(created_utc),
                        score=int(comment.get("score", 0) or 0),
                        text=body,
                        metadata={
                            "kind": "comment",
                            "subreddit": subreddit_name,
                            "post_id": post_id,
                            "parent_id": str(comment.get("parent_id", "")),
                            "scrape_mode": "unofficial",
                        },
                    )
                    docs.append(comment_doc)
        return docs


class QuoraScraper:
    def __init__(self, delay_s: float = 1.0) -> None:
        self.delay_s = delay_s
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _sleep(self) -> None:
        time.sleep(self.delay_s)

    def _extract_question_links(self, topic_url: str) -> list[str]:
        try:
            resp = self.session.get(topic_url, timeout=30)
            resp.raise_for_status()
        except Exception:
            return []
        self._sleep()

        soup = BeautifulSoup(resp.text, "html.parser")
        links: list[str] = []
        for a in soup.select("a[href]"):
            href = a.get("href", "").strip()
            if not href:
                continue
            if href.startswith("/"):
                url = f"https://www.quora.com{href}"
            elif href.startswith("https://www.quora.com/"):
                url = href
            else:
                continue
            path = url.replace("https://www.quora.com/", "")
            # Heuristic: keep question-like links, skip feed/profile/settings routes.
            if "/" in path:
                continue
            if path.startswith(("profile/", "topic/", "search", "settings", "about")):
                continue
            if "-" not in path:
                continue
            links.append(url)

        # Preserve order, de-dup.
        seen: set[str] = set()
        out: list[str] = []
        for link in links:
            if link not in seen:
                seen.add(link)
                out.append(link)
        return out

    def _extract_answers_from_jsonld(self, soup: BeautifulSoup) -> list[str]:
        answers: list[str] = []
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            raw = script.get_text(strip=True)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            stack = [data]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if node.get("@type") == "Answer":
                        txt = clean_text(node.get("text", ""))
                        if txt:
                            answers.append(txt)
                    for v in node.values():
                        stack.append(v)
                elif isinstance(node, list):
                    stack.extend(node)
        return answers

    def scrape(self, topic_urls: list[str], max_questions: int) -> list[RawDocument]:
        docs: list[RawDocument] = []
        links: list[str] = []
        for topic_url in topic_urls:
            links.extend(self._extract_question_links(topic_url))
            if len(links) >= max_questions:
                break
        links = links[:max_questions]

        for link in links:
            try:
                resp = self.session.get(link, timeout=30)
                resp.raise_for_status()
            except Exception:
                continue
            self._sleep()
            soup = BeautifulSoup(resp.text, "html.parser")

            title = ""
            title_meta = soup.find("meta", attrs={"property": "og:title"})
            if title_meta and title_meta.get("content"):
                title = clean_text(title_meta["content"])
            if not title:
                title = clean_text(soup.title.text if soup.title else "")
            if not title:
                title = "Quora question"

            answers = self._extract_answers_from_jsonld(soup)
            if not answers:
                fallback_blocks = soup.select("[class*='answer'], [class*='q-text']")
                for block in fallback_blocks:
                    txt = clean_text(block.get_text(" ", strip=True))
                    if len(txt) > 80:
                        answers.append(txt)

            for idx, ans in enumerate(answers):
                if len(ans) < 40:
                    continue
                doc = RawDocument(
                    doc_id=stable_id("quora_answer", link, str(idx)),
                    source="quora",
                    url=link,
                    title=title,
                    author="unknown",
                    created_at=datetime.now(tz=timezone.utc),
                    score=0,
                    text=ans,
                    metadata={"kind": "answer", "answer_index": idx},
                )
                docs.append(doc)

        return docs


class LocalHasherEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.token_re = re.compile(r"[a-z0-9']+")

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = self.token_re.findall((text or "").lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = -1.0 if ((h >> 8) & 1) else 1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class ActianCortexStore:
    def __init__(self, address: str, collection_name: str, dimension: int, recreate: bool = False) -> None:
        if CortexClient is None or DistanceMetric is None:
            raise RuntimeError(
                "Actian Cortex client not installed. Install the beta wheel "
                "(actiancortex-0.1.0b1-py3-none-any.whl) from the official repo."
            )
        self.address = address
        self.collection_name = collection_name
        self.dimension = dimension
        self.recreate = recreate

    def _ensure_collection(self, client: Any) -> None:
        if self.recreate:
            client.recreate_collection(
                name=self.collection_name,
                dimension=self.dimension,
                distance_metric=DistanceMetric.COSINE,
            )
            return
        if client.has_collection(self.collection_name):
            client.open_collection(self.collection_name)
            return
        client.create_collection(
            name=self.collection_name,
            dimension=self.dimension,
            distance_metric=DistanceMetric.COSINE,
        )

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]], batch_size: int = 128) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        with CortexClient(self.address) as client:
            self._ensure_collection(client)
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size]
                ids = [c.vector_id for c in batch_chunks]
                payloads = []
                for c in batch_chunks:
                    payloads.append(
                        {
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "source": c.source,
                            "url": c.url,
                            "title": c.title,
                            "text": c.text,
                            "metadata_json": json.dumps(c.metadata, ensure_ascii=True),
                        }
                    )
                client.batch_upsert(
                    self.collection_name,
                    ids=ids,
                    vectors=batch_embeddings,
                    payloads=payloads,
                )
            client.flush(self.collection_name)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        with CortexClient(self.address) as client:
            if source_filter and Filter is not None and Field is not None:
                filt = Filter().must(Field("source").eq(source_filter))
                results = client.search_filtered(
                    collection_name=self.collection_name,
                    query=query_vector,
                    filter=filt,
                    top_k=top_k,
                )
            else:
                results = client.search(
                    collection_name=self.collection_name,
                    query=query_vector,
                    top_k=top_k,
                    with_payload=True,
                )

        parsed: list[dict[str, Any]] = []
        for r in results:
            payload = getattr(r, "payload", {}) or {}
            if source_filter and payload.get("source") != source_filter:
                continue
            parsed.append(
                {
                    "id": getattr(r, "id", None),
                    "score": float(getattr(r, "score", 0.0)),
                    "source": payload.get("source"),
                    "url": payload.get("url"),
                    "title": payload.get("title"),
                    "text": payload.get("text"),
                    "doc_id": payload.get("doc_id"),
                    "chunk_id": payload.get("chunk_id"),
                    "metadata_json": payload.get("metadata_json"),
                }
            )
        return parsed[:top_k]


def build_chunks(docs: list[RawDocument], chunk_size_words: int, overlap_words: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for d in docs:
        pieces = chunk_text(d.text, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
        for i, piece in enumerate(pieces):
            chunks.append(
                ChunkRecord(
                    chunk_id=stable_id("chunk", d.doc_id, str(i), piece[:120]),
                    vector_id=stable_int_id("vec", d.doc_id, str(i), piece[:120]),
                    doc_id=d.doc_id,
                    source=d.source,
                    url=d.url,
                    title=d.title,
                    text=piece,
                    metadata={**d.metadata, "chunk_index": i},
                )
            )
    return chunks


def semantic_search(
    store: ActianCortexStore,
    embedder: LocalHasherEmbedder,
    query: str,
    top_k: int = 8,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    q_vec = embedder.embed_texts([query])[0]
    return store.search(query_vector=q_vec, top_k=top_k, source_filter=source_filter)


def ingest_documents(docs: list[RawDocument]) -> None:
    chunk_size_words = env_int("CHUNK_SIZE_WORDS", 220)
    chunk_overlap_words = env_int("CHUNK_OVERLAP_WORDS", 40)
    local_embed_dim = env_int("LOCAL_EMBED_DIM", 1024)
    cortex_address = os.getenv("CORTEX_ADDRESS", "localhost:50051")
    collection_name = os.getenv("ACTIAN_COLLECTION_NAME", "endgame_opinions")
    recreate_collection = os.getenv("ACTIAN_RECREATE_COLLECTION", "false").lower() == "true"

    if not docs:
        raise RuntimeError("No documents to ingest.")

    # De-dup by source + url + text hash.
    deduped: dict[str, RawDocument] = {}
    for d in docs:
        key = stable_id(d.source, d.url, d.text[:1500])
        deduped[key] = d
    docs = list(deduped.values())

    chunks = build_chunks(docs, chunk_size_words=chunk_size_words, overlap_words=chunk_overlap_words)
    if not chunks:
        raise RuntimeError("No chunks generated from scraped documents.")

    embedder = LocalHasherEmbedder(dim=local_embed_dim)
    embeddings = embedder.embed_texts([c.text for c in chunks])

    store = ActianCortexStore(
        address=cortex_address,
        collection_name=collection_name,
        dimension=local_embed_dim,
        recreate=recreate_collection,
    )
    store.upsert_chunks(chunks, embeddings)

    print(
        f"Ingest complete. docs={len(docs)}, chunks={len(chunks)}, "
        f"collection={collection_name}, address={cortex_address}"
    )


def ingest_pipeline() -> None:
    query = "Avengers Endgame opinions"
    subreddits = [s.strip() for s in os.getenv("REDDIT_SUBREDDITS", "marvelstudios,movies,marvel").split(",")]
    post_limit = env_int("REDDIT_POST_LIMIT", 120)
    comment_limit = env_int("REDDIT_COMMENT_LIMIT", 500)
    reddit_mode = os.getenv("REDDIT_SCRAPE_MODE", "unofficial").strip().lower()
    reddit_delay = env_float("REDDIT_REQUEST_DELAY_SECONDS", 0.6)
    topic_urls = [s.strip() for s in os.getenv("QUORA_TOPIC_URLS", "https://www.quora.com/topic/Avengers-Endgame").split(",")]
    max_questions = env_int("QUORA_MAX_QUESTIONS", 60)
    quora_delay = env_float("QUORA_REQUEST_DELAY_SECONDS", 1.0)

    use_praw = (
        reddit_mode == "praw"
        and bool(os.getenv("REDDIT_CLIENT_ID"))
        and bool(os.getenv("REDDIT_CLIENT_SECRET"))
        and bool(os.getenv("REDDIT_USER_AGENT"))
    )
    if use_praw:
        reddit_docs = RedditScraper().scrape(
            query=query,
            subreddits=subreddits,
            post_limit=post_limit,
            comment_limit=comment_limit,
        )
    else:
        reddit_docs = UnofficialRedditScraper(delay_s=reddit_delay).scrape(
            query=query,
            subreddits=subreddits,
            post_limit=post_limit,
            comment_limit=comment_limit,
        )
    quora_docs = QuoraScraper(delay_s=quora_delay).scrape(
        topic_urls=topic_urls,
        max_questions=max_questions,
    )

    all_docs = reddit_docs + quora_docs
    if not all_docs:
        raise RuntimeError("No documents scraped. Check API keys and source limits.")
    ingest_documents(all_docs)


def ingest_file_pipeline(path: str) -> None:
    docs: list[RawDocument] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = clean_text(str(obj.get("text", "")))
            if len(text) < 20:
                continue
            source = clean_text(str(obj.get("source", "dataset"))) or "dataset"
            url = clean_text(str(obj.get("url", ""))) or f"dataset://{source}"
            title = clean_text(str(obj.get("title", ""))) or "Dataset record"
            author = clean_text(str(obj.get("author", ""))) or "unknown"
            score = int(obj.get("score", 0) or 0)
            created_at_raw = obj.get("created_at")
            created_at = datetime.now(tz=timezone.utc)
            if isinstance(created_at_raw, str) and created_at_raw:
                try:
                    created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
                except Exception:
                    created_at = datetime.now(tz=timezone.utc)
            doc_id = str(obj.get("id", "")) or stable_id("dataset_doc", source, url, str(i))
            metadata = obj.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata = {**metadata, "dataset_row": i}
            docs.append(
                RawDocument(
                    doc_id=stable_id("dataset_doc_norm", doc_id),
                    source=source,
                    url=url,
                    title=title,
                    author=author,
                    created_at=created_at,
                    score=score,
                    text=text,
                    metadata=metadata,
                )
            )
    if not docs:
        raise RuntimeError(f"No valid documents found in JSONL file: {path}")
    ingest_documents(docs)


def run_search(query: str, top_k: int, source_filter: str | None) -> None:
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

    hits = semantic_search(store=store, embedder=embedder, query=query, top_k=top_k, source_filter=source_filter)
    print(json.dumps(hits, indent=2, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Avengers Endgame opinions and load into Actian for RAG.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Scrape Reddit+Quora, embed, and write into Actian")
    ingest_file_p = sub.add_parser("ingest-file", help="Ingest JSONL dataset file into Actian")
    ingest_file_p.add_argument("--path", required=True, help="Path to JSONL file (one JSON object per line)")

    search_p = sub.add_parser("search", help="Semantic search over stored chunks")
    search_p.add_argument("--query", required=True)
    search_p.add_argument("--top-k", type=int, default=8)
    search_p.add_argument("--source", default=None, choices=["reddit", "quora"])

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest_pipeline()
    elif args.cmd == "ingest-file":
        ingest_file_pipeline(path=args.path)
    elif args.cmd == "search":
        run_search(query=args.query, top_k=args.top_k, source_filter=args.source)


if __name__ == "__main__":
    main()
