"""Ranked search.

1. parse the query into a tree and evaluate it -> matching doc ids (lru_cache per index)
2. score the matches with the query's positive terms (Scorer protocol)
3. top-k with heapq.nlargest, then a snippet with highlighted terms for each result
"""

from __future__ import annotations

import heapq
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.index import Index
from findex.query import parse
from findex.scoring import SCORERS, Scorer, ScorerName
from findex.timing import timed
from findex.tokenize import iter_spans

log = logging.getLogger(__name__)
SNIPPET_RADIUS = 80


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    score: float  # first field: sorted(results) orders by score
    doc_id: int
    title: str
    path: str
    snippet: str = ""

    def to_dict(self) -> dict[str, object]:
        return {f: getattr(self, f) for f in ("doc_id", "score", "title", "path", "snippet")}


def match(index: Index, query: str) -> tuple[int, ...]:
    """Doc ids matching the Boolean query. Cached per index with lru_cache."""
    before = index.match.cache_info().hits
    ids = index.match(query)
    if index.match.cache_info().hits > before:
        log.debug("cache hit for %r", query)
    return ids


@timed
def search(
    index: Index, query: str, scorer: Scorer | ScorerName = "bm25", k: int = 10
) -> list[SearchResult]:
    """Top-k documents for a query. scorer: a Scorer object or its name ("bm25" | "tfidf")."""
    if isinstance(scorer, str):
        scorer = SCORERS[scorer]
    ids = match(index, query)
    if not ids:
        return []
    wanted = set(ids)
    terms = parse(query).terms()
    scores: defaultdict[int, float] = defaultdict(float)
    for d in ids:
        scores[d] = 0.0  # matched docs without positive terms (e.g. "NOT x") still appear
    for term in terms:
        for posting in index.postings.get(term, ()):
            if posting.doc_id in wanted:
                scores[posting.doc_id] += scorer(term, posting, index)
    top = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])  # O(n log k), not a full sort
    results: list[SearchResult] = []
    for doc_id, score in top:
        meta = index.doc_meta[doc_id]
        text = read_text(meta.path)
        results.append(SearchResult(score, doc_id, meta.title, meta.path, snippet(text, terms)))
    return results


def read_text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def snippet(text: str, terms: set[str], radius: int = SNIPPET_RADIUS) -> str:
    """A 2·radius window (±radius around the best occurrence) with the most distinct query
    terms; query terms are highlighted as **term**. Sliding window: O(number of hits)."""
    if not text or not terms:
        return ""
    nfc = unicodedata.normalize("NFC", text)  # iter_spans offsets point into the NFC text
    hits = [(s, e, t) for t, s, e in iter_spans(text) if t in terms]
    if not hits:
        return ""
    window: Counter[str] = Counter()
    best_i, best_j, best_n, j = 0, 0, 0, 0
    for i, (s, _, _) in enumerate(hits):
        while j < len(hits) and hits[j][1] <= s + 2 * radius:
            window[hits[j][2]] += 1
            j += 1
        if len(window) > best_n:
            best_i, best_j, best_n = i, j, len(window)
        window[hits[i][2]] -= 1
        if not window[hits[i][2]]:
            del window[hits[i][2]]
    center = (hits[best_i][0] + hits[best_j - 1][1]) // 2
    lo, hi = max(0, center - radius), min(len(nfc), center + radius)
    out: list[str] = []
    cur = lo
    for s, e, _ in hits[best_i:best_j]:
        if s >= cur and e <= hi:
            out += [nfc[cur:s], "**", nfc[s:e], "**"]
            cur = e
    out.append(nfc[cur:hi])
    body = re.sub(r"\s+", " ", "".join(out)).strip()
    return ("…" if lo > 0 else "") + body + ("…" if hi < len(nfc) else "")
