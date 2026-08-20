"""BM25 초벌 랭킹."""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from ._common import _tokenize


def bm25_rank(papers: list[dict], query_text: str, top_k: int) -> dict:
    if not papers:
        return {"selected": [], "avg_bm25": 0.0}

    corpus = [_tokenize(f"{p.get('title', '')} {p.get('abstract', '')}") for p in papers]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query_text))
    pairs = list(zip(papers, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_k]

    selected = []
    for paper, score in top:
        item = dict(paper)
        item["score_bm25"] = float(score)
        selected.append(item)

    avg = sum(p["score_bm25"] for p in selected) / len(selected) if selected else 0.0
    return {"selected": selected, "avg_bm25": avg}
