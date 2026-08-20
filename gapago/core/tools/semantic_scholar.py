"""Semantic Scholar 검색."""
from __future__ import annotations

import time
import requests

from ._common import _norm


_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_S2_FIELDS = "paperId,title,abstract,year,authors,url,externalIds,venue,publicationVenue,openAccessPdf,isOpenAccess"


def semantic_scholar_search(query: str, limit: int = 20, year: str = "") -> list[dict]:
    """Semantic Scholar Academic Graph API로 논문 검색."""
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": _S2_FIELDS,
    }
    if year:
        params["year"] = year  # e.g. "2020-2025" or "2023-"

    papers: list[dict] = []
    last_error = None
    for attempt in range(3):
        try:
            r = requests.get(_S2_API, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json().get("data", [])[:limit]
            for p in data:
                if not p.get("title"):
                    continue
                ext_ids = p.get("externalIds") or {}
                arxiv_id = ext_ids.get("ArXiv", "")
                doi = ext_ids.get("DOI", "")
                paper_id = f"arxiv:{arxiv_id}" if arxiv_id else f"s2:{p.get('paperId', '')}"
                authors = [a.get("name", "") for a in (p.get("authors") or []) if a.get("name")]
                # venue 추출: publicationVenue.name 우선, fallback으로 venue 필드
                pub_venue = p.get("publicationVenue") or {}
                venue = pub_venue.get("name", "") if isinstance(pub_venue, dict) else ""
                if not venue:
                    venue = p.get("venue") or ""
                if not venue and arxiv_id:
                    venue = "arXiv preprint"
                oa_pdf = (p.get("openAccessPdf") or {}).get("url", "")
                papers.append({
                    "paper_id": paper_id,
                    "title": _norm(p.get("title", "")),
                    "abstract": _norm(p.get("abstract") or ""),
                    "url": p.get("url") or "",
                    "year": p.get("year") or 0,
                    "authors": authors,
                    "score_bm25": 0.0,
                    "venue": venue,
                    "source": "semantic_scholar",
                    "full_text_sections": {"doi": doi, "pdf_url": oa_pdf},
                })
            last_error = None
            break
        except Exception as e:
            last_error = e
            time.sleep(2)
    if last_error:
        raise last_error
    return papers
