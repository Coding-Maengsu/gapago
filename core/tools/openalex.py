"""OpenAlex 검색."""
from __future__ import annotations

import requests

from ._common import _norm


_OPENALEX_API = "https://api.openalex.org/works"


def openalex_search(query: str, per_page: int = 20, year: str = "") -> list[dict]:
    """OpenAlex API로 논문 검색. API key 불필요."""
    params = {
        "search": query,
        "per_page": min(per_page, 200),
        "select": "id,title,publication_year,doi,authorships,abstract_inverted_index,primary_location,open_access,best_oa_location",
    }
    # 연도 필터: OpenAlex filter 파라미터 사용
    if year and "-" in year:
        parts = year.split("-")
        from_year = parts[0].strip()
        to_year = parts[1].strip() if parts[1].strip() else "2026"
        params["filter"] = f"publication_year:{from_year}-{to_year}"
    papers: list[dict] = []
    r = requests.get(
        _OPENALEX_API,
        params=params,
        headers={"User-Agent": "GAPAGO-Research-Agent/1.0 (mailto:gapago@research.dev)"},
        timeout=30,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    for p in results:
        title = _norm(p.get("title") or "")
        if not title:
            continue

        # abstract 복원 (inverted index → text)
        abstract = ""
        inv_idx = p.get("abstract_inverted_index")
        if inv_idx and isinstance(inv_idx, dict):
            word_positions = []
            for word, positions in inv_idx.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            abstract = " ".join(w for _, w in word_positions)

        doi_url = p.get("doi") or ""
        doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "") if doi_url else ""
        openalex_id = (p.get("id") or "").replace("https://openalex.org/", "")

        authors = []
        for a in (p.get("authorships") or []):
            name = (a.get("author") or {}).get("display_name", "")
            if name:
                authors.append(name)

        # venue 추출: primary_location.source.display_name
        venue = ""
        primary_loc = p.get("primary_location") or {}
        if isinstance(primary_loc, dict):
            source_info = primary_loc.get("source") or {}
            if isinstance(source_info, dict):
                venue = source_info.get("display_name", "")

        # OA PDF URL 추출
        oa_pdf = ""
        best_oa = p.get("best_oa_location") or {}
        if isinstance(best_oa, dict):
            oa_pdf = best_oa.get("pdf_url") or ""
        if not oa_pdf:
            oa_info = p.get("open_access") or {}
            if isinstance(oa_info, dict):
                oa_pdf = oa_info.get("oa_url") or ""

        papers.append({
            "paper_id": f"openalex:{openalex_id}",
            "title": title,
            "abstract": _norm(abstract),
            "url": doi_url or f"https://openalex.org/{openalex_id}",
            "year": p.get("publication_year") or 0,
            "authors": authors,
            "score_bm25": 0.0,
            "venue": venue,
            "source": "openalex",
            "full_text_sections": {"doi": doi, "pdf_url": oa_pdf},
        })
    return papers
