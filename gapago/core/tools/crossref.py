"""Crossref 검색."""
from __future__ import annotations

import requests

from ._common import _norm


_CROSSREF_API = "https://api.crossref.org/works"
_CROSSREF_HEADERS = {"User-Agent": "GAPAGO-Research-Agent/1.0 (mailto:gapago@research.org)"}


def crossref_search(query: str, rows: int = 40, year: str = "") -> list[dict]:
    """Crossref API로 학술 논문 검색. 1.5억+ 메타데이터, 넉넉한 rate limit."""
    params = {
        "query": query,
        "rows": min(rows, 1000),
        "sort": "relevance",
        "order": "desc",
        "mailto": "gapago@research.org",
    }

    # 연도 필터
    if year and "-" in year:
        parts = year.split("-")
        from_year = parts[0].strip()
        to_year = parts[1].strip() if parts[1].strip() else "2026"
        params["filter"] = f"from-pub-date:{from_year},until-pub-date:{to_year}"

    try:
        r = requests.get(_CROSSREF_API, params=params, headers=_CROSSREF_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [crossref] API 오류: {e}")
        return []

    items = data.get("message", {}).get("items", [])
    results: list[dict] = []
    for item in items:
        title_list = item.get("title", [])
        title = _norm(title_list[0]) if title_list else ""
        if not title:
            continue

        abstract = _norm(item.get("abstract", ""))

        # 저자
        authors = []
        for a in item.get("author", []):
            name_parts = [a.get("given", ""), a.get("family", "")]
            name = " ".join(p for p in name_parts if p).strip()
            if name:
                authors.append(name)

        # 연도
        date_parts = item.get("published-print", item.get("published-online", item.get("created", {})))
        year_val = 0
        if date_parts and date_parts.get("date-parts"):
            try:
                year_val = int(date_parts["date-parts"][0][0])
            except (IndexError, TypeError, ValueError):
                pass

        doi = item.get("DOI", "")
        url = f"https://doi.org/{doi}" if doi else ""

        # PDF URL 추출: link 필드에서 application/pdf 타입 우선
        # Elsevier TDM API URL (api.elsevier.com, httpAccept=text/xml) 등은 제외
        pdf_url = ""
        for link in (item.get("link") or []):
            ct = (link.get("content-type") or "").lower()
            link_url = link.get("URL", "")
            if "api.elsevier.com" in link_url or "httpAccept=text/xml" in link_url:
                continue
            if "pdf" in ct and link_url:
                pdf_url = link_url
                break
        if not pdf_url:
            for link in (item.get("link") or []):
                link_url = link.get("URL", "")
                if "api.elsevier.com" in link_url or "httpAccept=text/xml" in link_url:
                    continue
                if link_url:
                    pdf_url = link_url
                    break

        results.append({
            "paper_id": f"crossref:{doi}" if doi else f"crossref:{title[:50]}",
            "title": title,
            "abstract": abstract,
            "url": url,
            "year": year_val,
            "authors": authors,
            "doi": doi,
            "score_bm25": 0.0,
            "venue": _norm((item.get("container-title") or [""])[0]) if item.get("container-title") else "",
            "source": "crossref",
            "full_text_sections": {"doi": doi, "pdf_url": pdf_url} if doi else {},
        })

    return results
