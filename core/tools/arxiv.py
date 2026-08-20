"""arXiv Atom API 검색."""
from __future__ import annotations

import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

from ._common import _norm

_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _arxiv_url(search_query: str, start: int, max_results: int,
               sort_by: str = "relevance", sort_order: str = "descending") -> str:
    params = {
        "search_query": search_query,
        "start": int(start),
        "max_results": int(max_results),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    return f"{_ARXIV_API}?{urlencode(params)}"


def _parse_atom(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", _ATOM_NS)

    out: list[dict] = []
    for e in entries:
        arxiv_id_url = (e.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
        arxiv_id = arxiv_id_url.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "").strip()
        if not arxiv_id:
            continue

        title = _norm(e.findtext("atom:title", default="", namespaces=_ATOM_NS))
        abstract = _norm(e.findtext("atom:summary", default="", namespaces=_ATOM_NS))
        if not title or not abstract:
            continue

        authors = []
        for a in e.findall("atom:author", _ATOM_NS):
            name = (a.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
            if name:
                authors.append(name)

        published = (e.findtext("atom:published", default="", namespaces=_ATOM_NS) or "").strip()
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else 0

        url = arxiv_id_url
        for link in e.findall("atom:link", _ATOM_NS):
            if link.attrib.get("rel") == "alternate" and link.attrib.get("href"):
                url = link.attrib["href"]
                break

        out.append({
            "paper_id": f"arxiv:{arxiv_id}",
            "title": title,
            "abstract": abstract,
            "url": url,
            "year": year,
            "authors": authors,
            "score_bm25": 0.0,
            "venue": "arXiv preprint",
            "source": "arxiv",
            "full_text_sections": {},
        })

    return out


def arxiv_api_call(search_query: str, max_total: int, page_size: int, max_pages: int) -> list[dict]:
    raw: list[dict] = []
    for page in range(max_pages):
        start = page * page_size
        if start >= max_total:
            break

        url = _arxiv_url(
            search_query=search_query,
            start=start,
            max_results=min(page_size, max_total - start),
        )

        last_error = None
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 429:
                    wait = 3 * (attempt + 1)
                    print(f"  [arxiv] 429 rate limit → {wait}s 대기 후 재시도 ({attempt + 1}/3)")
                    print(f"  [arxiv] 응답 body: {r.text[:200]}")  # ← 이거 추가
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                batch = _parse_atom(r.text)
                raw.extend(batch)
                if not batch:
                    return list({p["paper_id"]: p for p in raw}.values())
                time.sleep(3)  # arXiv 권장 간격
                last_error = None
                break
            except Exception as e:
                last_error = e
                time.sleep(3)
        if last_error:
            raise last_error

    uniq = {p["paper_id"]: p for p in raw}
    return list(uniq.values())
