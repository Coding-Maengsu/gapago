from __future__ import annotations

from typing import Optional, List
import base64
import datetime
import json
import re
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from pydantic import BaseModel, Field

from Crypto.Cipher import AES
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.retrievers import ArxivRetriever
from rank_bm25 import BM25Okapi

from config import Configuration
from utils.tavily import TavilySearch


_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_SCIENCEON_OPENAPI = "https://apigateway.kisti.re.kr/openapicall.do"
_SCIENCEON_TOKEN_API = "https://apigateway.kisti.re.kr/tokenrequest.do"
_SCIENCEON_AES_IV = "jvHJ1EFA0IXBrxxz"
_SCIENCEON_TOKEN_CACHE: dict[str, Optional[str]] = {
    "access_token": None,
    "refresh_token": None,
}


# =====================================================================
# YearFilteredArxivRetriever: 특정 연도 논문만 필터링
# =====================================================================
class YearFilteredArxivRetriever(BaseRetriever):
    """arXiv 검색 결과를 특정 연도로 필터링하는 Retriever"""
    year: int
    load_max_docs: int = 20

    def _get_relevant_documents(self, query: str) -> List[Document]:
        retriever = ArxivRetriever(load_max_docs=self.load_max_docs)
        docs = retriever.invoke(query)
        return [
            doc for doc in docs
            if datetime.datetime.strptime(doc.metadata["Published"], "%Y-%m-%d").year == self.year
        ]


def _norm(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return _norm(text).lower().split()


def _safe_json_loads(text: str) -> dict | list | None:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text or "")
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


# ============================== arXiv ==============================
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


def arxiv_api_call(search_query: str, max_total: int, page_size: int, max_pages: int, year_filter: str = "") -> list[dict]:
    """
    arXiv API 호출 (연도 필터 지원).
    year_filter가 단일 연도(e.g. "2024")인 경우 YearFilteredArxivRetriever 사용.
    """
    # year_filter가 단일 연도인 경우 YearFilteredArxivRetriever 사용
    if year_filter and year_filter.isdigit():
        try:
            year = int(year_filter)
            print(f"  [arxiv] Using YearFilteredArxivRetriever for year={year}")
            retriever = YearFilteredArxivRetriever(year=year, load_max_docs=max_total)
            docs = retriever.invoke(search_query)

            results = []
            for doc in docs:
                metadata = doc.metadata
                arxiv_id = metadata.get("entry_id", "").split("/")[-1] if metadata.get("entry_id") else ""

                results.append({
                    "paper_id": f"arxiv:{arxiv_id}",
                    "title": _norm(metadata.get("Title", "")),
                    "abstract": _norm(doc.page_content or metadata.get("Summary", "")),
                    "url": metadata.get("entry_id", ""),
                    "year": datetime.datetime.strptime(metadata["Published"], "%Y-%m-%d").year,
                    "authors": metadata.get("Authors", "").split(", ") if metadata.get("Authors") else [],
                    "score_bm25": 0.0,
                    "venue": "arXiv preprint",
                    "source": "arxiv",
                    "full_text_sections": {},
                })

            print(f"  [arxiv] YearFilteredArxivRetriever returned {len(results)} papers")
            return results
        except Exception as e:
            print(f"  [arxiv] YearFilteredArxivRetriever failed: {e}, falling back to standard API")

    # 기존 방식 (API 직접 호출)
    raw: list[dict] = []
    for page in range(max_pages):
        start = page * page_size
        if start >= max_total:
            break

        # 모든 요청 전 5초 대기 (arXiv rate limit 준수 - 첫 페이지 포함)
        time.sleep(5)

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
                    wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                    print(f"  [arxiv] 429 rate limit → {wait}s 대기 후 재시도 ({attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                batch = _parse_atom(r.text)
                raw.extend(batch)
                if not batch:
                    return list({p["paper_id"]: p for p in raw}.values())
                last_error = None
                break
            except Exception as e:
                last_error = e
                time.sleep(3)
        if last_error:
            # 페이지 실패 시: 이전 페이지 결과라도 반환 (전체 실패보다 나음)
            print(f"  [arxiv] 페이지 {page+1} 실패, 이전까지 {len(raw)}개 논문 반환")
            break  # raise 대신 break로 변경

    uniq = {p["paper_id"]: p for p in raw}
    return list(uniq.values())


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


# =========================== ScienceON Helpers ==========================
def _scienceon_pad_pkcs7(text: str, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(text.encode("utf-8")) % block_size)
    return (text + chr(pad_len) * pad_len).encode("utf-8")


def _scienceon_encrypt_accounts(mac_address: str, key: str) -> str:
    timestamp = "".join(re.findall(r"\d", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    plain_txt = json.dumps({"datetime": timestamp, "mac_address": mac_address}, separators=(",", ":"), ensure_ascii=False)
    cipher = AES.new(key.encode("utf-8"), AES.MODE_CBC, _SCIENCEON_AES_IV.encode("utf-8"))
    encrypted_bytes = cipher.encrypt(_scienceon_pad_pkcs7(plain_txt))
    return base64.urlsafe_b64encode(encrypted_bytes).decode("utf-8")


def _scienceon_request_create_token(client_id: str, mac_address: str, key: str, timeout: int = 30) -> dict:
    encrypted_txt = _scienceon_encrypt_accounts(mac_address=mac_address, key=key)
    response = requests.get(
        _SCIENCEON_TOKEN_API,
        params={"client_id": client_id, "accounts": encrypted_txt},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def _scienceon_request_access_token(client_id: str, refresh_token: str, timeout: int = 30) -> dict:
    response = requests.get(
        _SCIENCEON_TOKEN_API,
        params={"refreshToken": refresh_token, "client_id": client_id},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def _scienceon_resolve_tokens(*, client_id: Optional[str], mac_address: Optional[str], key: Optional[str], timeout: int = 30) -> dict:
    access_token = _SCIENCEON_TOKEN_CACHE.get("access_token")
    refresh_token = _SCIENCEON_TOKEN_CACHE.get("refresh_token")
    events: list[str] = []

    if access_token:
        return {"access_token": access_token, "refresh_token": refresh_token, "events": events}

    if client_id and refresh_token:
        refreshed = _scienceon_request_access_token(client_id=client_id, refresh_token=refresh_token, timeout=timeout)
        new_access = refreshed.get("access_token")
        new_refresh = refreshed.get("refresh_token") or refresh_token
        if new_access:
            _SCIENCEON_TOKEN_CACHE["access_token"] = new_access
            _SCIENCEON_TOKEN_CACHE["refresh_token"] = new_refresh
            events.append("access_token_reissued_from_refresh_token")
            return {"access_token": new_access, "refresh_token": new_refresh, "events": events}

    if client_id and mac_address and key:
        created = _scienceon_request_create_token(client_id=client_id, mac_address=mac_address, key=key, timeout=timeout)
        new_access = created.get("access_token")
        new_refresh = created.get("refresh_token")
        if new_access:
            _SCIENCEON_TOKEN_CACHE["access_token"] = new_access
            _SCIENCEON_TOKEN_CACHE["refresh_token"] = new_refresh
            events.append("refresh_and_access_token_issued")
            return {"access_token": new_access, "refresh_token": new_refresh, "events": events}

    return {"access_token": None, "refresh_token": refresh_token, "events": events}


def _scienceon_item_values(record: ET.Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in record.findall("./item"):
        meta = item.attrib.get("metaCode")
        if meta:
            values[meta] = _norm(item.text or "")
    return values


def _scienceon_parse_search_xml(xml_text: str, target: str) -> dict:
    root = ET.fromstring(xml_text)

    total_count_text = _norm(root.findtext("./resultSummary/TotalCount", default="0"))
    service_datatype = _norm(root.findtext("./resultSummary/serviceDatatype", default=""))
    status_code = _norm(root.findtext("./resultSummary/statusCode", default=""))

    results = []
    for idx, record in enumerate(root.findall("./recordList/record"), start=1):
        values = _scienceon_item_values(record)
        cn = values.get("CN") or values.get("ArticleId") or f"{target}_{idx}"
        title = values.get("Title") or values.get("Title2") or ""
        abstract = values.get("Abstract") or values.get("Abstract2") or ""
        authors_raw = values.get("Author") or values.get("Author2") or ""
        authors = [a.strip() for a in re.split(r"[;|]", authors_raw) if a.strip()]
        year_text = values.get("Pubyear") or values.get("Pubdate") or ""
        year = int(year_text[:4]) if len(year_text) >= 4 and year_text[:4].isdigit() else 0
        url = values.get("FulltextURL") or values.get("ContentURL") or values.get("MobileURL") or values.get("DOI") or ""

        results.append({
            "paper_id": f"scienceon:{cn}",
            "title": title,
            "abstract": abstract,
            "url": url,
            "year": year,
            "authors": authors,
            "score_bm25": 0.0,
            "venue": values.get("JournalName", ""),
            "source": "scienceon",
            "full_text_sections": {},
            "journal": values.get("JournalName", ""),
            "publisher": values.get("Publisher", ""),
            "doi": values.get("DOI", ""),
            "keywords_text": values.get("Keyword") or values.get("Keyword2") or "",
            "content_url": values.get("ContentURL", ""),
            "fulltext_url": values.get("FulltextURL", ""),
            "raw": values,
        })

    return {
        "source": "scienceon",
        "target": target,
        "total_count": int(total_count_text) if total_count_text.isdigit() else 0,
        "service_datatype": service_datatype,
        "status_code": status_code,
        "results": results,
    }


def _scienceon_search_query(query: str, year: str = "") -> str:
    """Build ScienceON searchQuery JSON with optional PY (year) filter."""
    sq = {"BI": _norm(query)}
    if year and "-" in year:
        parts = year.split("-")
        from_year = parts[0].strip()
        to_year = parts[1].strip() if parts[1].strip() else "2026"
        sq["PY"] = f"{from_year}~{to_year}"
    return json.dumps(sq, ensure_ascii=False, separators=(",", ":"))


def scienceon_search(*, client_id: str, query: str, target: str = "ARTI", cur_page: int = 1, row_count: int = 10, mac_address: Optional[str] = None, key: Optional[str] = None, timeout: int = 30, year: str = "") -> dict:
    token_state = _scienceon_resolve_tokens(
        client_id=client_id,
        mac_address=mac_address,
        key=key,
        timeout=timeout,
    )
    access_token = token_state.get("access_token")
    refresh_token = token_state.get("refresh_token")
    events = list(token_state.get("events", []))

    if not access_token:
        raise RuntimeError(
            "ScienceON token is not available. Set SCIENCEON_CLIENT_ID and provide SCIENCEON_MAC_ADDRESS + SCIENCEON_KEY so the tool can issue a token at call time."
        )

    params = {
        "client_id": client_id,
        "token": access_token,
        "version": "1.0",
        "action": "search",
        "target": target,
        "searchQuery": _scienceon_search_query(query, year),
        "curPage": int(cur_page),
        "rowCount": int(row_count),
    }

    response = requests.get(_SCIENCEON_OPENAPI, params=params, timeout=timeout)
    response.raise_for_status()
    parsed = _scienceon_parse_search_xml(response.text, target=target)

    if parsed.get("status_code") != "200" and refresh_token:
        refreshed = _scienceon_request_access_token(client_id=client_id, refresh_token=refresh_token, timeout=timeout)
        new_access = refreshed.get("access_token")
        new_refresh = refreshed.get("refresh_token") or refresh_token
        if new_access:
            _SCIENCEON_TOKEN_CACHE["access_token"] = new_access
            _SCIENCEON_TOKEN_CACHE["refresh_token"] = new_refresh
            params["token"] = new_access
            response = requests.get(_SCIENCEON_OPENAPI, params=params, timeout=timeout)
            response.raise_for_status()
            parsed = _scienceon_parse_search_xml(response.text, target=target)
            events.append("access_token_reissued_after_non200_response")

    parsed["query"] = query
    parsed["cur_page"] = int(cur_page)
    parsed["row_count"] = int(row_count)
    parsed["token_events"] = events
    return parsed


def _scienceon_parse_patent_xml(xml_text: str) -> dict:
    """Parse ScienceON PATENT search XML into structured results."""
    root = ET.fromstring(xml_text)
    total_count_text = _norm(root.findtext("./resultSummary/TotalCount", default="0"))
    status_code = _norm(root.findtext("./resultSummary/statusCode", default=""))

    results = []
    for idx, record in enumerate(root.findall("./recordList/record"), start=1):
        values = _scienceon_item_values(record)
        cn = values.get("CN") or f"patent_{idx}"
        title = values.get("Title") or ""
        abstract = values.get("Abstract") or ""
        applicants = values.get("Applicants") or values.get("Applicant") or ""
        ipc = values.get("IPC") or ""
        nation = values.get("Nation") or ""
        appl_date = values.get("ApplDate") or ""
        publ_date = values.get("PublDate") or ""
        grant_date = values.get("GrantDate") or ""
        url = values.get("ContentURL") or ""

        year_text = publ_date or appl_date or ""
        year = int(year_text[:4]) if len(year_text) >= 4 and year_text[:4].isdigit() else 0

        results.append({
            "patent_id": f"scienceon_patent:{cn}",
            "title": title,
            "abstract": abstract,
            "url": url,
            "year": year,
            "applicants": applicants,
            "ipc": ipc,
            "nation": nation,
            "appl_date": appl_date,
            "publ_date": publ_date,
            "grant_date": grant_date,
            "source": "scienceon_patent",
            "raw": values,
        })

    return {
        "source": "scienceon_patent",
        "target": "PATENT",
        "total_count": int(total_count_text) if total_count_text.isdigit() else 0,
        "status_code": status_code,
        "results": results,
    }


def _scienceon_parse_report_xml(xml_text: str) -> dict:
    """Parse ScienceON REPORT search XML into structured results."""
    root = ET.fromstring(xml_text)
    total_count_text = _norm(root.findtext("./resultSummary/TotalCount", default="0"))
    status_code = _norm(root.findtext("./resultSummary/statusCode", default=""))

    results = []
    for idx, record in enumerate(root.findall("./recordList/record"), start=1):
        values = _scienceon_item_values(record)
        cn = values.get("CN") or f"report_{idx}"
        title = values.get("Title") or ""
        abstract = values.get("Abstract") or ""
        authors_raw = values.get("Author") or ""
        authors = [a.strip() for a in re.split(r"[;|]", authors_raw) if a.strip()]
        publisher = values.get("Publisher") or ""
        keywords = values.get("Keyword") or ""
        year_text = values.get("Pubyear") or values.get("Pubdate") or ""
        year = int(year_text[:4]) if len(year_text) >= 4 and year_text[:4].isdigit() else 0
        url = values.get("FulltextURL") or values.get("ContentURL") or ""

        results.append({
            "report_id": f"scienceon_report:{cn}",
            "title": title,
            "abstract": abstract,
            "url": url,
            "year": year,
            "authors": authors,
            "publisher": publisher,
            "keywords": keywords,
            "source": "scienceon_report",
            "raw": values,
        })

    return {
        "source": "scienceon_report",
        "target": "REPORT",
        "total_count": int(total_count_text) if total_count_text.isdigit() else 0,
        "status_code": status_code,
        "results": results,
    }


def scienceon_patent_search(*, client_id: str, query: str, cur_page: int = 1, row_count: int = 10,
                            mac_address: Optional[str] = None, key: Optional[str] = None, timeout: int = 30, year: str = "") -> dict:
    """Search ScienceON for patents (target=PATENT)."""
    token_state = _scienceon_resolve_tokens(client_id=client_id, mac_address=mac_address, key=key, timeout=timeout)
    access_token = token_state.get("access_token")
    refresh_token = token_state.get("refresh_token")
    events = list(token_state.get("events", []))

    if not access_token:
        raise RuntimeError("ScienceON token is not available. Set SCIENCEON_CLIENT_ID and provide SCIENCEON_MAC_ADDRESS + SCIENCEON_KEY.")

    params = {
        "client_id": client_id, "token": access_token, "version": "1.0",
        "action": "search", "target": "PATENT",
        "searchQuery": _scienceon_search_query(query, year),
        "curPage": int(cur_page), "rowCount": int(row_count),
    }

    response = requests.get(_SCIENCEON_OPENAPI, params=params, timeout=timeout)
    response.raise_for_status()
    parsed = _scienceon_parse_patent_xml(response.text)

    if parsed.get("status_code") != "200" and refresh_token:
        refreshed = _scienceon_request_access_token(client_id=client_id, refresh_token=refresh_token, timeout=timeout)
        new_access = refreshed.get("access_token")
        if new_access:
            _SCIENCEON_TOKEN_CACHE["access_token"] = new_access
            _SCIENCEON_TOKEN_CACHE["refresh_token"] = refreshed.get("refresh_token") or refresh_token
            params["token"] = new_access
            response = requests.get(_SCIENCEON_OPENAPI, params=params, timeout=timeout)
            response.raise_for_status()
            parsed = _scienceon_parse_patent_xml(response.text)
            events.append("access_token_reissued_after_non200_response")

    parsed["query"] = query
    parsed["cur_page"] = int(cur_page)
    parsed["row_count"] = int(row_count)
    parsed["token_events"] = events
    return parsed


def scienceon_report_search(*, client_id: str, query: str, cur_page: int = 1, row_count: int = 10,
                            mac_address: Optional[str] = None, key: Optional[str] = None, timeout: int = 30, year: str = "") -> dict:
    """Search ScienceON for national R&D reports (target=REPORT)."""
    token_state = _scienceon_resolve_tokens(client_id=client_id, mac_address=mac_address, key=key, timeout=timeout)
    access_token = token_state.get("access_token")
    refresh_token = token_state.get("refresh_token")
    events = list(token_state.get("events", []))

    if not access_token:
        raise RuntimeError("ScienceON token is not available. Set SCIENCEON_CLIENT_ID and provide SCIENCEON_MAC_ADDRESS + SCIENCEON_KEY.")

    params = {
        "client_id": client_id, "token": access_token, "version": "1.0",
        "action": "search", "target": "REPORT",
        "searchQuery": _scienceon_search_query(query, year),
        "curPage": int(cur_page), "rowCount": int(row_count),
    }

    response = requests.get(_SCIENCEON_OPENAPI, params=params, timeout=timeout)
    response.raise_for_status()
    parsed = _scienceon_parse_report_xml(response.text)

    if parsed.get("status_code") != "200" and refresh_token:
        refreshed = _scienceon_request_access_token(client_id=client_id, refresh_token=refresh_token, timeout=timeout)
        new_access = refreshed.get("access_token")
        if new_access:
            _SCIENCEON_TOKEN_CACHE["access_token"] = new_access
            _SCIENCEON_TOKEN_CACHE["refresh_token"] = refreshed.get("refresh_token") or refresh_token
            params["token"] = new_access
            response = requests.get(_SCIENCEON_OPENAPI, params=params, timeout=timeout)
            response.raise_for_status()
            parsed = _scienceon_parse_report_xml(response.text)
            events.append("access_token_reissued_after_non200_response")

    parsed["query"] = query
    parsed["cur_page"] = int(cur_page)
    parsed["row_count"] = int(row_count)
    parsed["token_events"] = events
    return parsed


# =========================== Semantic Scholar ===========================
_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_S2_FIELDS = "paperId,title,abstract,year,authors,url,externalIds,venue,publicationVenue"


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
                    "full_text_sections": {},
                    "doi": doi,
                })
            last_error = None
            break
        except Exception as e:
            last_error = e
            time.sleep(2)
    if last_error:
        raise last_error
    return papers


# =========================== OpenAlex ===========================
_OPENALEX_API = "https://api.openalex.org/works"


def openalex_search(query: str, per_page: int = 20, year: str = "") -> list[dict]:
    """OpenAlex API로 논문 검색. API key 불필요."""
    params = {
        "search": query,
        "per_page": min(per_page, 200),
        "select": "id,title,publication_year,doi,authorships,abstract_inverted_index,primary_location",
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
            "full_text_sections": {},
            "doi": doi,
        })
    return papers


# =========================== Tool APIs ==========================
class ArxivApiCallInput(BaseModel):
    search_query: str = Field(description="arXiv API search_query")
    max_total: int = Field(default=80, description="총 최대 결과 수")
    page_size: int = Field(default=40, description="페이지당 결과 수")
    max_pages: int = Field(default=3, description="최대 페이지 수")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026'). submittedDate 범위로 변환")


class WebSearchInput(BaseModel):
    query: str = Field(description="웹 검색 쿼리")


class SemanticScholarSearchInput(BaseModel):
    query: str = Field(description="Semantic Scholar 검색 쿼리")
    limit: int = Field(default=20, description="최대 결과 수 (max 100)")
    year: str = Field(default="", description="연도 필터 (e.g. '2020-2025', '2023-')")


class OpenAlexSearchInput(BaseModel):
    query: str = Field(description="OpenAlex 검색 쿼리")
    per_page: int = Field(default=20, description="최대 결과 수 (max 200)")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026')")


class ScienceOnSearchInput(BaseModel):
    query: str = Field(description="ScienceON 검색 쿼리")
    target: str = Field(default="ARTI", description="ScienceON target")
    cur_page: int = Field(default=1, description="현재 페이지 번호")
    row_count: int = Field(default=10, description="가져올 결과 수")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026')")


class ScienceOnPatentSearchInput(BaseModel):
    query: str = Field(description="ScienceON 특허 검색 쿼리")
    cur_page: int = Field(default=1, description="현재 페이지 번호")
    row_count: int = Field(default=10, description="가져올 결과 수")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026')")


class ScienceOnReportSearchInput(BaseModel):
    query: str = Field(description="ScienceON 국가 R&D 보고서 검색 쿼리")
    cur_page: int = Field(default=1, description="현재 페이지 번호")
    row_count: int = Field(default=10, description="가져올 결과 수")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026')")


def build_retrieval_tools(config: Optional[RunnableConfig] = None) -> List:
    """
    Retrieval Agent가 선택할 수 있는 외부 검색 툴만 노출한다.
    - web_search_tool
    - arxiv_api_call_tool
    - scienceon_search_tool (placeholder)
    """
    cfg = Configuration.from_runnable_config(config)
    tavily_tool = TavilySearch(max_results=cfg.tavily_max_results)

    @tool(args_schema=ArxivApiCallInput)
    def arxiv_api_call_tool(
        search_query: str,
        max_total: int = 80,
        page_size: int = 40,
        max_pages: int = 3,
        year: str = "",
    ) -> str:
        """Call arXiv API directly and return a paper list as JSON string. Use year param for date filtering (e.g. '2024' for single year or '2022-2026' for range)."""
        try:
            year_filter = ""
            # 단일 연도인 경우: YearFilteredArxivRetriever 사용
            if year and year.isdigit():
                year_filter = year
                print(f"  [arxiv_tool] Single year filter detected: {year}")
            # 연도 범위인 경우: submittedDate 필터 사용
            elif year and "-" in year:
                parts = year.split("-")
                from_year = parts[0].strip()
                to_year = parts[1].strip() if parts[1].strip() else "2026"
                date_filter = f" AND submittedDate:[{from_year}01010000 TO {to_year}12312359]"
                search_query = search_query + date_filter
                print(f"  [arxiv_tool] Year range filter: {from_year}-{to_year}")

            results = arxiv_api_call(
                search_query=search_query,
                max_total=max_total,
                page_size=page_size,
                max_pages=max_pages,
                year_filter=year_filter,
            )
            return json.dumps({
                "source": "arxiv",
                "query": search_query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Arxiv API call failed: {str(e)}</Error>"

    @tool(args_schema=WebSearchInput)
    def web_search_tool(query: str) -> str:
        """Search the web API and return results as JSON string."""
        try:
            results = tavily_tool.search(query)
            return json.dumps({
                "source": "web",
                "query": query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Web search failed: {str(e)}</Error>"

    @tool(args_schema=SemanticScholarSearchInput)
    def semantic_scholar_search_tool(query: str, limit: int = 20, year: str = "") -> str:
        """Search Semantic Scholar for academic papers. Returns papers with metadata. Good for finding highly-cited and cross-domain papers."""
        try:
            results = semantic_scholar_search(query=query, limit=limit, year=year)
            return json.dumps({
                "source": "semantic_scholar",
                "query": query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Semantic Scholar search failed: {str(e)}</Error>"

    @tool(args_schema=OpenAlexSearchInput)
    def openalex_search_tool(query: str, per_page: int = 20, year: str = "") -> str:
        """Search OpenAlex for academic papers. Covers 200M+ works across all disciplines. No API key needed. Use year param for filtering (e.g. '2022-2026')."""
        try:
            results = openalex_search(query=query, per_page=per_page, year=year)
            return json.dumps({
                "source": "openalex",
                "query": query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>OpenAlex search failed: {str(e)}</Error>"

    @tool(args_schema=ScienceOnSearchInput)
    def scienceon_search_tool(query: str, target: str = "ARTI", cur_page: int = 1, row_count: int = 10, year: str = "") -> str:
        """Search ScienceON paper records. Use year param for filtering (e.g. '2022-2026')."""
        if not cfg.scienceon_client_id:
            return "<Error>ScienceON client_id is not configured. Set SCIENCEON_CLIENT_ID.</Error>"
        try:
            result = scienceon_search(
                client_id=cfg.scienceon_client_id,
                query=query,
                target=target or cfg.scienceon_default_target,
                cur_page=cur_page,
                row_count=row_count or cfg.scienceon_default_row_count,
                mac_address=cfg.scienceon_mac_address,
                key=cfg.scienceon_key,
                year=year,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"<Error>ScienceON search failed: {str(e)}</Error>"

    @tool(args_schema=ScienceOnPatentSearchInput)
    def scienceon_patent_search_tool(query: str, cur_page: int = 1, row_count: int = 10, year: str = "") -> str:
        """Search ScienceON for Korean patents. Use year param for filtering (e.g. '2022-2026')."""
        if not cfg.scienceon_client_id:
            return "<Error>ScienceON client_id is not configured. Set SCIENCEON_CLIENT_ID.</Error>"
        try:
            result = scienceon_patent_search(
                client_id=cfg.scienceon_client_id,
                query=query,
                cur_page=cur_page,
                row_count=row_count or cfg.scienceon_default_row_count,
                mac_address=cfg.scienceon_mac_address,
                key=cfg.scienceon_key,
                year=year,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"<Error>ScienceON patent search failed: {str(e)}</Error>"

    @tool(args_schema=ScienceOnReportSearchInput)
    def scienceon_report_search_tool(query: str, cur_page: int = 1, row_count: int = 10, year: str = "") -> str:
        """Search ScienceON for Korean national R&D reports. Use year param for filtering (e.g. '2022-2026')."""
        if not cfg.scienceon_client_id:
            return "<Error>ScienceON client_id is not configured. Set SCIENCEON_CLIENT_ID.</Error>"
        try:
            result = scienceon_report_search(
                client_id=cfg.scienceon_client_id,
                query=query,
                cur_page=cur_page,
                row_count=row_count or cfg.scienceon_default_row_count,
                mac_address=cfg.scienceon_mac_address,
                key=cfg.scienceon_key,
                year=year,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"<Error>ScienceON report search failed: {str(e)}</Error>"

    return [
        web_search_tool,
        arxiv_api_call_tool,
        semantic_scholar_search_tool,
        openalex_search_tool,
        scienceon_search_tool,
        scienceon_patent_search_tool,
        scienceon_report_search_tool,
    ]


def build_role_tools(config: Optional[RunnableConfig] = None) -> dict:
    retrieval_tools = build_retrieval_tools(config)
    return {
        "QUERY_TOOLS": [],
        "RETRIEVAL_TOOLS": retrieval_tools,
        "LIMITATION_TOOLS": [],
        "GAP_INFER_TOOLS": [],
        "CRITIC_TOOLS": [],
        "RESPONSE_TOOLS": [],
    }
