"""ScienceON(KISTI) 논문 · 특허 · 보고서 검색."""
from __future__ import annotations

from typing import Optional
import base64
import json
import re
import datetime
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

from Crypto.Cipher import AES

from ._common import _norm

_SCIENCEON_OPENAPI = "https://apigateway.kisti.re.kr/openapicall.do"
_SCIENCEON_TOKEN_API = "https://apigateway.kisti.re.kr/tokenrequest.do"
_SCIENCEON_AES_IV = "jvHJ1EFA0IXBrxxz"
_SCIENCEON_TOKEN_CACHE: dict[str, Optional[str]] = {
    "access_token": None,
    "refresh_token": None,
}


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
