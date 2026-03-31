# 3-3) Limitation Extract Agent
from __future__ import annotations

import re
import time
import threading
import xml.etree.ElementTree as ET
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import hashlib
import json
import os
from pathlib import Path

import requests
import pymupdf as fitz
import pymupdf4llm

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from states import AgentState, Paper, LimitationItem
from llm import get_llm
from utils.parse_json import parse_json
from utils.progress import report_progress
from utils.cancel import is_cancelled

# =====================================================================
# 섹션 키워드 정의
# =====================================================================
SECTION_KEYWORDS = {
    # Track 1 - 저자 기술
    "conclusion":   ["conclusion", "concluding remarks", "summary"],
    "limitations":  ["limitation", "limitations", "weakness", "weaknesses"],
    "future_work":  ["future work", "future research", "future direction"],
    # Track 2 - 구조 분석
    "introduction": ["introduction", "background"],
    "method":       ["method", "methods", "methodology", "approach", "proposed method",
                     "our approach", "our method", "model", "framework", "architecture"],
    "experiment":   ["experiment", "experiments", "experimental setup", "experimental results",
                     "evaluation", "results", "empirical evaluation", "benchmarks"],
    "discussion":   ["discussion", "analysis", "ablation", "ablation study"],
}

TRACK1_KEYS = {"conclusion", "limitations", "future_work"}
TRACK2_KEYS = {"introduction", "method", "experiment", "discussion"}

MAX_SECTION_CHARS = 3000  # 섹션별 최대 글자 수 (토큰 비용 제한)


def _clean_doi(doi: str) -> str:
    """Strip URL prefixes from DOI, returning bare DOI like '10.1109/xxx'."""
    return doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()


# PMC BioC section_type → SECTION_KEYWORDS 키 매핑
_PMC_SECTION_MAP = {
    "INTRO": "introduction",
    "METHODS": "method",
    "RESULTS": "experiment",
    "DISCUSS": "discussion",
    "CONCL": "conclusion",
}

# S2 API rate limiter (1 req/s without key)
_S2_RATE_LOCK = threading.Lock()
_S2_LAST_CALL = 0.0


# =====================================================================
# 섹션 분리 유틸
# =====================================================================
def _split_sections(full_text: str) -> dict:
    """전체 텍스트에서 섹션별로 분리. plain text 및 Markdown 헤딩 모두 지원."""
    _kw_alt = "|".join(
        re.escape(kw)
        for kws in SECTION_KEYWORDS.values()
        for kw in kws
    )
    heading_pattern = re.compile(
        # "3.1 Conclusion:", "## Conclusions", "Experimental Results and Discussion" 등
        # 키워드가 포함된 줄을 헤딩으로 인식 (키워드 뒤 추가 단어 허용)
        r"(?m)^(?:#{1,4}\s+)?(?:\d+[\.\d]*\.?\s+)?(?:[^\n]{0,30}?\b)("
        + _kw_alt
        + r")s?\b[^\n]{0,60}$",
        re.IGNORECASE,
    )

    matches = list(heading_pattern.finditer(full_text))
    if not matches:
        return {}

    sections = {}
    for i, match in enumerate(matches):
        h = match.group(0).lower().strip()
        # 너무 긴 줄은 헤딩이 아닐 가능성이 높음
        if len(h) > 100:
            continue
        key = None
        for k, kws in SECTION_KEYWORDS.items():
            if any(kw in h for kw in kws):
                key = k
                break
        if not key or key in sections:
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()
        if len(text) > 100:
            sections[key] = text[:MAX_SECTION_CHARS]

    return sections


# =====================================================================
# Full text 로드 (소스별 분기)
# =====================================================================
_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://scholar.google.com/",
}

# ── HTTP 연결 풀 (TCP 재사용) ──
_http_session = requests.Session()
_http_session.headers.update(_REQUEST_HEADERS)

# ── Full text 캐시 ──
_CACHE_DIR = Path(".cache/fulltext")
_CACHE_TTL_DAYS = 7


def _cache_key(paper_id: str) -> str:
    return hashlib.sha256(paper_id.encode()).hexdigest()[:16]


def _cache_get(paper_id: str) -> Optional[dict]:
    """캐시에서 full text 섹션 로드. TTL 초과 시 None."""
    path = _CACHE_DIR / f"{_cache_key(paper_id)}.json"
    if not path.exists():
        return None
    try:
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days > _CACHE_TTL_DAYS:
            path.unlink(missing_ok=True)
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def _cache_put(paper_id: str, sections: dict) -> None:
    """섹션 데이터를 캐시에 저장."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _CACHE_DIR / f"{_cache_key(paper_id)}.json"
        path.write_text(json.dumps(sections, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _extract_arxiv_id(paper: Paper) -> Optional[str]:
    """paper_id ('arxiv:2401.12345v1') 에서 순수 arXiv ID를 추출."""
    pid = paper.paper_id or ""
    if pid.lower().startswith("arxiv:"):
        pid = pid[len("arxiv:"):]
    pid = pid.strip()
    if not pid:
        return None
    if not re.match(r"^(\d{4}\.\d{4,5}|[\w\-]+\.?[\w\-]*/\d{7})", pid):
        return None
    pid = re.sub(r"v\d+$", "", pid)
    return pid if pid else None


def _load_arxiv_html(paper: Paper) -> dict:
    """ar5iv HTML 버전에서 텍스트 추출 (PDF 다운로드 불필요)."""
    arxiv_id = _extract_arxiv_id(paper)
    if not arxiv_id:
        return {}

    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        from bs4 import BeautifulSoup

        resp = _http_session.get(url, timeout=8)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # 불필요한 요소 제거
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # 본문 영역 추출
        article = soup.find("article") or soup.find("div", class_="ltx_page_content") or soup
        full_text = article.get_text(separator="\n", strip=True)

        if len(full_text) < 200:
            return {}

        sections = _split_sections(full_text)
        if sections:
            print(f"  [fulltext:ar5iv] '{paper.title[:50]}' → {list(sections.keys())}")
        return sections

    except Exception as e:
        print(f"  [fulltext:ar5iv] 실패: {paper.title[:50]} ({e})")
        return {}


def _load_arxiv_pdf(paper: Paper) -> dict:
    """arXiv PDF fallback (ArxivLoader 사용)."""
    arxiv_id = _extract_arxiv_id(paper)
    if not arxiv_id:
        return {}
    try:
        from langchain_community.document_loaders import ArxivLoader

        loader = ArxivLoader(query=arxiv_id, load_max_docs=1, load_all_available_meta=True)
        docs = loader.load()
        if not docs:
            return {}

        sections = _split_sections(docs[0].page_content)
        print(f"  [fulltext:arxiv-pdf] '{paper.title[:50]}' → {list(sections.keys())}")
        return sections

    except Exception as e:
        print(f"  [fulltext:arxiv-pdf] 로드 실패: {paper.title[:50]} ({e})")
        return {}


def _load_arxiv_full_text(paper: Paper) -> dict:
    """arXiv 논문 full text 로드: ar5iv HTML 우선, 실패 시 PDF fallback."""
    sections = _load_arxiv_html(paper)
    if sections:
        return sections
    return _load_arxiv_pdf(paper)


_MAX_PDF_BYTES = 10 * 1024 * 1024  # 10MB

def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """pymupdf4llm으로 PDF 바이트에서 Markdown 텍스트 추출 (섹션 헤딩 보존)."""
    if len(pdf_bytes) > _MAX_PDF_BYTES:
        print(f"  [fulltext] PDF too large ({len(pdf_bytes) // 1024 // 1024}MB), skipping")
        return ""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        text = pymupdf4llm.to_markdown(doc)
    except Exception:
        # fallback: plain text
        text = ""
        for page in doc:
            text += page.get_text()
    doc.close()
    return text


def _publisher_pdf_url(final_url: str) -> Optional[str]:
    """출판사별 PDF URL 패턴을 이용해 직접 PDF URL을 생성."""
    from urllib.parse import urlparse
    parsed = urlparse(final_url)
    host = parsed.hostname or ""
    path = parsed.path

    # MDPI — Cloudflare 보호로 requests 접근 불가. HTML fallback으로 전환
    if "mdpi.com" in host:
        return None

    # PeerJ (peerj.com/articles/cs-XXXX → .../articles/cs-XXXX.pdf 대신 XML)
    if "peerj.com" in host and path.startswith("/articles/"):
        # PeerJ는 .pdf 직접 접근이 400이므로 HTML 본문 추출로 전환
        return None

    # Springer — /content/pdf/ 는 HTML을 반환하는 경우가 많으므로 HTML fallback 우선
    if "springer" in host:
        return None

    # ScienceDirect / Elsevier
    if "sciencedirect.com" in host and "/science/article/" in path:
        # ScienceDirect는 PDF 직접 링크 생성이 어려움 → HTML fallback
        return None

    return None


def _find_pdf_url_from_doi(doi: str) -> Optional[str]:
    """DOI 페이지에 접근하여 PDF 다운로드 링크를 찾는다."""
    doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    try:
        resp = _http_session.get(doi_url, timeout=8, allow_redirects=True)
        resp.raise_for_status()

        # 리다이렉트된 최종 URL이 PDF인 경우
        if resp.headers.get("content-type", "").startswith("application/pdf"):
            return resp.url

        # 출판사별 패턴으로 PDF URL 생성 시도
        publisher_url = _publisher_pdf_url(resp.url)
        if publisher_url:
            return publisher_url

        # HTML 페이지에서 PDF 링크 추출
        pdf_patterns = [
            r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
            r'href=["\']([^"\']*\/pdf[^"\']*)["\']',
            r'content=["\']([^"\']*\.pdf[^"\']*)["\']',
        ]
        for pattern in pdf_patterns:
            matches = re.findall(pattern, resp.text, re.IGNORECASE)
            for url in matches:
                if not url.startswith("http"):
                    from urllib.parse import urljoin
                    url = urljoin(resp.url, url)
                return url

    except Exception as e:
        print(f"  [fulltext:doi] DOI 페이지 접근 실패: {doi} ({e})")
    return None


def _extract_fulltext_from_html(html: str, title: str = "") -> dict:
    """HTML 페이지에서 논문 본문 텍스트를 추출하여 섹션 분리."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # 불필요한 요소 제거
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside",
                                   "figure", "table", "sup", "math"]):
            tag.decompose()
        # 참고문헌 영역 제거 (불필요한 텍스트 줄임)
        for ref_section in soup.find_all(["section", "div"], id=re.compile(r"ref|bib|citation", re.I)):
            ref_section.decompose()
        for ref_section in soup.find_all(["section", "div"], class_=re.compile(r"ref|bib|citation", re.I)):
            ref_section.decompose()

        # 본문 영역 탐색 (출판사별 패턴)
        article = (
            # PeerJ
            soup.find("div", class_="article-content")
            # MDPI
            or soup.find("div", class_="html-body")
            # 일반 패턴
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"article|paper|fulltext|main-content", re.I))
            or soup.find("div", id=re.compile(r"article|paper|fulltext|main-content", re.I))
            or soup.find("main")
            or soup
        )

        full_text = article.get_text(separator="\n", strip=True)
        if len(full_text) < 500:
            return {}

        return _split_sections(full_text)
    except Exception:
        return {}


def _find_pdf_link_in_html(html: str, base_url: str) -> str:
    """HTML 페이지에서 실제 PDF 다운로드 링크를 찾는다."""
    from urllib.parse import urljoin
    pdf_patterns = [
        r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
        r'href=["\']([^"\']*\/pdf[^"\']*)["\']',
        r'content=["\']([^"\']*\.pdf[^"\']*)["\']',
    ]
    for pattern in pdf_patterns:
        matches = re.findall(pattern, html, re.IGNORECASE)
        for url in matches:
            if not url.startswith("http"):
                url = urljoin(base_url, url)
            return url
    return ""


_BLOCKED_PDF_HOSTS = {
    "api.elsevier.com",       # TDM API (키 필요, XML 반환)
    "www.mdpi.com",           # Cloudflare 차단
    "www.tandfonline.com",    # 봇 차단
    "www.igi-global.com",     # 봇 차단 / 429
    "peerj.com",              # .pdf 엔드포인트 불안정 (400), HTML 추출이 안정적
}

_UNPAYWALL_EMAIL = "gapago-research@university.edu"


# Unpaywall에서 건너뛸 호스트 (메타 페이지이거나 DOI 리다이렉트)
_UNPAYWALL_SKIP_HOSTS = {"doaj.org", "dx.doi.org", "doi.org"}


def _unpaywall_find_accessible_url(doi: str) -> Optional[str]:
    """Unpaywall API로 접근 가능한 OA URL을 찾는다. PMC > biorxiv > 기타 순."""
    if not doi:
        return None
    clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    try:
        r = _http_session.get(
            f"https://api.unpaywall.org/v2/{clean_doi}",
            params={"email": _UNPAYWALL_EMAIL},
            timeout=5,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        locations = data.get("oa_locations") or []

        # 1순위: PMC (봇 차단 없음, HTML 본문 풍부)
        for loc in locations:
            url = loc.get("url_for_pdf") or loc.get("url") or ""
            if "pmc" in url.lower() or "ncbi.nlm.nih.gov/pmc" in url.lower():
                return _normalize_oa_url(url)

        # 2순위: biorxiv/medrxiv (프리프린트, HTML 접근 가능)
        for loc in locations:
            url = loc.get("url_for_pdf") or loc.get("url") or ""
            if "biorxiv.org" in url or "medrxiv.org" in url:
                return _normalize_oa_url(url)

        # 3순위: 기타 repository (DOAJ, doi.org 리다이렉트 제외)
        for loc in locations:
            if loc.get("host_type") == "repository":
                url = loc.get("url_for_pdf") or loc.get("url") or ""
                if url:
                    from urllib.parse import urlparse
                    host = (urlparse(url).hostname or "").lower()
                    if host not in _UNPAYWALL_SKIP_HOSTS and host not in _BLOCKED_PDF_HOSTS:
                        return _normalize_oa_url(url)

        # 4순위: publisher (봇 차단될 수 있지만 시도)
        for loc in locations:
            url = loc.get("url_for_pdf") or loc.get("url") or ""
            if url:
                from urllib.parse import urlparse
                host = (urlparse(url).hostname or "").lower()
                if host not in _BLOCKED_PDF_HOSTS and host not in _UNPAYWALL_SKIP_HOSTS:
                    return url
    except Exception:
        pass
    return None


def _normalize_oa_url(url: str) -> str:
    """OA URL을 접근 가능한 형태로 정규화."""
    # PMC: 다양한 형식 → 통일된 HTML 페이지 URL
    m = re.search(r"ncbi\.nlm\.nih\.gov/pmc/articles/(?:PMC)?(\d+)", url)
    if m:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{m.group(1)}/"
    m = re.search(r"pmc\.ncbi\.nlm\.nih\.gov/articles/PMC(\d+)", url)
    if m:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{m.group(1)}/"
    m = re.search(r"europepmc\.org/article/\w+/PMC(\d+)", url)
    if m:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{m.group(1)}/"

    # biorxiv/medrxiv: PDF URL → HTML 페이지 URL (PDF 직접 접근 403)
    m = re.search(r"(biorxiv|medrxiv)\.org/content/.*?/(\d+)\.full\.pdf", url)
    if m:
        server = m.group(1)
        paper_num = m.group(2)
        return f"https://www.{server}.org/content/10.1101/{paper_num}v1"

    return url


def _is_usable_pdf_url(url: str) -> bool:
    """다운로드 시도할 가치가 있는 PDF URL인지 판별."""
    if not url:
        return False
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    if host in _BLOCKED_PDF_HOSTS:
        return False
    # Elsevier XML API 패턴
    if "api.elsevier.com" in url or "httpAccept=text/xml" in url:
        return False
    return True


def _try_doi_landing_html(doi: str, title: str) -> dict:
    """DOI 랜딩 페이지(출판사 HTML)에서 직접 본문 추출 시도."""
    if not doi:
        return {}
    doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    try:
        resp = _http_session.get(doi_url, timeout=10, allow_redirects=True)
        resp.raise_for_status()
        ct = (resp.headers.get("content-type") or "").lower()

        # PDF로 리다이렉트된 경우
        if resp.content[:4] == b"%PDF":
            full_text = _extract_text_from_pdf_bytes(resp.content)
            if len(full_text) >= 200:
                sections = _split_sections(full_text)
                if sections:
                    print(f"  [fulltext:doi-redirect-pdf] '{title[:50]}' → {list(sections.keys())}")
                    return sections

        if "html" in ct:
            sections = _extract_fulltext_from_html(resp.text, title)
            if sections:
                print(f"  [fulltext:doi-html] '{title[:50]}' → {list(sections.keys())}")
                return sections
    except Exception:
        pass
    return {}


# =====================================================================
# S2 / PMC / EuropePMC fallback 함수들
# =====================================================================
def _s2_discover_alt_ids(doi: str) -> dict:
    """S2 batch API로 DOI에 대한 ArXiv/PMC ID를 발견. {arxiv_id, pmcid, oa_pdf_url}."""
    global _S2_LAST_CALL
    if not doi:
        return {}
    # Rate limit: 1 req/s
    with _S2_RATE_LOCK:
        elapsed = time.time() - _S2_LAST_CALL
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        _S2_LAST_CALL = time.time()
    try:
        resp = _http_session.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": "externalIds,openAccessPdf"},
            json={"ids": [f"DOI:{_clean_doi(doi)}"]},
            timeout=10,
        )
        if resp.status_code == 429:
            time.sleep(3)
            return {}
        resp.raise_for_status()
        data = resp.json()
        if not data or not data[0]:
            return {}
        paper_data = data[0]
        ext = paper_data.get("externalIds") or {}
        oa_pdf = (paper_data.get("openAccessPdf") or {}).get("url", "")
        return {
            "arxiv_id": ext.get("ArXiv", ""),
            "pmcid": ext.get("PubMedCentral", ""),
            "oa_pdf_url": oa_pdf,
        }
    except Exception:
        return {}


def _doi_to_pmcid(doi: str) -> Optional[str]:
    """NCBI ID Converter API로 DOI → PMCID 변환."""
    if not doi:
        return None
    try:
        resp = _http_session.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
            params={"ids": _clean_doi(doi), "format": "json"},
            timeout=5,
        )
        resp.raise_for_status()
        records = resp.json().get("records", [])
        if records and records[0].get("pmcid"):
            return records[0]["pmcid"]
    except Exception:
        pass
    return None


def _load_pmc_bioc_full_text(pmcid: str, title: str = "") -> dict:
    """PMC BioC API에서 구조화된 full text를 가져와 섹션으로 분리."""
    if not pmcid:
        return {}
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    try:
        resp = _http_session.get(
            f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode",
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # API returns either a list of collections or a single object
        if isinstance(data, list):
            data = data[0] if data else {}
        docs = data.get("documents", [])
        if not docs:
            return {}

        # passage별로 섹션 분류 후 텍스트 합치기
        section_texts: dict[str, list[str]] = {}
        for passage in docs[0].get("passages", []):
            infons = passage.get("infons", {})
            sec_type = infons.get("section_type", "")
            sec_heading = (infons.get("section", "") or "").lower()
            text = passage.get("text", "")
            if not text or len(text) < 20:
                continue

            # 1순위: section_type 매핑
            key = _PMC_SECTION_MAP.get(sec_type)

            # 2순위: 헤딩 텍스트로 SECTION_KEYWORDS 매칭
            if not key:
                for k, kws in SECTION_KEYWORDS.items():
                    if any(kw in sec_heading for kw in kws):
                        key = k
                        break

            if key:
                section_texts.setdefault(key, []).append(text)

        # 섹션별 텍스트 합치기 + 길이 제한
        sections = {}
        for key, texts in section_texts.items():
            combined = "\n".join(texts)
            if len(combined) > 100:
                sections[key] = combined[:MAX_SECTION_CHARS]

        if sections:
            print(f"  [fulltext:pmc-bioc] '{title[:50]}' → {list(sections.keys())}")
        return sections
    except Exception:
        return {}


def _load_europepmc_full_text(doi: str, title: str = "") -> dict:
    """EuropePMC에서 JATS XML full text를 가져와 섹션으로 분리."""
    if not doi:
        return {}
    clean = _clean_doi(doi)
    try:
        # Step 1: PMCID 검색
        resp = _http_session.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": f"DOI:{clean}", "format": "json", "resultType": "core"},
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])
        pmcid = None
        for r in results:
            if r.get("pmcid"):
                pmcid = r["pmcid"]
                break
        if not pmcid:
            return {}

        # Step 2: Full text XML
        xml_resp = _http_session.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
            timeout=15,
        )
        xml_resp.raise_for_status()

        # Step 3: JATS XML 파싱
        root = ET.fromstring(xml_resp.content)
        body = root.find(".//body")
        if body is None:
            return {}

        sections = {}
        for sec in body.iter("sec"):
            # sec-type 속성 또는 title 텍스트에서 섹션 키 결정
            sec_type = (sec.get("sec-type") or "").lower()
            title_el = sec.find("title")
            heading = (title_el.text or "").lower().strip() if title_el is not None else ""

            key = None
            # sec-type 매핑
            for kw, mapped_key in [
                ("intro", "introduction"), ("method", "method"), ("material", "method"),
                ("result", "experiment"), ("discuss", "discussion"),
                ("conclusion", "conclusion"), ("concl", "conclusion"),
            ]:
                if kw in sec_type:
                    key = mapped_key
                    break

            # heading 텍스트 매핑
            if not key:
                for k, kws in SECTION_KEYWORDS.items():
                    if any(kw in heading for kw in kws):
                        key = k
                        break

            if not key or key in sections:
                continue

            # 섹션 내 모든 텍스트 추출
            text = "".join(sec.itertext()).strip()
            if len(text) > 100:
                sections[key] = text[:MAX_SECTION_CHARS]

        if sections:
            print(f"  [fulltext:europepmc] '{title[:50]}' → {list(sections.keys())}")
        return sections
    except Exception:
        return {}


def _load_doi_full_text(paper: Paper) -> dict:
    """DOI를 통해 논문 full text를 로드. PDF → HTML → DOI 랜딩 순으로 시도."""
    # DOI 확보
    doi = ""
    if hasattr(paper, "full_text_sections") and isinstance(paper.full_text_sections, dict):
        doi = paper.full_text_sections.get("doi", "")
    if not doi:
        if paper.url and ("doi.org" in paper.url or "dx.doi.org" in paper.url):
            doi = paper.url

    # 1순위: API 응답에서 이미 확보된 pdf_url (S2 openAccessPdf, OpenAlex best_oa_location, Crossref link)
    pdf_url = ""
    if hasattr(paper, "full_text_sections") and isinstance(paper.full_text_sections, dict):
        pdf_url = paper.full_text_sections.get("pdf_url", "")

    # 쓸모없는 URL 필터링
    if not _is_usable_pdf_url(pdf_url):
        pdf_url = ""

    # 2순위: DOI 페이지 크롤링으로 PDF URL 탐색
    if not pdf_url and doi:
        pdf_url = _find_pdf_url_from_doi(doi)
        if not _is_usable_pdf_url(pdf_url):
            pdf_url = ""

    # PDF URL이 있으면 다운로드 시도
    if pdf_url:
        try:
            resp = _http_session.get(pdf_url, timeout=15)
            resp.raise_for_status()

            content_type = (resp.headers.get("content-type") or "").lower()

            # PDF 응답
            if resp.content[:4] == b"%PDF":
                full_text = _extract_text_from_pdf_bytes(resp.content)
                if len(full_text) >= 200:
                    sections = _split_sections(full_text)
                    print(f"  [fulltext:doi-pdf] '{paper.title[:50]}' → {list(sections.keys())}")
                    return sections

            # HTML 응답
            if "html" in content_type:
                sections = _extract_fulltext_from_html(resp.text, paper.title)
                if sections:
                    print(f"  [fulltext:html] '{paper.title[:50]}' → {list(sections.keys())}")
                    return sections
                # HTML에서 실제 PDF 링크를 찾아 재시도
                real_pdf = _find_pdf_link_in_html(resp.text, resp.url)
                if real_pdf and _is_usable_pdf_url(real_pdf):
                    try:
                        pdf_resp = _http_session.get(real_pdf, timeout=15)
                        pdf_resp.raise_for_status()
                        if pdf_resp.content[:4] == b"%PDF":
                            full_text = _extract_text_from_pdf_bytes(pdf_resp.content)
                            if len(full_text) >= 200:
                                sections = _split_sections(full_text)
                                print(f"  [fulltext:doi→pdf] '{paper.title[:50]}' → {list(sections.keys())}")
                                return sections
                    except Exception:
                        pass

        except Exception:
            pass  # PDF 실패 → DOI 랜딩 HTML fallback으로 진행

    # fallback 2: DOI 랜딩 페이지 HTML에서 직접 본문 추출
    sections = _try_doi_landing_html(doi, paper.title)
    if sections:
        return sections

    # fallback 3: Unpaywall API로 접근 가능한 OA 소스(PMC 등) 탐색
    if doi:
        oa_url = _unpaywall_find_accessible_url(doi)
        if oa_url:
            try:
                resp = _http_session.get(oa_url, timeout=15, allow_redirects=True)
                resp.raise_for_status()
                if resp.content[:4] == b"%PDF":
                    full_text = _extract_text_from_pdf_bytes(resp.content)
                    if len(full_text) >= 200:
                        sections = _split_sections(full_text)
                        if sections:
                            print(f"  [fulltext:unpaywall-pdf] '{paper.title[:50]}' → {list(sections.keys())}")
                            return sections
                ct = (resp.headers.get("content-type") or "").lower()
                if "html" in ct:
                    sections = _extract_fulltext_from_html(resp.text, paper.title)
                    if sections:
                        print(f"  [fulltext:unpaywall-html] '{paper.title[:50]}' → {list(sections.keys())}")
                        return sections
            except Exception:
                pass

    # fallback 4: Semantic Scholar Batch API → arXiv / PMC 경유
    if doi:
        try:
            alt = _s2_discover_alt_ids(doi)
            # S2에서 arXiv ID 발견 → ar5iv HTML로 full text 로드
            arxiv_id = alt.get("arxiv_id")
            if arxiv_id:
                from langchain_community.document_loaders import ArxivLoader
                ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
                try:
                    r = _http_session.get(ar5iv_url, timeout=15)
                    r.raise_for_status()
                    if "html" in (r.headers.get("content-type") or ""):
                        sections = _extract_fulltext_from_html(r.text, paper.title)
                        if sections:
                            print(f"  [fulltext:s2→ar5iv] '{paper.title[:50]}' → {list(sections.keys())}")
                            return sections
                except Exception:
                    pass
            # S2에서 PMCID 발견 → PMC BioC API
            pmcid = alt.get("pmcid")
            if pmcid:
                sections = _load_pmc_bioc_full_text(pmcid, paper.title)
                if sections:
                    print(f"  [fulltext:s2→pmc-bioc] '{paper.title[:50]}' → {list(sections.keys())}")
                    return sections
            # S2에서 OA PDF URL 발견
            s2_pdf = alt.get("oa_pdf_url")
            if s2_pdf and _is_usable_pdf_url(s2_pdf):
                try:
                    resp = _http_session.get(s2_pdf, timeout=15)
                    resp.raise_for_status()
                    if resp.content[:4] == b"%PDF":
                        full_text = _extract_text_from_pdf_bytes(resp.content)
                        if len(full_text) >= 200:
                            sections = _split_sections(full_text)
                            if sections:
                                print(f"  [fulltext:s2→pdf] '{paper.title[:50]}' → {list(sections.keys())}")
                                return sections
                except Exception:
                    pass
        except Exception:
            pass

    # fallback 5: NCBI ID Converter → PMCID → PMC BioC API
    if doi:
        pmcid = _doi_to_pmcid(doi)
        if pmcid:
            sections = _load_pmc_bioc_full_text(pmcid, paper.title)
            if sections:
                print(f"  [fulltext:ncbi→pmc-bioc] '{paper.title[:50]}' → {list(sections.keys())}")
                return sections

    # fallback 6: EuropePMC JATS XML
    if doi:
        sections = _load_europepmc_full_text(doi, paper.title)
        if sections:
            return sections

    if doi:
        print(f"  [fulltext:doi] full text 확보 불가 (abstract fallback): {doi}")
    return {}


def _load_scienceon_full_text(paper: Paper) -> dict:
    """ScienceON 논문 full text 로드. DOI가 있으면 DOI 경유, 없으면 빈 dict."""
    # paper.full_text_sections에 raw 메타데이터가 저장되어 있을 수 있음
    raw = {}
    if hasattr(paper, "full_text_sections") and isinstance(paper.full_text_sections, dict):
        raw = paper.full_text_sections

    doi = raw.get("doi", "")

    # Paper.url에서 DOI 추출 시도
    if not doi and paper.url and ("doi.org" in paper.url or "dx.doi.org" in paper.url):
        doi = paper.url

    if doi:
        return _load_doi_full_text(paper)

    print(f"  [fulltext:scienceon] DOI 없음 → abstract fallback: {paper.title[:50]}")
    return {}


def _load_scienceon_pdf_from_url(paper: Paper, url: str, label: str) -> dict:
    """ScienceON의 FulltextURL 또는 ContentURL에서 PDF를 다운로드하여 섹션 분리."""
    if not url:
        return {}
    try:
        resp = _http_session.get(url, timeout=15, allow_redirects=True)
        resp.raise_for_status()

        # 직접 PDF인 경우
        if resp.content[:4] == b"%PDF":
            full_text = _extract_text_from_pdf_bytes(resp.content)
            if len(full_text) >= 200:
                sections = _split_sections(full_text)
                print(f"  [fulltext:{label}] '{paper.title[:50]}' → {list(sections.keys())}")
                return sections

        # HTML 페이지인 경우: PDF 링크를 찾아서 재시도
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            pdf_patterns = [
                r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
                r'href=["\']([^"\']*\/pdf[^"\']*)["\']',
                r'href=["\']([^"\']*fulltext[^"\']*)["\']',
            ]
            for pattern in pdf_patterns:
                matches = re.findall(pattern, resp.text, re.IGNORECASE)
                for pdf_url in matches:
                    if not pdf_url.startswith("http"):
                        from urllib.parse import urljoin
                        pdf_url = urljoin(resp.url, pdf_url)
                    try:
                        pdf_resp = _http_session.get(pdf_url, timeout=15)
                        pdf_resp.raise_for_status()
                        if pdf_resp.content[:4] == b"%PDF":
                            full_text = _extract_text_from_pdf_bytes(pdf_resp.content)
                            if len(full_text) >= 200:
                                sections = _split_sections(full_text)
                                print(f"  [fulltext:{label}] '{paper.title[:50]}' (via HTML) → {list(sections.keys())}")
                                return sections
                    except Exception:
                        continue

    except Exception as e:
        print(f"  [fulltext:{label}] 다운로드 실패: {paper.title[:50]} ({e})")

    return {}


def _load_scienceon_patent_full_text(paper: Paper) -> dict:
    """ScienceON 특허 원문 로드. ContentURL에서 PDF/HTML을 시도."""
    raw = {}
    if hasattr(paper, "full_text_sections") and isinstance(paper.full_text_sections, dict):
        raw = paper.full_text_sections

    # FulltextURL 우선, ContentURL fallback
    for url_key in ("fulltext_url", "FulltextURL", "content_url", "ContentURL"):
        url = raw.get(url_key, "")
        if url:
            sections = _load_scienceon_pdf_from_url(paper, url, "scienceon_patent")
            if sections:
                return sections

    # paper.url fallback
    if paper.url:
        sections = _load_scienceon_pdf_from_url(paper, paper.url, "scienceon_patent")
        if sections:
            return sections

    print(f"  [fulltext:scienceon_patent] 원문 접근 불가 → abstract fallback: {paper.title[:50]}")
    return {}


def _load_scienceon_report_full_text(paper: Paper) -> dict:
    """ScienceON 국가 R&D 보고서 원문 로드. FulltextURL에서 PDF 다운로드."""
    raw = {}
    if hasattr(paper, "full_text_sections") and isinstance(paper.full_text_sections, dict):
        raw = paper.full_text_sections

    # FulltextURL 우선, ContentURL fallback
    for url_key in ("fulltext_url", "FulltextURL", "content_url", "ContentURL"):
        url = raw.get(url_key, "")
        if url:
            sections = _load_scienceon_pdf_from_url(paper, url, "scienceon_report")
            if sections:
                return sections

    # paper.url fallback
    if paper.url:
        sections = _load_scienceon_pdf_from_url(paper, paper.url, "scienceon_report")
        if sections:
            return sections

    print(f"  [fulltext:scienceon_report] 원문 접근 불가 → abstract fallback: {paper.title[:50]}")
    return {}


def _load_full_text_sections(paper: Paper) -> dict:
    """
    논문 소스별로 full text를 로드하고 섹션으로 분리.
    캐시 히트 시 네트워크 요청 없이 즉시 반환.
    """
    if paper.full_text_sections and any(
        k in paper.full_text_sections for k in SECTION_KEYWORDS
    ):
        return paper.full_text_sections

    # ── 캐시 체크 ──
    cached = _cache_get(paper.paper_id or "")
    if cached is not None:
        print(f"  [fulltext:cache] '{paper.title[:50]}' → cache hit")
        return cached

    pid = (paper.paper_id or "").lower()
    sections: dict = {}

    if pid.startswith("arxiv:"):
        sections = _load_arxiv_full_text(paper)
    elif pid.startswith("scienceon_patent:"):
        sections = _load_scienceon_patent_full_text(paper)
    elif pid.startswith("scienceon_report:"):
        sections = _load_scienceon_report_full_text(paper)
    elif pid.startswith("scienceon:"):
        sections = _load_scienceon_full_text(paper)
    elif pid.startswith(("crossref:", "s2:", "openalex:")):
        sections = _load_doi_full_text(paper)

    # ── 캐시 저장 (빈 dict도 저장하여 반복 실패 방지) ──
    _cache_put(paper.paper_id or "", sections)
    return sections


# =====================================================================
# 프롬프트 구성
# =====================================================================
def _build_prompt(paper: Paper, sections: dict) -> str:
    """
    2-track 방식 프롬프트 구성.
    Track 1: 저자 기술 섹션
    Track 2: 구조 분석 섹션
    fallback: abstract (Track 1+2 모두 없을 때)
    """
    track1 = {k: v for k, v in sections.items() if k in TRACK1_KEYS}
    track2 = {k: v for k, v in sections.items() if k in TRACK2_KEYS}

    # fallback: 모두 없으면 abstract 사용
    use_fallback = not track1 and not track2

    lines = [
        f"paper_id: {paper.paper_id}",
        f"title: {paper.title}",
        f"year: {paper.year}",
        "",
    ]

    if use_fallback:
        lines += [
            "## FALLBACK: Abstract Only",
            paper.abstract,
        ]
    else:
        if track1:
            lines.append("## Track 1: Author-Stated Sections")
            for k, v in track1.items():
                lines += [f"### [{k.upper()}]", v, ""]

        if track2:
            lines.append("## Track 2: Structural Analysis Sections")
            for k, v in track2.items():
                lines += [f"### [{k.upper()}]", v, ""]

    return "\n".join(lines)


SYSTEM_PROMPT = """ROLE: Limitation Extract Agent

You extract research limitations from academic papers using a 2-track approach.

## Track 1: Author-Stated Limitations
Sections: conclusion, limitations, future_work
→ Extract what the authors explicitly admit as limitations or future work.
→ Be critical: authors often write defensively. Note if a stated "future work" is actually avoiding a limitation.

## Track 2: Structural Limitations
Sections: introduction, method, experiment, discussion
→ Identify limitations NOT explicitly stated by authors but revealed by:
  - Narrow assumptions in the method (e.g., "we assume i.i.d. data")
  - Limited datasets or evaluation scope (e.g., single domain, small scale)
  - Missing baselines or comparisons
  - Scope restrictions mentioned in introduction

## Rules
1. Extract 1-3 limitations per paper. Prioritize Track 2 structural findings.
2. Each limitation MUST include:
   - claim: concise limitation statement (1-2 sentences). MUST be a complete, self-contained sentence. Never leave a claim unfinished or truncated.
   - evidence_quote: exact short quote from the provided text
   - track: "author_stated" or "structural"
   - source_section: section name (e.g., "conclusion", "method", "experiment")
3. Do NOT infer gaps. Only extract limitations from the provided text.
4. If only abstract is provided (FALLBACK), extract 1 limitation maximum.
5. If input text appears truncated, summarize and complete the limitation based on available context. Never output incomplete sentences.

## Output Format (strictly JSON list)
[
  {
    "paper_id": "<id>",
    "claim": "<limitation statement>",
    "evidence_quote": "<exact short quote>",
    "track": "author_stated" or "structural",
    "source_section": "<section name>"
  },
  ...
]
Output ONLY the JSON list. No explanation before or after.
"""


# =====================================================================
# Limitation 중복 제거
# =====================================================================
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "as", "into", "through", "during",
    "and", "or", "but", "not", "no", "nor", "so", "yet", "both", "either",
    "it", "its", "this", "that", "these", "those", "their", "our", "we",
    "they", "he", "she", "which", "who", "whom", "what", "where", "when",
    "how", "all", "each", "every", "any", "such", "than", "too", "very",
    "also", "only", "just", "more", "most", "other", "some", "about",
})


def _tokenize_claim(claim: str) -> set[str]:
    """claim을 소문자 토큰 집합으로 변환 (불용어 제거)."""
    tokens = set(re.sub(r"[^\w\s]", "", claim.lower()).split())
    return tokens - _STOP_WORDS


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """두 토큰 집합의 Jaccard 유사도."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedupe_limitations(limitations: list[dict], threshold: float = 0.55) -> list[dict]:
    """
    Limitation 중복 제거.
    1) 같은 paper 내: claim Jaccard >= threshold → 첫 번째만 유지
    2) 다른 paper 간: claim Jaccard >= threshold → 첫 번째만 유지
    evidence_quote가 더 긴 쪽을 우선 유지.
    """
    if not limitations:
        return []

    # evidence_quote 길이 내림차순 정렬 (더 근거가 풍부한 것 우선)
    sorted_lims = sorted(limitations, key=lambda x: len(x.get("evidence_quote", "")), reverse=True)

    kept: list[dict] = []
    kept_tokens: list[set[str]] = []

    for lim in sorted_lims:
        tokens = _tokenize_claim(lim.get("claim", ""))
        if not tokens:
            continue

        is_dup = False
        for existing_tokens in kept_tokens:
            if _jaccard_similarity(tokens, existing_tokens) >= threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(lim)
            kept_tokens.append(tokens)

    return kept


# =====================================================================
# limitation_extract_node
# =====================================================================
def limitation_extract_node(state: AgentState) -> AgentState:
    papers_raw = state.get("papers", [])
    errors = list(state.get("errors", []) or [])
    print(f"  [DEBUG] state['papers'] count: {len(papers_raw)}")
    print(f"  [DEBUG] state['papers'] type: {type(papers_raw)}")
    if papers_raw:
        print(f"  [DEBUG] first paper type: {type(papers_raw[0])}")
        print(f"  [DEBUG] first paper: {papers_raw[0]}")
    if not papers_raw:
        print("  ⚠️ [limitation] papers 없음 → 빈 limitations 반환")
        errors.append("[limitation_extract] No papers provided")
        return {
            "messages": [AIMessage(content="[]", name="limitation_extract")],
            "sender": "limitation_extract",
            "limitations": [],
            "errors": errors,
        }

    # dict → Paper 객체 변환
    papers = []
    for p in papers_raw:
        if isinstance(p, Paper):
            papers.append(p)
        else:
            try:
                papers.append(Paper(**p))
            except Exception as e:
                errors.append(f"[limitation_extract] Paper conversion failed: {e}")
                print(f"  ⚠️ Paper 변환 실패: {e}")
                continue

    print(f"  ✓ {len(papers)}편의 full text 접근 가능한 논문으로 limitation 추출 시작")

    provider = state.get("llm_provider")
    session_id = state.get("session_id", "")
    output_language = state.get("output_language", "auto")
    from prompts.system import get_language_instruction
    lang_instruction = get_language_instruction(output_language)

    all_limitations = []
    fulltext_fail_count = 0
    llm_fail_count = 0

    def _process_single_paper(paper: Paper) -> dict:
        """논문 1편에 대해 full text 로드 → LLM 추출 → 파싱을 수행."""
        print(f"\n  [limitation] 처리 중: {paper.paper_id}")
        result = {"paper_id": paper.paper_id, "limitations": [], "errors": [],
                  "fulltext_failed": False, "llm_failed": False, "content": "[]"}

        # 스레드 내에서 LLM 인스턴스 획득 (provider 변경 반영)
        llm = get_llm(provider=provider)

        # full text 로드
        sections = _load_full_text_sections(paper)
        if not sections:
            result["fulltext_failed"] = True

        # 프롬프트 구성
        paper_prompt = _build_prompt(paper, sections)

        # LLM 호출
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + lang_instruction},
            {"role": "user", "content": paper_prompt},
        ]

        try:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            # 콘텐츠 필터 등으로 실패 시 abstract만으로 재시도
            if "content_filter" in str(e) or "content management policy" in str(e):
                print(f"  ⚠️ 콘텐츠 필터 차단 → abstract fallback 재시도: {paper.paper_id}")
                fallback_prompt = _build_prompt(paper, {})
                fallback_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT + lang_instruction},
                    {"role": "user", "content": fallback_prompt},
                ]
                try:
                    response = llm.invoke(fallback_messages)
                    content = response.content if hasattr(response, "content") else str(response)
                    result["errors"].append(f"[limitation_extract] Content filter triggered for {paper.paper_id}, used abstract fallback")
                except Exception as e2:
                    result["llm_failed"] = True
                    result["errors"].append(f"[limitation_extract] LLM failed even with abstract fallback for {paper.paper_id}: {e2}")
                    print(f"  ⚠️ Abstract fallback도 실패: {paper.paper_id} ({e2})")
                    content = "[]"
            else:
                result["llm_failed"] = True
                result["errors"].append(f"[limitation_extract] LLM failed for {paper.paper_id}: {e}")
                print(f"  ⚠️ LLM 호출 실패: {paper.paper_id} ({e})")
                content = "[]"

        result["content"] = content

        # JSON 파싱
        parsed = parse_json(content)
        if not isinstance(parsed, list):
            parsed = []

        # LimitationItem 변환
        for item in parsed:
            try:
                lim = LimitationItem(
                    paper_id=item.get("paper_id", paper.paper_id),
                    claim=item.get("claim", ""),
                    evidence_quote=item.get("evidence_quote", ""),
                    track=item.get("track", "author_stated"),
                    source_section=item.get("source_section", ""),
                )
                if lim.claim:
                    result["limitations"].append(lim.model_dump())
            except Exception as e:
                result["errors"].append(f"[limitation_extract] LimitationItem parse failed for {paper.paper_id}: {e}")
                print(f"  ⚠️ LimitationItem 변환 실패: {e}")

        print(f"  ✓ {paper.paper_id}: {len(result['limitations'])}개 limitation 추출")
        return result

    # ── Step 1: Full text 로드 (병렬, LLM 호출 아님) ──
    print(f"  🔄 {len(papers)}편 논문 full text 로드 중...")
    paper_sections = {}
    with ThreadPoolExecutor(max_workers=min(3, len(papers))) as executor:
        futures = {executor.submit(_load_full_text_sections, p): p for p in papers}
        for future in as_completed(futures):
            if is_cancelled(session_id):
                executor.shutdown(wait=False, cancel_futures=True)
                print(f"  [limitation] 취소됨 — full text 로드 중단")
                return {"messages": [AIMessage(content="Cancelled.", name="limitation_extract")], "limitations": []}
            paper = futures[future]
            try:
                sections = future.result()
            except Exception:
                sections = {}
            paper_sections[paper.paper_id] = sections
            if not sections:
                fulltext_fail_count += 1

    # ── Step 1.5: Full text 실패 논문을 backup 논문으로 대체 ──
    if fulltext_fail_count > 0:
        backup_raw = state.get("backup_papers") or []
        if backup_raw:
            failed_ids = {pid for pid, sec in paper_sections.items() if not sec}
            # backup 중 full text 로드 가능한 논문으로 대체 (arXiv 우선, 그 외도 시도)
            replacements = []
            used_backup_ids = set()
            # arXiv 우선 정렬: arXiv가 앞으로
            sorted_backup = sorted(backup_raw, key=lambda bp: (0 if bp.get("paper_id", "").lower().startswith("arxiv:") else 1))
            for bp in sorted_backup:
                if len(replacements) >= fulltext_fail_count:
                    break
                bp_id = bp.get("paper_id", "")
                if bp_id in used_backup_ids or bp_id in failed_ids:
                    continue
                try:
                    replacement = Paper(**bp) if isinstance(bp, dict) else bp
                    sections = _load_full_text_sections(replacement)
                    if sections:
                        replacements.append((replacement, sections))
                        used_backup_ids.add(bp_id)
                except Exception:
                    continue

            if replacements:
                # 실패 논문 제거 + 대체 논문 추가
                original_count = len(papers)
                papers = [p for p in papers if p.paper_id not in failed_ids]
                for rep_paper, rep_sections in replacements:
                    papers.append(rep_paper)
                    paper_sections[rep_paper.paper_id] = rep_sections
                # 실패 카운트 업데이트
                new_fail = fulltext_fail_count - len(replacements)
                print(f"  [fulltext_replace] {len(replacements)}편 대체 완료 "
                      f"(실패 {fulltext_fail_count} → {max(0, new_fail)}편)")
                fulltext_fail_count = max(0, new_fail)

    report_progress(
        session_id, "limitation_extract",
        f"Full text loaded for {len(papers) - fulltext_fail_count}/{len(papers)} papers. Starting analysis...",
        current=0, total=len(papers),
    )

    # ── Step 2: 배치 LLM 호출 (3편씩 묶어서 호출 횟수 감소) ──
    BATCH_SIZE = 3
    batches = [papers[i:i + BATCH_SIZE] for i in range(0, len(papers), BATCH_SIZE)]
    max_workers = min(2, len(batches))
    print(f"  🔄 {len(papers)}편을 {len(batches)}개 배치로 나눠 {max_workers}개 워커로 처리")

    def _process_batch(batch: list[Paper]) -> dict:
        """논문 배치를 1회 LLM 호출로 처리."""
        llm = get_llm(provider=provider)
        batch_result = {"limitations": [], "errors": [], "llm_failed": False}

        # 배치 프롬프트 구성
        batch_prompt_parts = []
        for paper in batch:
            sections = paper_sections.get(paper.paper_id, {})
            paper_prompt = _build_prompt(paper, sections)
            # 배치 시 각 논문 텍스트를 4000자로 제한
            if len(paper_prompt) > 4000:
                paper_prompt = paper_prompt[:4000] + "\n... (truncated)"
            batch_prompt_parts.append(f"=== PAPER: {paper.paper_id} ===\n{paper_prompt}")

        combined_prompt = "\n\n".join(batch_prompt_parts)
        combined_prompt += (
            "\n\n=== INSTRUCTION ===\n"
            "Extract limitations from ALL papers above. "
            "Return a single JSON list containing limitations from every paper. "
            "Each item MUST include the correct paper_id."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + lang_instruction},
            {"role": "user", "content": combined_prompt},
        ]

        try:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            batch_result["llm_failed"] = True
            pids = [p.paper_id for p in batch]
            batch_result["errors"].append(f"[limitation_extract] LLM batch failed for {pids}: {e}")
            print(f"  ⚠️ 배치 LLM 호출 실패: {pids} ({e})")
            # fallback: 개별 호출 시도
            for paper in batch:
                single = _process_single_paper(paper)
                batch_result["limitations"].extend(single["limitations"])
                batch_result["errors"].extend(single["errors"])
            return batch_result

        # JSON 파싱
        parsed = parse_json(content)
        if not isinstance(parsed, list):
            parsed = []

        valid_pids = {p.paper_id for p in batch}
        for item in parsed:
            try:
                pid = item.get("paper_id", "")
                # paper_id가 배치에 없으면 가장 가까운 것으로 매칭
                if pid not in valid_pids:
                    pid = batch[0].paper_id
                lim = LimitationItem(
                    paper_id=pid,
                    claim=item.get("claim", ""),
                    evidence_quote=item.get("evidence_quote", ""),
                    track=item.get("track", "author_stated"),
                    source_section=item.get("source_section", ""),
                )
                if lim.claim:
                    batch_result["limitations"].append(lim.model_dump())
            except Exception as e:
                batch_result["errors"].append(f"[limitation_extract] LimitationItem parse failed: {e}")

        print(f"  ✓ 배치 ({len(batch)}편): {len(batch_result['limitations'])}개 limitation 추출")
        return batch_result

    processed_papers = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            if is_cancelled(session_id):
                executor.shutdown(wait=False, cancel_futures=True)
                print(f"  [limitation] 취소됨 — LLM 배치 처리 중단")
                return {"messages": [AIMessage(content="Cancelled.", name="limitation_extract")], "limitations": all_limitations}
            result = future.result()
            all_limitations.extend(result["limitations"])
            errors.extend(result["errors"])
            if result["llm_failed"]:
                llm_fail_count += 1
            processed_papers += len(futures[future])
            report_progress(
                session_id, "limitation_extract",
                f"Analyzing papers... ({processed_papers}/{len(papers)}) — {len(all_limitations)} limitations found",
                current=processed_papers, total=len(papers),
            )
            # Stream partial limitations to frontend
            if result["limitations"]:
                partial = [lim if isinstance(lim, dict) else lim.model_dump() if hasattr(lim, 'model_dump') else vars(lim)
                           for lim in result["limitations"]]
                report_progress(
                    session_id, "limitation_extract",
                    f"현재까지 {len(all_limitations)}개 한계점 추출",
                    type="partial_limitations", data=partial,
                )

    # fulltext/LLM 실패 요약을 errors에 기록
    if fulltext_fail_count:
        errors.append(f"[limitation_extract] Full text load failed for {fulltext_fail_count}/{len(papers)} papers (abstract fallback used)")
    if llm_fail_count:
        errors.append(f"[limitation_extract] LLM call failed for {llm_fail_count}/{len(papers)} papers")

    # 중복 제거
    before_dedup = len(all_limitations)
    all_limitations = _dedupe_limitations(all_limitations)
    dedup_removed = before_dedup - len(all_limitations)
    if dedup_removed:
        print(f"  [dedup] {before_dedup} → {len(all_limitations)} ({dedup_removed}개 중복 제거)")

    # 최종 summary 메시지
    summary = AIMessage(
        content=f"Extracted {len(all_limitations)} limitations from {len(papers)} papers (deduplicated from {before_dedup}).",
        name="limitation_extract",
    )

    print(f"\n  ✅ [limitation] 총 {len(all_limitations)}개 추출 완료 (중복 제거 {dedup_removed}개)")

    return {
        "messages": [summary],
        "sender": "limitation_extract",
        "limitations": all_limitations,
        "errors": errors,
    }