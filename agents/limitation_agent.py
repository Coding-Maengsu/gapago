# 3-3) Limitation Extract Agent
from __future__ import annotations

import re
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pymupdf as fitz

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from states import AgentState, Paper, LimitationItem
from llm import get_llm
from utils.parse_json import parse_json
from utils.progress import report_progress

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


# =====================================================================
# 섹션 분리 유틸
# =====================================================================
def _split_sections(full_text: str) -> dict:
    """전체 텍스트에서 섹션별로 분리."""
    heading_pattern = re.compile(
        r"(?m)^(\d+[\.\d]*\s+)?("
        + "|".join(
            kw.title()
            for kws in SECTION_KEYWORDS.values()
            for kw in kws
        )
        + r")(\s*:)?\s*$",
        re.IGNORECASE,
    )

    matches = list(heading_pattern.finditer(full_text))
    if not matches:
        return {}

    sections = {}
    for i, match in enumerate(matches):
        h = match.group(0).lower().strip()
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
_REQUEST_HEADERS = {"User-Agent": "GAPAGO-Research-Agent/1.0"}


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

        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=15)
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


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """PyMuPDF로 PDF 바이트에서 텍스트 추출."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def _find_pdf_url_from_doi(doi: str) -> Optional[str]:
    """DOI 페이지에 접근하여 PDF 다운로드 링크를 찾는다."""
    doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    try:
        resp = requests.get(doi_url, headers=_REQUEST_HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()

        # 리다이렉트된 최종 URL이 PDF인 경우
        if resp.headers.get("content-type", "").startswith("application/pdf"):
            return resp.url

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
                    # 상대 URL → 절대 URL
                    from urllib.parse import urljoin
                    url = urljoin(resp.url, url)
                return url

    except Exception as e:
        print(f"  [fulltext:doi] DOI 페이지 접근 실패: {doi} ({e})")
    return None


def _load_doi_full_text(paper: Paper) -> dict:
    """DOI를 통해 출판사 PDF에서 full text를 로드."""
    # paper에서 DOI 추출 (full_text_sections에 raw가 있을 수 있음)
    doi = ""
    if hasattr(paper, "full_text_sections") and isinstance(paper.full_text_sections, dict):
        doi = paper.full_text_sections.get("doi", "")
    if not doi:
        # Paper 객체의 url 필드가 DOI일 수 있음
        if paper.url and ("doi.org" in paper.url or "dx.doi.org" in paper.url):
            doi = paper.url

    if not doi:
        return {}

    pdf_url = _find_pdf_url_from_doi(doi)
    if not pdf_url:
        print(f"  [fulltext:doi] PDF 링크 없음: {doi}")
        return {}

    try:
        resp = requests.get(pdf_url, headers=_REQUEST_HEADERS, timeout=30)
        resp.raise_for_status()

        if not resp.content[:4] == b"%PDF":
            print(f"  [fulltext:doi] PDF가 아닌 응답: {pdf_url}")
            return {}

        full_text = _extract_text_from_pdf_bytes(resp.content)
        if len(full_text) < 200:
            return {}

        sections = _split_sections(full_text)
        print(f"  [fulltext:doi] '{paper.title[:50]}' → {list(sections.keys())}")
        return sections

    except Exception as e:
        print(f"  [fulltext:doi] PDF 다운로드 실패: {pdf_url} ({e})")
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
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=30, allow_redirects=True)
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
                        pdf_resp = requests.get(pdf_url, headers=_REQUEST_HEADERS, timeout=30)
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
    - arXiv → a5iv HTML / ArxivLoader PDF
    - ScienceON 논문 → DOI 경유 PDF 다운로드
    - ScienceON 특허 → ContentURL에서 PDF/HTML
    - ScienceON 보고서 → FulltextURL에서 PDF
    - 그 외 → abstract fallback (빈 dict)
    """
    if paper.full_text_sections and any(
        k in paper.full_text_sections for k in SECTION_KEYWORDS
    ):
        return paper.full_text_sections

    pid = (paper.paper_id or "").lower()

    if pid.startswith("arxiv:"):
        return _load_arxiv_full_text(paper)

    if pid.startswith("scienceon_patent:"):
        return _load_scienceon_patent_full_text(paper)

    if pid.startswith("scienceon_report:"):
        return _load_scienceon_report_full_text(paper)

    if pid.startswith("scienceon:"):
        return _load_scienceon_full_text(paper)

    # Semantic Scholar / OpenAlex: DOI 경유 시도, 없으면 abstract fallback
    if pid.startswith(("s2:", "openalex:")):
        sections = _load_doi_full_text(paper)
        if sections:
            return sections

    # 웹 소스 등은 abstract fallback
    return {}


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
   - claim: concise limitation statement (1-2 sentences)
   - evidence_quote: exact short quote from the provided text
   - track: "author_stated" or "structural"
   - source_section: section name (e.g., "conclusion", "method", "experiment")
3. Do NOT infer gaps. Only extract limitations from the provided text.
4. If only abstract is provided (FALLBACK), extract 1 limitation maximum.

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
    with ThreadPoolExecutor(max_workers=min(5, len(papers))) as executor:
        futures = {executor.submit(_load_full_text_sections, p): p for p in papers}
        for future in as_completed(futures):
            paper = futures[future]
            try:
                sections = future.result()
            except Exception:
                sections = {}
            paper_sections[paper.paper_id] = sections
            if not sections:
                fulltext_fail_count += 1

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
            # 배치 시 각 논문 텍스트를 2000자로 제한
            if len(paper_prompt) > 2000:
                paper_prompt = paper_prompt[:2000] + "\n... (truncated)"
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