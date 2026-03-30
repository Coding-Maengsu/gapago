# 3-3) Paper Retrieval Agent (tool-using selector)
from __future__ import annotations

import numpy as np

from states import AgentState, Paper
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from tools import build_role_tools, bm25_rank, _safe_json_loads, _tokenize, _norm
from rank_bm25 import BM25Okapi
from prompts.system import make_system_prompt
from llm import get_llm
from config import Configuration
from utils.parse_json import parse_json
from utils.progress import report_progress
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import re

ROLE_TOOLS = build_role_tools()
RETRIEVAL_TOOLS = ROLE_TOOLS["RETRIEVAL_TOOLS"]

_RETRIEVAL_PROMPT_TEMPLATE = """ROLE: Paper Retrieval Agent
You are a retrieval orchestrator. Your job is to SELECT and CALL the most appropriate search tools.

Available tools (ALL academic tools MUST be used every time):
- arxiv_api_call_tool: arXiv API direct search. Best for CS, ML, physics, math preprints. Returns papers with arxiv: ID for guaranteed full text access (ar5iv HTML). Use arxiv_query_candidates from Meaning Expansion.
- crossref_search_tool: PRIMARY search tool, 60 results default, most reliable. Covers 150M+ scholarly works with DOI and citation metadata. Very reliable, no rate limit issues.
- semantic_scholar_search_tool: Semantic Scholar search, 50 results, excellent for AI/ML. Covers 200M+ papers across all disciplines with citation data. Returns arXiv papers via arxiv: ID for full text access.
- openalex_search_tool: OpenAlex search, 40 results, broad interdisciplinary coverage. Covers 200M+ works across all disciplines including humanities, social sciences, and engineering. No API key needed.
- scienceon_search_tool: ScienceON/KISTI academic paper search. Covers both domestic and international journal articles, conference papers, and theses indexed by KISTI.
- scienceon_patent_search_tool: ScienceON/KISTI patent search. Returns patent title, abstract, applicants, IPC classification, and application/publication dates. Useful for finding related prior art and technology trends.
- scienceon_report_search_tool: ScienceON/KISTI national R&D report search. Returns government-funded research reports with full-text links. Useful for finding national research projects, policy-driven studies, and R&D trends.
- web_search_tool: General web search for latest trends, news, and community discussions. Do NOT use for academic paper retrieval.

{year_instruction}

Inputs may include a previous Meaning Expansion Agent message containing:
- keywords
- expanded_terms
- arxiv_query_candidates
- web_query_candidates
- scienceon_query_candidates

Rules:
1) Do not perform meaning expansion yourself.
2) Use only the available tools above.
3) You MUST call ALL of these academic search tools for every retrieval: arxiv_api_call_tool, crossref_search_tool, semantic_scholar_search_tool, openalex_search_tool, and scienceon_search_tool. Do NOT skip any. Use different query variations per source to maximize diversity and coverage. arXiv papers have guaranteed full text access, so prioritize them when available.
4) Use scienceon_patent_search_tool and scienceon_report_search_tool when the topic involves applied technology, engineering, or industry applications. Patent data reveals prior art and technology gaps. R&D reports reveal government-funded research directions and policy-driven gaps.
5) Use web_search_tool ONLY for discovering latest trends, emerging issues, recent developments, and community discussions related to the research topic. Do NOT use it to search for academic papers.
6) Normalize academic results into one combined papers list. Keep web trend results separate in web_results. Keep patent and report results separate in patent_results and report_results.

Output JSON with fields:
- selected_tools: [..]
- tool_rationale: <string>
- papers: list of {{paper_id,title,year,url,abstract,authors,source}}
- web_results: list of latest trend/issue items from web search (NOT papers)
- patent_results: list of patent items from scienceon_patent_search_tool
- report_results: list of R&D report items from scienceon_report_search_tool
- scienceon_results: list
- notes: list[str]
Do NOT infer limitations or gaps.
"""


def _build_year_instruction(year_range: str, resolved_year: str) -> str:
    """Build year filter instruction for system prompt."""
    if resolved_year:
        return (
            f"[YEAR FILTER: {resolved_year}]\n"
            f"All search tools support a 'year' parameter. You MUST pass year='{resolved_year}' "
            f"to every academic search tool call (arxiv, semantic_scholar, openalex, scienceon, etc.)."
        )
    # auto: let agent decide
    return (
        "[YEAR FILTER: AUTO]\n"
        "All search tools support a 'year' parameter (format: 'YYYY-YYYY', e.g. '2023-2026').\n"
        "Decide an appropriate year range based on the research field:\n"
        "- AI/LLM/deep learning: '2023-2026' (fast-moving field)\n"
        "- Applied engineering: '2021-2026'\n"
        "- Basic science/medicine: '2018-2026'\n"
        "You MUST pass the year parameter to every academic search tool call."
    )


def _build_retrieval_agent(provider: str = None, year_range: str = "auto", resolved_year: str = ""):
    """Build the paper retrieval agent with year filter baked into system prompt."""
    llm = get_llm(provider=provider)
    year_instruction = _build_year_instruction(year_range, resolved_year)
    system_prompt = make_system_prompt(
        _RETRIEVAL_PROMPT_TEMPLATE.format(year_instruction=year_instruction)
    )
    return create_agent(
        llm,
        tools=RETRIEVAL_TOOLS,
        system_prompt=system_prompt,
    )


def _parse_papers_from_ai_message(content: str) -> list[dict]:
    """
    AIMessage content (JSON)에서 papers 리스트 파싱.
    LLM이 output JSON 안에 papers 필드로 반환하는 경우 처리.
    """
    data = _safe_json_loads(content)
    if not data or not isinstance(data, dict):
        return []

    papers = []
    # ✅ LLM output JSON의 papers 필드
    for p in data.get("papers", []):
        if not isinstance(p, dict):
            continue
        papers.append({
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "url": p.get("url", ""),
            "year": p.get("year", 0) or 0,
            "authors": p.get("authors") or [],
            "score_bm25": p.get("score_bm25", 0.0),
            "source": p.get("source", "arxiv"),
            "full_text_sections": {},
        })

    # web_results는 별도로 반환하지 않음 (papers에 합치지 않음)
    return papers


def _parse_papers_from_tool_messages(messages: list) -> list[dict]:
    papers = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if not content or getattr(msg, "type", "") != "tool":
            continue

        data = _safe_json_loads(content)
        if not data or not isinstance(data, dict):
            continue

        source = data.get("source", "")
        if source in ("arxiv", "crossref", "scienceon", "semantic_scholar", "openalex"):
            papers.extend(data.get("results", []))
        elif source in ("scienceon_patent", "scienceon_report"):
            # 특허/보고서 결과를 papers 형식으로 정규화
            for r in data.get("results", []):
                paper_id = r.get("patent_id") or r.get("report_id") or ""
                papers.append({
                    "paper_id": paper_id,
                    "title": r.get("title", ""),
                    "abstract": r.get("abstract", ""),
                    "url": r.get("url", ""),
                    "year": r.get("year", 0),
                    "authors": r.get("authors", []) if "authors" in r else [r.get("applicants", "")],
                    "score_bm25": 0.0,
                    "source": source,
                    "full_text_sections": r.get("raw", {}),
                })
        # web source는 별도 처리 (papers에 합치지 않음)
    return papers


def _parse_web_results_from_tool_messages(messages: list) -> list[dict]:
    """ToolMessage에서 웹 검색 결과만 별도로 파싱."""
    web_results = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if not content or getattr(msg, "type", "") != "tool":
            continue
        data = _safe_json_loads(content)
        if not data or not isinstance(data, dict):
            continue
        if data.get("source") == "web":
            for r in data.get("results", []):
                web_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content") or r.get("snippet", ""),
                    "source": "web",
                })
    return web_results


def _parse_web_results_from_ai_message(content: str) -> list[dict]:
    """AIMessage JSON에서 web_results를 별도로 파싱."""
    data = _safe_json_loads(content)
    if not data or not isinstance(data, dict):
        return []
    web_results = []
    for r in data.get("web_results", []):
        if not isinstance(r, dict):
            continue
        web_results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content") or r.get("snippet", ""),
            "source": "web",
        })
    return web_results


def _dedupe_papers(raw_papers: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for p in raw_papers:
        key = (
            (p.get("doi") or "").lower().strip(),
            (p.get("title") or "").lower().strip(),
            int(p.get("year", 0) or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


RERANKER_PROMPT = """You are a research paper relevance assessor for GAP analysis.

Research Query: "{query}"

Below are {n} candidate papers. Select the {top_k} most relevant papers for identifying research gaps, limitations, and future directions.

Prioritize:
1. Papers directly addressing the research topic (methodology, dataset, application domain match)
2. Papers with substantive abstracts that likely contain discussable limitations
3. Diversity of perspectives (different approaches/subfields within the topic)
4. Recent papers (more likely to have current limitations)

Papers:
{papers_text}

Return a JSON object:
{{
  "selected_indices": [1, 3, 5, ...],
  "rationale": "Brief explanation of selection strategy"
}}

IMPORTANT: Return ONLY the JSON object. Select exactly {top_k} papers (or fewer if fewer are truly relevant)."""


def _llm_rerank(papers: list[dict], query: str, top_k: int, provider: str = None) -> list[dict]:
    """LLM Reranker: BM25 필터링된 논문에서 GAP 분석에 가장 적합한 논문 선별."""
    if len(papers) <= top_k:
        return papers

    papers_text = "\n".join(
        f"[{i+1}] Title: {p.get('title', '')}\n"
        f"    Year: {p.get('year', 'N/A')}\n"
        f"    Abstract: {(p.get('abstract', '') or '')[:300]}\n"
        for i, p in enumerate(papers)
    )

    prompt = RERANKER_PROMPT.format(
        query=query,
        n=len(papers),
        top_k=top_k,
        papers_text=papers_text,
    )

    try:
        llm = get_llm(provider=provider)
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        parsed = parse_json(content)

        if not isinstance(parsed, dict) or "selected_indices" not in parsed:
            print(f"  [WARN] LLM Reranker 파싱 실패 → BM25 top-{top_k} fallback")
            return papers[:top_k]

        indices = parsed["selected_indices"]
        # 유효한 인덱스만 필터 (1-indexed → 0-indexed)
        seen = set()
        selected = []
        for idx in indices:
            try:
                idx = int(idx)
            except (ValueError, TypeError):
                continue
            if idx < 1 or idx > len(papers) or idx in seen:
                continue
            seen.add(idx)
            selected.append(papers[idx - 1])
            if len(selected) >= top_k:
                break

        if not selected:
            print(f"  [WARN] LLM Reranker 유효 결과 없음 → BM25 top-{top_k} fallback")
            return papers[:top_k]

        rationale = parsed.get("rationale", "")
        print(f"  [reranker] {len(papers)}편 → {len(selected)}편 선별 ({rationale[:80]})")
        return selected

    except Exception as e:
        print(f"  [WARN] LLM Reranker 실패: {e} → BM25 top-{top_k} fallback")
        return papers[:top_k]


def _resolve_year_range(year_range: str) -> str:
    """Convert year_range setting to 'YYYY-YYYY' format for tools."""
    from datetime import datetime
    current_year = datetime.now().year
    if year_range == "1y":
        return f"{current_year - 1}-{current_year}"
    elif year_range == "3y":
        return f"{current_year - 3}-{current_year}"
    elif year_range == "5y":
        return f"{current_year - 5}-{current_year}"
    # "auto" or unknown → return empty (agent decides)
    return ""


# =====================================================================
# Full text 접근 가능 여부 체크 (빠른 테스트)
# =====================================================================
_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
}


def _extract_arxiv_id(paper: dict) -> str:
    """paper_id에서 arXiv ID 추출."""
    pid = paper.get("paper_id", "")
    if pid.lower().startswith("arxiv:"):
        pid = pid[len("arxiv:"):]
    pid = pid.strip()
    if not pid or not re.match(r"^(\d{4}\.\d{4,5}|[\w\-]+\.?[\w\-]*/\d{7})", pid):
        return ""
    pid = re.sub(r"v\d+$", "", pid)
    return pid


def _find_pdf_url_from_doi(doi: str) -> str:
    """DOI 페이지에서 PDF URL 찾기 (다운로드 없음)."""
    doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    try:
        resp = requests.get(doi_url, headers=_REQUEST_HEADERS, timeout=10, allow_redirects=True)
        resp.raise_for_status()

        # 리다이렉트된 최종 URL이 PDF인 경우
        if resp.headers.get("content-type", "").startswith("application/pdf"):
            return resp.url

        # HTML 페이지에서 PDF 링크 추출
        pdf_patterns = [
            r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
            r'href=["\']([^"\']*\/pdf[^"\']*)["\']',
        ]
        for pattern in pdf_patterns:
            matches = re.findall(pattern, resp.text, re.IGNORECASE)
            for url in matches:
                if not url.startswith("http"):
                    from urllib.parse import urljoin
                    url = urljoin(resp.url, url)
                return url
    except Exception:
        pass
    return ""


# full text 확보 신뢰도 등급
_FULLTEXT_CONFIDENCE = {
    "guaranteed": 3,   # arXiv (ar5iv HTML 거의 100%)
    "likely": 2,       # OA PDF URL 확보됨
    "maybe": 1,        # DOI만 있음 (출판사 차단 가능성)
    "none": 0,
}


def _fulltext_confidence(paper: dict) -> int:
    """논문의 full text 확보 신뢰도를 반환."""
    pid = paper.get("paper_id", "").lower()

    # arXiv: ar5iv HTML이 거의 항상 존재
    if pid.startswith("arxiv:"):
        if _extract_arxiv_id(paper):
            return _FULLTEXT_CONFIDENCE["guaranteed"]
        return _FULLTEXT_CONFIDENCE["none"]

    # ScienceON 특허/보고서: URL 존재 여부
    if pid.startswith(("scienceon_patent:", "scienceon_report:")):
        fts = paper.get("full_text_sections", {})
        for key in ("fulltext_url", "FulltextURL", "content_url", "ContentURL"):
            if fts.get(key):
                return _FULLTEXT_CONFIDENCE["likely"]
        return _FULLTEXT_CONFIDENCE["maybe"] if paper.get("url") else _FULLTEXT_CONFIDENCE["none"]

    # API가 OA PDF URL을 제공한 경우
    pdf_url = paper.get("pdf_url") or paper.get("full_text_sections", {}).get("pdf_url", "")
    if pdf_url:
        return _FULLTEXT_CONFIDENCE["likely"]

    # DOI 기반 소스: DOI가 있으면 가능성은 있지만 확실하지 않음
    if pid.startswith(("crossref:", "s2:", "openalex:", "scienceon:")):
        doi = paper.get("doi", "") or paper.get("full_text_sections", {}).get("doi", "")
        if not doi and paper.get("url") and "doi.org" in paper.get("url", ""):
            doi = paper.get("url")
        if doi:
            return _FULLTEXT_CONFIDENCE["maybe"]

    return _FULLTEXT_CONFIDENCE["none"]


def _can_load_full_text_fast(paper: dict) -> bool:
    """Full text 접근 가능 여부를 메타데이터만으로 빠르게 판별."""
    return _fulltext_confidence(paper) > _FULLTEXT_CONFIDENCE["none"]


def _filter_fulltext_available(papers: list[dict], target_count: int = 30) -> list[dict]:
    """
    논문 리스트에서 full text 접근 가능한 논문만 필터링.
    arXiv(guaranteed) → OA PDF(likely) → DOI(maybe) 순으로 우선 배치.
    각 신뢰도 그룹 내에서는 BM25 순서 유지.
    """
    if not papers:
        return []

    print(f"  [fulltext_filter] {len(papers)}편 중 full text 접근 가능 여부 확인 중...")

    # 각 논문의 신뢰도 계산
    scored = []
    for i, paper in enumerate(papers):
        conf = _fulltext_confidence(paper)
        if conf > 0:
            scored.append((conf, i, paper))

    # 신뢰도 내림차순, 동일 신뢰도 내에서는 원래 순서(BM25) 유지
    scored.sort(key=lambda x: (-x[0], x[1]))

    fulltext_available = [paper for _, _, paper in scored[:target_count]]

    # 소스별 통계 출력
    stats = {"guaranteed": 0, "likely": 0, "maybe": 0}
    for conf, _, _ in scored[:target_count]:
        if conf == 3:
            stats["guaranteed"] += 1
        elif conf == 2:
            stats["likely"] += 1
        else:
            stats["maybe"] += 1

    print(f"  [fulltext_filter] {len(fulltext_available)}/{len(papers)}편 선별 "
          f"(guaranteed={stats['guaranteed']}, likely={stats['likely']}, maybe={stats['maybe']})")
    return fulltext_available


def paper_retrieval_node(state: AgentState) -> AgentState:
    provider = state.get("llm_provider")
    session_id = state.get("session_id", "")

    # state에서 연도 필터 읽기 → 시스템 프롬프트에 반영
    year_range = state.get("year_range", "auto")
    resolved_year = _resolve_year_range(year_range)
    print(f"  [retrieval] year_range={year_range}, resolved_year={resolved_year}")

    paper_retrieval_agent = _build_retrieval_agent(
        provider=provider,
        year_range=year_range,
        resolved_year=resolved_year,
    )

    messages = state.get("messages", [])
    print(f"  [DEBUG] 전체 messages 수: {len(messages)}")
    for i, m in enumerate(messages):
        print(f"  [DEBUG] msg[{i}] type={type(m).__name__} name={getattr(m, 'name', None)} content={getattr(m, 'content', '')[:80]}")
    # ✅ meaning_expand 메시지 우선, 없으면 query_analysis, 그것도 없으면 user_question
    query_messages = [
        m for m in state.get("messages", [])
        if getattr(m, "name", None) == "meaning_expand"
    ]
    if not query_messages:
        query_messages = [
            m for m in state.get("messages", [])
            if getattr(m, "name", None) == "query_analysis"
        ]
    if not query_messages:
        fallback_text = state.get("refined_query") or state.get("user_question", "")
        query_messages = [HumanMessage(content=fallback_text)]

    # 가장 최신 메시지 1개만 전달
    query_messages = query_messages[-1:]

    result = paper_retrieval_agent.invoke({
        **state,
        "messages": query_messages,
    })

    messages = result.get("messages", [])
    last_content = messages[-1].content if messages else "{}"

    # ✅ 1순위: tool 메시지에서 직접 파싱
    raw_papers = _parse_papers_from_tool_messages(messages)

    # ✅ 웹 결과 별도 파싱
    web_results = _parse_web_results_from_tool_messages(messages)

    # ✅ 2순위: LLM output JSON의 papers 필드에서 파싱 (현재 구조 대응)
    if not raw_papers:
        raw_papers = _parse_papers_from_ai_message(last_content)
        if not web_results:
            web_results = _parse_web_results_from_ai_message(last_content)

    raw_papers = _dedupe_papers(raw_papers)
    total_candidates_count = len(raw_papers)  # BM25 전 전체 후보 수
    print(f"  [DEBUG] raw_papers count (after dedup): {total_candidates_count}")

    # ✅ 연도 필터 (resolved_year가 있으면 범위 밖 논문 제거)
    if resolved_year and "-" in resolved_year:
        parts = resolved_year.split("-")
        from_year = int(parts[0].strip())
        to_year = int(parts[1].strip()) if parts[1].strip() else 9999
        before_filter = len(raw_papers)
        raw_papers = [
            p for p in raw_papers
            if not p.get("year") or from_year <= int(p.get("year", 0) or 0) <= to_year
        ]
        filtered_count = before_filter - len(raw_papers)
        if filtered_count:
            print(f"  [year_filter] {before_filter} → {len(raw_papers)} ({filtered_count}편 범위 밖 제거, {from_year}-{to_year})")

    # ✅ BM25 1차 필터 (동적 k: 점수 분포 기반 적응적 컷오프)
    total_found = len(raw_papers)
    cfg = Configuration()
    query = state.get("refined_query") or state.get("user_question", "")
    if raw_papers and query:
        # 동적 k 계산: BM25 점수 분포에서 평균 + 0.5*표준편차 이상인 논문 수
        corpus = [_tokenize(f"{p.get('title', '')} {p.get('abstract', '')}") for p in raw_papers]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))
        if len(scores) > 0:
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            threshold = mean_score + 0.5 * std_score
            dynamic_k = int(np.sum(scores > threshold))
            dynamic_k = max(10, min(dynamic_k, cfg.bm25_top_k))  # 10 ~ bm25_top_k 범위
            print(f"  [BM25] dynamic_k={dynamic_k} (mean={mean_score:.2f}, std={std_score:.2f}, threshold={threshold:.2f})")
        else:
            dynamic_k = cfg.bm25_top_k
        ranked = bm25_rank(raw_papers, query, top_k=dynamic_k)
        raw_papers = ranked.get("selected", raw_papers)
    print(f"  [DEBUG] BM25 1st stage: {len(raw_papers)} papers")

    # ✅ Full text 접근 가능 여부 필터링 (BM25와 Reranker 사이)
    raw_papers = _filter_fulltext_available(raw_papers, target_count=cfg.fulltext_target_count)
    print(f"  [DEBUG] Full text filter: {len(raw_papers)} papers")

    # ✅ LLM Reranker 2차 선별 (정밀) - full text 가능한 논문만 대상
    # 선별되지 않은 논문은 backup으로 보관 (full text 실패 시 대체용)
    backup_raw = []
    if raw_papers and query and len(raw_papers) > cfg.reranker_top_k:
        reranked = _llm_rerank(raw_papers, query, top_k=cfg.reranker_top_k, provider=provider)
        reranked_ids = {p.get("paper_id") for p in reranked}
        backup_raw = [p for p in raw_papers if p.get("paper_id") not in reranked_ids]
        # backup은 arXiv(guaranteed) 우선, 동일 그룹 내에서는 BM25 순위(원래 순서) 유지
        # enumerate로 원래 인덱스 보존 → stable sort
        backup_raw = sorted(
            backup_raw,
            key=lambda p: (
                0 if p.get("paper_id", "").lower().startswith("arxiv:") else 1,
                -p.get("score_bm25", 0.0),  # BM25 점수 높은 순
            ),
        )
        raw_papers = reranked
    print(f"  [DEBUG] LLM Reranker 2nd stage: {len(raw_papers)} papers, backup: {len(backup_raw)} papers")
    report_progress(
        session_id, "paper_retrieval",
        f"{total_found}편 중 가장 관련도 높은 {len(raw_papers)}편을 선별했습니다",
    )

    # ✅ Paper 객체로 변환
    papers = []
    for p in raw_papers:
        if not p.get("title") or not p.get("paper_id"):
            continue
        try:
            # ScienceON 등 외부 소스의 doi를 full_text_sections에 보존
            fts = dict(p.get("full_text_sections") or {})
            if p.get("doi") and "doi" not in fts:
                fts["doi"] = p["doi"]
            if p.get("pdf_url") and "pdf_url" not in fts:
                fts["pdf_url"] = p["pdf_url"]
            # venue 결정: 명시적 venue > journal > source 라벨
            venue = p.get("venue", "")
            if not venue:
                venue = p.get("journal", "")
            if not venue:
                source = p.get("source", "")
                venue_map = {
                    "arxiv": "arXiv preprint",
                    "semantic_scholar": "Semantic Scholar",
                    "openalex": "OpenAlex",
                    "scienceon": "ScienceON",
                    "scienceon_patent": "Patent",
                    "scienceon_report": "R&D Report",
                }
                venue = venue_map.get(source, source)
            papers.append(Paper(
                paper_id=p.get("paper_id", ""),
                title=p.get("title", ""),
                abstract=p.get("abstract", ""),
                url=p.get("url", ""),
                year=p.get("year", 0) or 0,
                authors=p.get("authors") or [],
                score_bm25=p.get("score_bm25", 0.0),
                venue=venue,
                full_text_sections=fts,
            ))
        except Exception as e:
            print(f"  ⚠️ Paper 변환 실패: {e}")
            continue

    # backup_papers를 dict 형태로 변환 (state에 저장)
    backup_papers = []
    for p in backup_raw:
        try:
            fts = dict(p.get("full_text_sections") or {})
            if p.get("doi") and "doi" not in fts:
                fts["doi"] = p["doi"]
            if p.get("pdf_url") and "pdf_url" not in fts:
                fts["pdf_url"] = p["pdf_url"]
            backup_papers.append({
                "paper_id": p.get("paper_id", ""),
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "url": p.get("url", ""),
                "year": p.get("year", 0) or 0,
                "authors": p.get("authors") or [],
                "score_bm25": p.get("score_bm25", 0.0),
                "venue": p.get("venue", ""),
                "full_text_sections": fts,
            })
        except Exception:
            continue
    arxiv_backup = sum(1 for bp in backup_papers if bp.get("paper_id", "").lower().startswith("arxiv:"))
    print(f"  ✓ Retrieved {len(papers)}/{total_candidates_count} papers + {len(web_results)} web results")
    print(f"  ✓ Backup pool: {len(backup_papers)} papers ({arxiv_backup} arXiv)")

    last = AIMessage(content=last_content, name="paper_retrieval")
    return {
        "messages": [last],
        "sender": "paper_retrieval",
        "papers": papers,
        "backup_papers": backup_papers,
        "total_candidates_count": total_candidates_count,
        "web_results": web_results,
    }