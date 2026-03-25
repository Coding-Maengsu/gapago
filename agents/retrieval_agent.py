# 3-3) Paper Retrieval Agent (tool-using selector)
from __future__ import annotations

from states import AgentState, Paper
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from tools import build_role_tools, bm25_rank, _safe_json_loads
from prompts.system import make_system_prompt
from llm import get_llm
from config import Configuration
from utils.parse_json import parse_json

llm = get_llm()

ROLE_TOOLS = build_role_tools()
RETRIEVAL_TOOLS = ROLE_TOOLS["RETRIEVAL_TOOLS"]

paper_retrieval_agent = create_agent(
    llm,
    tools=RETRIEVAL_TOOLS,
    system_prompt=make_system_prompt(
        """ROLE: Paper Retrieval Agent
You are a retrieval orchestrator. Your job is to SELECT and CALL the most appropriate search tools.

Available tools:
- arxiv_api_call_tool: Primary academic paper search (arXiv). Best for CS, physics, math papers.
- semantic_scholar_search_tool: Broad academic search (Semantic Scholar). Covers all disciplines, good for highly-cited papers.
- openalex_search_tool: Comprehensive academic search (OpenAlex, 200M+ works). Good for cross-domain and interdisciplinary coverage.
- scienceon_search_tool: Korean academic paper search (ScienceON/KISTI). Best for Korean-language papers and domestic journal articles.
- scienceon_patent_search_tool: Korean patent search (ScienceON/KISTI). Returns patent title, abstract, applicants, IPC classification, and application/publication dates. Useful for finding related prior art and technology trends.
- scienceon_report_search_tool: Korean national R&D report search (ScienceON/KISTI). Returns government-funded research reports with full-text links. Useful for finding national research projects, policy-driven studies, and R&D trends.
- web_search_tool: General web search for tracking latest trends, news, blog posts, and community discussions. Do NOT use this tool for academic paper retrieval.

Inputs may include a previous Meaning Expansion Agent message containing:
- keywords
- expanded_terms
- arxiv_query_candidates
- web_query_candidates
- scienceon_query_candidates

Rules:
1) Do not perform meaning expansion yourself.
2) Use only the available tools above.
3) For academic paper retrieval, use AT LEAST 2-3 academic search tools together (arxiv + semantic_scholar + openalex) to maximize coverage. Use different query variations per source for diversity.
4) Use scienceon_patent_search_tool and scienceon_report_search_tool when the topic involves applied technology, engineering, or industry applications. Patent data reveals prior art and technology gaps. R&D reports reveal government-funded research directions and policy-driven gaps.
5) Use web_search_tool ONLY for discovering latest trends, emerging issues, recent developments, and community discussions related to the research topic. Do NOT use it to search for academic papers.
6) Normalize academic results into one combined papers list. Keep web trend results separate in web_results. Keep patent and report results separate in patent_results and report_results.

Output JSON with fields:
- selected_tools: [..]
- tool_rationale: <string>
- papers: list of {paper_id,title,year,url,abstract,authors,source}
- web_results: list of latest trend/issue items from web search (NOT papers)
- patent_results: list of patent items from scienceon_patent_search_tool
- report_results: list of R&D report items from scienceon_report_search_tool
- scienceon_results: list
- notes: list[str]
Do NOT infer limitations or gaps.
"""
    ),
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
        if source in ("arxiv", "scienceon", "semantic_scholar", "openalex"):
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


def _llm_rerank(papers: list[dict], query: str, top_k: int) -> list[dict]:
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
        llm = get_llm()
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


def paper_retrieval_node(state: AgentState) -> AgentState:
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
    print(f"  [DEBUG] raw_papers count (after dedup): {len(raw_papers)}")

    # ✅ BM25 1차 필터 (넓게)
    cfg = Configuration()
    query = state.get("refined_query") or state.get("user_question", "")
    if raw_papers and query:
        ranked = bm25_rank(raw_papers, query, top_k=cfg.bm25_top_k)
        raw_papers = ranked.get("selected", raw_papers)
    print(f"  [DEBUG] BM25 1st stage: {len(raw_papers)} papers")

    # ✅ LLM Reranker 2차 선별 (정밀)
    if raw_papers and query and len(raw_papers) > cfg.reranker_top_k:
        raw_papers = _llm_rerank(raw_papers, query, top_k=cfg.reranker_top_k)
    print(f"  [DEBUG] LLM Reranker 2nd stage: {len(raw_papers)} papers")

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
            papers.append(Paper(
                paper_id=p.get("paper_id", ""),
                title=p.get("title", ""),
                abstract=p.get("abstract", ""),
                url=p.get("url", ""),
                year=p.get("year", 0) or 0,
                authors=p.get("authors") or [],
                score_bm25=p.get("score_bm25", 0.0),
                full_text_sections=fts,
            ))
        except Exception as e:
            print(f"  ⚠️ Paper 변환 실패: {e}")
            continue

    print(f"  ✓ Retrieved {len(papers)} papers + {len(web_results)} web results")

    last = AIMessage(content=last_content, name="paper_retrieval")
    return {
        "messages": [last],
        "sender": "paper_retrieval",
        "papers": papers,
        "web_results": web_results,
    }