# Recency Check Agent
# limitation의 최신성을 웹 검색 결과와 대조하여 검증
# 1) LLM이 refined_query를 보고 연구 도메인 판단 + 맞춤 검색 쿼리 생성
# 2) 도메인별 적합한 웹 소스에서 Tavily 검색
# 3) 검색 결과와 limitation을 대조하여 최신성 판정
from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from gapago.core.states import AgentState
from gapago.core.llm import get_llm, get_llm_for_agent
from gapago.utils.parse_json import parse_json
from gapago.utils.tavily import TavilySearch

# ── 도메인별 검색 소스 매핑 ──────────────────────────────────────
DOMAIN_SOURCES = {
    "ai_cs": {
        "label": "AI / Computer Science",
        "include": ["paperswithcode.com", "github.com", "huggingface.co",
                     "medium.com", "towardsdatascience.com"],
        "exclude": ["arxiv.org"],
    },
    "biomedical": {
        "label": "Biomedical / Life Sciences",
        "include": ["pubmed.ncbi.nlm.nih.gov", "biorxiv.org", "medrxiv.org",
                     "nature.com", "sciencedirect.com"],
        "exclude": ["arxiv.org"],
    },
    "materials_chemistry": {
        "label": "Materials / Chemistry",
        "include": ["nature.com", "sciencedirect.com", "acs.org",
                     "rsc.org", "materialsproject.org"],
        "exclude": ["arxiv.org"],
    },
    "physics": {
        "label": "Physics",
        "include": ["nature.com", "phys.org", "sciencedirect.com",
                     "aps.org", "iop.org"],
        "exclude": ["arxiv.org"],
    },
    "general": {
        "label": "General Science & Technology",
        "include": [],
        "exclude": ["arxiv.org"],
    },
}

# ── 도메인 판단 + 검색 쿼리 생성 프롬프트 ──────────────────────
QUERY_GEN_PROMPT = """You are given a research query and a list of research limitations.

## Task
1. Determine the research domain from the query. Choose ONE:
   - ai_cs: AI, machine learning, computer science, NLP, computer vision
   - biomedical: biology, medicine, drug discovery, genomics, clinical
   - materials_chemistry: materials science, chemistry, polymers, catalysis
   - physics: physics, quantum, optics, astrophysics
   - general: interdisciplinary or unclear domain

2. Generate 3-5 targeted web search queries to check if these limitations have been recently addressed.
   - Each query should focus on a cluster of related limitations
   - Add time markers like "2024" or "2025" or "latest" or "state-of-the-art"
   - Focus on solutions, new methods, new benchmarks, or breakthroughs

## Output Format (strictly JSON)
{
  "domain": "<one of: ai_cs, biomedical, materials_chemistry, physics, general>",
  "search_queries": ["query1", "query2", ...]
}

Output ONLY the JSON. No explanation."""

# ── 최신성 판정 프롬프트 ──────────────────────────────────────
RECENCY_PROMPT = """ROLE: Recency Check Agent

You verify whether research limitations are still relevant by cross-referencing with recent web sources.

## Task
For each limitation (identified by limitation_id), determine if recent developments have addressed it:
- "unresolved": No evidence that this limitation has been addressed
- "partial": Some progress found, but not fully resolved
- "resolved": Clear evidence that this limitation has been overcome

## Output Format (strictly JSON list)
[
  {
    "limitation_id": <integer, must match the input limitation_id exactly>,
    "recency_status": "unresolved" or "partial" or "resolved",
    "evidence": "<brief explanation referencing web source, or 'No relevant web evidence found'>"
  },
  ...
]

## Rules
1. Be conservative: only mark "resolved" if there is clear, specific evidence.
2. If no web result is relevant to a limitation, mark it "unresolved".
3. Do NOT invent or hallucinate web sources. Only use the provided web results.
4. You MUST use the exact limitation_id from the input. Do NOT renumber or skip.
5. Output ONLY the JSON list. No explanation before or after."""


def _search_for_recency(limitations: list, refined_query: str, existing_web: list,
                        user_domain: str = "", provider: str = None, state: dict = None) -> list:
    """Limitation 맞춤 Tavily 검색 수행. 도메인 자동 판단 + 쿼리 생성."""
    if state:
        llm = get_llm_for_agent(state, "recency_check")
    else:
        llm = get_llm(provider=provider)

    # limitation 요약
    lim_summary = "\n".join(
        f"  - [{l.get('paper_id', '')}] {l.get('claim', '')}"
        for l in limitations[:15]
    )

    # Step 1: LLM으로 도메인 판단 + 검색 쿼리 생성
    messages = [
        SystemMessage(content=QUERY_GEN_PROMPT),
        HumanMessage(content=f"## Research Query\n{refined_query}\n\n## Limitations\n{lim_summary}"),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        parsed = parse_json(content)
    except Exception as e:
        print(f"  ⚠️ [recency] 쿼리 생성 실패: {e}")
        return existing_web

    if not isinstance(parsed, dict):
        return existing_web

    # 사용자 지정 도메인 우선, 없으면 LLM 판단 사용
    domain = parsed.get("domain", "general")
    if user_domain and user_domain != "auto" and user_domain in DOMAIN_SOURCES:
        print(f"  [recency] 사용자 지정 도메인: {user_domain} (LLM 판단: {domain})")
        domain = user_domain

    search_queries = parsed.get("search_queries", [])

    if not search_queries:
        return existing_web

    # 도메인 소스 결정
    sources = DOMAIN_SOURCES.get(domain, DOMAIN_SOURCES["general"])
    print(f"  [recency] 도메인: {sources['label']} ({domain})")
    print(f"  [recency] 검색 쿼리 {len(search_queries)}개 생성")

    # Step 2: Tavily 검색
    tavily = TavilySearch(max_results=3)
    all_results = list(existing_web)  # 기존 웹 결과도 포함

    for query in search_queries[:5]:
        try:
            results = tavily.search(
                query=query,
                include_domains=sources["include"] or None,
                exclude_domains=sources["exclude"] or None,
                max_results=3,
                include_raw_content=False,
            )
            for r in results:
                all_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "source": "recency_search",
                    "query": query,
                })
            print(f"  [recency] '{query[:50]}...' → {len(results)}개 결과")
        except Exception as e:
            print(f"  ⚠️ [recency] 검색 실패: {query[:50]} ({e})")

    print(f"  [recency] 총 {len(all_results)}개 웹 결과 수집")
    return all_results


def recency_check_node(state: AgentState) -> AgentState:
    web_results = state.get("web_results", [])
    limitations = state.get("limitations", [])
    refined_query = state.get("refined_query", "")

    if not limitations:
        return {
            "messages": [AIMessage(content="No limitations to check.", name="recency_check")],
            "sender": "recency_check",
            "limitations": [],
        }

    # Step 1: limitation 맞춤 웹 검색
    user_domain = state.get("research_domain", "")
    all_web = _search_for_recency(limitations, refined_query, web_results, user_domain, state=state)

    # 웹 결과가 여전히 없으면 전체 unresolved
    if not all_web:
        print("  [recency] 웹 결과 없음 → 전체 unresolved로 통과")
        for lim in limitations:
            lim["recency_status"] = "unresolved"
            lim["recency_evidence"] = "No web results available"

        return {
            "messages": [AIMessage(
                content=f"Recency check: no web results. {len(limitations)} limitations unresolved.",
                name="recency_check",
            )],
            "sender": "recency_check",
            "limitations": limitations,
            "web_results": all_web,
        }

    # Step 2: LLM으로 최신성 판정
    web_context = "\n".join(
        f"  - [{r.get('title', 'N/A')}] ({r.get('url', 'N/A')}) {r.get('content', '')[:400]}"
        for r in all_web[:15]
    )

    lim_context = "\n".join(
        f"  limitation_id={i}: [{l.get('paper_id', '')}] {l.get('claim', '')}"
        for i, l in enumerate(limitations)
    )

    llm = get_llm_for_agent(state, "recency_check")
    messages = [
        SystemMessage(content=RECENCY_PROMPT),
        HumanMessage(content=(
            f"## Limitations to verify ({len(limitations)}개)\n{lim_context}\n\n"
            f"## Recent web sources ({len(all_web)}개)\n{web_context}"
        )),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"  ⚠️ [recency] LLM 판정 실패: {e} → 전체 unresolved 처리")
        for lim in limitations:
            lim["recency_status"] = "unresolved"
            lim["recency_evidence"] = "LLM call failed"
        return {
            "messages": [AIMessage(content="Recency check failed. All unresolved.", name="recency_check")],
            "sender": "recency_check",
            "limitations": limitations,
            "web_results": all_web,
        }

    # Step 3: 파싱 및 매핑
    parsed = parse_json(content)
    if not isinstance(parsed, list):
        parsed = []

    recency_map = {}
    for item in parsed:
        lid = item.get("limitation_id")
        if lid is not None:
            try:
                recency_map[int(lid)] = {
                    "status": item.get("recency_status", "unresolved"),
                    "evidence": item.get("evidence", ""),
                }
            except (ValueError, TypeError):
                continue

    resolved_count = 0
    partial_count = 0
    for i, lim in enumerate(limitations):
        match = recency_map.get(i)
        if match:
            lim["recency_status"] = match["status"]
            lim["recency_evidence"] = match["evidence"]
            if match["status"] == "resolved":
                resolved_count += 1
            elif match["status"] == "partial":
                partial_count += 1
        else:
            lim["recency_status"] = "unresolved"
            lim["recency_evidence"] = "No matching recency check result"

    summary = (
        f"Recency check complete: {len(limitations)} limitations verified. "
        f"resolved={resolved_count}, partial={partial_count}, "
        f"unresolved={len(limitations) - resolved_count - partial_count}"
    )
    print(f"  ✅ [recency] {summary}")

    return {
        "messages": [AIMessage(content=summary, name="recency_check")],
        "sender": "recency_check",
        "limitations": limitations,
        "web_results": all_web,
    }
