from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm import get_llm
from states import AgentState


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text or "")
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def _extract_query_context(state: AgentState) -> dict[str, Any]:
    refined_query = _norm(state.get("refined_query", ""))
    keywords = [_norm(k) for k in (state.get("keywords", []) or []) if _norm(k)]
    negative_keywords = [_norm(k) for k in (state.get("negative_keywords", []) or []) if _norm(k)]

    if refined_query and keywords:
        return {
            "refined_query": refined_query,
            "keywords": keywords,
            "negative_keywords": negative_keywords,
        }

    messages = state.get("messages") or []
    for msg in reversed(messages):
        name = getattr(msg, "name", "") or ""
        content = getattr(msg, "content", "") or ""
        if name != "query_analysis":
            continue
        parsed = _safe_json_loads(content)
        if isinstance(parsed, dict):
            refined_query = refined_query or _norm(
                parsed.get("refined_query") or parsed.get("query_proposal") or ""
            )
            raw_keywords = parsed.get("keywords") or []
            if not keywords and isinstance(raw_keywords, list):
                keywords = [_norm(k) for k in raw_keywords if isinstance(k, str) and _norm(k)]
            raw_negative = parsed.get("negative_keywords") or []
            if not negative_keywords and isinstance(raw_negative, list):
                negative_keywords = [_norm(k) for k in raw_negative if isinstance(k, str) and _norm(k)]
        if refined_query or keywords:
            break

    if not refined_query:
        refined_query = _norm(state.get("user_question", ""))
    if not keywords and refined_query:
        keywords = [refined_query]

    return {
        "refined_query": refined_query,
        "keywords": keywords,
        "negative_keywords": negative_keywords,
    }


def meaning_expand_node(state: AgentState) -> AgentState:
    llm = get_llm(provider=state.get("llm_provider"))

    ctx = _extract_query_context(state)
    refined_query = ctx["refined_query"]
    keywords = ctx["keywords"]
    negative_keywords = ctx["negative_keywords"]

    trace = state.get("trace") or {}
    errors = state.get("errors") or []

    trace.setdefault("retrieval", {})
    trace["retrieval"].setdefault("meaning_expand", {})
    mem = trace.get("memory", {}) or {}

    prompt = f"""You are an information retrieval engineer.

INPUT:
- refined_query: {refined_query}
- keywords: {keywords}
- negative_keywords: {negative_keywords}
- user_memory (optional): {json.dumps(mem, ensure_ascii=False)[:1500]}

GOAL:
1) Expand keywords with acronyms, synonyms, spelling variants, and close technical terms.
2) Prepare search candidates for the Retrieval Agent.
3) Do not call any search tool. This stage only prepares retrieval hints.

OUTPUT JSON ONLY with keys:
{{
    "refined_query": "string",
    "keywords": ["..."],
    "negative_keywords": ["..."],
    "expanded_terms": ["<=12 terms"],
    "arxiv_query_candidates": ["<=4 queries"],
    "web_query_candidates": ["<=4 queries"],
    "scienceon_query_candidates": ["<=3 queries"],
    "notes": ["optional notes for retrieval agent"]
}}
"""

    messages = [
        SystemMessage(content="You generate retrieval-ready keyword expansions with high recall."),
        HumanMessage(content=prompt),
    ]

    try:
        resp = llm.invoke(messages)
        data = _safe_json_loads(getattr(resp, "content", "") or "")
    except Exception as e:
        errors.append(f"[meaning_expand] LLM expansion failed: {e}")
        data = {}

    expanded_terms = data.get("expanded_terms", []) or []
    expanded_terms = [_norm(x) for x in expanded_terms if isinstance(x, str) and _norm(x)][:12]

    def _clean_list(key: str, limit: int) -> list[str]:
        values = data.get(key, []) or []
        if not isinstance(values, list):
            return []
        return [_norm(v) for v in values if isinstance(v, str) and _norm(v)][:limit]

    arxiv_candidates = _clean_list("arxiv_query_candidates", 4)
    web_candidates = _clean_list("web_query_candidates", 4)
    scienceon_candidates = _clean_list("scienceon_query_candidates", 3)
    notes = _clean_list("notes", 6)

    if not arxiv_candidates and refined_query:
        arxiv_candidates = [refined_query]
    if not web_candidates and refined_query:
        web_candidates = [refined_query]
    if not scienceon_candidates and refined_query:
        scienceon_candidates = [refined_query]

    payload = {
        "refined_query": refined_query,
        "keywords": keywords,
        "negative_keywords": negative_keywords,
        "expanded_terms": expanded_terms,
        "arxiv_query_candidates": arxiv_candidates,
        "web_query_candidates": web_candidates,
        "scienceon_query_candidates": scienceon_candidates,
        "notes": notes,
    }

    trace["retrieval"]["meaning_expand"] = payload

    last = AIMessage(content=json.dumps(payload, ensure_ascii=False), name="meaning_expand")
    return {
        "messages": [last],
        "sender": "meaning_expand",
        "trace": trace,
        "errors": errors,
        "refined_query": refined_query,
        "keywords": keywords,
        "negative_keywords": negative_keywords,
    }
