"""
================================================== 2. QUERY REFINEMENT (v2) ==================================================

[적용 논문 및 코드 매핑]

① APA — Alignment with Perceived Ambiguity (Kim et al., EMNLP 2024 | arXiv:2404.11972)
   논문 근거: "we propose Alignment with Perceived Ambiguity (APA), a novel pipeline
               that aligns LLMs to manage ambiguous queries by leveraging their own
               assessment of ambiguity (i.e., perceived ambiguity)."
   코드 반영:
   - query_analysis_node가 state["ambiguity_signals"]에 저장한 perceived_ambiguous,
     dominant_interpretation을 활용하여 refined_query 생성
   - APA의 "dominant_interpretation"을 refined_query의 출발점으로 사용
   - 모호하지 않은 경우(not perceived_ambiguous) → query_analysis의 suggested_query 직접 사용

[기존 대비 변경점]
- 기존: create_agent + JSON 파싱 (불안정)
- 신규: structured_output (안정적) + APA dominant_interpretation 활용
        → 만약 query_analysis 단계에서 이미 APA가 dominant_interpretation을 확정했다면
          그것을 refined_query 베이스로 사용 (불필요한 LLM 호출 절감)
=======================================================================================================================
"""

import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List

from gapago.core.states import AgentState
from gapago.core.llm import get_llm_for_agent
from gapago.core.prompts import make_system_prompt
from gapago.utils.parse_json import parse_json


# ── Structured output 스키마 ────────────────────────────────────────────────
class RefinedQuery(BaseModel):
    """Query Refinement Agent의 structured output"""
    refined_query: str = Field(
        description=(
            "A precise, retrieval-ready academic search query. "
            "Should be specific enough for arXiv/PubMed searches. "
            "If a dominant_interpretation was provided, build upon it."
        )
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="2~5 core search keywords (domain terms, task terms, method terms)."
    )
    negative_keywords: List[str] = Field(
        default_factory=list,
        description="Optional 1~3 exclusion terms."
    )
    refinement_note: str = Field(
        default="",
        description="Brief note on what was refined or clarified from the original query."
    )


REFINE_SYSTEM_PROMPT = make_system_prompt(
    "ROLE: Query Refinement Agent (v2)\n"
    "Transform the clarified user research question into a retrieval-ready academic search query.\n\n"

    "INPUT CONTEXT (from previous agents):\n"
    "  - original_query: The user's original input.\n"
    "  - suggested_query: Query Analysis Agent's suggested reformulation.\n"
    "  - dominant_interpretation: APA's most plausible interpretation of the query.\n"
    "    → If provided, USE THIS as the foundation for refined_query.\n"
    "  - keywords: Keywords already extracted by Query Analysis Agent.\n"
    "  - perceived_ambiguous / other_interpretations: APA self-assessment.\n"
    "    → Resolve toward dominant_interpretation. Do NOT blend other readings in.\n\n"

    "REFINEMENT RULES:\n"
    "1. If dominant_interpretation is non-empty, start from it (it's the most plausible reading).\n"
    "2. Make the query specific enough for academic database search (arXiv, PubMed, ScienceON).\n"
    "3. Preserve the user's core intent — do not over-expand or change the topic.\n"
    "4. keywords: 2~5 terms, prioritize domain + task + method terms.\n"
    "5. negative_keywords: only include if the query needs explicit exclusion.\n"
    "6. refinement_note: briefly explain what you improved (1 sentence).\n"
)


def query_refinement_node(state: AgentState) -> AgentState:
    """
    Query Refinement Node (v2)

    변경점:
    - 기존 create_agent + JSON 파싱 → structured_output으로 안정화
    - APA의 dominant_interpretation을 refined_query 기반으로 활용
    - ambiguity_signals가 있으면 context로 제공
    """

    # ── 1. state에서 컨텍스트 수집 ──────────────────────────────────────────
    original_query = state.get("user_question", "")
    suggested_query = state.get("refined_query", "")
    existing_keywords = state.get("keywords", [])
    existing_neg_keywords = state.get("negative_keywords", [])
    ambiguity_signals = state.get("ambiguity_signals", {})

    # APA — query_analysis 가 채운 ambiguity_signals 를 그대로 사용한다
    # (gapago/core/states.py 의 AmbiguitySignals 스키마)
    dominant_interpretation = ambiguity_signals.get("dominant_interpretation", "")
    perceived_ambiguous = bool(ambiguity_signals.get("perceived_ambiguous", False))

    # ── 2. 모호하지 않으면 LLM 호출을 생략한다 ───────────────────────────────
    # APA 의 핵심: 모델이 스스로 모호하지 않다고 판단했다면 재해석이 불필요하다.
    is_apa_clear = (not perceived_ambiguous) and bool(suggested_query)

    llm = get_llm_for_agent(state, "query_refine")
    structured_refine_llm = llm.with_structured_output(RefinedQuery)

    if is_apa_clear and existing_keywords:
        # APA + CLAMBER 모두 문제없음 → suggested_query를 그대로 refined_query로 사용
        refined = suggested_query
        note = "APA: not perceived ambiguous — suggested_query 를 그대로 사용 (LLM 호출 생략)."
        keywords = existing_keywords
        negative_keywords = existing_neg_keywords

        content_out = json.dumps({
            "refined_query": refined,
            "keywords": keywords,
            "negative_keywords": negative_keywords,
            "refinement_note": note,
        }, ensure_ascii=False)
    else:
        # ── 3. LLM refinement (structured output) ───────────────────────────
        context_parts = []
        if original_query:
            context_parts.append(f"original_query: {original_query}")
        if suggested_query:
            context_parts.append(f"suggested_query: {suggested_query}")
        if dominant_interpretation:
            context_parts.append(f"dominant_interpretation (APA): {dominant_interpretation}")
        if existing_keywords:
            context_parts.append(f"keywords: {existing_keywords}")
        if perceived_ambiguous:
            context_parts.append(
                "perceived_ambiguous: true — resolve toward dominant_interpretation"
            )
            others = [i for i in ambiguity_signals.get("interpretations", [])
                      if i != dominant_interpretation]
            if others:
                context_parts.append(f"other_interpretations (do NOT mix in): {others}")
            if ambiguity_signals.get("ambiguity_rationale"):
                context_parts.append(f"ambiguity_rationale: {ambiguity_signals['ambiguity_rationale']}")

        context_text = "\n".join(context_parts)

        input_messages = [
            SystemMessage(content=REFINE_SYSTEM_PROMPT),
            HumanMessage(content=context_text),
        ]

        try:
            parsed: RefinedQuery = structured_refine_llm.invoke(input_messages)
            refined = parsed.refined_query
            keywords = [k.strip() for k in (parsed.keywords or []) if k.strip()][:5]
            negative_keywords = [k.strip() for k in (parsed.negative_keywords or []) if k.strip()][:3]
            note = parsed.refinement_note
            content_out = parsed.model_dump_json()
        except Exception as e:
            # fallback: suggested_query 유지
            refined = suggested_query or original_query
            keywords = existing_keywords
            negative_keywords = existing_neg_keywords
            note = f"Refinement failed ({e}). Using suggested_query as fallback."
            content_out = json.dumps({
                "refined_query": refined,
                "keywords": keywords,
                "negative_keywords": negative_keywords,
                "refinement_note": note,
            }, ensure_ascii=False)

    last = AIMessage(content=content_out, name="query_refinement")

    return {
        "messages": [last],
        "sender": "query_refinement",
        "refined_query": refined,
        "keywords": keywords if keywords else existing_keywords,
        "negative_keywords": negative_keywords if negative_keywords else existing_neg_keywords,
    }