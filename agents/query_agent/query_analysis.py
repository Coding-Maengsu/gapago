"""
================================================== QUERY ANALYSIS (v3) ==================================================

[설계 원칙]
목적: 사용자의 연구 방향성을 받아 논문 검색이 가능한 쿼리를 만든다.
     GAP 분석과 무관. 검색 가능성만 판단.

[적용 논문]

① SemRank (Zhang et al., EMNLP 2025 | arXiv:2505.21815)
   논문 근거:
     "Each paper is indexed using multi-granular scientific concepts,
      including general research topics and detailed key phrases."
     "broad topics aim to cover overall themes not explicitly mentioned
      such as 'natural language generation' and 'automatic evaluation',
      while key phrases capture detailed information specific to the paper."
   코드 반영:
     - ScopeAssessment.general_topic   : 큰 연구 분야 (이것만 있으면 TOO_BROAD)
     - ScopeAssessment.specific_phrases: 실제 검색에 쓸 구체적 키워드
     - specific_phrases 없음 → TOO_BROAD
     - specific_phrases 1개 이상 → SEARCHABLE

② CoQuest (Liu et al., CHI 2024 | arXiv:2310.06155)
   논문 근거:
     "breadth-first design made users feel more creative and gained
      more trust from users"
     "AI Thoughts: explaining AI's rationale of why each RQ is generated"
     "Refine and Re-scope" 단계가 Human-AI co-creation의 핵심
   코드 반영:
     - TOO_BROAD  → breadth_candidates 3개 동시 제시 (breadth-first)
     - SEARCHABLE → rationale 함께 출력 (AI Thoughts)
     - 사용자 확인 후 Re-scope 루프 (Human-in-the-loop)
=========================================================================================================================
"""

import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from states import AgentState, QueryResult
from llm import get_llm

# =====================================================================================================================
# 0) 시스템 프롬프트
# =====================================================================================================================

SYSTEM_PROMPT = """You are a Query Analysis Agent for an academic paper search system.

Your ONLY job: determine if a user's research direction can be used to search for academic papers on arXiv.
You do NOT care about research design, methodology choices, or GAP analysis suitability.
You ONLY care: "Can we search for papers with this input?"

=== SCOPE ASSESSMENT (based on SemRank, Zhang et al. EMNLP 2025) ===

Classify the input into ONE of three levels:

1. TOO_BROAD
   - Only a general research field is mentioned, no specific concept
   - Examples: "natural language processing", "computer vision", "deep learning", "AI"
   - Expected arXiv results: hundreds of thousands → meaningful search impossible

2. SEARCHABLE
   - At least one specific phrase/concept that can directly retrieve papers
   - Examples: "deepfake detection", "medical image segmentation", "multimodal emotion recognition"
   - Expected arXiv results: hundreds to thousands → proceed

3. TOO_NARROW
   - The combination of specific phrases is too unusual → almost no papers exist
   - Examples: "quantum computing for Korean hate speech detection"
   - Expected arXiv results: near zero

=== CLASSIFICATION RULE ===

Extract from the input:
  general_topic   : The broad research field (e.g., "deepfake detection")
  specific_phrases: Concrete keywords directly usable for arXiv search

Decision:
  - specific_phrases is EMPTY                        → TOO_BROAD
  - specific_phrases has 1+ entries                  → SEARCHABLE
  - specific_phrases exist but combination too rare  → TOO_NARROW

=== OUTPUT BY LEVEL ===

TOO_BROAD:
  - breadth_candidates: exactly 3 sub-directions (each SEARCHABLE level)
    Each: direction (str), rationale (why searchable), sample_keywords (list)
  - rationale: why the input is too broad

SEARCHABLE:
  - refined_query: clean academic search query preserving user intent
  - keywords: 2~5 arXiv search keywords (NO generic: AI, model, system, method)
  - negative_keywords: only if clearly needed (1~3 max)
  - rationale: why searchable (AI Thoughts)
  - breadth_candidates: leave EMPTY

TOO_NARROW:
  - expansion_suggestion: why almost no papers exist + broader angle suggestion
  - rationale: explanation
  - refined_query, keywords: leave EMPTY

=== RATIONALE WRITING RULES ===

When writing the `rationale` field, follow these strict rules:
- Focus ONLY on what is missing or unclear in the user's input (e.g., no task, no domain, no application).
- Do NOT mention arXiv, search engines, databases, or retrieval systems by name.
- Do NOT describe expected search result volume (e.g., "hundreds of thousands of papers").
- Do NOT use phrases like "meaningful search impossible" or "retrieval impractical".
- Keep it short (1~2 sentences max) and user-friendly.

Example of BAD rationale (forbidden):
  "The input only names a broad model family without any task or domain. Searching arXiv with this alone would return an extremely large and unfocused set of papers, making meaningful retrieval impractical."

Example of GOOD rationale (allowed):
  "입력하신 내용만으로는 구체적인 연구 과제나 적용 분야를 알 수 없습니다."
  "No specific task, domain, or application is mentioned."

=== LANGUAGE ===
Match the language of the user's input.
Korean input → all text fields in Korean. Keywords always in English.
"""


# =====================================================================================================================
# 1) 헬퍼 함수
# =====================================================================================================================

def _collect_user_input(state: AgentState) -> str:
    """누적된 HumanMessage를 하나로 합산"""
    return "\n".join(
        m.content.strip()
        for m in state["messages"]
        if isinstance(m, HumanMessage)
    )


def _build_scope_message(result: QueryResult) -> str:
    """
    판정 결과 → 사용자에게 보여줄 메시지
    - TOO_BROAD  : CoQuest breadth-first — 후보 3개 동시 제시
    - SEARCHABLE : CoQuest AI Thoughts  — 근거 + 키워드 표시
    - TOO_NARROW : 확장 제안
    """
    sa = result.scope_assessment
    level = sa.scope_level

    if level == "TOO_BROAD":
        lines = [
            "입력하신 연구 방향이 너무 넓어서 논문 검색이 어렵습니다.",
            f"\n[판정 근거] {sa.rationale}",
            "\n아래 세 가지 방향 중 하나를 선택하시거나, 원하는 방향을 직접 입력해주세요:\n",
        ]
        for i, cand in enumerate(sa.breadth_candidates, 1):
            lines.append(f"  {i}. {cand.direction}")
            lines.append(f"     └ {cand.rationale}")
            if cand.sample_keywords:
                lines.append(f"     └ 예상 키워드: {', '.join(cand.sample_keywords)}")
            lines.append("")
        return "\n".join(lines)

    elif level == "TOO_NARROW":
        return "\n".join([
            "입력하신 방향은 관련 논문이 거의 없을 것으로 예상됩니다.",
            f"\n[판정 근거] {sa.rationale}",
            f"\n[제안] {sa.expansion_suggestion}",
            "\n방향을 조금 더 넓혀서 다시 입력해주시겠어요?",
        ])

    else:  # SEARCHABLE
        lines = [
            "입력하신 방향으로 논문 검색이 가능합니다.",
            f"\n[판정 근거] {sa.rationale}",
            f"\n[검색 쿼리] {result.refined_query}",
            f"[검색 키워드] {', '.join(result.keywords)}",
        ]
        if result.negative_keywords:
            lines.append(f"[제외 키워드] {', '.join(result.negative_keywords)}")
        lines.append(
            "\n이 방향으로 논문 검색을 시작할까요? "
            "(계속하려면 Enter, 수정하려면 다른 방향을 입력해주세요)"
        )
        return "\n".join(lines)


# =====================================================================================================================
# 2) 노드 함수
# =====================================================================================================================

def query_analysis_node(state: AgentState) -> AgentState:
    """
    Query Analysis 메인 노드

    1. 사용자 입력 수집
    2. LLM → Scope Assessment (SemRank 기준)
    3. 판정 메시지 생성 (CoQuest 방식)
    4. SEARCHABLE → keywords 확정 / 아니면 사용자 입력 대기
    """
    it = state.get("iteration", 0) + 1
    max_it = state.get("max_iterations", 3)

    user_input = _collect_user_input(state)

    # 숫자 입력 → breadth_candidates 선택으로 치환 (TOO_BROAD 재질문 시)
    candidates = state.get("breadth_candidates", [])
    last_human = [m.content.strip() for m in state["messages"] if isinstance(m, HumanMessage)]
    if candidates and last_human:
        choice = last_human[-1]
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            selected = candidates[int(choice) - 1]
            direction = selected.get("direction", "") if isinstance(selected, dict) else selected.direction
            user_input = direction

    from prompts.system import get_language_instruction
    output_language = state.get("output_language", "auto")
    lang_instruction = get_language_instruction(output_language)

    llm = get_llm(provider=state.get("llm_provider"))
    structured_llm = llm.with_structured_output(QueryResult)
    result: QueryResult = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT + lang_instruction),
        HumanMessage(content=user_input),
    ])
    sa = result.scope_assessment

    scope_msg = _build_scope_message(result)
    out_messages = [
        AIMessage(
            content=json.dumps(result.model_dump(), ensure_ascii=False),
            name="query_analysis",
        ),
        AIMessage(content=scope_msg, name="clarify_prompt"),
    ]

    is_searchable = (sa.scope_level == "SEARCHABLE")
    needs_user_input = (not is_searchable) and (it < max_it)

    return {
        "messages": out_messages,
        "sender": "query_analysis",
        "iteration": it,
        "scope_level": sa.scope_level,
        "scope_rationale": sa.rationale,
        "breadth_candidates": [c.model_dump() for c in sa.breadth_candidates],
        "expansion_suggestion": sa.expansion_suggestion,
        "keywords": result.keywords if is_searchable else [],
        "negative_keywords": result.negative_keywords if is_searchable else [],
        "refined_query": result.refined_query if is_searchable else "",
        "user_question": user_input,
        "needs_user_input": needs_user_input,
    }


def human_clarify_node(state: AgentState) -> AgentState:
    """Human-in-the-loop interrupt 전용 노드"""
    return {"sender": "human_clarify"}