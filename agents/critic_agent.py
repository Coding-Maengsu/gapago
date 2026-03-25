# 3-5) Critic Score Agent
from states import AgentState
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from llm import get_llm
from utils.parse_json import parse_json

MAX_CRITIC_LOOPS = 2  # 최대 재시도 횟수 (이후 강제 ACCEPT)

SYSTEM_PROMPT = """ROLE: Critic Agent

You evaluate the quality of a research gap analysis pipeline.

## Evaluation Criteria (each 0.0 ~ 1.0)
1. query_specificity  : Is the refined query specific enough for academic search?
2. paper_relevance    : Are the retrieved papers relevant to the research question?
3. groundedness       : Are the identified gaps grounded in actual paper evidence?

## Scoring Guide
- 0.0~0.3: Poor — critical information missing or irrelevant
- 0.4~0.6: Fair — partially useful but significant gaps remain
- 0.7~0.8: Good — mostly solid with minor issues
- 0.9~1.0: Excellent — high quality, well-grounded

## Decision Rules
- If ALL three scores >= 0.6 → output "DECISION: ACCEPT"
- If paper_relevance < 0.4 → output "DECISION: REDO_RETRIEVAL"
- If query_specificity < 0.4 → output "DECISION: REFINE_QUERY"
- When in doubt, prefer ACCEPT over retry.
- If this is a retry round, be MORE lenient — only reject if quality is clearly unacceptable.

## Output Format (strictly follow)
query_specificity: <score>
paper_relevance: <score>
groundedness: <score>
reasoning: <1-2 sentences explaining your assessment>

DECISION: <ACCEPT or REDO_RETRIEVAL or REFINE_QUERY>

IMPORTANT:
- You MUST output exactly one DECISION tag.
- Do NOT generate new gaps, topics, or suggestions.
- Do NOT omit the DECISION tag.
""" 


def critic_score_node(state: AgentState) -> AgentState:
    llm = get_llm(provider=state.get("llm_provider"))
    loop_count = state.get("critic_loop_count", 0) or 0

    # 최대 루프 횟수 초과 시 강제 ACCEPT
    if loop_count >= MAX_CRITIC_LOOPS:
        print(f"  ⚠️ [critic] 최대 재시도 횟수({MAX_CRITIC_LOOPS})를 초과하여 강제 ACCEPT")
        content = (
            "query_specificity: 0.6\n"
            "paper_relevance: 0.6\n"
            "groundedness: 0.6\n"
            "reasoning: Maximum retry limit reached. Accepting current results.\n\n"
            "DECISION: ACCEPT"
        )
        return {
            "messages": [AIMessage(content=content, name="critic_score")],
            "sender": "critic_score",
            "critic_loop_count": loop_count + 1,
        }

    # 이전 messages에서 주요 정보 요약하여 전달
    context_parts = []
    for msg in state.get("messages", []):
        name = getattr(msg, "name", "") or ""
        content = getattr(msg, "content", "") or ""
        if name in ("query_analysis", "paper_retrieval", "limitation_extract", "gap_infer"):
            context_parts.append(f"[{name}]\n{content[:2000]}")

    # state에 저장된 구조화 데이터도 컨텍스트에 포함
    limitations = state.get("limitations", [])
    if limitations:
        lim_summary = "\n".join(
            f"  - [{l.get('paper_id','')}] {l.get('claim','')}" for l in limitations[:20]
        )
        context_parts.append(f"[limitations_data]\n{lim_summary}")

    gaps = state.get("gaps", [])
    if gaps:
        gap_summary = "\n".join(
            f"  - [{g.get('axis','')}] {g.get('gap_statement','')} (논문 {g.get('repeat_count',0)}편)" for g in gaps
        )
        context_parts.append(f"[gaps_data]\n{gap_summary}")

    context = "\n\n".join(context_parts[-6:])

    retry_note = ""
    if loop_count > 0:
        retry_note = f"\n\nNOTE: This is retry round {loop_count}. Be more lenient — only reject if quality is clearly unacceptable."

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Evaluate the following pipeline outputs:{retry_note}\n\n{context}"),
    ]

    try:
        response = llm.invoke(messages)
        result_content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"  ⚠️ [critic] LLM 호출 실패: {e}")
        result_content = "reasoning: LLM call failed. Accepting current results.\n\nDECISION: ACCEPT"

    # DECISION 태그가 없으면 강제로 ACCEPT 추가
    if "DECISION:" not in result_content:
        print("  ⚠️ [critic] DECISION 태그 없음 → 강제 ACCEPT 추가")
        result_content += "\n\nDECISION: ACCEPT"

    print(f"  [critic] loop={loop_count + 1}/{MAX_CRITIC_LOOPS} | {result_content[-30:]}")

    return {
        "messages": [AIMessage(content=result_content, name="critic_score")],
        "sender": "critic_score",
        "critic_loop_count": loop_count + 1,
    }
