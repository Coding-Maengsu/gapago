"""
GAPAGO Orchestrator Agent
LLM 기반 오케스트레이터 — 현재 state를 읽고 다음 실행할 에이전트를 결정.
Command(goto=...) 반환으로 LangGraph 동적 라우팅.
"""

import json
from collections import Counter
from langchain_core.messages import AIMessage
from langgraph.types import Command
from gapago.core.llm import get_llm_for_agent

# ── 필수 실행 순서 (코드 레벨 강제) ──
MANDATORY_SEQUENCE = [
    "meaning_expand",
    "paper_retrieval",
    "limitation_extract",
    "gap_infer",
    "final_response",
]

# 필수 단계 선행 조건
DEPS = {
    "meaning_expand": [],
    "paper_retrieval": ["meaning_expand"],
    "limitation_extract": ["paper_retrieval"],
    "gap_infer": ["limitation_extract"],
    "final_response": ["gap_infer"],
}

# 선택적 에이전트 (오케스트레이터가 삽입 가능)
OPTIONAL_AGENTS = [
    "limitation_eval",
    "recency_check",
    "critic_score",
]

# 모든 유효한 에이전트 이름
ALL_AGENTS = set(MANDATORY_SEQUENCE + OPTIONAL_AGENTS)

MAX_ORCHESTRATOR_STEPS = 15
MAX_AGENT_RERUNS = 2


def _dependencies_met(stage: str, completed: list) -> bool:
    """필수 단계의 선행 조건이 모두 완료되었는지 확인"""
    return all(dep in completed for dep in DEPS.get(stage, []))


def _next_mandatory(completed: list) -> str | None:
    """완료되지 않은 다음 필수 단계 반환"""
    for stage in MANDATORY_SEQUENCE:
        if stage not in completed and _dependencies_met(stage, completed):
            return stage
    return None


def _build_orchestrator_prompt(
    completed: list,
    next_mandatory: str | None,
    feedback: dict,
    state: dict,
) -> str:
    papers_count = len(state.get("papers", []) or [])
    limitations_count = len(state.get("limitations", []) or [])
    gaps_count = len(state.get("gaps", []) or [])
    fast_mode = state.get("fast_mode", False)

    feedback_str = json.dumps(feedback, ensure_ascii=False) if feedback else "None"

    available_actions = []
    recommended_action = None

    if next_mandatory:
        available_actions.append(
            f'- Proceed to "{next_mandatory}" (continue pipeline)'
        )

    # optional 삽입 — 품질 게이트 강력 권장
    if "limitation_extract" in completed and "gap_infer" not in completed:
        if "limitation_eval" not in completed:
            available_actions.append(
                '- **RECOMMENDED**: Insert "limitation_eval" (validates extraction quality — skipping risks low-quality gaps)'
            )
            if not recommended_action:
                recommended_action = "limitation_eval"
        if "recency_check" not in completed:
            available_actions.append(
                '- **RECOMMENDED**: Insert "recency_check" (verifies limitations against recent research — enriches web_results for gap inference)'
            )
            if not recommended_action:
                recommended_action = "recency_check"

    if "gap_infer" in completed and "final_response" not in completed:
        if "critic_score" not in completed:
            available_actions.append(
                '- **RECOMMENDED**: Insert "critic_score" (quality gate — can trigger re-retrieval if analysis is weak)'
            )
            if not recommended_action:
                recommended_action = "critic_score"

    # 재실행 가능 여부
    counts = Counter(completed)
    if counts.get("meaning_expand", 0) < MAX_AGENT_RERUNS and feedback:
        available_actions.append(
            '- Re-run "meaning_expand" (feedback suggests query improvement needed)'
        )
    if counts.get("paper_retrieval", 0) < MAX_AGENT_RERUNS and feedback:
        available_actions.append(
            '- Re-run "paper_retrieval" (feedback suggests more/better papers needed)'
        )

    actions_str = "\n".join(available_actions) if available_actions else "No actions available — proceed to END."

    # fast_mode 컨텍스트 — 강제가 아닌 가이드라인 제공
    fast_hint = ""
    if fast_mode:
        fast_hint = (
            "\n## ⚡ FAST MODE 활성화됨\n"
            "사용자가 빠른 분석을 요청했습니다. 속도와 품질의 균형을 판단하세요:\n"
            "- optional 에이전트(limitation_eval, recency_check, critic_score)는 **스킵 가능**하나,\n"
            "  현재 데이터 품질이 낮다고 판단되면 실행해도 됩니다.\n"
            "- 예: limitation이 매우 적거나 feedback에 품질 문제가 있으면 eval 실행 고려.\n"
            "- 판단 기준: 스킵해도 최종 결과에 큰 영향이 없는가?\n"
        )

    # 품질 게이트 권장 메시지
    quality_hint = ""
    if recommended_action:
        quality_hint = f"""
## Quality Policy
You SHOULD insert quality-check stages when they are available.
The recommended next action is "{recommended_action}".
Only skip a recommended stage if you have a strong, specific reason (e.g., feedback confirms quality is already high).
Skipping quality gates degrades final output — the system was designed to use them."""

    return f"""You are the Pipeline Orchestrator for GAPAGO, a research gap analysis system.
Your goal is to produce the HIGHEST QUALITY gap analysis, not just the fastest one.
You decide which agent to run next based on the current state.

## Current State Summary
- Completed stages: {completed}
- Total steps taken: {len(completed)} / {MAX_ORCHESTRATOR_STEPS}
- Next mandatory stage: {next_mandatory or "ALL DONE"}
- Agent feedback: {feedback_str}
- Papers found: {papers_count}
- Limitations extracted: {limitations_count}
- Gaps inferred: {gaps_count}
{fast_hint}{quality_hint}

## Available Actions
{actions_str}

## Decision Rules
1. You CANNOT skip mandatory stages.
2. When a **RECOMMENDED** action is available, you SHOULD choose it unless there is a concrete reason not to.
3. Quality-check stages (limitation_eval, recency_check, critic_score) catch errors early and improve downstream results.
4. If agent feedback reports quality issues, re-run the relevant stage.
5. If all mandatory stages are done, output "__end__".

Respond with ONLY a JSON object:
{{"next_agent": "<agent_name>", "reasoning": "<1 sentence>"}}"""


def _parse_decision(response_text: str, fallback: str | None) -> str:
    """LLM 응답에서 next_agent 파싱"""
    try:
        # JSON 블록 추출
        text = response_text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            agent = data.get("next_agent", "")
            if agent == "__end__":
                return "__end__"
            if agent in ALL_AGENTS:
                return agent
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    return fallback or "__end__"


def orchestrator_node(state: dict) -> Command:
    """
    오케스트레이터 노드 — LLM이 다음 실행할 에이전트를 결정.
    필수 경로 제약은 코드 레벨에서 강제.
    """
    completed = list(state.get("completed_stages", []) or [])
    feedback = state.get("agent_feedback", {}) or {}
    counts = Counter(completed)

    # ── 안전장치: 최대 스텝 도달 ──
    if len(completed) >= MAX_ORCHESTRATOR_STEPS:
        if "final_response" not in completed:
            return Command(
                goto="final_response",
                update={
                    "orchestrator_plan": ["final_response"],
                    "agent_feedback": {},
                },
            )
        return Command(goto="__end__")

    # ── 다음 필수 단계 확인 ──
    next_mandatory = _next_mandatory(completed)

    # ── 모든 필수 단계 완료 → END ──
    if next_mandatory is None:
        return Command(goto="__end__")

    # ── LLM에게 결정 요청 ──
    prompt = _build_orchestrator_prompt(completed, next_mandatory, feedback, state)

    try:
        llm = get_llm_for_agent(state, "orchestrator")
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        decision = _parse_decision(response_text, fallback=next_mandatory)
    except Exception as e:
        print(f"  [orchestrator] LLM 실패, fallback 사용: {e}")
        decision = next_mandatory

    # ── 결정 검증 ──

    # 1) __end__ 결정인데 필수 단계 남아있으면 거부
    if decision == "__end__" and next_mandatory:
        decision = next_mandatory

    # 2) 유효하지 않은 에이전트명이면 fallback
    if decision not in ALL_AGENTS and decision != "__end__":
        decision = next_mandatory

    # 3) 재실행 횟수 초과 체크
    if counts.get(decision, 0) >= MAX_AGENT_RERUNS and decision in OPTIONAL_AGENTS:
        decision = next_mandatory
    if decision in MANDATORY_SEQUENCE and decision in completed and counts.get(decision, 0) >= MAX_AGENT_RERUNS:
        decision = next_mandatory

    # 4) 선행 조건 미충족 에이전트면 fallback
    if decision in DEPS and not _dependencies_met(decision, completed):
        decision = next_mandatory

    # 5) final_response 이후 아직 END가 아니면 END
    if decision == "__end__":
        return Command(goto="__end__")

    print(f"  [orchestrator] step={len(completed)+1}/{MAX_ORCHESTRATOR_STEPS} → {decision}")

    return Command(
        goto=decision,
        update={
            "orchestrator_plan": [decision],
            "agent_feedback": {},  # 피드백 소비 후 초기화
        },
    )
