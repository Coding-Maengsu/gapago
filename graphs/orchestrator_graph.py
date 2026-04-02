"""
GAPAGO Orchestrator Graph
오케스트레이터 패턴 기반 동적 그래프 — GAPAGO_ORCHESTRATOR=1 일 때 활성화.
기존 에이전트를 래퍼로 감싸서 completed_stages / agent_feedback 자동 업데이트.
"""

import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from states import AgentState
from .query_subgraph import build_subgraph

from agents import (
    meaning_expand_node,
    paper_retrieval_node,
    limitation_extract_node,
    limitation_eval_node,
    recency_check_node,
    gap_infer_node,
    critic_score_node,
    final_response_node,
)
from agents.orchestrator_agent import orchestrator_node

_ALLOWED_MODULES = [
    ("states", "Paper"),
    ("states", "LimitationItem"),
    ("states", "GapCandidate"),
    ("states", "CriticScores"),
    ("states", "DimensionScore"),
    ("states", "EvaluationResult"),
    ("states", "ScopeCandidate"),
    ("states", "ScopeAssessment"),
    ("states", "QueryResult"),
]

# ── 에이전트 노드 매핑 ──
AGENT_NODES = {
    "meaning_expand": meaning_expand_node,
    "paper_retrieval": paper_retrieval_node,
    "limitation_extract": limitation_extract_node,
    "limitation_eval": limitation_eval_node,
    "recency_check": recency_check_node,
    "gap_infer": gap_infer_node,
    "critic_score": critic_score_node,
    "final_response": final_response_node,
}


def _extract_feedback(agent_name: str, result: dict, prev_state: dict) -> dict:
    """에이전트 결과에서 오케스트레이터용 피드백 추출"""
    if agent_name == "limitation_eval":
        eval_data = result.get("limitation_eval", {})
        if isinstance(eval_data, dict) and eval_data.get("decision") == "RETRY":
            return {
                "from": "limitation_eval",
                "type": "quality_low",
                "detail": eval_data.get("retry_guidance", ""),
                "suggestion": "limitation_extract 재실행 권장",
            }

    elif agent_name == "critic_score":
        msgs = result.get("messages", [])
        if msgs:
            last_msg = msgs[-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            if "REDO_RETRIEVAL" in content:
                return {
                    "from": "critic_score",
                    "type": "relevance_low",
                    "suggestion": "meaning_expand + paper_retrieval 재실행 권장",
                }
            elif "REFINE_QUERY" in content:
                return {
                    "from": "critic_score",
                    "type": "query_weak",
                    "suggestion": "query 재정제 필요",
                }

    return {}


def _wrap_agent(agent_fn, agent_name: str):
    """기존 에이전트 노드를 래퍼로 감싸서 orchestrator 호환 상태 업데이트 추가"""

    if asyncio.iscoroutinefunction(agent_fn):
        async def wrapped(state: AgentState) -> dict:
            result = await agent_fn(state)
            result["completed_stages"] = [agent_name]
            result["agent_feedback"] = _extract_feedback(agent_name, result, state)
            return result

        wrapped.__name__ = f"{agent_name}_wrapped"
    else:
        def wrapped(state: AgentState) -> dict:
            result = agent_fn(state)
            result["completed_stages"] = [agent_name]
            result["agent_feedback"] = _extract_feedback(agent_name, result, state)
            return result

        wrapped.__name__ = f"{agent_name}_wrapped"

    return wrapped


def build_orchestrator_graph():
    """오케스트레이터 패턴 기반 동적 그래프 빌드"""
    query_subgraph = build_subgraph()
    workflow = StateGraph(AgentState)

    # ── query_subgraph (기존 그대로) ──
    workflow.add_node("query_subgraph", query_subgraph)

    # ── 오케스트레이터 노드 ──
    workflow.add_node("orchestrator", orchestrator_node)

    # ── 에이전트 노드들 (래퍼 적용) ──
    for name, fn in AGENT_NODES.items():
        workflow.add_node(name, _wrap_agent(fn, name))

    # ── 엣지: START → query_subgraph → orchestrator ──
    workflow.add_edge(START, "query_subgraph")
    workflow.add_edge("query_subgraph", "orchestrator")

    # ── 모든 에이전트 → orchestrator (실행 후 항상 오케스트레이터로 복귀) ──
    for name in AGENT_NODES:
        workflow.add_edge(name, "orchestrator")

    # orchestrator는 Command(goto=...) 반환으로 동적 라우팅
    # → add_conditional_edges 불필요

    serde = JsonPlusSerializer(allowed_json_modules=_ALLOWED_MODULES)
    graph = workflow.compile(
        checkpointer=MemorySaver(serde=serde),
    )

    return graph
