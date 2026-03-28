from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
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


def route_after_eval(state: AgentState) -> str:
    """
    limitation_eval →
      - PASS  → recency_check (계속 진행)
      - RETRY → limitation_extract (재추출)
    """
    eval_result = state.get("limitation_eval", {})
    decision = eval_result.get("decision", "PASS")

    if decision == "RETRY":
        print("  [graph] limitation_eval → RETRY → limitation_extract")
        return "limitation_extract"

    return "recency_check"


def route_after_critic(state: AgentState) -> str:
    """
    critic_score ->
      - ACCEPT          -> final_response
      - REDO_RETRIEVAL  -> meaning_expand -> paper_retrieval
      - REFINE_QUERY    -> query_analysis
      - FINAL ANSWER    -> END
      - fallback        -> final_response (루프 방지)
    """
    last = state["messages"][-1].content or ""

    if "FINAL ANSWER" in last:
        return END

    if "DECISION: ACCEPT" in last:
        return "final_response"

    if "DECISION: REDO_RETRIEVAL" in last:
        return "meaning_expand"

    if "DECISION: REFINE_QUERY" in last:
        return "query_subgraph"

    # 태그 매칭 실패 시 루프 방지를 위해 final_response로 이동
    return "final_response"


def build_graph():
    query_subgraph = build_subgraph()

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("query_subgraph", query_subgraph)
    workflow.add_node("meaning_expand", meaning_expand_node)
    workflow.add_node("paper_retrieval", paper_retrieval_node)
    workflow.add_node("limitation_extract", limitation_extract_node)
    workflow.add_node("limitation_eval", limitation_eval_node)
    workflow.add_node("recency_check", recency_check_node)
    workflow.add_node("gap_infer", gap_infer_node)
    workflow.add_node("critic_score", critic_score_node)
    workflow.add_node("final_response", final_response_node)

    # Define edges
    # start -> query_analysis
    workflow.add_edge(START, "query_subgraph")
    workflow.add_edge("query_subgraph", "meaning_expand")
    workflow.add_edge("meaning_expand", "paper_retrieval")
    workflow.add_edge("paper_retrieval", "limitation_extract")
    workflow.add_edge("limitation_extract", "limitation_eval")

    # limitation_eval → PASS: recency_check, RETRY: limitation_extract
    workflow.add_conditional_edges(
        "limitation_eval",
        route_after_eval,
        {
            "recency_check": "recency_check",
            "limitation_extract": "limitation_extract",
        },
    )

    workflow.add_edge("recency_check", "gap_infer")
    workflow.add_edge("gap_infer", "critic_score")

    workflow.add_conditional_edges(
        "critic_score",
        route_after_critic,
        {
            "meaning_expand": "meaning_expand",
            "query_subgraph": "query_subgraph",
            "final_response": "final_response",
            END: END,
        },
    )

    workflow.add_edge("final_response", END)

    graph = workflow.compile(
        checkpointer=MemorySaver(),
    )

    return graph


"""
체크포인터(memory)
- 각 노드간 실행결과를 추적하기 위한 메모리
- 체크포인터를 활용하여 특정 시점(snapshot)으로 되돌리기 기능도 가능!
- multi turn 대화에도 유용함(thread_id 만 변경하면 새로운 대화로 바꿔줌)
- compile 지정하여 그래프 생성
- Human-In-The-Loop 를 위해 필수 요소

-`get_state_history` 메서드를 사용하여 상태 기록을 가져오는 방법
- 상태 기록을 통해 원하는 상태를 지정하여 해당 지점에서 다시 시작 가능(Replay 기능)
"""
