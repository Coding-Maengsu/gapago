from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from gapago.core.states import AgentState
from gapago.agents import human_clarify_node, query_analysis_node, query_refinement_node


def route_after_query_analysis(state: AgentState) -> str:
    """
    SEARCHABLE 이면 query_refine 으로 넘겨 APA 기반 최종 정제를 수행한다.
    TOO_BROAD / TOO_NARROW 이면 사용자에게 되묻는다.
    """
    if state.get("needs_user_input", False):
        return "human_clarify"
    return "query_refine"


def build_subgraph():
    builder = StateGraph(AgentState)

    builder.add_node("query_analysis", query_analysis_node)
    builder.add_node("human_clarify", human_clarify_node)
    builder.add_node("query_refine", query_refinement_node)

    builder.add_edge(START, "query_analysis")
    builder.add_conditional_edges(
        "query_analysis",
        route_after_query_analysis,
        {
            "human_clarify": "human_clarify",
            "query_refine": "query_refine",
        },
    )
    builder.add_edge("human_clarify", "query_analysis")
    builder.add_edge("query_refine", END)

    graph = builder.compile(
        interrupt_before=["human_clarify"],
    )
    return graph