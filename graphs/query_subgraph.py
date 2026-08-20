from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from core.states import AgentState
from agents import human_clarify_node, query_analysis_node


def route_after_query_analysis(state: AgentState) -> str:
    if state.get("needs_user_input", False):
        return "human_clarify"
    return "next"


def build_subgraph():
    builder = StateGraph(AgentState)

    builder.add_node("query_analysis", query_analysis_node)
    builder.add_node("human_clarify", human_clarify_node)

    builder.add_edge(START, "query_analysis")
    builder.add_conditional_edges(
        "query_analysis",
        route_after_query_analysis,
        {
            "human_clarify": "human_clarify",
            "next": END,
        },
    )
    builder.add_edge("human_clarify", "query_analysis")

    graph = builder.compile(
        interrupt_before=["human_clarify"],
    )
    return graph