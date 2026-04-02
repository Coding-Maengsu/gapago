# 3-6) Final Response Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm_for_agent

ROLE_TOOLS = build_role_tools()
RESPONSE_TOOLS = ROLE_TOOLS["RESPONSE_TOOLS"]

RESPONSE_SYSTEM_PROMPT = (
    "You are a research gap analysis report writer.\n"
    "Write a structured final report based on all prior messages.\n"
    "CRITICAL: Use standard Markdown ONLY.\n"
    "  - Do NOT use unicode box-drawing characters (─ ━ ═ │ etc.).\n"
    "  - Use --- for horizontal rules.\n"
    "  - Use standard pipe | syntax for tables.\n"
    "  - Every table MUST have a separator row (|---|---| ...) directly after the header row.\n\n"

    "## Research GAP Analysis Report\n\n"
    "**Query:** <original user query>\n\n"
    "**Refined Query:** <refined query>\n\n"

    "---\n\n"
    "### 1. Related Papers\n\n"
    "| # | paper_id | Title | Year | Relevance |\n"
    "|---|----------|-------|------|-----------|\n"
    "| 1 | <id> | <title> | <year> | <one-line relevance> |\n"
    "(list ALL retrieved papers, one row per paper)\n\n"

    "---\n\n"
    "### 2. Key Limitations (by axis)\n\n"
    "- **[<axis>]** <one-line summary of recurring limitation> (<N>건)\n"
    "(one bullet per axis group)\n\n"

    "---\n\n"
    "### 3. Research Gaps Overview\n\n"
    "List in URGENCY ORDER (highest urgency_score first).\n\n"
    "For EACH gap, output ONLY the following compact block — no extra prose:\n\n"
    "#### <stars> GAP #<N> — <axis_label> (<count>개 논문)\n\n"
    "**<gap_statement>**\n\n"
    "<elaboration: one sentence only>\n\n"
    "📌 **Proposed Topic:** *<proposed_topic>*\n\n"
    "Stars: ★★★ rank 1 | ★★☆ ranks 2-3 | ★☆☆ ranks 4+\n"
    "Append '⚠️ 근거 단일 논문' if repeat_count == 1\n\n"

    "---\n\n"
    "### 4. Detailed Gap Analysis\n\n"
    "Repeat for EACH gap in the same order as Section 3.\n\n"
    "#### GAP #<N> — <axis_label>: Deep Dive\n\n"
    "**Gap Statement:** <gap_statement>\n\n"
    "**Why This Gap Persists (Technical Barriers):**\n"
    "- <barrier 1>\n"
    "- <barrier 2>\n"
    "- <barrier 3>\n\n"
    "**What Has Already Been Tried (and failed):**\n"
    "- <what_was_tried 1>\n"
    "- <what_was_tried 2>\n\n"
    "**Proposed Research Direction:**\n\n"
    "<detail field — include Core Insight, Why this direction, First experiment, Alternative directions>\n\n"
    "**Supporting Evidence:**\n"
    "- Papers: <supporting_papers list>\n"
    "- Key quotes:\n"
    "  > <supporting_quotes, one per line>\n\n"
    "**Urgency:** <urgency_score>/10 — <urgency_rationale>\n\n"
    "**Novelty Score:** <novelty_score>/10\n\n"

    "---\n\n"
    "### 5. Critic Scores\n\n"
    "| Metric | Score | Status |\n"
    "|--------|-------|--------|\n"
    "| <metric> | <score> | ✅ or ⚠️ |\n\n"
    "**DECISION:** <ACCEPT/REDO/REFINE>\n\n"
    "**Flags:** <list any flags from critic>\n\n"

    "End your output with exactly: FINAL ANSWER\n"
)
def _build_data_context(state: AgentState) -> str:
    """state에 저장된 구조화 데이터를 텍스트로 변환하여 LLM에 전달."""
    parts = []

    # papers (사실 데이터 — 제목, 연도, 저자 등)
    papers = state.get("papers", [])
    if papers:
        lines = ["[papers_data]"]
        for i, p in enumerate(papers, 1):
            if isinstance(p, dict):
                pid, title, year, authors = p.get("paper_id",""), p.get("title",""), p.get("year",0), p.get("authors",[])
            else:
                pid, title, year, authors = p.paper_id, p.title, p.year, p.authors
            lines.append(f"  {i}. {pid} | {title} | {year} | {', '.join(authors[:3]) if authors else 'N/A'}")
        parts.append("\n".join(lines))

    # limitations
    limitations = state.get("limitations", [])
    if limitations:
        lines = ["[limitation_extract_data]"]
        for lim in limitations:
            lines.append(
                f"  - [{lim.get('paper_id','')}][{lim.get('track','')}/{lim.get('source_section','')}] "
                f"{lim.get('claim','')}"
            )
        parts.append("\n".join(lines))

    # gaps
    gaps = state.get("gaps", [])
    if gaps:
        lines = ["[gap_infer_data]"]
        for i, g in enumerate(gaps, 1):
            lines.append(
                f"  GAP #{i} [{g.get('axis','')} · {g.get('axis_label','')} · {g.get('repeat_count',0)}개 논문]"
            )
            lines.append(f"    gap_statement: {g.get('gap_statement','')}")
            lines.append(f"    elaboration: {g.get('elaboration','')}")
            lines.append(f"    proposed_topic: {g.get('proposed_topic','')}")
            lines.append(f"    detail: {g.get('detail','')}")
            lines.append(f"    barriers: {g.get('barriers',[])}")
            lines.append(f"    barrier_type: {g.get('barrier_type','')}")
            lines.append(f"    what_was_tried: {g.get('what_was_tried',[])}")
            lines.append(f"    alt_topics: {g.get('alt_topics',[])}")
            lines.append(f"    novelty_score: {g.get('novelty_score',0)}")
            lines.append(f"    urgency_score: {g.get('urgency_score',0)}")
            lines.append(f"    urgency_rationale: {g.get('urgency_rationale','')}")
            lines.append(f"    supporting_papers: {g.get('supporting_papers',[])}")
            lines.append(f"    supporting_quotes: {g.get('supporting_quotes',[])}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def final_response_node(state: AgentState) -> AgentState:
    # 런타임에 LLM + agent 생성 (사용자 선택 provider 반영)
    llm = get_llm_for_agent(state, "response")
    agent = create_agent(
        model=llm,
        tools=RESPONSE_TOOLS,
        system_prompt=make_system_prompt(RESPONSE_SYSTEM_PROMPT),
    )

    # 구조화 데이터를 messages에 주입
    data_context = _build_data_context(state)
    enriched_state = dict(state)
    if data_context:
        from langchain_core.messages import HumanMessage
        enriched_state["messages"] = list(state.get("messages", [])) + [
            HumanMessage(content=f"아래는 파이프라인에서 생성된 구조화 데이터입니다. 이 데이터를 기반으로 보고서를 작성하세요.\n\n{data_context}")
        ]

    result = agent.invoke(enriched_state)
    last = AIMessage(content=result["messages"][-1].content, name="final_response")
    return {"messages": [last], "sender": "final_response"}