"""
GAPAGO - Gradio Demo UI
Research GAP Analysis Multi-Agent System
For Hugging Face Spaces deployment
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr

import config  # noqa: F401
from graphs.graph import build_graph
from langchain_core.messages import HumanMessage
from llm import get_llm

OUTPUT_DIR = Path("/tmp/gapago_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

NODE_LABELS = {
    "query_subgraph":     "Query Analysis",
    "meaning_expand":     "Meaning Expansion",
    "paper_retrieval":    "Paper Retrieval",
    "limitation_extract": "Limitation Extraction",
    "limitation_eval":    "Limitation Evaluation",
    "recency_check":      "Recency Check",
    "gap_infer":          "GAP Inference",
    "critic_score":       "Critic Scoring",
    "final_response":     "Final Report",
}

PIPELINE_NODES = list(NODE_LABELS.keys())


def run_pipeline(query, provider, domain, progress=gr.Progress()):
    """Run the GAPAGO pipeline with real-time progress updates."""
    if not query.strip():
        yield "Please enter a research question.", "", "", ""
        return

    os.environ["LLM_PROVIDER"] = provider
    get_llm.cache_clear()

    graph = build_graph()
    config_dict = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 30,
    }

    inputs = {
        "messages": [HumanMessage(content=query)],
        "max_iterations": 3,
        "research_domain": domain,
    }

    progress(0, desc="Starting pipeline...")

    status_log = ""
    papers_md = ""
    gaps_md = ""
    report_md = ""
    completed = 0

    try:
        for event in graph.stream(inputs, config_dict, subgraphs=True):
            path, update = event

            if path:
                continue

            for node, values in update.items():
                if node.startswith("__"):
                    continue
                if not isinstance(values, dict):
                    continue

                completed += 1
                label = NODE_LABELS.get(node, node)
                pct = min(completed / len(PIPELINE_NODES), 1.0)
                progress(pct, desc=f"{label} complete")

                # Status log
                status_log += f"### {label}\n"

                # Node-specific rendering
                if node == "query_subgraph":
                    rq = values.get("refined_query", "")
                    kw = values.get("keywords", [])
                    if rq:
                        status_log += f"**Refined Query:** {rq}\n\n"
                    if kw:
                        status_log += f"**Keywords:** {', '.join(kw)}\n\n"

                elif node == "paper_retrieval":
                    papers = values.get("papers", [])
                    web = values.get("web_results", [])
                    status_log += f"Papers: {len(papers)}, Web results: {len(web)}\n\n"

                    papers_md = f"## Papers ({len(papers)})\n\n"
                    papers_md += "| # | Title | Year | Authors |\n|---|-------|------|--------|\n"
                    for i, p in enumerate(papers):
                        if isinstance(p, dict):
                            title = p.get("title", "")[:60]
                            year = p.get("year", "")
                            authors = ", ".join(p.get("authors", [])[:3])
                            papers_md += f"| {i+1} | {title} | {year} | {authors} |\n"

                elif node == "limitation_extract":
                    lims = values.get("limitations", [])
                    status_log += f"Extracted: {len(lims)} limitations\n\n"

                elif node == "limitation_eval":
                    eval_data = values.get("limitation_eval", {})
                    decision = eval_data.get("decision", "N/A")
                    lims = values.get("limitations", [])
                    status_log += f"Decision: **{decision}** ({len(lims)} passed)\n\n"

                elif node == "recency_check":
                    lims = values.get("limitations", [])
                    counts = {"unresolved": 0, "partial": 0, "resolved": 0}
                    for lim in lims:
                        s = lim.get("recency_status", "unresolved")
                        counts[s] = counts.get(s, 0) + 1
                    status_log += f"Unresolved: {counts['unresolved']}, Partial: {counts['partial']}, Resolved: {counts['resolved']}\n\n"

                elif node == "gap_infer":
                    gaps = values.get("gaps", [])
                    status_log += f"Found: {len(gaps)} research gaps\n\n"

                    gaps_md = f"## Research GAPs ({len(gaps)})\n\n"
                    for i, gap in enumerate(gaps):
                        axis_label = gap.get("axis_label", gap.get("axis", ""))
                        axis_type = "Fixed" if gap.get("axis_type") == "fixed" else "Dynamic"
                        count = gap.get("repeat_count", 0)
                        gaps_md += f"### GAP #{i+1} ({axis_type}: {axis_label}, {count} papers)\n\n"
                        gaps_md += f"**{gap.get('gap_statement', '')}**\n\n"
                        gaps_md += f"{gap.get('elaboration', '')}\n\n"
                        gaps_md += f"*Proposed Topic:* {gap.get('proposed_topic', '')}\n\n"
                        gaps_md += "---\n\n"

                elif node == "critic_score":
                    msgs = values.get("messages", [])
                    for msg in msgs:
                        content = getattr(msg, "content", "")
                        if content:
                            status_log += f"```\n{content[:500]}\n```\n\n"

                elif node == "final_response":
                    msgs = values.get("messages", [])
                    for msg in msgs:
                        content = getattr(msg, "content", "")
                        if content:
                            report_md = content

                status_log += "---\n\n"
                yield status_log, papers_md, gaps_md, report_md

    except Exception as e:
        status_log += f"\n\n**Error:** {str(e)}"
        yield status_log, papers_md, gaps_md, report_md
        return

    progress(1.0, desc="Pipeline complete!")

    # Save result
    try:
        final_state = graph.get_state(config_dict)
        state_values = final_state.values if final_state else {}
        messages_out = []
        for msg in state_values.get("messages", []):
            messages_out.append({
                "type": msg.type,
                "name": getattr(msg, "name", None),
                "content": getattr(msg, "content", ""),
            })
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "refined_query": state_values.get("refined_query", ""),
            "keywords": state_values.get("keywords", []),
            "limitations": state_values.get("limitations", []),
            "gaps": state_values.get("gaps", []),
            "messages": messages_out,
        }
        fname = f"gapago_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        (OUTPUT_DIR / fname).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    yield status_log, papers_md, gaps_md, report_md


# ── Gradio UI ────────────────────────────────────────────────────────

with gr.Blocks(title="GAPAGO - Research GAP Analyzer") as demo:

    gr.Markdown("# GAPAGO", elem_classes="main-title")
    gr.Markdown("Research GAP Analysis Multi-Agent System", elem_classes="sub-title")

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Research Question",
                placeholder="e.g., Domain adaptation for fault detection in smart manufacturing",
                lines=2,
            )
        with gr.Column(scale=1):
            provider_input = gr.Dropdown(
                choices=["azure", "claude", "gemini"],
                value="azure",
                label="LLM Provider",
            )
            domain_input = gr.Dropdown(
                choices=["auto", "ai_cs", "biomedical", "materials_chemistry", "physics", "general"],
                value="auto",
                label="Research Domain",
            )

    run_btn = gr.Button("Analyze", variant="primary", size="lg")

    with gr.Tabs():
        with gr.Tab("Progress"):
            status_output = gr.Markdown(label="Pipeline Status")
        with gr.Tab("Papers"):
            papers_output = gr.Markdown(label="Retrieved Papers")
        with gr.Tab("Research GAPs"):
            gaps_output = gr.Markdown(label="Identified GAPs")
        with gr.Tab("Final Report"):
            report_output = gr.Markdown(label="Final Report")

    run_btn.click(
        fn=run_pipeline,
        inputs=[query_input, provider_input, domain_input],
        outputs=[status_output, papers_output, gaps_output, report_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    )
