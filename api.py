"""
GAPAGO - FastAPI Backend
Wraps the LangGraph pipeline as a REST API for remote frontends.
"""

import os
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config  # noqa: F401  (.env, LangSmith)

from graphs.graph import build_graph
from langchain_core.messages import HumanMessage
from llm import get_llm

app = FastAPI(title="GAPAGO API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PIPELINE_NODES = [
    "query_subgraph", "meaning_expand", "paper_retrieval",
    "limitation_extract", "limitation_eval", "recency_check",
    "gap_infer", "critic_score", "final_response",
]

# ── In-memory session store ──
sessions: dict[str, dict] = {}


# ── Request / Response models ──
class AnalyzeRequest(BaseModel):
    query: str
    provider: str = "azure"
    domain: str = "auto"


class ClarifyRequest(BaseModel):
    response: str


# ── Pipeline runner (runs in background thread) ──
def _run_pipeline(session_id: str, query: str, provider: str, domain: str):
    """Execute pipeline in a background thread, updating session state."""
    session = sessions[session_id]

    os.environ["LLM_PROVIDER"] = provider
    get_llm.cache_clear()

    graph = build_graph()
    config_dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 30,
    }
    inputs = {
        "messages": [HumanMessage(content=query)],
        "max_iterations": 3,
        "research_domain": domain,
    }

    session["graph"] = graph
    session["config_dict"] = config_dict
    session["status"] = "running"

    try:
        _stream_pipeline(session, graph, inputs, config_dict)
    except Exception as e:
        session["status"] = "error"
        session["error"] = str(e)


def _stream_pipeline(session: dict, graph, stream_input, config_dict):
    """Stream graph events, updating session progress."""
    interrupted = False
    clarify_prompt = None

    for event in graph.stream(stream_input, config_dict, subgraphs=True):
        path, update = event

        if path:
            for node, values in update.items():
                if node == "__interrupt__":
                    interrupted = True
                if isinstance(values, dict):
                    for msg in values.get("messages", []):
                        if getattr(msg, "name", None) == "scope_prompt":
                            clarify_prompt = msg.content
            continue

        for node, values in update.items():
            if node == "__interrupt__":
                interrupted = True
                continue
            if node.startswith("__"):
                continue

            session["completed_nodes"].append(node)

            # Extract serializable summary per node
            if isinstance(values, dict):
                session["node_summaries"][node] = _summarize_node(node, values)

    if interrupted:
        session["status"] = "interrupted"
        session["clarify_prompt"] = clarify_prompt
    else:
        # Pipeline completed
        final_state = graph.get_state(config_dict)
        state_values = final_state.values if final_state else {}
        result = _save_result(session["query"], state_values)
        session["status"] = "completed"
        session["result"] = result


def _summarize_node(node: str, values: dict) -> dict:
    """Extract a JSON-serializable summary from node output."""
    summary: dict = {"node": node}

    if node == "query_subgraph":
        summary["refined_query"] = values.get("refined_query", "")
        summary["keywords"] = values.get("keywords", [])
        summary["scope_level"] = values.get("scope_level", "")

    elif node == "paper_retrieval":
        papers = values.get("papers", [])
        summary["paper_count"] = len(papers)
        rows = []
        for p in papers:
            if isinstance(p, dict):
                rows.append({
                    "paper_id": p.get("paper_id", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "source": p.get("paper_id", "").split(":")[0] if ":" in p.get("paper_id", "") else "",
                })
            else:
                rows.append({
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "year": p.year,
                    "source": p.paper_id.split(":")[0] if ":" in p.paper_id else "",
                })
        summary["papers"] = rows

    elif node == "limitation_extract":
        limitations = values.get("limitations", [])
        summary["limitation_count"] = len(limitations)
        summary["limitations"] = limitations

    elif node == "limitation_eval":
        eval_data = values.get("limitation_eval", {})
        summary["decision"] = eval_data.get("decision", "N/A")
        summary["call1_results"] = eval_data.get("call1_results", [])
        call2 = eval_data.get("call2_result", {})
        summary["type_distribution"] = call2.get("type_distribution", {})
        summary["limitation_count"] = len(values.get("limitations", []))
        summary["eval_warnings"] = values.get("eval_warnings", [])

    elif node == "recency_check":
        limitations = values.get("limitations", [])
        status_counts = {"unresolved": 0, "partial": 0, "resolved": 0}
        for lim in limitations:
            s = lim.get("recency_status", "unresolved")
            status_counts[s] = status_counts.get(s, 0) + 1
        summary["recency_counts"] = status_counts
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                summary["message"] = content
                break

    elif node == "gap_infer":
        gaps = values.get("gaps", [])
        summary["gap_count"] = len(gaps)
        summary["gaps"] = gaps

    elif node == "critic_score":
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                summary["message"] = content[:1000]
                break

    elif node == "final_response":
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                summary["report"] = content
                break

    elif node == "meaning_expand":
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                summary["message"] = content[:500]
                break

    return summary


def _save_result(query: str, state_values: dict) -> dict:
    """Save pipeline result to JSON and return the data."""
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
        "web_results": state_values.get("web_results", []),
        "limitation_eval": state_values.get("limitation_eval", {}),
        "eval_warnings": state_values.get("eval_warnings", []),
        "messages": messages_out,
    }

    fname = f"gapago_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = OUTPUT_DIR / fname
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


# =====================================================================
# API Endpoints
# =====================================================================

@app.post("/api/analyze")
def start_analyze(req: AnalyzeRequest):
    """Start a new analysis pipeline. Returns session_id for polling."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "session_id": session_id,
        "query": req.query,
        "provider": req.provider,
        "domain": req.domain,
        "status": "starting",
        "completed_nodes": [],
        "node_summaries": {},
        "clarify_prompt": None,
        "error": None,
        "result": None,
        "graph": None,
        "config_dict": None,
        "created_at": datetime.now().isoformat(),
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(session_id, req.query, req.provider, req.domain),
        daemon=True,
    )
    thread.start()

    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """Poll session status, completed nodes, and results."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "query": session["query"],
        "status": session["status"],
        "completed_nodes": session["completed_nodes"],
        "node_summaries": session["node_summaries"],
        "clarify_prompt": session["clarify_prompt"],
        "error": session["error"],
        "result": session["result"],
        "total_nodes": len(PIPELINE_NODES),
    }


@app.post("/api/sessions/{session_id}/clarify")
def clarify(session_id: str, req: ClarifyRequest):
    """Send user clarification to resume an interrupted pipeline."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "interrupted":
        raise HTTPException(status_code=400, detail="Session is not interrupted")

    graph = session["graph"]
    config_dict = session["config_dict"]

    graph.update_state(
        config_dict,
        {"messages": [HumanMessage(content=req.response)]},
    )

    session["status"] = "running"
    session["clarify_prompt"] = None

    thread = threading.Thread(
        target=_resume_pipeline,
        args=(session_id,),
        daemon=True,
    )
    thread.start()

    return {"status": "resumed"}


def _resume_pipeline(session_id: str):
    """Resume pipeline after clarification."""
    session = sessions[session_id]
    graph = session["graph"]
    config_dict = session["config_dict"]

    try:
        _stream_pipeline(session, graph, None, config_dict)
    except Exception as e:
        session["status"] = "error"
        session["error"] = str(e)


@app.get("/api/history")
def get_history():
    """List saved analysis results."""
    files = sorted(OUTPUT_DIR.glob("gapago_result_*.json"), reverse=True)
    history = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            history.append({
                "file": f.name,
                "query": data.get("query", ""),
                "timestamp": data.get("timestamp", ""),
                "refined_query": data.get("refined_query", ""),
                "gaps_count": len(data.get("gaps", [])),
            })
        except Exception:
            continue
    return {"history": history}


@app.get("/api/history/{filename}")
def get_history_detail(filename: str):
    """Get a specific saved result."""
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.name.startswith("gapago_result_"):
        raise HTTPException(status_code=404, detail="File not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
