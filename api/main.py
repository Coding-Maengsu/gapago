"""
GAPAGO - FastAPI Server
Research GAP Analysis Multi-Agent System
"""

import os
import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import config  # noqa: F401  (.env load, LangSmith)

from graphs.graph import build_graph
from langchain_core.messages import HumanMessage
from llm import AVAILABLE_PROVIDERS, get_llm

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="GAPAGO", description="Research GAP Analysis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ── Startup warm-up ──────────────────────────────────────────────────
@app.on_event("startup")
async def warmup():
    """Pre-initialize LLM and graph in background to avoid blocking server start."""
    async def _warmup():
        await asyncio.sleep(1)
        try:
            print("[startup] Warming up LLM...")
            get_llm()
            print("[startup] Warming up graph...")
            build_graph()
            print("[startup] Warm-up complete.")
        except Exception as e:
            print(f"[startup] Warm-up failed (non-fatal): {e}")

    asyncio.create_task(_warmup())


@app.get("/api/health")
async def health():
    """Health check endpoint for uptime monitoring."""
    return {"status": "ok"}


# ── In-memory session store ──────────────────────────────────────────
# Each session: {
#   "status": "running" | "completed" | "stopped" | "error" | "interrupted",
#   "query": str, "user_id": str, "started_at": str,
#   "graph": LangGraph, "config": dict,
#   "events": list[dict],        # accumulated events for replay
#   "event_signal": asyncio.Event,  # notify new event available
#   "cancelled": asyncio.Event,  # stop signal
#   "filename": str | None,      # result filename when completed
#   "clarify_prompt": str | None,
# }
_sessions: dict = {}


# ── Request / Response Models ───────────────────────────────────────
class HistoryItem(BaseModel):
    filename: str
    query: str
    timestamp: str
    refined_query: str = ""
    gaps_count: int = 0
    status: str = "completed"
    session_id: str = ""


# ── Utility ─────────────────────────────────────────────────────────
def _serialize_messages(state_values: dict) -> list[dict]:
    out = []
    for msg in state_values.get("messages", []):
        out.append({
            "type": msg.type,
            "name": getattr(msg, "name", None),
            "content": getattr(msg, "content", ""),
        })
    return out


def _save_result(query: str, state_values: dict, user_id: str = "") -> str:
    messages_out = _serialize_messages(state_values)

    papers = state_values.get("papers", [])
    papers_out = []
    for p in papers:
        d = p if isinstance(p, dict) else (p.model_dump() if hasattr(p, "model_dump") else p.__dict__)
        papers_out.append({
            "paper_id": d.get("paper_id", ""),
            "title": d.get("title", ""),
            "year": d.get("year", 0),
            "authors": d.get("authors", []),
            "url": d.get("url", ""),
        })

    result = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "refined_query": state_values.get("refined_query", ""),
        "keywords": state_values.get("keywords", []),
        "papers": papers_out,
        "limitations": state_values.get("limitations", []),
        "limitation_eval": state_values.get("limitation_eval", {}),
        "eval_warnings": state_values.get("eval_warnings", []),
        "gaps": state_values.get("gaps", []),
        "web_results": state_values.get("web_results", []),
        "messages": messages_out,
    }

    fname = f"gapago_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = OUTPUT_DIR / fname
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return fname


def _push_event(session_id: str, event: dict):
    """Add event to session and signal waiting consumers."""
    session = _sessions.get(session_id)
    if not session:
        return
    session["events"].append(event)
    session["event_signal"].set()


# ── Background pipeline runner ───────────────────────────────────────
async def _run_pipeline(session_id: str, graph, config_dict: dict, inputs: dict):
    """Run the analysis pipeline as a background task."""
    session = _sessions.get(session_id)
    if not session:
        return

    try:
        async for event in graph.astream(inputs, config_dict, subgraphs=True):
            # Check cancellation
            if session["cancelled"].is_set():
                session["status"] = "stopped"
                _push_event(session_id, {"event": "stopped"})
                return

            path, update = event

            # Subgraph internal events
            if path:
                for node, values in update.items():
                    if node == "__interrupt__":
                        session["status"] = "interrupted"
                    if isinstance(values, dict):
                        for msg in values.get("messages", []):
                            if getattr(msg, "name", None) == "clarify_prompt":
                                session["clarify_prompt"] = msg.content
                continue

            # Root graph events
            for node, values in update.items():
                if node == "__interrupt__":
                    session["status"] = "interrupted"
                    continue
                if node.startswith("__"):
                    continue

                payload = _build_node_payload(node, values)
                _push_event(session_id, payload)

        if session["cancelled"].is_set():
            session["status"] = "stopped"
            _push_event(session_id, {"event": "stopped"})
            return

        if session["status"] == "interrupted":
            _push_event(session_id, {
                "event": "interrupt",
                "session_id": session_id,
                "clarify_prompt": session.get("clarify_prompt", ""),
            })
        else:
            # Pipeline complete — save result
            final_state = graph.get_state(config_dict)
            state_values = final_state.values if final_state else {}
            fname = _save_result(session["query"], state_values, session["user_id"])
            session["status"] = "completed"
            session["filename"] = fname
            _push_event(session_id, {"event": "complete", "filename": fname})

    except Exception as e:
        session["status"] = "error"
        _push_event(session_id, {"event": "error", "message": str(e)})


# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/api/providers")
async def get_providers():
    """Available LLM providers."""
    return {
        k: {"id": v[0], "name": v[1]}
        for k, v in AVAILABLE_PROVIDERS.items()
    }


@app.get("/api/history")
async def get_history(user_id: str = ""):
    """List saved analysis results + running sessions, filtered by user_id."""
    items = []

    # Running/interrupted sessions
    for sid, session in _sessions.items():
        if session["status"] not in ("running", "interrupted"):
            continue
        if user_id and session.get("user_id", "") != user_id:
            continue
        items.append(HistoryItem(
            filename="",
            query=session.get("query", ""),
            timestamp=session.get("started_at", ""),
            status=session["status"],
            session_id=sid,
        ))

    # Completed results from files
    files = sorted(OUTPUT_DIR.glob("gapago_result_*.json"), reverse=True)
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if user_id and data.get("user_id", "") != user_id:
                continue
            items.append(HistoryItem(
                filename=f.name,
                query=data.get("query", "(no query)"),
                timestamp=data.get("timestamp", ""),
                refined_query=data.get("refined_query", ""),
                gaps_count=len(data.get("gaps", [])),
                status="completed",
            ))
        except Exception:
            continue
    return items


@app.get("/api/history/{filename}")
async def get_history_detail(filename: str):
    """Get a saved analysis result."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Result not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/analyze")
async def analyze(query: str, provider: str = "azure", domain: str = "auto", user_id: str = ""):
    """
    Start a new analysis pipeline in background. Returns session_id.
    Client should connect to /api/stream/{session_id} for SSE updates.
    """
    session_id = str(uuid.uuid4())
    graph = build_graph()
    config_dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 30,
    }

    inputs = {
        "messages": [HumanMessage(content=query)],
        "max_iterations": 3,
        "research_domain": domain,
        "llm_provider": provider,
    }

    _sessions[session_id] = {
        "status": "running",
        "query": query,
        "user_id": user_id,
        "started_at": datetime.now().isoformat(),
        "graph": graph,
        "config": config_dict,
        "events": [],
        "event_signal": asyncio.Event(),
        "cancelled": asyncio.Event(),
        "filename": None,
        "clarify_prompt": None,
    }

    # Launch pipeline in background
    asyncio.create_task(_run_pipeline(session_id, graph, config_dict, inputs))

    return {"session_id": session_id}


@app.get("/api/stream/{session_id}")
async def stream(session_id: str, from_idx: int = 0):
    """
    SSE stream for a running session. Replays past events, then streams new ones.
    Supports reconnection — client can reconnect after refresh.
    Use from_idx to skip already-consumed events (e.g. after clarify resume).
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    async def event_stream():
        cursor = from_idx
        while True:
            # Send any buffered events
            while cursor < len(session["events"]):
                evt = session["events"][cursor]
                cursor += 1
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

            # Check if session is done
            if session["status"] in ("completed", "stopped", "error"):
                break
            if session["status"] == "interrupted" and cursor >= len(session["events"]):
                break

            # Wait for new events
            session["event_signal"].clear()
            try:
                await asyncio.wait_for(session["event_signal"].wait(), timeout=30)
            except asyncio.TimeoutError:
                # Send keepalive
                yield f"data: {json.dumps({'event': 'keepalive'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/stop/{session_id}")
async def stop(session_id: str):
    """Stop a running analysis and clean up resources."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if session["status"] == "running":
        session["cancelled"].set()
        # Wait briefly for the task to notice cancellation
        await asyncio.sleep(0.5)

    # Clean up
    _sessions.pop(session_id, None)
    return {"status": "stopped", "session_id": session_id}


@app.get("/api/status/{session_id}")
async def status(session_id: str):
    """Check current status of a session."""
    session = _sessions.get(session_id)
    if not session:
        return {"status": "not_found", "session_id": session_id}
    return {
        "status": session["status"],
        "session_id": session_id,
        "query": session.get("query", ""),
        "filename": session.get("filename"),
    }


@app.get("/api/clarify")
async def clarify(session_id: str, response: str):
    """Resume pipeline after human clarification."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired")

    graph = session["graph"]
    config_dict = session["config"]

    # Inject user response into the interrupted subgraph
    current_state = graph.get_state(config_dict, subgraphs=True)
    target_config = config_dict
    state_stack = current_state
    while state_stack and state_stack.tasks:
        task = state_stack.tasks[0]
        if hasattr(task, "state") and task.state:
            target_config = task.state.config
            state_stack = task.state
        else:
            break

    graph.update_state(
        target_config,
        {"messages": [HumanMessage(content=response)]},
    )

    # Resume pipeline in background
    session["status"] = "running"
    session["event_signal"] = asyncio.Event()
    asyncio.create_task(_run_pipeline(session_id, graph, config_dict, None))

    return {"session_id": session_id, "status": "resumed", "events_count": len(session["events"])}


def _build_node_payload(node: str, values: dict) -> dict:
    """Build a JSON-serializable payload for a node result."""
    payload: dict = {"event": "node", "node": node}

    if not isinstance(values, dict):
        return payload

    if node == "query_subgraph":
        payload["refined_query"] = values.get("refined_query", "")
        payload["keywords"] = values.get("keywords", [])
        payload["scope_level"] = values.get("scope_level", "")

    elif node == "meaning_expand":
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                payload["expansion"] = content[:1000]

    elif node == "paper_retrieval":
        papers = values.get("papers", [])
        payload["papers_count"] = len(papers)
        payload["papers"] = []
        for p in papers:
            d = p if isinstance(p, dict) else (p.model_dump() if hasattr(p, "model_dump") else p.__dict__)
            payload["papers"].append({
                "paper_id": d.get("paper_id", ""),
                "title": d.get("title", ""),
                "year": d.get("year", ""),
                "authors": (d.get("authors") or [])[:3],
                "url": d.get("url", ""),
            })
        payload["web_results_count"] = len(values.get("web_results", []))

    elif node == "limitation_extract":
        limitations = values.get("limitations", [])
        payload["limitations_count"] = len(limitations)
        payload["limitations"] = []
        for lim in limitations:
            payload["limitations"].append({
                "paper_id": lim.get("paper_id", ""),
                "claim": lim.get("claim", ""),
                "track": lim.get("track", ""),
                "source_section": lim.get("source_section", ""),
                "evidence_quote": lim.get("evidence_quote", "")[:200],
            })

    elif node == "limitation_eval":
        eval_data = values.get("limitation_eval", {})
        payload["decision"] = eval_data.get("decision", "N/A")
        payload["call1_results"] = eval_data.get("call1_results", [])
        payload["call2_result"] = eval_data.get("call2_result", {})
        payload["eval_warnings"] = values.get("eval_warnings", [])
        payload["limitations_count"] = len(values.get("limitations", []))

    elif node == "recency_check":
        limitations = values.get("limitations", [])
        status_counts = {"unresolved": 0, "partial": 0, "resolved": 0}
        for lim in limitations:
            s = lim.get("recency_status", "unresolved")
            status_counts[s] = status_counts.get(s, 0) + 1
        payload["recency_status"] = status_counts
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                payload["summary"] = content[:500]

    elif node == "gap_infer":
        gaps = values.get("gaps", [])
        payload["gaps_count"] = len(gaps)
        payload["gaps"] = gaps

    elif node == "critic_score":
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                payload["critic_output"] = content[:1500]

    elif node == "final_response":
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                payload["report"] = content

    return payload


# ── Static frontend ─────────────────────────────────────────────────
@app.get("/debug/paths")
async def debug_paths():
    """Temporary debug endpoint."""
    index = FRONTEND_DIR / "index.html"
    return {
        "FRONTEND_DIR": str(FRONTEND_DIR),
        "index_exists": index.exists(),
        "cwd": str(Path.cwd()),
        "__file__": str(Path(__file__).resolve()),
        "dir_listing": [str(p.name) for p in FRONTEND_DIR.parent.iterdir()] if FRONTEND_DIR.parent.exists() else "parent not found",
    }


@app.get("/logo.png")
async def logo():
    logo_path = FRONTEND_DIR / "logo.png"
    if logo_path.exists():
        return FileResponse(str(logo_path), media_type="image/png")
    raise HTTPException(404, "Logo not found")


@app.get("/new_logo.png")
async def new_logo():
    path = FRONTEND_DIR / "new_logo.png"
    if path.exists():
        return FileResponse(str(path), media_type="image/png")
    raise HTTPException(404, "new_logo not found")


@app.get("/middle_image.png")
async def middle_image():
    path = FRONTEND_DIR / "middle_image.png"
    if path.exists():
        return FileResponse(str(path), media_type="image/png")
    raise HTTPException(404, "middle_image not found")


@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"message": f"Frontend not found. FRONTEND_DIR={FRONTEND_DIR}, exists={FRONTEND_DIR.exists()}"}
