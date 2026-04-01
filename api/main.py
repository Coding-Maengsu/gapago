"""
GAPAGO - FastAPI Server
Research GAP Analysis Multi-Agent System
"""

import gc
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
from langchain_core.messages import HumanMessage, AIMessage
from llm import AVAILABLE_PROVIDERS, get_llm
from agents.gap_chat_agent import gap_chat_respond
from agents.retrieval_agent import preload_models
from utils.progress import init_progress, drain_progress, cleanup_progress, mark_stage_start, mark_stage_done
from utils.session_store import init_db as init_session_db, save_session, update_session_status
from utils import cancel as cancel_registry

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
LANDING_DIR = Path(__file__).resolve().parent.parent / "landing" / "dist"


# ── Startup warm-up ──────────────────────────────────────────────────
@app.on_event("startup")
async def warmup():
    """Pre-initialize LLM, graph, and session DB in background."""
    init_session_db()
    asyncio.create_task(_session_reaper())
    async def _warmup():
        await asyncio.sleep(1)
        try:
            print("[startup] Warming up LLM...")
            get_llm()
            print("[startup] Warming up graph...")
            _get_graph()  # 캐시 채우기 — 이후 analyze/explore는 즉시 반환
            # ── 모델 사전 로딩 ──────────────────────────────────────────
            # workers=1 단일 프로세스이므로 여기서 한 번만 로딩하면
            # 이후 모든 요청이 캐시된 모델을 재사용 (메모리 절약 + 첫 요청 지연 제거)
            print("[startup] Preloading Embedding + CrossEncoder models...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, preload_models, "light")
            print("[startup] Warm-up complete.")
        except Exception as e:
            print(f"[startup] Warm-up failed (non-fatal): {e}")

    asyncio.create_task(_warmup())


@app.get("/api/health")
async def health():
    """Health check endpoint for uptime monitoring."""
    return {"status": "ok"}


# ── Graph 캐시 (프로세스당 1회만 빌드) ───────────────────────────────────
# build_graph()는 StateGraph + MemorySaver 초기화로 수백ms 소요.
# 매 요청마다 호출하면 event loop를 블로킹하여 /api/stream 404를 유발.
_graph_cache = None

def _get_graph():
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = build_graph()
    return _graph_cache


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

# ── Session reaper (TTL-based cleanup) ────────────────────────────────
_SESSION_TTL_SECONDS = 300  # 완료 후 5분 보관

async def _session_reaper():
    """Background task: clean up finished sessions after TTL."""
    while True:
        await asyncio.sleep(120)  # 2분마다 체크
        now = datetime.now()
        to_delete = []
        for sid, session in list(_sessions.items()):
            if session["status"] not in ("completed", "stopped", "error"):
                continue
            completed_at = session.get("completed_at")
            if not completed_at:
                session["completed_at"] = now
                continue
            elapsed = (now - completed_at).total_seconds()
            if elapsed > _SESSION_TTL_SECONDS:
                to_delete.append(sid)

        for sid in to_delete:
            session = _sessions.pop(sid, None)
            fname = session.get("filename", "") if session else ""
            if session:
                _clear_checkpoints(sid)
                session.clear()
            _chat_histories.pop(sid, None)
            if fname:
                _chat_histories.pop(fname, None)

        if to_delete:
            gc.collect()
            print(f"[reaper] Cleaned {len(to_delete)} sessions: {to_delete}")


def _clear_checkpoints(thread_id: str):
    """MemorySaver에서 해당 thread_id의 체크포인트 삭제."""
    graph = _graph_cache
    if graph is None:
        return
    try:
        checkpointer = graph.checkpointer
        if hasattr(checkpointer, 'storage'):
            checkpointer.storage.pop(thread_id, None)
        if hasattr(checkpointer, 'writes'):
            checkpointer.writes.pop(thread_id, None)
    except Exception as e:
        print(f"[reaper] checkpoint cleanup error for {thread_id}: {e}")


# ── Request / Response Models ───────────────────────────────────────
class HistoryItem(BaseModel):
    filename: str
    query: str
    timestamp: str
    refined_query: str = ""
    gaps_count: int = 0
    status: str = "completed"
    session_id: str = ""
    parent_session_id: str = ""


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


def _save_result(query: str, state_values: dict, user_id: str = "", parent_session_id: str = "", session_id: str = "") -> str:
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
            "venue": d.get("venue", ""),
        })

    result = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "session_id": session_id,
        "parent_session_id": parent_session_id,
        "refined_query": state_values.get("refined_query", ""),
        "keywords": state_values.get("keywords", []),
        "papers": papers_out,
        "total_searched": state_values.get("total_candidates_count", len(papers)),
        "limitations": state_values.get("limitations", []),
        "limitation_eval": state_values.get("limitation_eval", {}),
        "eval_warnings": state_values.get("eval_warnings", []),
        "gaps": state_values.get("gaps", []),
        "web_results": state_values.get("web_results", []),
        "messages": messages_out,
    }

    fname = f"gapago_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = OUTPUT_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return fname


_MAX_EVENTS_PER_SESSION = 500

def _push_event(session_id: str, event: dict):
    """Add event to session and signal waiting consumers."""
    session = _sessions.get(session_id)
    if not session:
        return
    events = session["events"]
    if len(events) >= _MAX_EVENTS_PER_SESSION:
        del events[:_MAX_EVENTS_PER_SESSION // 10]
    events.append(event)
    session["event_signal"].set()


# ── Background pipeline runner ───────────────────────────────────────
async def _run_pipeline(session_id: str, graph, config_dict: dict, inputs: dict | None):
    """Run the analysis pipeline as a background task."""
    session = _sessions.get(session_id)
    if not session:
        return

    init_progress(session_id)
    cancel_registry.register(session_id)

    async def _drain_loop():
        """Drain progress events from agents running in threads."""
        while True:
            items = drain_progress(session_id)
            for item in items:
                _push_event(session_id, item)
            await asyncio.sleep(0.3)

    drainer = asyncio.create_task(_drain_loop())

    try:
        async for event in graph.astream(inputs, config_dict, subgraphs=True):
            # Check cancellation
            if session["cancelled"].is_set():
                session["status"] = "stopped"
                session["completed_at"] = datetime.now()
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

                # Track pipeline stage completion for ETA
                mark_stage_done(session_id, node)
                payload = _build_node_payload(node, values)
                _push_event(session_id, payload)

        if session["cancelled"].is_set():
            session["status"] = "stopped"
            session["completed_at"] = datetime.now()
            update_session_status(session_id, "stopped")
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
            fname = _save_result(session["query"], state_values, session["user_id"], session.get("parent_session_id", ""), session_id)
            session["status"] = "completed"
            session["filename"] = fname
            session["completed_at"] = datetime.now()
            update_session_status(session_id, "completed")
            _push_event(session_id, {"event": "complete", "filename": fname})

    except Exception as e:
        session["status"] = "error"
        session["completed_at"] = datetime.now()
        update_session_status(session_id, "error")
        _push_event(session_id, {"event": "error", "message": str(e)})
    finally:
        drainer.cancel()
        cleanup_progress(session_id)
        cancel_registry.cleanup(session_id)


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
            parent_session_id=session.get("parent_session_id", ""),
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
                session_id=data.get("session_id", ""),
                parent_session_id=data.get("parent_session_id", ""),
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


@app.delete("/api/history/{filename}")
async def delete_history(filename: str):
    """Delete a saved analysis result and its session record."""
    # 파일명 검증 (path traversal 방지)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Result not found")
    # 파일에서 session_id 읽어서 SQLite도 정리
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sid = data.get("session_id", "")
        if sid:
            from utils.session_store import delete_session
            delete_session(sid)
            _chat_histories.pop(sid, None)
    except Exception:
        pass
    # filename 키로 저장된 채팅 기록도 정리
    _chat_histories.pop(filename, None)
    path.unlink()
    return {"status": "deleted", "filename": filename}


@app.get("/api/analyze")
async def analyze(query: str, provider: str = "azure", domain: str = "auto", year_range: str = "auto", output_language: str = "auto", user_id: str = "", fast_mode: bool = False):
    """
    Start a new analysis pipeline in background. Returns session_id.
    Client should connect to /api/stream/{session_id} for SSE updates.
    """
    session_id = str(uuid.uuid4())

    # ── 세션을 graph 빌드보다 먼저 등록 ──────────────────────────────
    # graph = build_graph()가 동기 함수라 수백ms 소요될 수 있음.
    # 프론트는 /api/analyze 응답 즉시 /api/stream을 호출하므로,
    # 세션 등록이 늦으면 stream에서 404가 발생함.
    # → 세션을 먼저 "pending" 상태로 등록하고, graph는 이후에 채움.
    started_at = datetime.now().isoformat()
    _sessions[session_id] = {
        "status": "running",
        "query": query,
        "user_id": user_id,
        "started_at": started_at,
        "graph": None,  # graph는 아래에서 채움
        "config": None,
        "events": [],
        "event_signal": asyncio.Event(),
        "cancelled": asyncio.Event(),
        "filename": None,
        "clarify_prompt": None,
    }
    # Persist to SQLite
    save_session(session_id, "running", query, user_id=user_id, started_at=started_at)

    graph = _get_graph()  # 캐시된 graph 반환 (최초 1회만 빌드)
    config_dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 30,
    }
    _sessions[session_id]["graph"] = graph
    _sessions[session_id]["config"] = config_dict

    inputs = {
        "messages": [HumanMessage(content=query)],
        "max_iterations": 3,
        "research_domain": domain,
        "llm_provider": provider,
        "year_range": year_range,
        "output_language": output_language,
        "session_id": session_id,
        "fast_mode": fast_mode,
    }

    # Launch pipeline in background
    asyncio.create_task(_run_pipeline(session_id, graph, config_dict, inputs))

    return {"session_id": session_id}


@app.get("/api/explore")
async def explore(topic: str, session_id: str = "", provider: str = "azure", domain: str = "auto", year_range: str = "auto", output_language: str = "auto", user_id: str = ""):
    """
    Start an exploration (chain re-execution) based on a proposed topic from a previous analysis.
    Links the new session to the parent session for hierarchical history.
    """
    new_session_id = str(uuid.uuid4())

    # 세션 먼저 등록 (analyze와 동일한 race condition 방지)
    _sessions[new_session_id] = {
        "status": "running",
        "query": topic,
        "user_id": user_id,
        "started_at": datetime.now().isoformat(),
        "graph": None,
        "config": None,
        "events": [],
        "event_signal": asyncio.Event(),
        "cancelled": asyncio.Event(),
        "filename": None,
        "clarify_prompt": None,
        "parent_session_id": session_id,
    }

    graph = _get_graph()
    config_dict = {
        "configurable": {"thread_id": new_session_id},
        "recursion_limit": 30,
    }
    _sessions[new_session_id]["graph"] = graph
    _sessions[new_session_id]["config"] = config_dict

    inputs = {
        "messages": [HumanMessage(content=topic)],
        "max_iterations": 3,
        "research_domain": domain,
        "llm_provider": provider,
        "year_range": year_range,
        "output_language": output_language,
    }

    asyncio.create_task(_run_pipeline(new_session_id, graph, config_dict, inputs))

    return {"session_id": new_session_id, "parent_session_id": session_id}


# ── Chat history (in-memory, per session) ────────────────────────────
_chat_histories: dict[str, list[dict]] = {}
_MAX_CHAT_MESSAGES = 100


class ChatRequest(BaseModel):
    session_id: str
    message: str
    filename: str = ""


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat about analysis results. Works with both active sessions and saved results."""
    # Build minimal AgentState from session or saved file
    state: dict = {}

    session = _sessions.get(req.session_id)
    if session and session.get("filename"):
        # Completed session — load from file
        path = OUTPUT_DIR / session["filename"]
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            state = _build_chat_state(data, session.get("query", ""))
    elif req.filename:
        # Load from filename directly
        path = OUTPUT_DIR / req.filename
        if not path.exists():
            raise HTTPException(404, "Result not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        state = _build_chat_state(data)
    else:
        raise HTTPException(400, "No completed analysis found for this session")

    # Restore chat history
    chat_key = req.session_id or req.filename
    if chat_key not in _chat_histories:
        _chat_histories[chat_key] = []
    if len(_chat_histories[chat_key]) > _MAX_CHAT_MESSAGES:
        _chat_histories[chat_key] = _chat_histories[chat_key][-_MAX_CHAT_MESSAGES:]

    # Add previous chat messages to state
    for msg in _chat_histories[chat_key]:
        if msg["role"] == "user":
            state.setdefault("messages", []).append(HumanMessage(content=msg["content"]))
        else:
            state.setdefault("messages", []).append(AIMessage(content=msg["content"], name="gap_chat"))

    try:
        response = gap_chat_respond(state, req.message)
    except Exception as e:
        raise HTTPException(500, f"Chat error: {e}")

    # Save to history
    _chat_histories[chat_key].append({"role": "user", "content": req.message})
    _chat_histories[chat_key].append({"role": "assistant", "content": response})

    return {"response": response}


def _build_chat_state(data: dict, query: str = "") -> dict:
    """Build a minimal AgentState dict from saved result data."""
    return {
        "refined_query": data.get("refined_query", query or data.get("query", "")),
        "gaps": data.get("gaps", []),
        "limitations": data.get("limitations", []),
        "papers": data.get("papers", []),
        "messages": [],
        "llm_provider": data.get("llm_provider", "azure"),
    }


@app.get("/api/stream/{session_id}")
async def stream(session_id: str, from_idx: int = 0):
    """
    SSE stream for a running session. Replays past events, then streams new ones.
    Supports reconnection — client can reconnect after refresh.
    Use from_idx to skip already-consumed events (e.g. after clarify resume).
    """
    # 세션이 아직 등록 안 됐을 수 있으므로 잠시 대기 (race condition 방어)
    session = _sessions.get(session_id)
    if not session:
        for _ in range(10):  # 최대 1초 대기
            await asyncio.sleep(0.1)
            session = _sessions.get(session_id)
            if session:
                break
    if not session:
        # SQLite fallback — 세션이 존재했는지 확인
        from utils.session_store import get_session
        db_session = get_session(session_id)
        if db_session:
            db_status = db_session["status"]
            # running 상태인데 메모리에 없음 = 서버 재시작으로 소실된 세션
            if db_status == "running":
                update_session_status(session_id, "interrupted")
                db_status = "interrupted"
            async def ended_stream():
                msg = {"event": "session_ended", "reason": db_status,
                       "message": "서버가 재시작되어 이전 분석이 중단되었습니다. 새로운 분석을 시작해주세요."}
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            return StreamingResponse(ended_stream(), media_type="text/event-stream")
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
        cancel_registry.cancel(session_id)
        # Wait briefly for the task to notice cancellation
        await asyncio.sleep(0.5)

    # 즉시 삭제하지 않고 상태만 변경 — reaper가 TTL 후 정리
    session["status"] = "stopped"
    session["completed_at"] = datetime.now()
    update_session_status(session_id, "stopped")
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
        keywords = values.get("keywords", [])
        payload["keywords"] = keywords
        payload["scope_level"] = values.get("scope_level", "")
        if keywords:
            payload["detail"] = f"Keywords: {', '.join(keywords[:5])}"

    elif node == "meaning_expand":
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                payload["expansion"] = content[:1000]
        payload["detail"] = "Search terms expanded"

    elif node == "paper_retrieval":
        papers = values.get("papers", [])
        web_results = values.get("web_results", [])
        payload["papers_count"] = len(papers)
        payload["total_searched"] = values.get("total_candidates_count", len(papers))
        payload["papers"] = []
        for p in papers:
            d = p if isinstance(p, dict) else (p.model_dump() if hasattr(p, "model_dump") else p.__dict__)
            payload["papers"].append({
                "paper_id": d.get("paper_id", ""),
                "title": d.get("title", ""),
                "year": d.get("year", ""),
                "authors": (d.get("authors") or [])[:3],
                "url": d.get("url", ""),
                "venue": d.get("venue", ""),
            })
        payload["web_results_count"] = len(values.get("web_results", []))

    elif node == "limitation_extract":
        limitations = values.get("limitations", [])
        payload["limitations_count"] = len(limitations)
        paper_ids = list({lim.get("paper_id", "") for lim in limitations})
        payload["detail"] = f"{len(paper_ids)}편 논문에서 {len(limitations)}개 한계점 추출"
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
        decision = eval_data.get("decision", "N/A")
        payload["decision"] = decision
        payload["call1_results"] = eval_data.get("call1_results", [])
        payload["call2_result"] = eval_data.get("call2_result", {})
        payload["eval_warnings"] = values.get("eval_warnings", [])
        lim_count = len(values.get("limitations", []))
        payload["limitations_count"] = lim_count
        payload["detail"] = f"{lim_count} limitations evaluated — {decision}"

    elif node == "recency_check":
        limitations = values.get("limitations", [])
        status_counts = {"unresolved": 0, "partial": 0, "resolved": 0}
        for lim in limitations:
            s = lim.get("recency_status", "unresolved")
            status_counts[s] = status_counts.get(s, 0) + 1
        payload["recency_status"] = status_counts
        payload["detail"] = f"Unresolved: {status_counts['unresolved']}, Partial: {status_counts['partial']}, Resolved: {status_counts['resolved']}"
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                payload["summary"] = content[:500]

    elif node == "gap_infer":
        gaps = values.get("gaps", [])
        payload["gaps_count"] = len(gaps)
        payload["gaps"] = gaps
        payload["detail"] = f"{len(gaps)}개의 연구 갭 도출 완료"

    elif node == "critic_score":
        msgs = values.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                payload["critic_output"] = content[:1500]
        payload["detail"] = "Quality check passed"

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
    # 랜딩페이지 우선 서빙
    landing_index = LANDING_DIR / "index.html"
    if landing_index.exists():
        return FileResponse(str(landing_index), media_type="text/html")
    # 랜딩페이지 미빌드 시 기존 앱으로 fallback
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"message": "Frontend not found"}


@app.get("/app")
async def app_page():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    raise HTTPException(404, "App not found")


# ── Landing page static assets (mounted at startup for reliability) ──
@app.on_event("startup")
async def mount_landing_assets():
    _landing_assets = LANDING_DIR / "assets"
    if _landing_assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_landing_assets)), name="landing-assets")