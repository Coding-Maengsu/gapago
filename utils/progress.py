"""
Thread-safe progress reporting for mid-node SSE updates.

Agents call report_progress() from sync threads.
The API drains the queue asynchronously and pushes SSE events.
"""
from __future__ import annotations

import queue
import time
from typing import Dict, List, Optional

_queues: Dict[str, queue.Queue] = {}

# ── Pipeline-level ETA tracking ─────────────────────────────────────────
STAGE_WEIGHTS = {
    "query_subgraph": 5,
    "meaning_expand": 5,
    "paper_retrieval": 20,
    "limitation_extract": 25,
    "limitation_eval": 5,
    "recency_check": 10,
    "gap_infer": 25,
    "critic_score": 3,
    "final_response": 2,
}
_TOTAL_WEIGHT = sum(STAGE_WEIGHTS.values())

# Per-session pipeline tracking state
_pipeline_state: Dict[str, dict] = {}


def init_progress(session_id: str):
    _queues[session_id] = queue.Queue()
    _pipeline_state[session_id] = {
        "started_at": time.time(),
        "completed_stages": [],
        "current_stage": None,
    }


def mark_stage_start(session_id: str, stage: str):
    """Called when a pipeline stage begins."""
    ps = _pipeline_state.get(session_id)
    if not ps:
        return
    ps["current_stage"] = stage


def mark_stage_done(session_id: str, stage: str):
    """Called when a pipeline stage completes."""
    ps = _pipeline_state.get(session_id)
    if not ps:
        return
    if stage not in ps["completed_stages"]:
        ps["completed_stages"].append(stage)
    ps["current_stage"] = None


def get_pipeline_eta(session_id: str) -> dict:
    """Compute pipeline-level progress percentage and ETA."""
    ps = _pipeline_state.get(session_id)
    if not ps:
        return {}

    done_weight = sum(STAGE_WEIGHTS.get(s, 0) for s in ps["completed_stages"])
    pct = round(done_weight / _TOTAL_WEIGHT * 100)

    elapsed = time.time() - ps["started_at"]
    if pct > 0:
        estimated_total = elapsed / (pct / 100)
        remaining = max(0, estimated_total - elapsed)
    else:
        remaining = 0

    return {
        "pipeline_progress_pct": pct,
        "elapsed_s": round(elapsed, 1),
        "estimated_remaining_s": round(remaining, 1),
    }


def report_progress(session_id: str, node: str, detail: str,
                    current: int = 0, total: int = 0):
    q = _queues.get(session_id)
    if not q:
        return
    payload = {
        "event": "progress",
        "node": node,
        "detail": detail,
    }
    if total > 0:
        payload["current"] = current
        payload["total"] = total
        payload["progress"] = round(current / total * 100)
    # Attach pipeline-level ETA
    eta = get_pipeline_eta(session_id)
    if eta:
        payload.update(eta)
    q.put(payload)


def drain_progress(session_id: str) -> list:
    q = _queues.get(session_id)
    if not q:
        return []
    items = []
    try:
        while True:
            items.append(q.get_nowait())
    except queue.Empty:
        pass
    return items


def cleanup_progress(session_id: str):
    _queues.pop(session_id, None)
    _pipeline_state.pop(session_id, None)
