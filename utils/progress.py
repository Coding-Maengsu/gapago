"""
Thread-safe progress reporting for mid-node SSE updates.

Agents call report_progress() from sync threads.
The API drains the queue asynchronously and pushes SSE events.
"""
from __future__ import annotations

import queue
from typing import Dict, List

_queues: Dict[str, queue.Queue] = {}


def init_progress(session_id: str):
    _queues[session_id] = queue.Queue()


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
