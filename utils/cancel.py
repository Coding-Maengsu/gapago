"""
Lightweight cancellation registry for pipeline sessions.

Agents can check `is_cancelled(session_id)` without importing api.main.
"""
import threading

_cancelled: dict[str, threading.Event] = {}


def register(session_id: str) -> threading.Event:
    """Register a cancellation event for a session."""
    evt = threading.Event()
    _cancelled[session_id] = evt
    return evt


def cancel(session_id: str):
    """Signal cancellation for a session."""
    evt = _cancelled.get(session_id)
    if evt:
        evt.set()


def is_cancelled(session_id: str) -> bool:
    """Check if a session has been cancelled. Safe to call from any thread."""
    evt = _cancelled.get(session_id)
    return evt.is_set() if evt else False


def cleanup(session_id: str):
    """Remove a session from the registry."""
    _cancelled.pop(session_id, None)
