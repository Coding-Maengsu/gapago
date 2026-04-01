"""
SQLite-based session persistence.

Stores session metadata so sessions survive process restarts.
Transient data (events, signals) remain in-memory.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sessions.db"
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db():
    """Create sessions table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'running',
            query TEXT NOT NULL DEFAULT '',
            user_id TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL,
            completed_at TEXT,
            metadata TEXT DEFAULT '{}'
        )
    """)
    conn.commit()
    # Mark any leftover "running" sessions as "interrupted" (server restart)
    conn.execute("""
        UPDATE sessions SET status = 'interrupted',
        completed_at = COALESCE(completed_at, ?)
        WHERE status = 'running'
    """, (datetime.now().isoformat(),))
    conn.commit()


def save_session(
    session_id: str,
    status: str,
    query: str,
    user_id: str = "",
    started_at: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """Insert or update a session record."""
    conn = _get_conn()
    now = started_at or datetime.now().isoformat()
    meta_json = json.dumps(metadata or {}, ensure_ascii=False)
    conn.execute("""
        INSERT INTO sessions (session_id, status, query, user_id, started_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            status = excluded.status,
            query = excluded.query,
            metadata = excluded.metadata
    """, (session_id, status, query, user_id, now, meta_json))
    conn.commit()


def update_session_status(session_id: str, status: str):
    """Update session status. Sets completed_at for terminal states."""
    conn = _get_conn()
    completed_at = datetime.now().isoformat() if status in ("completed", "error", "stopped") else None
    conn.execute("""
        UPDATE sessions SET status = ?, completed_at = COALESCE(?, completed_at)
        WHERE session_id = ?
    """, (status, completed_at, session_id))
    conn.commit()


def get_session(session_id: str) -> Optional[dict]:
    """Get a session record."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["metadata"] = json.loads(result.get("metadata") or "{}")
    return result


def list_sessions(user_id: str = "", limit: int = 50) -> list[dict]:
    """List recent sessions, optionally filtered by user_id."""
    conn = _get_conn()
    if user_id:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        results.append(d)
    return results


def delete_session(session_id: str):
    """Delete a session record."""
    conn = _get_conn()
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
