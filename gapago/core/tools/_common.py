"""검색 소스 모듈 공통 헬퍼."""
from __future__ import annotations

import json
import re


def _norm(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return _norm(text).lower().split()


def _safe_json_loads(text: str) -> dict | list | None:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text or "")
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
