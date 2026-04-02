import json
import re


def _strip_thinking_tags(text: str) -> str:
    """
    Groq Qwen3 / QwQ 등 추론 모델이 출력하는 <think>...</think> 블록을 제거.
    JSON 파싱 전처리로 사용.
    """
    # <think>...</think> (중첩 포함, non-greedy)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 간혹 닫는 태그 없이 끝나는 경우 대비
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def parse_json(text: str) -> dict | list:
    """
    Parse JSON from LLM response with error recovery.
    Groq Qwen3 / QwQ 추론 모델의 <think> 블록을 자동으로 제거한 뒤 파싱.

    Args:
        text: Raw LLM response

    Returns:
        Parsed JSON (dict or list)
    """
    # ── Step 0: <think> 태그 제거 (추론 모델 CoT 출력 처리) ──────────────
    cleaned = _strip_thinking_tags(text)

    # ── Step 1: 직접 파싱 시도 ──────────────────────────────────────────
    for candidate in (cleaned, text):  # cleaned 먼저, 실패 시 원본 재시도
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # ── Step 2: 코드블록 안의 JSON 추출 ────────────────────────────
        json_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", candidate, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # ── Step 3: 텍스트에서 JSON 오브젝트/배열 추출 ─────────────────
        json_match = re.search(r"\{.*\}|\[.*\]", candidate, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    print(f"⚠️ Could not parse JSON from response: {text[:200]}...")
    return {}