# ModelRouter 브랜치 — 웹 실행 시 문제점 분석

**분석 기준 브랜치**: `worktree-typed-questing-llama`
**작성일**: 2026-04-02

---

## [HIGH] 1. `/api/explore` inputs에 `session_id` 누락

**파일**: `api/main.py` — explore 엔드포인트의 inputs dict

analyze에는 `session_id`가 inputs에 포함되지만, explore에는 빠져 있음:

```python
# analyze — O
inputs = {
    ...
    "session_id": session_id,
    "fast_mode": fast_mode,
}

# explore — X
inputs = {
    ...
    "output_language": output_language,
    # session_id 누락!
}
```

**영향**: explore 실행 시 에이전트의 `report_progress(state)` 호출에서 session_id가 없어 SSE 진행률 이벤트가 프론트엔드에 전달되지 않음. 분석은 완료되지만 실시간 진행 상태가 표시 안 됨.

**수정 방법**: inputs에 `"session_id": new_session_id,` 추가

---

## [LOW] 2. `output_language` 선택 안 하면 빈 문자열 전송

**파일**: `frontend/index.html` — `startAnalysis()`, `exploreDirection()`

```javascript
const outputLanguage = document.getElementById('outputLanguage').value;
// select에서 아무것도 선택하지 않으면 빈 문자열 ""이 전송됨
```

API 기본값은 `"auto"`이지만, 빈 문자열이 전달되면 기본값이 적용되지 않음.

**영향**: `output_language: ""` 상태에서 언어 감지가 동작하지 않을 수 있음.

**수정 방법**: `document.getElementById('outputLanguage').value || 'auto'` fallback 추가 (두 곳 모두)

---

## [INFO] 3. 프론트엔드 `api/chat`에서 `routing_profile` 미전달

**파일**: `frontend/index.html` — `sendChatMessage()`

chat은 POST body로 `session_id`, `message`, `filename`만 전송. `routing_profile`, `output_language` 등은 전송하지 않음.

**실질적 영향 없음**: `_build_chat_state()`가 저장된 결과 파일에서 `model_routing`, `output_language`를 읽어오므로, 분석 완료 후 저장된 결과가 있으면 정상 동작함. 수정 불필요.

---

## [INFO] 4. `api/clarify`에 파라미터 미전달 (기존 동작)

**파일**: `frontend/index.html` — `resumePipeline()`

clarify는 `session_id`와 `response`만 전송. 이것은 main 브랜치에서도 동일한 패턴이며, clarify는 기존 graph state를 resume하므로 이미 state에 `model_routing`이 있음. **이번 브랜치의 신규 문제 아님.**

---

## 수정 요약

| 우선순위 | 항목 | 파일 | 수정 내용 |
|---|---|---|---|
| HIGH | explore session_id 누락 | `api/main.py` | inputs에 `"session_id": new_session_id` 추가 |
| LOW | outputLanguage 빈 문자열 | `frontend/index.html` (2곳) | `\|\| 'auto'` fallback 추가 |
| — | chat routing_profile | — | 수정 불필요 (저장 결과에서 복원됨) |
| — | clarify 파라미터 | — | 수정 불필요 (기존 동작, graph state에 존재) |

---

## 검증 방법

1. **analyze 실행** → SSE 진행률 정상 표시 확인
2. **explore 실행** → SSE 진행률 정상 표시 확인 (수정 전에는 진행률 미표시)
3. **채팅** → 라우팅 프로파일 유지 확인 (gap_chat이 올바른 provider 사용)
4. **output_language 미선택** 상태로 분석 → 언어 감지 정상 동작 확인
5. 각 프로파일(balanced/optimized/quality/speed)로 분석 실행 → provider 드롭다운 활성화/비활성화 확인
