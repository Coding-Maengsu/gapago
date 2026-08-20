# GAPAGO 메모리 관리 & 404 에러 통합 수정 계획

> 404 "Session not found" 에러와 Render 4GB OOM을 하나의 세션 생명주기 관리로 해결

## 1. 현황 분석

### Render 환경

- **메모리 한도:** 4GB (초과 시 `Instance failed: OOM`)
- **서버:** `uvicorn --workers 1` (단일 프로세스)
- **모델 티어:** `RERANK_MODELS=light` (MiniLM 계열, ~80MB)

### 메모리 사용 구조

| 구성 요소 | 위치 | 예상 메모리 | 정리 방식 | 위험도 |
|-----------|------|------------|----------|--------|
| `_sessions` dict (이벤트 무한 누적) | `gapago/api/main.py:100` | **무제한 증가** | `/api/stop`만 | **Critical** |
| **MemorySaver 체크포인트** (공유 그래프) | `gapago/graphs/graph.py:126` | **무제한 증가** | **없음** | **Critical** |
| **PDF 동시 다운로드 (8워커)** | `limitation_agent.py:1280` | **피크 ~240MB** | 함수 종료 | **Critical** |
| `_chat_histories` dict | `gapago/api/main.py:423` | **무제한 증가** | 없음 | **High** |
| `messages` 리스트 (LangGraph state) | `states.py:177` | **노드마다 누적** | 없음 | **High** |
| ML 모델 (MiniLM + CrossEncoder light) | `retrieval_agent.py:32-35` | ~80MB (고정) | 없음 (설계상) | Low |
| progress 큐 | `gapago/utils/progress.py:13` | 소량 | ✅ 정리됨 | Low |

### 핵심 발견: MemorySaver 누적

`_graph_cache`가 프로세스 전체에서 **단일 인스턴스**로 공유됨 (`gapago/api/main.py:80-86`).
이 graph 안의 `MemorySaver`도 하나이며, 모든 세션의 체크포인트가 `thread_id`별로 누적됨.

```
세션 A (thread_id=aaa) → MemorySaver에 8개 노드 체크포인트 저장 (~250KB)
세션 B (thread_id=bbb) → 같은 MemorySaver에 추가 (~250KB)
세션 C ...
→ _sessions에서 세션을 삭제해도 MemorySaver 체크포인트는 남아있음
→ 세션 100개 누적 시: ~25MB, 1000개: ~250MB+
```

### 핵심 발견: PDF 동시 다운로드 메모리 스파이크

`limitation_agent.py:1280`에서 ThreadPoolExecutor **8개 워커**가 동시에 PDF 다운로드 + PyMuPDF 파싱.

```
워커 8개 × arXiv PDF (5-30MB) = 최대 240MB 순간 스파이크
PyMuPDF: PDF 전체를 메모리에 올린 뒤 마크다운 변환
다른 메모리 사용과 합산 시 4GB 돌파 가능
```

### 핵심 발견: MemorySaver × 노드 수 = state N배 복제

파이프라인 **8개 노드** 각각에서 MemorySaver가 전체 state를 스냅샷.

```
State 1회분: papers(20편) + limitations(100개) + messages + gaps ≈ 300KB
× 8 노드 체크포인트 = ~2.4MB/세션
× critic 루프 재실행 시 추가 복제
```

### 에러 시나리오

```
[404 "Session not found"]
1. 서버 재시작 → _sessions = {} 초기화 → 프론트 재연결 시 404
2. /api/stop → _sessions.pop() → 스트림 아직 연결 중 → 404
3. 히스토리 "분석 중..." 클릭 → status 미확인 → 바로 stream 연결 → 404

[OOM (4GB 초과)]
1. 세션 완료 후 _sessions에 events, graph, config 잔존 (pop 안 됨)
2. MemorySaver에 모든 세션 체크포인트 영구 누적 (노드 × 세션 수)
3. _chat_histories 무한 증가
4. PDF 동시 다운로드 8개 스파이크 + 위 3개 합산 → 4GB 초과
5. Explore 기능으로 세션 증식 시 가속
```

---

## 2. 통합 수정 계획

핵심 원칙: **세션 생명주기를 단일 시스템으로 관리하고, 모든 메모리 참조를 확실히 해제**

### Phase 1: 세션 생명주기 정리 (404 + OOM 동시 해결)

#### 1-1. startup 시 interrupted 세션 정리

**파일:** `gapago/utils/session_store.py` — `init_db()`

**현재:** running → `error`
**변경:** running → `interrupted` (에러가 아니라 서버 재시작으로 중단된 것)

```python
# 변경 전 (line 47-48)
conn.execute("UPDATE sessions SET status = 'error' WHERE status = 'running'")

# 변경 후
conn.execute("""
    UPDATE sessions SET status = 'interrupted',
    completed_at = COALESCE(completed_at, ?)
    WHERE status = 'running'
""", (datetime.now().isoformat(),))
```

#### 1-2. `/api/stop` — pop() → 상태 변경

**파일:** `gapago/api/main.py:535-549`

**현재:** `_sessions.pop(session_id, None)` — 즉시 삭제 → 스트림 연결 중 404
**변경:** 상태만 변경, 정리는 reaper에 위임

```python
# 변경 전 (line 548)
_sessions.pop(session_id, None)

# 변경 후
session["status"] = "stopped"
session["completed_at"] = datetime.now()
update_session_status(session_id, "stopped")
```

#### 1-3. 세션 자동 정리 reaper + MemorySaver 체크포인트 해제 (핵심)

**파일:** `gapago/api/main.py` (신규 함수)

완료/중단/에러 상태 세션을 TTL 기반으로 자동 정리하는 백그라운드 태스크.
**MemorySaver 체크포인트도 함께 삭제.**

```python
import gc

_SESSION_TTL_SECONDS = 300  # 완료 후 5분 보관 (Render 4GB 환경이라 짧게)

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
            if session:
                # ── MemorySaver 체크포인트 해제 ──
                # _graph_cache는 공유 인스턴스이므로,
                # 해당 thread_id의 체크포인트만 삭제
                _clear_checkpoints(sid)
                # 참조 해제
                session.clear()
            # chat history도 함께 정리
            _chat_histories.pop(sid, None)

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
            # MemorySaver의 내부 storage dict에서 thread_id 키 제거
            checkpointer.storage.pop(thread_id, None)
        if hasattr(checkpointer, 'writes'):
            checkpointer.writes.pop(thread_id, None)
    except Exception as e:
        print(f"[reaper] checkpoint cleanup error for {thread_id}: {e}")
```

startup에 등록:
```python
@app.on_event("startup")
async def warmup():
    init_session_db()
    asyncio.create_task(_session_reaper())  # ← 추가
    # ... 기존 warmup 코드
```

#### 1-4. 파이프라인 완료 시 completed_at 기록

**파일:** `gapago/api/main.py` — `_run_pipeline()` 내

```python
# 완료 경로 (line 245-248)
session["status"] = "completed"
session["filename"] = fname
session["completed_at"] = datetime.now()  # ← 추가

# 에러 경로 (line 251-253)
session["status"] = "error"
session["completed_at"] = datetime.now()  # ← 추가

# 중단(stop) 경로 (line 198-200)
session["status"] = "stopped"
session["completed_at"] = datetime.now()  # ← 추가
```

### Phase 2: 이벤트 메모리 상한

#### 2-1. 세션당 이벤트 수 제한

**파일:** `gapago/api/main.py:166-172`

```python
_MAX_EVENTS_PER_SESSION = 500

def _push_event(session_id: str, event: dict):
    session = _sessions.get(session_id)
    if not session:
        return
    events = session["events"]
    if len(events) >= _MAX_EVENTS_PER_SESSION:
        # 앞쪽 10% 제거 (오래된 이벤트 드롭)
        del events[:_MAX_EVENTS_PER_SESSION // 10]
    events.append(event)
    session["event_signal"].set()
```

### Phase 3: chat_histories 정리

#### 3-1. 메시지 수 상한

**파일:** `gapago/api/main.py:456-458`

```python
_MAX_CHAT_MESSAGES = 100

# chat() 함수 내, _chat_histories 접근 부분
if chat_key not in _chat_histories:
    _chat_histories[chat_key] = []
if len(_chat_histories[chat_key]) > _MAX_CHAT_MESSAGES:
    _chat_histories[chat_key] = _chat_histories[chat_key][-_MAX_CHAT_MESSAGES:]
```

### Phase 4: 프론트엔드 404 방어

**파일:** `frontend/index.html`

#### 4-1. reconnectToSession에 status 체크 추가

```javascript
async function reconnectToSession(sessionId, query) {
    // ── status 확인 먼저 ──
    try {
        const statusResp = await fetch(`gapago/api/status/${sessionId}`);
        const statusData = await statusResp.json();
        if (['not_found', 'stopped', 'error', 'completed'].includes(statusData.status)) {
            showError('이 세션은 더 이상 사용할 수 없습니다. 새로운 분석을 시작해주세요.');
            localStorage.removeItem('gapago_active_session');
            loadHistory();
            return;
        }
    } catch (e) {
        showError('서버에 연결할 수 없습니다.');
        return;
    }
    // ── 기존 reconnect 로직 (line 2035~) ──
    currentSessionId = sessionId;
    localStorage.setItem('gapago_active_session', sessionId);
    // ... (나머지 기존 코드 유지)
}
```

#### 4-2. connectToStream 404 재시도 축소

```javascript
// status 체크를 이미 했으므로 재시도 불필요
async function connectToStream(sessionId, fromIdx, retries = 1) {
    // ... (기존 로직, retries 기본값만 변경)
}
```

### Phase 5: `/api/stream` SQLite fallback (선택)

**파일:** `gapago/api/main.py:491-507`

완료/중단된 세션의 스트림 요청 시 404 대신 적절한 SSE 응답 반환.

```python
@app.get("/api/stream/{session_id}")
async def stream(session_id: str, from_idx: int = 0):
    session = _sessions.get(session_id)
    if not session:
        for _ in range(10):
            await asyncio.sleep(0.1)
            session = _sessions.get(session_id)
            if session:
                break
    if not session:
        # SQLite fallback — 세션이 존재했는지 확인
        from utils.session_store import get_session
        db_session = get_session(session_id)
        if db_session:
            status = db_session["status"]
            async def ended_stream():
                msg = {"event": "session_ended", "reason": status,
                       "message": f"세션이 {status} 상태입니다. 새로운 분석을 시작해주세요."}
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            return StreamingResponse(ended_stream(), media_type="text/event-stream")
        raise HTTPException(404, "Session not found")
    # ... 기존 스트림 로직
```

### Phase 6: 파이프라인 실행 중 메모리 스파이크 억제 (신규)

#### 6-1. PDF 동시 다운로드 워커 축소

**파일:** `gapago/agents/limitation_agent.py:1280`

**현재:** `ThreadPoolExecutor(max_workers=min(8, len(papers)))` — 8개 동시 다운로드
**변경:** `max_workers=3`으로 축소

```python
# 변경 전
with ThreadPoolExecutor(max_workers=min(8, len(papers))) as executor:

# 변경 후
with ThreadPoolExecutor(max_workers=min(3, len(papers))) as executor:
```

**효과:** 피크 메모리 ~240MB → ~90MB (동시 PDF 3개로 제한)
**트레이드오프:** full text 로딩 시간 약간 증가 (8병렬 → 3병렬), 안정성 확보가 더 중요

#### 6-2. PDF 크기 제한 추가

**파일:** `gapago/agents/limitation_agent.py` — PDF 다운로드 함수들

```python
_MAX_PDF_BYTES = 10 * 1024 * 1024  # 10MB 제한

# requests.get() 호출 부분에 추가
resp = requests.get(url, timeout=8)
if len(resp.content) > _MAX_PDF_BYTES:
    print(f"  [fulltext] PDF too large ({len(resp.content) // 1024 // 1024}MB), skipping")
    return {}
```

**효과:** 30MB+ 대형 PDF가 메모리를 점유하는 것을 방지

#### 6-3. `_save_result()` 스트리밍 직렬화

**파일:** `gapago/api/main.py:127-163`

**현재:** `json.dumps(result)` → 전체 결과를 문자열로 메모리에 올림
**변경:** `json.dump(result, f)`로 파일에 직접 스트리밍

```python
# 변경 전
path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

# 변경 후
with open(path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
```

**효과:** 직렬화 결과 문자열 (~500KB-2MB) 메모리 더블링 방지

### Phase 7: HTTP 연결 풀링 (신규)

#### 7-1. requests.Session() 사용

**파일:** `gapago/agents/limitation_agent.py`, `gapago/agents/retrieval_agent.py`

**현재:** 매번 `requests.get()` 새 연결 생성 (TCP 핸드셰이크 + 메모리 누적)
**변경:** 모듈 레벨 `requests.Session()` 공유

```python
# agents/limitation_agent.py 상단
import requests

_http_session = requests.Session()
_http_session.headers.update({"User-Agent": "GAPAGO/1.0"})

# 기존 requests.get() 호출을 모두 _http_session.get()으로 변경
# 예: requests.get(url, timeout=8) → _http_session.get(url, timeout=8)
```

**효과:**
- TCP 연결 재사용 → 연결 생성/해제 오버헤드 제거
- GC 대기 상태의 미사용 연결 메모리 ~10MB 절감
- 응답 시간도 개선 (keep-alive)

### Phase 8: ONNX Runtime GPU 탐색 비활성화 (신규)

#### 8-1. 환경변수 추가

**파일:** `render.yaml`

```yaml
- key: ORT_DISABLE_GPU_DEVICE_ENUMERATION
  value: "1"
```

**효과:** Render(GPU 없음)에서 불필요한 GPU 디바이스 탐색 경고 제거
```
[W:onnxruntime:Default, device_discovery.cc:164] GPU device discovery failed
```

---

## 3. 구현 순서 및 우선순위

| 순서 | 항목 | Phase | 해결 문제 | 난이도 |
|------|------|-------|----------|--------|
| 1 | startup interrupted 정리 | 1-1 | 404 방지 | 1줄 |
| 2 | `/api/stop` pop → 상태변경 | 1-2 | 404 방지 | 3줄 |
| 3 | completed_at 기록 | 1-4 | reaper 의존 | 3줄 |
| 4 | **세션 reaper + MemorySaver 정리** | 1-3 | **OOM 핵심** | ~40줄 |
| 5 | **PDF 워커 축소 8→3 + 크기 제한** | 6-1, 6-2 | **OOM 스파이크** | 2줄 |
| 6 | 이벤트 상한 500개 | 2-1 | OOM 방지 | 5줄 |
| 7 | chat_histories 상한 100개 | 3-1 | OOM 방지 | 3줄 |
| 8 | `_save_result` 스트리밍 직렬화 | 6-3 | OOM 피크 | 2줄 |
| 9 | HTTP 연결 풀링 | 7-1 | 메모리+성능 | 5줄 |
| 10 | 프론트 reconnect status 체크 | 4-1 | 404 UX | 10줄 |
| 11 | connectToStream 재시도 축소 | 4-2 | 불필요한 404 제거 | 1줄 |
| 12 | stream SQLite fallback | 5 | 완성도 (선택) | 15줄 |
| 13 | ONNX GPU 탐색 비활성화 | 8-1 | 로그 정리 | 1줄 |

---

## 4. 메모리 예상 효과

### 변경 전 (Render 4GB 환경, 최악 시나리오)

```
Python + FastAPI 기본:           ~200MB
ML 모델 (light):                 ~80MB
세션 10개 × events 누적:         ~500MB  (정리 안 됨)
MemorySaver 10개 세션 × 8노드:   ~50MB   (정리 안 됨, 계속 증가)
chat_histories:                  ~100MB  (정리 안 됨)
PDF 동시 다운로드 피크 (8개):     ~240MB  (순간 스파이크)
_save_result 직렬화 더블링:       ~2MB    (피크)
HTTP 미사용 연결 잔존:            ~10MB
──────────────────────────────────────────
합계:                            ~1,182MB (계속 증가 → 결국 4GB 초과)
```

### 변경 후

```
Python + FastAPI 기본:           ~200MB
ML 모델 (light):                 ~80MB
활성 세션 1-2개:                 ~50MB   (reaper가 5분 후 정리)
MemorySaver:                     ~5MB    (완료 세션 체크포인트 삭제)
chat_histories:                  ~5MB    (100개 상한)
PDF 동시 다운로드 피크 (3개):     ~90MB   (워커 축소)
HTTP 연결 풀 (재사용):            ~2MB
──────────────────────────────────────────
합계:                            ~432MB  (안정적 상한, 피크 시 ~520MB)
```

**예상 효과: 무제한 증가 ~1.2GB+ → ~432MB 안정 (Render 4GB 대비 87% 여유)**

---

## 5. 수정 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `gapago/utils/session_store.py` | `init_db()` — running → interrupted |
| `gapago/api/main.py` | reaper 추가, stop pop→상태변경, 이벤트 상한, completed_at, stream fallback, _save_result 스트리밍 |
| `gapago/agents/limitation_agent.py` | PDF 워커 8→3, PDF 크기 제한, HTTP 세션 풀링 |
| `gapago/agents/retrieval_agent.py` | HTTP 세션 풀링 |
| `frontend/index.html` | reconnect status 체크, 재시도 축소 |
| `render.yaml` | ORT_DISABLE_GPU_DEVICE_ENUMERATION 환경변수 |
| `gapago/graphs/graph.py` | 변경 없음 (reaper에서 checkpointer 접근) |

---

## 6. 검증 방법

1. **OOM 방지**: 5개+ 세션 연속 실행 후 `[reaper] Cleaned N sessions` 로그 확인
2. **PDF 스파이크**: limitation 단계에서 메모리 피크가 기존 대비 낮은지 확인
3. **404 방지**: 서버 재시작 후 프론트에서 이전 세션 접근 → "세션이 interrupted 상태" 안내
4. **stop 안정성**: 분석 중 중지 → 스트림 정상 종료 (404 없음)
5. **메모리 안정**: Render 대시보드에서 메모리가 ~500MB 이하 유지 확인
6. **MemorySaver 해제**: reaper 실행 후 `checkpointer.storage` 크기 감소 확인
7. **ONNX 경고 해소**: 서버 시작 시 GPU discovery 경고 없음 확인

---

## 7. 리스크 및 참고

### Explore 세션 증식
`/api/explore`로 GAP 탐색을 반복하면 독립 세션이 계속 생성됨.
→ reaper가 완료 세션을 5분 후 정리하므로 대응되지만, 동시에 여러 explore를 실행하면 피크 증가 가능.

### messages 리스트 누적
`AgentState.messages`는 `add_messages` 리듀서로 모든 노드의 LLM 응답이 계속 추가됨 (20-50개, ~250KB).
MemorySaver가 노드마다 이를 복제하므로 실질적으로 8배 (8노드 × 250KB = ~2MB/세션).
→ 현재 계획의 reaper + checkpoint 정리로 세션 종료 후 해제되지만, **실행 중에는 줄일 수 없음**.
→ 향후 개선: 노드 간 message 요약/정리 (LangGraph 구조 변경 필요, 현재 스코프 밖).

### _save_result 시점 메모리 더블링
`graph.get_state()` + `json.dump()` 시 state가 일시적으로 복제됨.
→ Phase 6-3 스트리밍 직렬화로 문자열 더블링은 해결하지만, `get_state()` 자체의 복제는 불가피.
→ 여러 세션이 동시 완료되면 스파이크 가능 (실제로는 단일 사용자 환경이라 낮은 확률).
