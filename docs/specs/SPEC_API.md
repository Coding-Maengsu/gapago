# GAPAGO API 스펙

## 1. 개요

GAPAGO API는 FastAPI 기반의 비동기 REST API + SSE(Server-Sent Events) 서버이다. 연구 갭 분석 파이프라인의 실행, 세션 관리, 실시간 스트리밍, 결과 조회 기능을 제공한다.

---

## 2. 기술 스택

| 컴포넌트 | 기술 |
|---------|------|
| 웹 프레임워크 | FastAPI |
| ASGI 서버 | Uvicorn |
| 실시간 통신 | Server-Sent Events (SSE) |
| 세션 관리 | 인메모리 딕셔너리 |
| 파이프라인 엔진 | LangGraph StateGraph |
| 결과 저장 | JSON 파일 (`outputs/`) |
| 배포 | Render.com |

---

## 3. 엔드포인트

### 3.1 `GET /api/health` — 헬스 체크

**응답:**
```json
{"status": "ok"}
```

---

### 3.2 `GET /api/providers` — LLM 프로바이더 목록

**응답** (키는 숫자 문자열 — `AVAILABLE_PROVIDERS` 기반):
```json
{
  "1": {"id": "azure", "name": "Azure OpenAI (GPT)"},
  "2": {"id": "claude", "name": "Claude (AWS Bedrock)"},
  "3": {"id": "exaone", "name": "LG EXAONE (Local GPU)"}
}
```

> **Note:** Gemini는 사용자 선택 목록에서 제거됨. 코드에서는 하위 호환을 위해 유지되나 `AVAILABLE_PROVIDERS`에 노출되지 않음.

---

### 3.3 `GET /api/explore` — 추가 탐색 (체인 재실행)

이전 분석의 proposed_topic을 기반으로 새 분석을 시작하고 부모 세션에 연결한다.

**쿼리 파라미터:**

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `topic` | string | O | - | 탐색할 주제 (proposed_topic) |
| `session_id` | string | X | `""` | 부모 세션 ID |
| `provider` | string | X | `"azure"` | LLM 프로바이더 |
| `domain` | string | X | `"auto"` | 연구 도메인 |
| `year_range` | string | X | `"auto"` | 연도 범위 |
| `output_language` | string | X | `"auto"` | 출력 언어 |
| `user_id` | string | X | `""` | 사용자 식별자 |
| `routing_profile` | string | X | `"optimized"` | 라우팅 프로파일 (`optimized`, `quality`) |
| `fast_mode` | bool | X | `false` | 빠른 분석 모드 |

**응답:**
```json
{"session_id": "새_세션_ID", "parent_session_id": "부모_세션_ID"}
```

---

### 3.4 `GET /api/analyze` — 분석 시작

새로운 연구 갭 분석을 시작하고 세션 ID를 반환한다.

**쿼리 파라미터:**

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `query` | string | O | - | 연구 질문 |
| `provider` | string | X | `"azure"` | LLM 프로바이더 (`azure`, `claude`, `exaone`) |
| `domain` | string | X | `"auto"` | 연구 도메인 (`auto`, `ai_cs`, `biomedical`, `materials_chemistry`, `physics`, `general`) |
| `year_range` | string | X | `"auto"` | 연도 범위 (`auto`, `1y`, `3y`, `5y`) |
| `output_language` | string | X | `"auto"` | 출력 언어 (`auto`, `ko`, `en`) |
| `user_id` | string | X | `""` | 사용자 식별자 |
| `fast_mode` | bool | X | `false` | 빠른 분석 모드 (CrossEncoder 스킵, 상위 3개 축만 분석) |
| `routing_profile` | string | X | `"optimized"` | 라우팅 프로파일 (`optimized`, `quality`) |

**응답:**
```json
{"session_id": "uuid-string"}
```

**내부 동작:**
1. 세션 생성 (인메모리)
2. `ModelRouter(default_provider, profile)` 생성 → `model_routing` dict 주입
3. LangGraph 파이프라인 초기화
4. 백그라운드 태스크로 파이프라인 실행 시작
5. 즉시 `session_id` 반환

**파이프라인 입력 State:**
```python
{
    "messages": [HumanMessage(content=query)],
    "max_iterations": 3,
    "research_domain": domain,
    "llm_provider": provider,
    "year_range": year_range,
    "output_language": output_language,
    "session_id": session_id,       # 진행률 리포팅용
    "fast_mode": fast_mode,         # 빠른 분석 모드
    "model_routing": router.to_dict(),  # 에이전트별 LLM 라우팅 설정
}
```

---

### 3.5 `POST /api/chat` — 결과 기반 대화

분석 완료 후 결과에 대한 질문/답변 대화를 수행한다.

**요청 Body (JSON):**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `session_id` | string | O | 세션 ID |
| `message` | string | O | 사용자 질문 |
| `filename` | string | X | 저장된 결과 파일명 (세션 없이 히스토리에서 로드 시) |

**응답:**
```json
{"response": "AI 답변 텍스트"}
```

**내부 동작:**
1. 완료된 세션이면 결과 파일에서 state 재구성
2. `filename`만 제공 시 해당 파일에서 state 로드
3. `gap_chat_respond(state, message)` 호출
4. 세션별 대화 히스토리 유지 (최대 100개 메시지)

---

### 3.6 `DELETE /api/history/{filename}` — 히스토리 삭제

저장된 분석 결과를 삭제한다.

**응답:**
```json
{"status": "deleted", "filename": "..."}
```

**내부 동작:**
1. 파일명 검증 (path traversal 방지)
2. 파일에서 `session_id` 읽어서 SQLite 세션 레코드도 삭제
3. 관련 채팅 히스토리도 함께 정리

---

### 3.7 `GET /api/stream/{session_id}` — SSE 스트림

실시간 파이프라인 진행 상황을 SSE로 스트리밍한다.

**경로 파라미터:**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `session_id` | string | 분석 세션 ID |

**쿼리 파라미터:**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `from_idx` | int | `0` | 이벤트 재생 시작 인덱스 (재연결 시 사용) |

**응답:** `text/event-stream`

**이벤트 형식:**

#### `node` 이벤트 (노드 완료)
```json
{"event": "node", "node": "<node_name>", ...payload}
```

#### `interrupt` 이벤트 (명확화 필요)
```json
{"event": "interrupt", "session_id": "...", "clarify_prompt": "질문 텍스트"}
```

#### `complete` 이벤트 (분석 완료)
```json
{"event": "complete", "filename": "gapago_result_YYYYMMDD_HHMMSS.json"}
```

#### `error` 이벤트
```json
{"event": "error", "message": "에러 설명"}
```

#### `stopped` 이벤트 (사용자 중지)
```json
{"event": "stopped"}
```

#### `progress` 이벤트 (노드 내 중간 진행률)
```json
{
  "event": "progress",
  "node": "limitation_extract",
  "detail": "Analyzing papers... (5/15) — 12 limitations found",
  "current": 5,
  "total": 15,
  "progress": 33
}
```
- `current`, `total`, `progress`는 `total > 0`일 때만 포함
- `utils/progress.py`의 스레드 안전 큐를 통해 에이전트 스레드에서 비동기 SSE로 전달
- API 서버의 `_drain_loop` 태스크가 0.3초 간격으로 큐를 폴링

#### `keepalive` 이벤트 (30초 간격)
```json
{"event": "keepalive"}
```

**재연결 메커니즘:**
- 모든 이벤트는 서버에 버퍼링됨
- `from_idx` 파라미터로 누락된 이벤트 재생
- 명확화(clarify) 후 재연결 시 사용

---

### 3.8 `GET /api/status/{session_id}` — 세션 상태 조회

**응답:**
```json
{
  "status": "running|interrupted|completed|stopped|error|not_found",
  "query": "원본 쿼리",
  "filename": "결과 파일명 (완료 시)"
}
```

---

### 3.9 `GET /api/stop/{session_id}` — 분석 중지

실행 중인 분석을 중지한다.

**응답:**
```json
{"status": "stopped", "session_id": "..."}
```

---

### 3.10 `GET /api/clarify` — 명확화 응답 제출

인터럽트된 파이프라인에 사용자 응답을 전달하고 재개한다.

**쿼리 파라미터:**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `session_id` | string | O | 세션 ID |
| `response` | string | O | 사용자의 명확화 응답 |

**응답 (성공):**
```json
{"session_id": "...", "status": "resumed", "events_count": 5}
```

**에러 응답:**
- `404`: 존재하지 않는 세션
- `410`: 서버 재시작으로 세션이 소실됨 (SQLite에 기록은 있으나 인메모리에 없음)

**내부 동작:**
1. 인메모리 세션 조회 → 없으면 SQLite fallback으로 원인 구분 (404 vs 410)
2. 인터럽트된 서브그래프 상태에 사용자 응답 주입 (`app.update_state()`)
3. 백그라운드에서 파이프라인 재개 (`_run_pipeline(session_id, graph, config, None)` — inputs=None으로 resume)
4. 누적된 이벤트 수 반환

---

### 3.11 `GET /api/history` — 분석 히스토리 목록

**쿼리 파라미터:**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `user_id` | string | `""` | 사용자 필터 (빈 문자열이면 전체) |

**응답:**
```json
[
  {
    "filename": "gapago_result_20260328_143022.json",
    "query": "연구 질문 미리보기",
    "timestamp": "2026-03-28T14:30:22",
    "refined_query": "정제된 쿼리",
    "gaps_count": 7,
    "status": "completed",
    "session_id": "uuid-string",
    "parent_session_id": ""
  }
]
```

- 활성 세션 (`running`/`interrupted`) + 저장된 파일 결과 모두 포함
- `HistoryItem` Pydantic 모델로 구조화

---

### 3.12 `GET /api/history/{filename}` — 저장된 결과 상세

**응답:** 저장된 JSON 결과 파일 전체 내용

```json
{
  "query": "원본 질문",
  "timestamp": "ISO8601",
  "user_id": "사용자 ID",
  "session_id": "세션 ID",
  "parent_session_id": "부모 세션 ID (추가 탐색 시)",
  "refined_query": "정제된 검색 쿼리",
  "keywords": ["keyword1", "keyword2"],
  "papers": [...],
  "limitations": [...],
  "limitation_eval": {...},
  "eval_warnings": [...],
  "gaps": [...],
  "web_results": [...],
  "messages": [...]
}
```

---

### 3.13 `GET /` — 정적 파일 서빙

- 랜딩 페이지(`landing/dist/index.html`) 우선 서빙
- 랜딩 미빌드 시 `frontend/index.html`로 fallback
- `/logo.png`, `/new_logo.png`, `/middle_image.png` 정적 이미지 서빙

### 3.14 `GET /app` — 앱 페이지 서빙

- `frontend/index.html` 반환 (메인 분석 SPA)
- 랜딩 페이지와 분리된 앱 진입점

---

## 4. SSE 노드별 페이로드 구조

### 4.1 `query_subgraph`
```json
{
  "event": "node",
  "node": "query_subgraph",
  "refined_query": "정제된 쿼리",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "scope_level": "SEARCHABLE|TOO_BROAD|TOO_NARROW"
}
```

### 4.2 `paper_retrieval`
```json
{
  "event": "node",
  "node": "paper_retrieval",
  "papers_count": 15,
  "total_searched": 130,
  "papers": [
    {
      "paper_id": "arxiv:2305.12345",
      "title": "논문 제목",
      "year": 2024,
      "authors": ["Author A", "Author B"],
      "url": "https://arxiv.org/abs/2305.12345",
      "venue": "arXiv preprint"
    }
  ],
  "web_results_count": 5
}
```

### 4.3 `limitation_extract`
```json
{
  "event": "node",
  "node": "limitation_extract",
  "limitations_count": 12,
  "limitations": [
    {
      "paper_id": "arxiv:2305.12345",
      "claim": "한계점 설명",
      "track": "author_stated|structural",
      "source_section": "Conclusion",
      "evidence_quote": "원문 인용"
    }
  ]
}
```

### 4.4 `limitation_eval`
```json
{
  "event": "node",
  "node": "limitation_eval",
  "decision": "PASS|RETRY",
  "call1_results": [...],
  "call2_result": {...},
  "eval_warnings": [...],
  "limitations_count": 12,
  "detail": "12 limitations evaluated — PASS"
}
```

### 4.5 `recency_check`
```json
{
  "event": "node",
  "node": "recency_check",
  "recency_status": {
    "unresolved": 12,
    "partial": 5,
    "resolved": 3
  },
  "summary": "요약 텍스트"
}
```

### 4.6 `gap_infer`
```json
{
  "event": "node",
  "node": "gap_infer",
  "gaps_count": 7,
  "gaps": [
    {
      "axis": "data",
      "axis_label": "Data & Dataset",
      "axis_type": "dynamic",
      "gap_statement": "갭 설명 (25단어 이내)",
      "elaboration": "상세 설명",
      "proposed_topic": "제안 연구 방향",
      "repeat_count": 3,
      "supporting_papers": ["arxiv:xxx", "s2:yyy"],
      "supporting_quotes": ["인용1", "인용2"]
    }
  ]
}
```

### 4.7 `critic_score`
```json
{
  "event": "node",
  "node": "critic_score",
  "critic_output": "평가 텍스트 (DECISION: ACCEPT|REDO_RETRIEVAL|REFINE_QUERY 포함)"
}
```

### 4.8 `final_response`
```json
{
  "event": "node",
  "node": "final_response",
  "report": "마크다운 형식의 최종 보고서"
}
```

---

## 5. 세션 관리

### 5.1 세션 구조 (인메모리)

```python
_sessions[session_id] = {
    "status": str,          # running, interrupted, completed, stopped, error
    "graph": CompiledGraph, # LangGraph 인스턴스
    "config": RunnableConfig,
    "events": list,         # 버퍼링된 SSE 이벤트 (최대 500개, 초과 시 앞쪽 10% 제거)
    "event_signal": Event,  # 새 이벤트 알림용
    "query": str,           # 원본 쿼리
    "user_id": str,         # 사용자 ID
    "started_at": str,      # ISO8601 시작 시간
    "completed_at": datetime, # 완료 시각 (reaper TTL 기준)
    "filename": str,        # 결과 파일명 (완료 시)
    "cancelled": Event,     # 중지 시그널
    "clarify_prompt": str,  # 명확화 프롬프트 (인터럽트 시)
    "parent_session_id": str,  # 추가 탐색 시 부모 세션 (/api/explore에서만)
}
```

**채팅 히스토리 (별도 관리):**
- `_chat_histories[session_id]` — 세션별 대화 메시지 리스트 (최대 100개)
- 세션 또는 filename을 키로 사용

**진행률 큐 (별도 관리):**
- `utils/progress.py`의 `_queues[session_id]`로 관리
- `init_progress(session_id)`: 파이프라인 시작 시 초기화
- `cleanup_progress(session_id)`: 파이프라인 종료 시 정리

### 5.2 라이프사이클

```
생성 (/api/analyze)
  → 실행 중 (running)
  → [인터럽트 (interrupted)] → 재개 (/api/clarify) → 실행 중
  → 완료 (completed) | 중지 (stopped) | 오류 (error)
```

---

## 6. 환경변수

### 6.1 필수 (LLM 프로바이더별)

#### Azure OpenAI (기본)
| 변수 | 설명 | 예시 |
|------|------|------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API 키 | - |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI 엔드포인트 | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | 배포 이름 | `gpt-5.1-chat` |
| `AZURE_OPENAI_API_VERSION` | API 버전 | `2024-12-01-preview` |

#### Claude (AWS Bedrock)
| 변수 | 설명 | 기본값 |
|------|------|--------|
| `AWS_REGION` | AWS 리전 | `us-east-1` |
| `AWS_ACCESS_KEY_ID` | AWS 액세스 키 | - |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 키 | - |
| `BEDROCK_CLAUDE_MODEL` | 모델 ID | `us.anthropic.claude-sonnet-4-20250514-v1:0` |

#### Google Gemini (Vertex AI)
| 변수 | 설명 | 기본값 |
|------|------|--------|
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | GCP 서비스 계정 JSON (임시 파일로 저장됨) | - |
| `GOOGLE_CLOUD_PROJECT` | GCP 프로젝트 ID | `coding-beast` |
| `GOOGLE_CLOUD_LOCATION` | GCP 리전 | `us-central1` |

#### LG EXAONE (로컬 GPU)
| 변수 | 설명 | 기본값 |
|------|------|--------|
| `EXAONE_MODEL_PATH` | 모델 경로 | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` |

#### Groq (GAP 추론 단계 전용)
| 변수 | 설명 | 기본값 |
|------|------|--------|
| `GROQ_API_KEY` | Groq API 키 | - |
| `GROQ_MODEL` | 모델 ID | `qwen/qwen3-32b` |
| `GROQ_REASONING_EFFORT` | 추론 노력 수준 | `default` (`default`/`none`) |

#### QwQ (GAP 추론 단계 전용, 로컬 GPU)
| 변수 | 설명 | 기본값 |
|------|------|--------|
| `QWQ_MODEL_PATH` | 모델 경로 | `Qwen/QwQ-32B` |

### 6.2 필수 (공통)
| 변수 | 설명 |
|------|------|
| `TAVILY_API_KEY` | Tavily 웹 검색 API 키 |
| `LANGCHAIN_API_KEY` 또는 `LANGSMITH_API_KEY` | LangSmith 트레이싱 키 |

### 6.3 선택

| 변수 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `LLM_PROVIDER` | string | `azure` | 기본 LLM 프로바이더 |
| `LLM_MODEL` | string | - | 모델/배포 이름 |
| `GAP_REASONING_PROVIDER` | string | - | GAP 추론 단계 전용 프로바이더 (`groq`, `qwq`) |
| `TAVILY_MAX_RESULTS` | int | `5` | 최대 웹 검색 결과 수 |
| `ARXIV_MAX_RESULTS` | int | `20` | 최대 arXiv 논문 수 |
| `BM25_TOP_K` | int | `50` | BM25 1차 필터 수 |
| `RERANKER_TOP_K` | int | `15` | CrossEncoder/LLM 리랭커 2차 선택 수 |
| `RERANK_MODELS` | string | `auto` | 랭킹 모델 tier (`auto`/`light`/`full`) |
| `FULLTEXT_TARGET_COUNT` | int | `30` | Full text 필터 후 최대 논문 수 |
| `SCIENCEON_CLIENT_ID` | string | - | ScienceON API 클라이언트 ID |
| `SCIENCEON_MAC_ADDRESS` | string | - | ScienceON 토큰 생성용 MAC |
| `SCIENCEON_KEY` | string | - | ScienceON AES 암호화 키 |
| `SCIENCEON_DEFAULT_TARGET` | string | `ARTI` | ScienceON 검색 대상 |
| `SCIENCEON_DEFAULT_ROW_COUNT` | int | `20` | ScienceON 기본 결과 수 |

---

## 7. 설정 (Configuration)

**파일:** `core/config.py`

`Configuration` dataclass로 환경변수에서 로딩되며, LangGraph `RunnableConfig`를 통한 요청별 오버라이드를 지원한다.

```python
@dataclass
class Configuration:
    tavily_max_results: int    # 1-50, 기본 5
    arxiv_max_docs: int        # 1-50, 기본 20
    scienceon_client_id: str
    scienceon_mac_address: str
    scienceon_key: str
    scienceon_default_target: str  # 기본 "ARTI"
    scienceon_default_row_count: int  # 기본 20
    fulltext_target_count: int # 15-60, 기본 30
    bm25_top_k: int            # 10-100, 기본 50
    reranker_top_k: int        # 5-50, 기본 15
    rerank_models: str         # "auto"/"light"/"full", 기본 "auto"
```

**접근 방식:** `Configuration.from_runnable_config(config)` 메서드로 요청별 설정 조회

---

## 8. LLM 프로바이더

**파일:** `core/llm.py`

`get_llm(provider, model)` 팩토리 함수, `@lru_cache(maxsize=8)` 캐싱.

**기본 파이프라인 프로바이더** (사용자 선택 가능):

| 프로바이더 | 라이브러리 | 기본 모델 | 반환 타입 |
|-----------|-----------|----------|----------|
| `azure` | `langchain-openai` | `gpt-5.1-chat` | `AzureChatOpenAI` |
| `claude` / `anthropic` | `langchain-aws` | `claude-sonnet-4-20250514-v1:0` | `ChatBedrockConverse` (read_timeout=300s) |
| `exaone` | `transformers` + `langchain` | `EXAONE-3.5-7.8B-Instruct` | `ChatHuggingFace` |

> **Note:** `gemini` / `google` 프로바이더는 코드에 유지되나 사용자 선택 목록에서 제거됨.

**에이전트별 LLM 라우팅** (`core/model_router.py`):

`get_llm_for_agent(state, agent_name)` 함수로 에이전트별 최적 provider를 자동 배정.
`model_routing` state 필드가 없으면 기존 `llm_provider` fallback.

| 프로파일 | 설명 | 대부분 에이전트 | 핵심 추출/보고서 | 추론 | orchestrator |
|---------|------|----------------|----------------|------|-------------|
| `optimized` | 에이전트별 최적화 (기본) | azure (기본 provider) | claude | groq | groq |
| `quality` | 최고 품질 | azure | claude | groq | groq |

> **Note (2026-04-03):** optimized 프로파일에서 경량 작업(query_analysis, meaning_expand, critic_score 등)의 Groq 라우팅이 환각 방지를 위해 기본 provider(azure)로 변경됨. orchestrator와 gap_reasoning만 Groq 유지.

**GAP 추론 전용 프로바이더** (`GAP_REASONING_PROVIDER` 환경변수로 선택, 내부 라우팅 전용):

| 프로바이더 | 라이브러리 | 기본 모델 | 특징 |
|-----------|-----------|----------|------|
| `groq` | `langchain-groq` | `qwen/qwen3-32b` | ~535 tok/s, Thinking Mode 지원, `reasoning_effort` 파라미터 |
| `qwq` | `transformers` + `langchain` | `Qwen/QwQ-32B` | 로컬 GPU (A100 권장), CoT 추론 특화 |

**대화형 선택:** `select_provider_interactive()` 함수로 CLI에서 기본 프로바이더 선택. GAP 추론 프로바이더는 환경변수로만 설정.

---

## 9. 검색 도구

**파일:** `core/tools/`

### 9.1 검색 함수 목록

> **참고:** LLM tool 호출이 아닌, `retrieval_agent.py`의 `_parallel_search()`가 직접 호출하는 Python 함수.

| 함수 | 데이터 소스 | API 키 필요 | 설명 |
|------|-----------|------------|------|
| `arxiv_api_call` | arXiv | X | 직접 API (XML/Atom), `threading.Lock` 직렬화 + 5초 간격 |
| `crossref_search` | Crossref | X | 1.5억+ 메타데이터, PDF URL 추출, venue 포함 |
| `semantic_scholar_search` | Semantic Scholar | X | 학술 그래프 API (2억+ 논문) |
| `openalex_search` | OpenAlex | X | 학술 데이터 (2억+ 저작물), inverted index abstract |
| `TavilySearch.search` | Tavily | O | 웹 검색 (트렌드용) |
| `scienceon_search` | ScienceON (KISTI) | O | 한국 학술 DB |
| `scienceon_patent_search` | ScienceON | O | 특허 검색 |
| `scienceon_report_search` | ScienceON | O | 국가 R&D 보고서 |

### 9.2 논문 정규화 스키마

모든 소스의 결과는 통일된 스키마로 정규화된다:

```json
{
  "paper_id": "source:id",
  "title": "논문 제목",
  "abstract": "초록",
  "url": "https://...",
  "year": 2024,
  "authors": ["Author A", "Author B"],
  "doi": "10.xxxx/yyyy",
  "venue": "arXiv preprint",
  "source": "arxiv|crossref|semantic_scholar|openalex|scienceon|web",
  "full_text_sections": {"doi": "...", "pdf_url": "..."}
}
```

`paper_id` 접두사: `arxiv:`, `crossref:`, `s2:`, `openalex:`, `scienceon:`, `web:`

### 9.3 BM25 랭킹

```python
bm25_rank(papers, query_text, top_k=30) -> List[dict]
```
- `rank_bm25` 라이브러리 (BM25Okapi 알고리즘)
- title + abstract를 토큰화하여 점수 계산
- 내림차순 정렬 후 상위 `top_k` 반환

### 9.4 검색 호출 방식

> LLM ReAct 에이전트 기반 tool 호출에서 **직접 병렬 함수 호출**로 전환됨.

`retrieval_agent.py`의 `_parallel_search()`가 `ThreadPoolExecutor(max_workers=len(tasks))`로 8개 검색 함수를 동시 실행. LangChain `@tool` 데코레이터나 `build_role_tools()`는 더 이상 사용하지 않음.

### 9.5 ScienceON 인증 흐름

```
1. 토큰 캐시 확인 (access + refresh)
2. access_token 없고 refresh_token 있음 → 갱신
3. 토큰 없고 인증 정보 있음 → 신규 생성 (AES 암호화)
4. 401/token_error 시 → 자동 갱신 후 재시도
```

---

## 10. 데이터 모델

**파일:** `core/states.py`

### 10.1 Pydantic 모델

| 모델 | 용도 | 주요 필드 |
|------|------|----------|
| `ScopeAssessment` | 쿼리 범위 평가 | `scope_level`, `general_topic`, `specific_phrases`, `breadth_candidates` |
| `QueryResult` | 쿼리 분석 결과 | `scope_assessment`, `refined_query`, `keywords`, `negative_keywords` |
| `Paper` | 논문 메타데이터 | `paper_id`, `title`, `abstract`, `url`, `year`, `authors`, `score_bm25`, `venue` |
| `LimitationItem` | 한계점 항목 | `paper_id`, `claim`, `evidence_quote`, `track`, `source_section` |
| `GapCandidate` | 연구 갭 후보 | `axis`, `gap_statement`, `elaboration`, `proposed_topic`, `repeat_count`, `supporting_papers` |
| `CriticScores` | 비평 점수 | `query_specificity`, `paper_relevance`, `groundedness` (각 0.0-1.0) |

### 10.2 AgentState (TypedDict) 필드 요약

| 섹션 | 필드 | 타입 |
|------|------|------|
| 오케스트레이션 | `messages`, `sender`, `errors` | `Sequence[BaseMessage]`, `str`, `List[str]` |
| 쿼리 | `iteration`, `max_iterations`, `scope_level`, `refined_query`, `keywords`, `negative_keywords`, `needs_user_input` | `int`, `int`, `str`, `str`, `List[str]`, `List[str]`, `bool` |
| 검색 | `papers`, `total_candidates_count`, `web_results`, `research_domain`, `llm_provider`, `year_range`, `output_language`, `model_routing`, `fast_mode`, `session_id` | `List[dict]`, `int`, `List[dict]`, `str`, `str`, `str`, `str`, `dict`, `bool`, `str` |
| 오케스트레이터 | `completed_stages`, `agent_feedback`, `orchestrator_plan` | `List[str]`, `dict`, `List[str]` |
| 한계점 | `limitations` | `List[dict]` |
| 한계점 평가 | `limitation_eval`, `eval_warnings`, `eval_retry_count` | `dict`, `List[str]`, `int` |
| 갭 추론 | `gaps` | `List[dict]` |
| 비평 | `critic`, `critic_loop_count` | `Optional[dict]`, `int` |
| 트레이싱 | `trace` | `dict` |

---

## 11. 배포

### 11.1 Render.com (`render.yaml`)

```yaml
services:
  - type: web
    name: gapago
    runtime: python
    buildCommand: cd landing && npm install && npm run build && cd .. && pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
    envVars:
      - key: PYTHON_VERSION
        value: "3.10.12"
      - key: NODE_VERSION
        value: "20.18.0"
      - key: LLM_PROVIDER
        value: azure
      - key: RERANK_MODELS
        value: light
      - key: ORT_DISABLE_GPU_DEVICE_ENUMERATION
        value: "1"
```

### 11.2 주요 의존성

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| `fastapi` | latest | 웹 프레임워크 |
| `uvicorn` | - | ASGI 서버 |
| `langchain` | 1.2.10 | 에이전트 프레임워크 |
| `langgraph` | 1.0.8 | 상태 그래프 오케스트레이션 |
| `langchain-openai` | 1.1.9 | Azure OpenAI |
| `langchain-aws` | 1.4.0 | AWS Bedrock (Claude) |
| `langchain-google-vertexai` | - | Gemini (Vertex AI) |
| `langchain-groq` | - | Groq (Qwen3-32B) |
| `anthropic` | 0.86.0 | Claude SDK |
| `sentence-transformers` | - | SPECTER2/MiniLM 임베딩 + CrossEncoder |
| `faiss-cpu` / `faiss-gpu` | - | FAISS 벡터 검색 |
| `rank-bm25` | 0.2.2 | BM25 랭킹 |
| `tavily` | 1.1.0 | Tavily 웹 검색 |
| `pydantic` | 2.12.5 | 데이터 검증 |
| `python-dotenv` | 1.2.1 | .env 로딩 |
| `pycryptodome` | 3.23.0 | AES 암호화 (ScienceON) |
| `torch` | 2.10.0 | GPU 지원 (EXAONE) |
| `transformers` | 5.1.0 | HuggingFace 모델 |
| `beautifulsoup4` | 4.14.3 | HTML 파싱 |
| `PyMuPDF` | 1.24.3 | PDF 텍스트 추출 |

---

## 12. 유틸리티

### 12.1 `utils/parse_json.py` — JSON 파싱
- 직접 JSON → 마크다운 코드 블록 → 정규식 추출 순서로 시도
- 실패 시 빈 dict/list 반환

### 12.2 `utils/session_store.py` — SQLite 세션 영속화
- 서버 재시작 시에도 세션 상태 유지
- `init_db()`: DB 초기화, 기존 running 세션을 interrupted로 마킹
- `save_session()`, `update_session_status()`, `get_session()`, `delete_session()`

### 12.3 `utils/cancel.py` — 파이프라인 취소 레지스트리
- 세션별 취소 시그널 관리
- `register()`, `cancel()`, `is_cancelled()`, `cleanup()`

### 12.4 `utils/tavily.py` — Tavily 검색 래퍼
- 도메인 필터링 (include/exclude)
- 파라미터: `search_depth`, `topic`, `max_results`, `days`

### 12.5 `utils/logging.py` — LangSmith 트레이싱
- 환경변수 기반 LangSmith 초기화
