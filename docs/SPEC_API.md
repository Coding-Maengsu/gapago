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

**응답:**
```json
{
  "azure": {"id": "azure", "name": "Azure OpenAI"},
  "claude": {"id": "claude", "name": "Claude (Bedrock)"},
  "gemini": {"id": "gemini", "name": "Google Gemini"},
  "exaone": {"id": "exaone", "name": "LG EXAONE"}
}
```

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
| `provider` | string | X | `"azure"` | LLM 프로바이더 (`azure`, `claude`, `gemini`, `exaone`) |
| `domain` | string | X | `"auto"` | 연구 도메인 (`auto`, `ai_cs`, `biomedical`, `materials_chemistry`, `physics`, `general`) |
| `year_range` | string | X | `"auto"` | 연도 범위 (`auto`, `1y`, `3y`, `5y`) |
| `output_language` | string | X | `"auto"` | 출력 언어 (`auto`, `ko`, `en`) |
| `user_id` | string | X | `""` | 사용자 식별자 |

**응답:**
```json
{"session_id": "uuid-string"}
```

**내부 동작:**
1. 세션 생성 (인메모리)
2. LangGraph 파이프라인 초기화
3. 백그라운드 태스크로 파이프라인 실행 시작
4. 즉시 `session_id` 반환

**파이프라인 입력 State:**
```python
{
    "messages": [HumanMessage(content=query)],
    "max_iterations": 3,
    "research_domain": domain,
    "llm_provider": provider,
    "year_range": year_range,
    "output_language": output_language,
}
```

---

### 3.4 `GET /api/stream/{session_id}` — SSE 스트림

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

#### `keepalive` 이벤트 (30초 간격)
```json
{"event": "keepalive"}
```

**재연결 메커니즘:**
- 모든 이벤트는 서버에 버퍼링됨
- `from_idx` 파라미터로 누락된 이벤트 재생
- 명확화(clarify) 후 재연결 시 사용

---

### 3.5 `GET /api/status/{session_id}` — 세션 상태 조회

**응답:**
```json
{
  "status": "running|interrupted|completed|stopped|error|not_found",
  "query": "원본 쿼리",
  "filename": "결과 파일명 (완료 시)"
}
```

---

### 3.6 `GET /api/stop/{session_id}` — 분석 중지

실행 중인 분석을 중지한다.

**응답:**
```json
{"status": "stopped", "session_id": "..."}
```

---

### 3.7 `GET /api/clarify` — 명확화 응답 제출

인터럽트된 파이프라인에 사용자 응답을 전달하고 재개한다.

**쿼리 파라미터:**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `session_id` | string | O | 세션 ID |
| `response` | string | O | 사용자의 명확화 응답 |

**응답:**
```json
{"status": "resumed", "accumulated_events": 5}
```

**내부 동작:**
1. 인터럽트된 서브그래프 상태에 사용자 응답 주입 (`app.update_state()`)
2. 백그라운드에서 파이프라인 재개
3. 누적된 이벤트 수 반환

---

### 3.8 `GET /api/history` — 분석 히스토리 목록

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
    "gaps_count": 7,
    "status": "completed"
  }
]
```

- 활성 세션 + 저장된 파일 결과 모두 포함
- 타임스탬프 역순 정렬

---

### 3.9 `GET /api/history/{filename}` — 저장된 결과 상세

**응답:** 저장된 JSON 결과 파일 전체 내용

```json
{
  "query": "원본 질문",
  "timestamp": "ISO8601",
  "user_id": "사용자 ID",
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

### 3.10 `GET /` — 정적 파일 서빙

- `frontend/index.html` 반환
- `/logo.png`, `/new_logo.png`, `/middle_image.png` 정적 이미지 서빙

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
  "warnings": [...]
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
      "axis_type": "fixed|dynamic",
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
    "events": list,         # 버퍼링된 SSE 이벤트
    "event_signal": Event,  # 새 이벤트 알림용
    "query": str,           # 원본 쿼리
    "user_id": str,         # 사용자 ID
    "filename": str,        # 결과 파일명 (완료 시)
    "cancelled": Event,     # 중지 시그널
}
```

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

#### Google Gemini
| 변수 | 설명 |
|------|------|
| `GOOGLE_API_KEY` | Google API 키 |

#### LG EXAONE (로컬 GPU)
| 변수 | 설명 | 기본값 |
|------|------|--------|
| `EXAONE_MODEL_PATH` | 모델 경로 | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` |

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
| `TAVILY_MAX_RESULTS` | int | `5` | 최대 웹 검색 결과 수 |
| `ARXIV_MAX_RESULTS` | int | `10` | 최대 arXiv 논문 수 |
| `BM25_TOP_K` | int | `30` | BM25 1차 필터 수 |
| `RERANKER_TOP_K` | int | `15` | LLM 리랭커 2차 선택 수 |
| `SCIENCEON_CLIENT_ID` | string | - | ScienceON API 클라이언트 ID |
| `SCIENCEON_MAC_ADDRESS` | string | - | ScienceON 토큰 생성용 MAC |
| `SCIENCEON_KEY` | string | - | ScienceON AES 암호화 키 |
| `SCIENCEON_DEFAULT_TARGET` | string | `ARTI` | ScienceON 검색 대상 |
| `SCIENCEON_DEFAULT_ROW_COUNT` | int | `10` | ScienceON 기본 결과 수 |

---

## 7. 설정 (Configuration)

**파일:** `config.py`

`Configuration` dataclass로 환경변수에서 로딩되며, LangGraph `RunnableConfig`를 통한 요청별 오버라이드를 지원한다.

```python
@dataclass
class Configuration:
    tavily_max_results: int    # 1-50, 기본 5
    arxiv_max_docs: int        # 1-50, 기본 10
    bm25_top_k: int            # 10-100, 기본 30
    reranker_top_k: int        # 5-50, 기본 15
    scienceon_client_id: str
    scienceon_mac_address: str
    scienceon_key: str
    scienceon_default_target: str  # 기본 "ARTI"
    scienceon_default_row_count: int  # 기본 10
```

**접근 방식:** `Configuration.from_runnable_config(config)` 메서드로 요청별 설정 조회

---

## 8. LLM 프로바이더

**파일:** `llm.py`

`get_llm(provider, model)` 팩토리 함수, `@lru_cache(maxsize=8)` 캐싱.

| 프로바이더 | 라이브러리 | 기본 모델 | 반환 타입 |
|-----------|-----------|----------|----------|
| `azure` | `langchain-openai` | `gpt-5.1-chat` | `AzureChatOpenAI` |
| `claude` / `anthropic` | `langchain-aws` | `claude-sonnet-4-20250514-v1:0` | `ChatBedrockConverse` |
| `gemini` / `google` | `langchain-google-genai` | `gemini-2.0-flash` | `ChatGoogleGenerativeAI` |
| `exaone` | `transformers` + `langchain` | `EXAONE-3.5-7.8B-Instruct` | `ChatHuggingFace` |

**대화형 선택:** `select_provider_interactive()` 함수로 CLI에서 프로바이더 선택

---

## 9. 검색 도구

**파일:** `tools.py`

### 9.1 도구 목록

| 도구 | 데이터 소스 | API 키 필요 | 설명 |
|------|-----------|------------|------|
| `arxiv_api_call_tool` | arXiv | X | 직접 API (XML/Atom 파싱) |
| `semantic_scholar_search_tool` | Semantic Scholar | X | 학술 그래프 API (2억+ 논문) |
| `openalex_search_tool` | OpenAlex | X | 학술 데이터 (2억+ 저작물) |
| `web_search_tool` | Tavily | O | 웹 검색 |
| `scienceon_search_tool` | ScienceON (KISTI) | O | 한국 학술 DB |
| `scienceon_patent_search_tool` | ScienceON | O | 특허 검색 |
| `scienceon_report_search_tool` | ScienceON | O | 국가 R&D 보고서 |

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
  "source": "arxiv|semantic_scholar|openalex|scienceon|web"
}
```

`paper_id` 접두사: `arxiv:`, `s2:`, `openalex:`, `scienceon:`, `web:`

### 9.3 BM25 랭킹

```python
bm25_rank(papers, query_text, top_k=30) -> List[dict]
```
- `rank_bm25` 라이브러리 (BM25Okapi 알고리즘)
- title + abstract를 토큰화하여 점수 계산
- 내림차순 정렬 후 상위 `top_k` 반환

### 9.4 도구 그룹핑

```python
build_role_tools(config) -> dict:
    "QUERY_TOOLS": []                    # 도구 없음
    "RETRIEVAL_TOOLS": [7개 검색 도구]    # 모든 검색 도구
    "LIMITATION_TOOLS": []               # 도구 없음
    "GAP_INFER_TOOLS": []               # 도구 없음
    "CRITIC_TOOLS": []                   # 도구 없음
    "RESPONSE_TOOLS": []                # 도구 없음
```

### 9.5 ScienceON 인증 흐름

```
1. 토큰 캐시 확인 (access + refresh)
2. access_token 없고 refresh_token 있음 → 갱신
3. 토큰 없고 인증 정보 있음 → 신규 생성 (AES 암호화)
4. 401/token_error 시 → 자동 갱신 후 재시도
```

---

## 10. 데이터 모델

**파일:** `states.py`

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
| 검색 | `papers`, `total_candidates_count`, `web_results`, `research_domain`, `llm_provider`, `year_range`, `output_language` | `List[dict]`, `int`, `List[dict]`, `str`, `str`, `str`, `str` |
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
    buildCommand: pip install -r requirements_deploy.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 3
    envVars:
      - key: PYTHON_VERSION
        value: "3.10.12"
      - key: LLM_PROVIDER
        value: azure
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
| `langchain-google-genai` | 4.2.1 | Gemini |
| `anthropic` | 0.86.0 | Claude SDK |
| `rank-bm25` | 0.2.2 | BM25 랭킹 |
| `arxiv` | 2.4.1 | arXiv API 클라이언트 |
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

### 12.2 `utils/tavily.py` — Tavily 검색 래퍼
- 도메인 필터링 (include/exclude)
- 파라미터: `search_depth`, `topic`, `max_results`, `days`

### 12.3 `utils/logging.py` — LangSmith 트레이싱
- 환경변수 기반 LangSmith 초기화

### 12.4 `utils/vis_graph.py` — 그래프 시각화
- LangGraph 파이프라인 시각화 유틸리티
