# GAPAGO 기술보고서 — 개발 내용

## 1. 시스템 개요

### 1-1. 전체 아키텍처
- **프레임워크**: FastAPI (백엔드) + Vanilla JS SPA (프론트엔드)
- **파이프라인 엔진**: LangGraph StateGraph (조건부 루프 + 서브그래프)
- **통신**: Server-Sent Events (SSE) 실시간 스트리밍
- **세션 관리**: 인메모리 dict + SQLite 영속화
- **배포**: Render (단일 워커, CPU-only)

### 1-2. Agent 구성 요소 (파이프라인 노드 8개)

| 순서 | 노드 | 파일 | 역할 |
|------|------|------|------|
| 1 | query_subgraph | `agents/query_agent/` | 연구 질문 범위 평가 + 정제 (SemRank) |
| 2 | meaning_expand | `agents/meaning_expand_agent.py` | 검색 키워드 확장 (동의어, 약어, 변형) |
| 3 | paper_retrieval | `agents/retrieval_agent.py` | 다중 소스 논문 검색 + 임베딩 리랭킹 |
| 4 | limitation_extract | `agents/limitation_agent.py` | 전문(full-text) 기반 한계점 추출 |
| 5 | limitation_eval | `agents/limitation_eval_agent.py` | 한계점 품질 검증 (FActScore + Prometheus) |
| 6 | recency_check | `agents/recency_agent.py` | 최신성 검증 (웹 검색 기반) |
| 7 | gap_infer | `agents/gap_agent.py` | 연구 갭 도출 (동적 축 생성 + 긴급도 채점) |
| 8 | critic_score → final_response | `agents/critic_agent.py`, `agents/response_agent.py` | 품질 게이트 + 최종 리포트 생성 |

### 1-3. 보조 Agent

| Agent | 파일 | 역할 |
|-------|------|------|
| orchestrator | `agents/orchestrator_agent.py` | LLM 기반 동적 라우팅 (선택적 노드 삽입) |
| gap_chat | `agents/gap_chat_agent.py` | 분석 결과 대화형 Q&A |

### 1-4. 데이터 흐름도

```
사용자 질문 입력
  ↓
[query_subgraph] → scope 평가 → (TOO_BROAD → 사용자 확인 인터럽트)
  ↓ refined_query, keywords
[meaning_expand] → 키워드 확장, 플랫폼별 쿼리 후보 생성
  ↓ expanded_terms, query_candidates
[paper_retrieval] → 8개 소스 병렬 검색 → BM25 → 임베딩 → CrossEncoder 리랭킹
  ↓ papers (최대 30편, full-text 접근 가능 기준)
[limitation_extract] → 전문 다운로드 (8단계 폴백) → 섹션 분할 → 2트랙 추출
  ↓ limitations[]
[limitation_eval] → Call1: 원자적 팩트 검증 + 루브릭 채점 → Call2: 전체 품질 판단
  ↓ (RETRY → limitation_extract 재실행 / PASS → 다음)
[recency_check] → 도메인 자동 감지 → Tavily 웹 검색 → unresolved/partial/resolved 분류
  ↓ limitations[] (recency_status 부착)
[gap_infer] → 동적 축 생성 → 한계점 분류 → 긴급도 채점 → 기술 장벽 분석 → 창의적 방향 제안
  ↓ gaps[]
[critic_score] → query_specificity / paper_relevance / groundedness 평가
  ↓ (ACCEPT → final_response / REDO_RETRIEVAL → meaning_expand / REFINE_QUERY → query_subgraph)
[final_response] → 마크다운 리포트 생성
  ↓
결과 저장 (JSON) + 사용자에게 SSE 전달
```

### 1-5. 품질 제어 루프
- **critic_score → meaning_expand**: 논문 관련성 부족 시 재검색 (최대 2회)
- **critic_score → query_subgraph**: 질문 구체성 부족 시 재정제
- **limitation_eval → limitation_extract**: 추출 품질 미달 시 재추출
- **Human-in-the-Loop**: query_subgraph에서 scope=TOO_BROAD 시 사용자 선택 인터럽트

---

## 2. 핵심 알고리즘 및 기술적 특징

### 2-1. 질문 분석 (Query Analysis)
- **SemRank** (Zhang et al., EMNLP 2025): 다중 입도(multi-granular) 과학 개념 인덱싱
- **APA** (Kim et al., EMNLP 2024): 모호성 인지 기반 쿼리 정제 (CLAMBER 탐지 + INFOGAIN 임계값)
- Scope 3분류: TOO_BROAD (3 하위유형 A/B/C) / SEARCHABLE / TOO_NARROW
- 구성요소 추출: Domain [D], Task [T], Modality [M], Problem [P]

### 2-2. 논문 검색 및 리랭킹 (Retrieval)
- **8개 소스 병렬 검색** (ThreadPoolExecutor):
  - arXiv (100편) / Crossref (60편) / Semantic Scholar (50편) / OpenAlex (40편)
  - ScienceON 논문 (15편) + 특허 (10편) + 보고서 (10편)
  - Tavily 웹 검색 (3쿼리 × 3결과)
- **3단계 리랭킹 파이프라인**:
  1. BM25 초벌 필터링 (top-50)
  2. SPECTER2 / MiniLM-L6-v2 임베딩 유사도
  3. BGE Reranker v2-m3 / MiniLM CrossEncoder 정밀 리랭킹
- **ONNX Runtime 최적화**: CPU 환경에서 2-3× 추론 가속
- **Full-text 접근성 티어링**: guaranteed (ar5iv) > likely (OA PDF) > maybe (DOI) > none
- **중복 제거**: DOI + 제목 + 연도 기준

### 2-3. 전문 기반 한계점 추출 (Limitation Extraction)
- **8단계 전문 획득 폴백 체인**:
  1. arXiv ar5iv HTML
  2. arXiv PDF (LangChain)
  3. DOI 랜딩 페이지
  4. Unpaywall OA 탐색
  5. S2 Batch API (ArXiv/PMC ID)
  6. PMC BioC XML
  7. EuropePMC JATS XML
  8. Direct PDF (pymupdf4llm)
- **섹션 분할**: 정규식 기반 (introduction, method, experiment, discussion, conclusion, limitations, future_work)
- **2트랙 병렬 추출**:
  - Track 1: 저자 명시 한계점 (conclusion, limitations, future_work 섹션)
  - Track 2: 구조적 분석 (method/experiment/discussion 마이닝)
- **캐싱**: 전문 섹션 7일 TTL, 실패 캐시 6시간

### 2-4. 한계점 평가 (Limitation Evaluation)
- **4개 논문 앙상블 프레임워크**:
  - **Call 1 — 원자적 검증 + 루브릭 채점**:
    - FActScore 기반: 주장 분해 → 개별 팩트 검증
    - Prometheus 기반: 3차원 루브릭 (Groundedness / Specificity / Relevance, 각 1-5점)
    - 병렬 배치 처리 (10개/배치, 최대 3 워커)
  - **Call 2 — 전체 판단**:
    - LimAgents 기반: strong / weak / remove 판정
    - Xu et al. 분류 체계: methodology / data / scope / evaluation / theoretical / resource
    - Set-level 분석: type_distribution, coverage_warning (>75% 편중), diversity_score (1-5)
  - **RETRY 조건**: >50% weak, 평균 groundedness <3.0, 평균 fact_score <0.6, diversity ≤2

### 2-5. 최신성 검증 (Recency Check)
- **도메인 자동 감지**: LLM 기반 (ai_cs / biomedical / materials_chemistry / physics / general)
- **도메인 특화 웹 검색**: Tavily 5개 쿼리, 시간 마커 포함 ("2024", "2025", "latest")
- **3단계 분류**: unresolved (미해결) / partial (부분 해결) / resolved (해결됨)

### 2-6. 연구 갭 도출 (Gap Inference)
- **동적 축 생성** (귀납적, 사전 정의 카테고리 없음):
  - LLM이 한계점 집합에서 도메인 특화 축을 도출 (3-7개)
  - 축당 최소 2개 한계점, 상호 배타적 범위 검증
- **긴급도 채점**: recency 가중치 (unresolved 1.0 > partial 0.5 > resolved 0.0)
- **기술 장벽 분석**: Reasoning 모델 (QwQ-32B / Qwen3-32B)로 지속 이유 분석
- **창의적 방향 제안**: 웹 검색 결과 맥락 활용, novelty_score (1-10) 채점

### 2-7. 품질 게이트 (Critic Score)
- **LLM-as-a-Judge**: 3개 메트릭 (0.0-1.0)
  - query_specificity / paper_relevance / groundedness
- **결정 규칙**: 모두 ≥0.6 → ACCEPT / paper_relevance <0.4 → REDO_RETRIEVAL / query_specificity <0.4 → REFINE_QUERY
- **루프 제한**: MAX_CRITIC_LOOPS = 2

### 2-8. Fast Mode 최적화
- CrossEncoder 리랭킹 스킵
- limitation_eval Call 1 스킵
- 검색 논문 수 축소
- 경량 임베딩 모델 사용
- 전체 provider를 Groq로 전환 (speed 프로파일)

---

## 3. 사용된 LLM, 데이터 및 도구

### 3-1. LLM 모델

| Provider | 모델 | 용도 |
|----------|------|------|
| Azure OpenAI | GPT-5.2 | 기본 파이프라인 전 에이전트 |
| Anthropic (Bedrock) | Claude Sonnet 4 | Quality 프로파일 / 고품질 추출 |
| Google | Gemini 3.1 Flash Lite | 경량 작업 대안 |
| Groq | Qwen3-32B | Gap reasoning + Speed 프로파일 |
| LG AI | EXAONE 3.5-7.8B | 로컬 GPU 추론 옵션 |

### 3-2. 임베딩 및 리랭킹 모델

| 모델 | 용도 | 환경 |
|------|------|------|
| SPECTER2 | 논문 임베딩 | GPU (full 모드) |
| MiniLM-L6-v2 | 논문 임베딩 | CPU (light 모드) |
| BGE Reranker v2-m3 | CrossEncoder 리랭킹 | GPU (full 모드) |
| MiniLM CrossEncoder | CrossEncoder 리랭킹 | CPU (light 모드) |

### 3-3. 외부 데이터 소스 / API

| 서비스 | 용도 | 최대 건수 |
|--------|------|-----------|
| arXiv API | 논문 검색 | 100편 |
| Crossref API | 교차 도메인 논문 | 60편 |
| Semantic Scholar API | 학술 그래프 | 50편 |
| OpenAlex API | 오픈 리서치 메타데이터 | 40편 |
| ScienceON API | 국내 논문/특허/보고서 | 15+10+10편 |
| Tavily Web Search | 최신 동향 검색 | 3쿼리 × 3결과 |
| Unpaywall API | OA 전문 탐색 | 논문별 1회 |
| PMC BioC API | 구조화된 전문 (XML) | 논문별 1회 |
| EuropePMC API | JATS XML 전문 | 논문별 1회 |
| ar5iv | arXiv HTML 렌더링 | 논문별 1회 |

### 3-4. 주요 라이브러리

| 카테고리 | 라이브러리 |
|----------|------------|
| 파이프라인 | LangGraph 1.0.8, LangChain 1.2.10 |
| 웹 서버 | FastAPI, Uvicorn |
| LLM 연동 | langchain-openai, langchain-anthropic, langchain-google-genai, langchain-groq |
| 벡터 검색 | FAISS (CPU), sentence-transformers |
| 문서 처리 | PyMuPDF, pymupdf4llm, BeautifulSoup4 |
| 텍스트 랭킹 | rank-bm25, ONNX Runtime |
| 시각화 | Cytoscape.js (프론트엔드 관계도) |
| DB | SQLite (세션 영속화) |

---

## 4. 개발 환경

### 4-1. 런타임 환경
- **Python**: 3.10.12
- **PyTorch**: CPU-only 빌드 (배포 환경 경량화, ~200MB)
- **ONNX Runtime**: CPU 추론 가속 (GPU enumeration 비활성화)

### 4-2. GPU 활용 내역
- **개발/테스트 시**: CUDA GPU 활용
  - SPECTER2 임베딩 + BGE Reranker v2-m3 CrossEncoder (full 모드)
  - EXAONE 3.5-7.8B 로컬 추론
- **배포 환경 (Render)**: CPU-only
  - MiniLM-L6-v2 + MiniLM CrossEncoder (light 모드)
  - ONNX Runtime CPU 가속
  - 자동 디바이스 감지: CUDA > MPS > CPU

### 4-3. 배포 구성
- **플랫폼**: Render (Web Service)
- **서버**: Uvicorn 단일 워커 (workers=1)
- **모델 티어**: light (CPU 최적화)
- **모니터링**: LangSmith 트레이싱 (GAPAGO 프로젝트)

### 4-4. 주요 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| FULLTEXT_TARGET_COUNT | 30 | 전문 다운로드 목표 논문 수 |
| BM25_TOP_K | 50 | BM25 초벌 필터링 상위 N |
| RERANKER_TOP_K | 15 | CrossEncoder 최종 선별 수 |
| MAX_ITERATIONS | 2 | 질문 정제 최대 반복 |
| MAX_CRITIC_LOOPS | 2 | 품질 게이트 최대 루프 |
| RERANK_MODELS | light / full / auto | 리랭킹 모델 티어 선택 |

---

## 5. 기술적 특징 요약

- **Multi-Agent 아키텍처**: LangGraph 기반 8단계 파이프라인 + 3개 품질 제어 루프
- **다중 소스 검색**: 8개 학술 DB/API 병렬 검색 + 3단계 리랭킹
- **전문 기반 분석**: 8단계 폴백 체인으로 최대한 full-text 확보
- **동적 축 생성**: 사전 정의 카테고리 없이 데이터에서 귀납적 연구 축 도출
- **Human-in-the-Loop**: 모호한 질문 시 사용자 확인 인터럽트
- **멀티 LLM 라우팅**: 에이전트별 최적 모델 자동 배정 (4개 프로파일)
- **실시간 스트리밍**: SSE 기반 파이프라인 진행 상황 실시간 전달
- **Fast Mode**: 경량 모델 + 분석 간소화로 속도 최적화
