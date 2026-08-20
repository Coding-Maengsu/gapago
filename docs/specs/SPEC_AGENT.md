# GAPAGO 에이전트(코드) 스펙

## 1. 개요

GAPAGO의 핵심은 LangGraph StateGraph 기반의 멀티 에이전트 파이프라인이다. 10개의 에이전트 노드가 순차적/조건부로 실행되며(+ 1개 오케스트레이터 에이전트, 1개 파이프라인 외부 대화 에이전트), 사용자의 연구 질문에서 시작하여 논문 검색 → 한계점 추출 → 갭 추론 → 품질 평가 → 최종 보고서를 생성한다.

**LLM 라우팅**: 모든 에이전트는 `get_llm_for_agent(state, agent_name)`을 사용하여 `ModelRouter`가 에이전트별 최적 LLM provider를 자동 배정한다 (`gapago/core/model_router.py`). `model_routing` state 필드가 없으면 기존 `llm_provider`로 fallback.

---

## 2. 파이프라인 흐름도

### 2.1 메인 파이프라인 (`gapago/graphs/graph.py`)

```
START
  │
  ▼
query_subgraph ──────────────────────────────┐
  │                                          │ (REFINE_QUERY 루프)
  ▼                                          │
meaning_expand ◄─────────────────────────────┤
  │                                          │ (REDO_RETRIEVAL 루프)
  ▼                                          │
paper_retrieval                              │
  │                                          │
  ▼                                          │
limitation_extract ◄──────┐                  │
  │                       │ (RETRY 루프)     │
  ▼                       │                  │
limitation_eval ──────────┘                  │
  │ (PASS)                                   │
  ▼                                          │
recency_check                                │
  │                                          │
  ▼                                          │
gap_infer                                    │
  │                                          │
  ▼                                          │
critic_score ────────────────────────────────┘
  │ (ACCEPT)
  ▼
final_response
  │
  ▼
END
```

### 2.2 쿼리 서브그래프 (`gapago/graphs/query_subgraph.py`)

```
START
  │
  ▼
query_analysis
  │
  ├─ (needs_user_input = true) ──► [INTERRUPT] human_clarify ──► query_analysis (루프)
  │
  └─ (needs_user_input = false) ──► END
```

- `interrupt_before=["human_clarify"]`: 명확화 노드 진입 전 인터럽트
- 최대 반복: `max_iterations = 3`

### 2.3 오케스트레이터 파이프라인 (`gapago/graphs/orchestrator_graph.py`)

`GAPAGO_ORCHESTRATOR=1` 환경변수로 활성화. LLM 기반 동적 파이프라인 조율.

```
START
  │
  ▼
query_subgraph
  │
  ▼
orchestrator ◄──────────────────────────┐
  │                                     │
  ├─► meaning_expand ───────────────────┤
  ├─► paper_retrieval ──────────────────┤
  ├─► limitation_extract ───────────────┤
  ├─► limitation_eval (optional) ───────┤
  ├─► recency_check (optional) ─────────┤
  ├─► gap_infer ────────────────────────┤
  ├─► critic_score (optional) ──────────┤
  ├─► final_response ───────────────────┘
  │
  ▼
END
```

**필수 순서**: meaning_expand → paper_retrieval → limitation_extract → gap_infer → final_response
**선택적 에이전트**: limitation_eval, recency_check, critic_score (데이터 품질에 따라 LLM이 판단)
**Fast Mode**: `⚡ FAST MODE 활성화됨` 가이드라인을 프롬프트에 주입 — 강제 스킵이 아닌 LLM 유동 판단
**제한**: 최대 15 orchestrator 스텝, 에이전트당 최대 2회 재실행

### 2.4 조건부 라우팅

| 라우팅 함수 | 조건 | 대상 노드 |
|------------|------|----------|
| `route_after_eval` | `decision == "RETRY"` | `limitation_extract` |
| | `decision == "PASS"` | `recency_check` |
| `route_after_critic` | `DECISION: ACCEPT` | `final_response` |
| | `DECISION: REDO_RETRIEVAL` | `meaning_expand` |
| | `DECISION: REFINE_QUERY` | `query_subgraph` |
| | 태그 매칭 실패 | `final_response` (fallback) |
| `critic_score_node` 내부 | `critic_loop_count >= 2` | 강제 `ACCEPT` 생성 (노드 내부에서 판정) |

### 2.4 체크포인팅

- `MemorySaver()`: 인메모리 상태 영속화
- `thread_id` 기반 세션 관리 (Human-in-the-Loop 지원)

---

## 3. 에이전트 노드 상세

### 3.1 Query Analysis Node

**파일:** `gapago/agents/query_agent/query_analysis.py`

**역할:** 연구 질문의 검색 가능성 평가 (SemRank 프레임워크)

**분류 체계:**

| 분류 | 조건 | 후속 동작 |
|------|------|----------|
| `TOO_BROAD` | 일반적 토픽, 구체적 키워드 부족 | `breadth_candidates` (3개 옵션) → 인터럽트 |
| `SEARCHABLE` | 도메인 + 태스크 + 방법론 명확 | `refined_query` + `keywords` 생성 |
| `TOO_NARROW` | 지나치게 구체적, 결과 부족 우려 | `expansion_suggestion` 제안 → 인터럽트 |

**학술적 기반:**
- **SemRank** (Zhang et al., EMNLP 2025): `general_topic` vs `specific_phrases` 추출
- **CoQuest** (Liu et al., CHI 2024): 폭 우선(breadth-first) 후보 제시

**구현:**
- `llm.with_structured_output(QueryResult)` 사용
- SemRank 기반 질적 판정 규칙: `general_topic` vs `specific_phrases` 존재 여부로 분류
- `specific_phrases` 없음 → `TOO_BROAD`, 1개+ → `SEARCHABLE`, 조합이 너무 희귀 → `TOO_NARROW`

**출력 상태 업데이트:**
```python
{
    "scope_level": "SEARCHABLE",
    "refined_query": "정제된 쿼리",
    "keywords": ["kw1", "kw2", "kw3"],
    "negative_keywords": ["neg1"],
    "needs_user_input": False
}
```

---

### 3.2 Human Clarify Node

**파일:** `gapago/agents/query_agent/query_analysis.py`

**역할:** 쿼리 명확화를 위한 인터럽트 포인트

**동작:**
1. `interrupt_before` 설정으로 노드 진입 전 파이프라인 중단
2. 사용자 응답을 상태에 주입 (`app.update_state()`)
3. `needs_user_input = False`로 설정하여 루프 탈출 준비
4. `query_analysis` 노드로 재진입

**명확화 유형:**
- `TOO_BROAD`: 3개의 구체적 방향 중 선택 (breadth_candidates)
- `TOO_NARROW`: 확장 제안 수용 여부 결정

---

### 3.3 Meaning Expand Node

**파일:** `gapago/agents/meaning_expand_agent.py`

**역할:** 키워드 확장 및 검색 전략 수립 (도구 호출 없음 — 준비 단계)

**출력:**

| 필드 | 최대 수 | 설명 |
|------|---------|------|
| `expanded_terms` | 12 | 동의어, 약어, 변형 |
| `arxiv_query_candidates` | 4 | arXiv 검색용 쿼리 |
| `web_query_candidates` | 4 | 웹 검색용 쿼리 |
| `scienceon_query_candidates` | 3 | 한국 학술 DB 검색용 쿼리 |

---

### 3.4 Paper Retrieval Node

**파일:** `gapago/agents/retrieval_agent.py`

**역할:** 멀티소스 병렬 논문 검색 + 다단계 랭킹

**아키텍처:** LLM ReAct 에이전트가 아닌 **직접 병렬 검색** (`_parallel_search` + `ThreadPoolExecutor`). `async def paper_retrieval_node`이 `run_in_executor`로 CPU-bound 작업을 offload.

**검색 소스 (8개, `_parallel_search`에서 동시 호출):**

| 함수 | 소스 | 기본 결과 수 | 특징 |
|------|------|------------|------|
| `arxiv_api_call` | arXiv | 100 | XML/Atom 파싱, `threading.Lock` 직렬화, 5초 간격, 지수 백오프 |
| `crossref_search` | Crossref | 60 | 1.5억+ 메타데이터, PDF URL 추출, venue 포함 |
| `semantic_scholar_search` | Semantic Scholar | 50 | 2억+ 논문, citation 데이터 |
| `openalex_search` | OpenAlex | 40 | 2억+ 저작물, inverted index abstract 재구성 |
| `scienceon_search` | ScienceON | 15 | 한국 학술 DB, AES 암호화 인증 |
| `scienceon_patent_search` | ScienceON | 10 | 특허 (IPC 분류) |
| `scienceon_report_search` | ScienceON | 10 | 국가 R&D 보고서 |
| `TavilySearch.search` | Tavily | 5 | 웹 검색 (트렌드용, `web_results`로 분리) |

**랭킹 파이프라인 (6단계):**

1. **병렬 검색** (`_parallel_search`): 8개 소스 동시 호출 (`ThreadPoolExecutor(max_workers=len(tasks))`)
2. **중복 제거** (`_dedupe_papers`): DOI + title + year 기반 → `total_candidates_count` 기록
3. **연도 필터**: `year_range` → YYYY-YYYY 형식 변환 (`_resolve_year_range`)
4. **1단계: BM25 + FAISS 병렬** → union → 중복 제거
   - BM25 동적 k: `max(10, min(sum(scores > threshold), bm25_top_k))` (기본 50)
   - FAISS (`_faiss_filter`): SPECTER2/MiniLM 임베딩 + 코사인 유사도 (ONNX 지원)
   - 두 결과의 합집합으로 후보 풀 확대
5. **Full text 접근 가능 필터** (`_filter_fulltext_available`): 메타데이터 기반 신뢰도 등급
   - `guaranteed` (3): arXiv (ar5iv HTML 거의 100%)
   - `likely` (2): OA PDF URL 확보됨
   - `maybe` (1): DOI만 있음 (출판사 차단 가능성)
   - 신뢰도 내림차순 정렬, 동일 등급 내 BM25 순서 유지
6. **2단계: CrossEncoder reranking** (`_cross_encoder_rerank`) → 실패 시 LLM Reranker fallback (`_llm_rerank`)
   - 모델 tier: `auto` (GPU→full `BGE-reranker-v2-m3`, CPU→light `ms-marco-MiniLM`)
   - `reranker_top_k` (기본 15) 최종 선별

**Embedding/CrossEncoder 모델 (lazy load, ONNX 지원):**

| 용도 | Light (CPU) | Full (GPU) |
|------|-------------|------------|
| Embedding (FAISS) | `all-MiniLM-L6-v2` | `allenai/specter2_base` |
| CrossEncoder | `ms-marco-MiniLM-L-6-v2` | `BAAI/bge-reranker-v2-m3` |

- `RERANK_MODELS` 환경변수 또는 `config.rerank_models`로 tier 선택 (`auto`/`light`/`full`)
- `preload_models()`: 서버 시작 시 사전 로딩

**venue 정규화:**
- 결정 우선순위: 명시적 venue > journal > source 라벨
- source 라벨 매핑: `arxiv→"arXiv preprint"`, `crossref→저널명`, `semantic_scholar→"Semantic Scholar"`, `openalex→"OpenAlex"`, `scienceon→"ScienceON"`, `scienceon_patent→"Patent"`, `scienceon_report→"R&D Report"`

**진행률 보고:**
- `report_progress(session_id, "paper_retrieval", ...)` 호출

**출력:**
```python
{
    "papers": [정규화된 논문 목록],
    "total_candidates_count": 130,   # 중복 제거 후 전체 후보 수
    "web_results": [웹 검색 결과 (별도)]
}
```

---

### 3.5 Limitation Extract Node

**파일:** `gapago/agents/limitation_agent.py`

**역할:** 2-트랙 한계점 추출

#### Track 1: 저자 명시 한계점 (Author-Stated)
- **검색 섹션**: Conclusion, Limitations, Future Work
- 저자가 직접 인정한 한계점 추출

#### Track 2: 구조적 한계점 (Structural)
- **검색 섹션**: Introduction, Method, Experiment, Discussion
- 숨겨진 제약 조건: 가정, 데이터셋 범위, 누락된 베이스라인

**전체 텍스트 로딩 (소스별):**

| 소스 | 로딩 방식 |
|------|----------|
| arXiv | ar5iv HTML 우선 → ArxivLoader PDF 폴백 (`_wait_arxiv_rate_limit`: 스레드 안전 3초 간격) |
| ScienceON | DOI 기반 PDF 또는 ContentURL 직접 접근 |
| 특허/보고서 | ContentURL/FulltextURL (HTML → PDF 캐스케이드) |
| Semantic Scholar / OpenAlex | DOI 기반 PDF |

> **Full-text 전용**: abstract fallback은 제거됨. Full text 섹션을 확보하지 못한 논문은 추출을 스킵한다.

**섹션 추출:**
- 정규식 기반 헤딩 매칭 (예: "Method", "Methodology", "Approach")
- `MAX_SECTION_CHARS = 3000` (섹션당 토큰 비용 제어)

**언어 설정:**
- `get_language_instruction(output_language)`로 생성된 `lang_instruction`을 모든 LLM 시스템 프롬프트에 주입
- `SYSTEM_PROMPT + lang_instruction` 형태로 3곳 (개별 호출, 콘텐츠 필터 fallback, 배치 호출)에 적용

**진행률 보고:**
- `report_progress(session_id, "limitation_extract", ...)` 호출
- Full text 로드 완료 후, 배치 처리 진행 시 SSE progress 이벤트 발행

**Full text 품질 검증:**
- 섹션 합산 문자 수가 `_MIN_FULLTEXT_CHARS = 500`자 미만이면 파싱 실패로 간주 (쓰레기 데이터 필터링)
- 실패 처리된 논문은 `sections = {}`, `source_tag = "none"`으로 리셋
- 기존 "Step 1.5: 백업 논문 대체" 로직이 자동 발동 — arXiv 논문 우선으로 대체 후보 선정 및 full text 재로드

**배치 처리:**
1. 병렬 전체 텍스트 로딩 (`ThreadPoolExecutor`, `min(8, len(papers))` 워커)
2. Full text 품질 검증 (500자 미만 필터링 → 백업 논문 자동 교체)
3. 배치 LLM 호출 (3 논문/배치, 2 워커)
4. 배치 실패 시 개별 논문 재시도

**교차 검증** (`_verify_limitations()`):
- `model_routing.profile`이 `optimized`, `quality`, `speed`일 때만 실행 (`balanced`에서는 스킵)
- 논문별로 limitation을 그룹핑 → full text sections와 함께 검증 LLM에 전달
- 검증 항목: evidence_quote 원문 존재 여부 (FOUND/NOT_FOUND) + claim 도출 가능성 (VALID/INVALID)
- 결과: `verified: true/false` 플래그 추가 (제거하지 않고 하류 에이전트가 판단)
- 검증 provider: `limitation_verify` 에이전트 이름으로 라우팅

**중복 제거:**
- Jaccard 유사도 (토큰화된 claim) — 임계값 0.55
- 더 긴 evidence_quote 우선 (더 상세)

**출력:**
```python
{
    "limitations": [
        {
            "paper_id": "arxiv:xxx",
            "claim": "한계점 설명",
            "evidence_quote": "원문 인용",
            "track": "author_stated|structural",
            "source_section": "Conclusion"
        }
    ]
}
```

---

### 3.6 Limitation Eval Node

**파일:** `gapago/agents/limitation_eval_agent.py`

**역할:** 이중 호출 한계점 품질 평가

#### Call 1: 개별 한계점 품질 (원자적 + 루브릭)

**FActScore 기반:**
- claim을 원자적 사실(atomic facts)로 분해
- 사실별 판정: `SUPPORTED` / `NOT_SUPPORTED` / `IRRELEVANT`
- `fact_score = supported_count / total_count`

**Prometheus 기반:** 3차원 점수 (1-5 척도)

| 차원 | 평가 기준 |
|------|----------|
| Groundedness | 증거가 claim을 뒷받침하는가? |
| Specificity | 정량적 세부 수준은? |
| Relevance | 연구 질문과의 연결성은? |

#### Call 2: 세트 수준 판단 (총체적 + 유형 분류)

**LimAgents 기반:**
- 품질 판단: `strong` / `weak` / `remove`

**Xu et al. 분류 체계:** 6가지 유형
- `methodology`, `data`, `scope`, `evaluation`, `theoretical`, `resource`

**커버리지 분석:**
- 유형 분포 + 다양성 점수 (1-5)

**PASS/RETRY 판정 기준:**

| 조건 | 판정 |
|------|------|
| 모든 기준 충족 | `PASS` |
| >50% weak | `RETRY` |
| avg_groundedness < 3.0 | `RETRY` |
| avg_fact_score < 0.6 | `RETRY` |
| diversity_score ≤ 2 | `RETRY` |

**후처리:**
- groundedness < 2 또는 fact_score < 0.4인 한계점 필터링
- 살아남은 한계점에 평가 메타데이터 첨부
- `MAX_EVAL_RETRIES = 1`

---

### 3.7 Recency Check Node

**파일:** `gapago/agents/recency_agent.py`

**역할:** 한계점의 최근 해결 여부 웹 검증

**도메인별 검색 소스 (`DOMAIN_SOURCES` 매핑):**

| 도메인 | 검색 대상 |
|--------|----------|
| `ai_cs` | paperswithcode, github, huggingface, medium, towardsdatascience |
| `biomedical` | pubmed, biorxiv, medrxiv, nature, sciencedirect |
| `materials_chemistry` | nature, sciencedirect, acs, rsc, materialsproject |
| `physics` | nature, phys.org, sciencedirect, aps, iop |
| `general` | 특정 소스 없음 |

**워크플로우:**
1. LLM: 쿼리 + 한계점 → 도메인 분류 + 검색 쿼리 (3-5개) 생성
2. Tavily 검색 (도메인 필터, 쿼리당 3 결과)
3. LLM: 웹 결과 vs 한계점 교차 참조
4. 상태 할당:

| 상태 | 의미 | 가중치 |
|------|------|--------|
| `unresolved` | 최근 해결 증거 없음 | 1.0 |
| `partial` | 부분적 해결 | 0.5 |
| `resolved` | 명확한 해결 증거 있음 | 0.0 |

- **보수적 점수**: 명확한 증거가 있을 때만 `resolved`

---

### 3.8 Gap Infer Node

**파일:** `gapago/agents/gap_agent.py`

**역할:** 4단계 연구 갭 추론 파이프라인 (완전 동적 축 — 고정 축 없음)

**핵심 설계 원칙:**
1. limitation의 단순 반전(반사적 반전) 금지
2. recency_status 활용 → "아직 아무도 안 푼" limitation만 GAP 후보로 승격
3. web_results 맥락 주입 → 창의적 방향 제안 시 최신 동향 반영
4. 사전 정의 카테고리 없이 LLM이 귀납적으로 도메인 특화 축 도출

**파라미터:**

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `GAP_AXES_DYNAMIC_MIN` | 3 | 최소 축 생성 수 |
| `GAP_AXES_DYNAMIC_MAX` | 7 | 최대 축 생성 수 |
| `GAP_AXES_DYNAMIC_MIN_PAPERS` | 2 | 축 후보 인정 최소 limitation 수 |

#### Step 1: 완전 동적 축 생성 (`_generate_all_axes`)

- **고정 축 완전 제거** — 사전 정의된 "Data", "Methodology" 등의 카테고리를 사용하지 않음
- LLM이 전체 limitations를 분석하여 연구 질문과 논문 도메인에 특화된 축을 자유롭게 생성
- 각 축은 limitations의 실제 패턴에서 귀납적으로 도출
- 상호 배타적 범위 (낮은 중복)

**좋은 축 예시** (도메인 특화):
  - "Clinical Note Heterogeneity" (not "Data")
  - "Discharge-to-Admission Temporal Lag" (not "Generalizability")
  - "ICD Coding Ambiguity Handling" (not "Evaluation")

**축 생성 출력 구조:**
```json
{
  "name": "snake_case_key",
  "label": "Human-readable Label (3-6 words)",
  "description": "어떤 limitation이 이 축에 속하는지 한 문장 설명",
  "rationale": "이 축이 연구 질문에 중요한 이유",
  "type": "dynamic"
}
```

#### Step 1 Fallback: 동적 축 실패 시 재시도 (`_generate_fallback_axes`)

- 동적 축 생성이 실패하거나 결과가 2개 미만이면 단순 프롬프트로 재시도
- 3-5개 thematic cluster로 그룹핑
- 최종 fallback: `"general_limitation"` 단일 축

#### Step 2: 최종 축 딕셔너리 구성 (`_build_final_axes`)

- 동적 축 리스트를 `{name: info_dict}` 형태로 변환
- 모든 축에 `type: "dynamic"` 태깅

#### Step 3: 배치 분류 + recency 가중치 적용

**배치 분류** (`_classify_limitations_batch`):
- 각 한계점을 최종 축 중 하나로 분류 (배치 LLM, BATCH_SIZE=20)
- 분류 실패 시 첫 번째 축으로 fallback

**recency 가중치** (`_build_axis_groups_with_recency`):

| 상태 | 가중치 | 설명 |
|------|--------|------|
| `unresolved` | 1.0 | GAP 후보 승격 |
| `partial` | 0.5 | 부분 기여 |
| `resolved` | 0.0 | GAP 후보에서 제외 (카운트는 유지) |

- `axis_groups` 구축: `weighted_count`, `unresolved_lims`, `total_count`

#### Step 4a: 긴급도 점수 (`_score_axis_urgency`)

- LLM 점수 (0-10) 기준:
  1. 미해결 한계점 수 (recency-weighted)
  2. 연구 질문과의 직접적 관련성
  3. 연쇄 영향 (cascade impact — 다른 축에 미치는 영향)
  4. 현재 진전 가능성
- **최종 점수 = 60% LLM 긴급도 + 40% 정규화된 weighted_count**
- 모든 활성 축(weighted_count > 0)에 대해 계산

#### Step 4b: 장벽 분석 (`_analyze_barriers`)

- LLM: N편의 논문이 이 문제를 인정하면서도 왜 해결하지 못했는가?
- resolved된 limitation과 대조하여 분석

| 출력 | 설명 |
|------|------|
| `gap_statement` | 정확한 미해결 문제 (≤25단어) |
| `barriers` | 3가지 기술적/구조적 이유 |
| `barrier_type` | 분류 (`data_scarcity`, `benchmark_absence`, `computational_cost`, `evaluation_mismatch`, `methodological_gap`, `domain_shift`, `conflicting_objectives`, `other`) |
| `what_was_tried` | 이전 시도된 접근법 2-3가지 (반복 방지) |

#### Step 4c: 창의적 방향 생성 (`_generate_creative_directions`)

- LLM: 3가지 독립적 후보 방향 생성
- **반 단순 뒤집기(Anti-simple-reversal)**: 한계점을 단순히 뒤집는 것 금지
  - "small data" → "use more data" (금지)
  - 장벽을 우회하는 예상치 못한 각도 필요
- 이미 시도된 접근(`what_was_tried`) 재사용 금지
- 최신 웹 결과(`web_results`) 반영 — 최신 동향 활용하되 이미 다룬 것 제외
- cascade_impact 활용 — 다른 축에도 이점이 있는 방향 우선
- 최고 후보 선택: 신규성(novelty_score) + 실현 가능성 + 영향력
- **함수 파라미터**: `lang_instruction: str = ""`, `provider: str = None`

**후보별 출력:**
```json
{
  "direction_id": 1,
  "core_insight": "핵심 인사이트 한 문장",
  "proposed_topic": "논문 스타일 제목: 방법 + 데이터셋 + 베이스라인 + 목표",
  "methodology_hint": "구현 계획 2-3문장",
  "novelty_score": 8
}
```

**언어 설정:**
- `gap_infer_node`에서 `get_language_instruction(output_language)`로 `lang_instruction` 생성
- 모든 내부 함수(`_generate_all_axes`, `_classify_limitations_batch`, `_score_axis_urgency`, `_analyze_barriers`, `_generate_creative_directions`)에 `lang_instruction` 파라미터 전달

**진행률 보고:**
- `report_progress(session_id, "gap_infer", ...)` 호출
- 축별 처리 진행 시 SSE progress 이벤트 발행

**최종 출력 (긴급도순 정렬):**
```python
{
    "gaps": [
        {
            # GapCandidate 기본 필드
            "axis": "clinical_note_heterogeneity",
            "axis_label": "Clinical Note Heterogeneity",
            "axis_type": "dynamic",
            "gap_statement": "갭 설명 (≤25단어)",
            "elaboration": "핵심 인사이트 (1-2문장)",
            "proposed_topic": "제안 연구 방향 (논문 스타일 제목)",
            "repeat_count": 3,
            "supporting_papers": ["arxiv:xxx", "s2:yyy"],
            "supporting_quotes": ["인용1", "인용2"],
            # 확장 필드 (GapCandidate 스키마 외 병합)
            "detail": "리포트 상세 섹션용 풍부한 컨텍스트 (마크다운)",
            "barriers": ["장벽1", "장벽2", "장벽3"],
            "barrier_type": "benchmark_absence",
            "what_was_tried": ["이전 접근1", "이전 접근2"],
            "alt_topics": ["대안 방향1", "대안 방향2"],
            "novelty_score": 8,
            "urgency_score": 7.4,
            "urgency_rationale": "긴급도 근거 한 문장"
        }
    ]
}
```

---

### 3.9 Critic Score Node

**파일:** `gapago/agents/critic_agent.py`

**역할:** 품질 게이트키퍼 + 조건부 라우팅

**평가 차원 (각 0.0 - 1.0):**

| 차원 | 평가 기준 |
|------|----------|
| `query_specificity` | 정제된 쿼리가 충분히 구체적인가? |
| `paper_relevance` | 검색된 논문이 질문과 관련 있는가? |
| `groundedness` | 갭이 논문 증거에 근거하는가? |

**판정 규칙:**

| 조건 | 결정 |
|------|------|
| 세 차원 모두 ≥ 0.6 | `ACCEPT` → `final_response` |
| `paper_relevance` < 0.4 | `REDO_RETRIEVAL` → `meaning_expand` |
| `query_specificity` < 0.4 | `REFINE_QUERY` → `query_subgraph` |
| 판단 어려울 때 | `ACCEPT` 선호 |
| 재시도 라운드 | 더 관대 (명백히 불합격일 때만 거부) |

**루프 방지:** `MAX_CRITIC_LOOPS = 2` → 2회 재시도 후 강제 `ACCEPT`

---

### 3.10 Final Response Node

**파일:** `gapago/agents/response_agent.py`

**역할:** 구조화된 마크다운 최종 보고서 생성

**데이터 컨텍스트 구축:**
- `papers_data`: 검색된 논문 (ID, 제목, 연도, 저자)
- `limitation_extract_data`: 추출된 한계점 (논문별)
- `gap_infer_data`: 식별된 갭 (모든 메타데이터)

**보고서 형식:**
- 표준 마크다운 (유니코드 테이블 문자 미사용)
- 섹션: `## Related Papers`, `## Key Limitations`, `## Research Gaps & Proposed Topics`
- `FINAL ANSWER`로 종료

**구현 특징:**
- `create_agent` 사용
- **새로운 메시지**로 시작 (메시지 누적 없음 — 토큰 절약)

---

### 3.11 Gap Chat Agent (결과 검토 대화)

**파일:** `gapago/agents/gap_chat_agent.py`

**역할:** 파이프라인 완료 후 사용자와 대화형 결과 검토

**파이프라인 외부 에이전트** — 메인 StateGraph에 포함되지 않으며, `main.py`에서 파이프라인 완료 후 별도 호출.

**핵심 함수:**

| 함수 | 역할 |
|------|------|
| `detect_user_intent(user_input, num_gaps)` | LLM 기반 의도 분류 (`exit`, `help`, `show_gap_detail`, `question`) |
| `gap_chat_respond(state, user_question)` | GAP/limitation/papers 컨텍스트 기반 답변 생성 |
| `format_gap_details(gap)` | GAP 상세 정보 포맷팅 (축, 진술, 제안 주제, 지지 논문, 인용구) |
| `interactive_chat_loop(state)` | 대화형 루프 실행 (의도 파악 → 처리 반복) |

**대화 루프 흐름:**
```
사용자 입력 → LLM 의도 분류 →
  ├─ exit → 대화 종료
  ├─ help → 도움말 표시
  ├─ show_gap_detail → N번 GAP 상세 표시
  └─ question → LLM 답변 생성 (GAP/limitation 컨텍스트 활용)
```

**`main.py` 연동:**
- 파이프라인 완료 후 "결과에 대해 질문하시나요?" 프롬프트 표시
- LLM이 사용자 응답의 긍정/부정 의도 파악
- 긍정 시 `interactive_chat_loop(state_values)` 호출

---

## 4. 상태 관리 (AgentState)

**파일:** `gapago/core/states.py`

### 4.1 AgentState TypedDict 전체 구조

```python
class AgentState(TypedDict):
    # -0- 오케스트레이션
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sender: str
    errors: List[str]
    completed_stages: Annotated[List[str], _append_stages]  # 오케스트레이터 실행 이력
    agent_feedback: dict              # 에이전트 간 신호 (eval RETRY 등)
    orchestrator_plan: List[str]      # 오케스트레이터 결정 기록

    # -1- 쿼리 에이전트
    iteration: int
    max_iterations: int          # 기본 3
    scope_level: str             # TOO_BROAD / SEARCHABLE / TOO_NARROW
    scope_rationale: str
    breadth_candidates: List[dict]
    expansion_suggestion: str
    keywords: List[str]          # 2-5개
    negative_keywords: List[str] # 1-3개
    refined_query: str
    user_question: str
    needs_user_input: bool

    # -2- 검색 에이전트
    papers: List[dict]
    web_results: List[dict]
    research_domain: str         # auto/ai_cs/biomedical/materials_chemistry/physics/general
    llm_provider: str            # azure/claude/gemini/exaone
    year_range: str              # auto/1y/3y/5y
    output_language: str         # auto/ko/en
    session_id: str              # SSE 진행률 리포팅용 세션 ID
    fast_mode: bool              # True면 빠른 분석 (CrossEncoder 스킵, 상위 3개 축만)
    model_routing: dict          # {"default_provider": "azure", "profile": "optimized"}

    # -3- 한계점 에이전트
    limitations: List[dict]
    paper_extraction_status: List[dict]  # 논문별 full text 추출 상태 (status/fulltext_source/sections)

    # -3.5- 한계점 평가 에이전트
    limitation_eval: dict
    eval_warnings: List[str]
    eval_retry_count: int

    # -4- 갭 추론 에이전트
    gaps: List[dict]

    # -5- 비평 에이전트
    critic: Optional[dict]
    critic_loop_count: int

    # 트레이싱
    trace: dict
```

### 4.2 메시지 네이밍 규약

각 에이전트가 자신의 출력 메시지에 이름을 태깅:
```python
AIMessage(content=..., name="query_analysis")
AIMessage(content=..., name="meaning_expand")
AIMessage(content=..., name="paper_retrieval")
AIMessage(content=..., name="limitation_extract")
AIMessage(content=..., name="gap_infer")
AIMessage(content=..., name="critic_score")
AIMessage(content=..., name="final_response")
```

하위 에이전트가 이름으로 상위 메시지를 파싱하여 컨텍스트 추출 (폴백 메커니즘).

### 4.3 상태 통신 패턴

| 통신 방식 | 용도 | 예시 |
|---------|------|------|
| State 필드 | 구조화된 데이터 | `papers`, `limitations`, `gaps` |
| 메시지 히스토리 | 비구조적 컨텍스트 | `AIMessage(name="meaning_expand")` |
| 이름 기반 파싱 | 폴백 | 하위 에이전트가 상위 메시지 검색 |

---

## 5. Pydantic 데이터 모델

### 5.1 ScopeAssessment

```python
class ScopeAssessment(BaseModel):
    scope_level: Literal["TOO_BROAD", "SEARCHABLE", "TOO_NARROW"]
    general_topic: str                    # 넓은 분야 이름
    specific_phrases: List[str]           # 구체적 키워드
    rationale: str                        # AI Thoughts
    breadth_candidates: List[ScopeCandidate]  # TOO_BROAD 시 3개 옵션
    expansion_suggestion: str             # TOO_NARROW 시 확장 제안
```

### 5.2 QueryResult

```python
class QueryResult(BaseModel):
    scope_assessment: ScopeAssessment
    refined_query: str          # 정제된 학술 쿼리 (SEARCHABLE 시)
    keywords: List[str]         # 2-5개 arXiv 키워드
    negative_keywords: List[str] # 1-3개 제외 키워드
```

### 5.3 Paper

```python
class Paper(BaseModel):
    paper_id: str               # source:id 형식
    title: str
    abstract: str
    url: str
    year: int
    authors: List[str]
    score_bm25: float = 0.0     # BM25 랭킹 점수
    venue: str = ""             # 게재지/소스 (저널명, "arXiv preprint" 등)
    full_text_sections: dict    # 전체 텍스트 (있는 경우)
```

### 5.4 LimitationItem

```python
class LimitationItem(BaseModel):
    paper_id: str
    claim: str                  # 한계점 설명
    evidence_quote: str         # 원문 인용
    track: Literal["author_stated", "structural"]
    source_section: str         # Abstract, Conclusion 등
```

### 5.5 GapCandidate

```python
class GapCandidate(BaseModel):
    axis: str                   # 동적 축 키 (snake_case)
    axis_label: str = ""        # 표시 이름 (Human-readable)
    axis_type: str = "dynamic"  # 항상 "dynamic" (고정 축 제거됨)
    gap_statement: str          # ≤25단어 갭 설명
    elaboration: str = ""       # 핵심 인사이트 (1-2문장)
    proposed_topic: str = ""    # 제안 연구 방향 (논문 스타일 제목)
    repeat_count: int = 0       # 해당 축에 매핑된 limitation 수
    supporting_papers: List[str] # 뒷받침 논문 ID
    supporting_quotes: List[str] # 증거 인용
```

**확장 필드** (GapCandidate 스키마 외, `gap_dict.update()`로 병합):

| 필드 | 타입 | 설명 |
|------|------|------|
| `detail` | `str` | 리포트 상세 섹션용 마크다운 (core_insight + rationale + methodology_hint + alt_topics) |
| `barriers` | `List[str]` | 3가지 기술적/구조적 장벽 |
| `barrier_type` | `str` | 장벽 분류 (data_scarcity, benchmark_absence 등) |
| `what_was_tried` | `List[str]` | 이전 시도된 접근법 |
| `alt_topics` | `List[str]` | 채택되지 않은 대안 연구 방향 |
| `novelty_score` | `int` | 신규성 점수 (1-10) |
| `urgency_score` | `float` | 긴급도 최종 점수 (60% LLM + 40% weighted_count) |
| `urgency_rationale` | `str` | 긴급도 근거 |

### 5.6 DimensionScore

```python
class DimensionScore(BaseModel):
    dimension: str              # 평가 차원 키
    label: str                  # 표시 이름
    score: int = Field(0, ge=0, le=10)  # 0-10 점수
    reasoning: str = ""         # 점수 근거
```

### 5.7 EvaluationResult

```python
class EvaluationResult(BaseModel):
    dimension_scores: list[DimensionScore]  # 차원별 점수
    average_score: float = Field(0.0, ge=0.0, le=10.0)  # 평균
    summary: str = ""           # 전체 평가 요약
```

### 5.8 CriticScores

```python
class CriticScores(BaseModel):
    query_specificity: float    # 0.0 - 1.0
    paper_relevance: float      # 0.0 - 1.0
    groundedness: float         # 0.0 - 1.0
```

---

## 6. 프롬프트 엔지니어링

### 6.1 기본 시스템 프롬프트 (`gapago/core/prompts.py`)

`make_system_prompt(suffix)` 함수로 공통 기반 + 에이전트별 접미사 결합:

- 역할 기반 책임 범위
- 도구 사용 제약
- 구조화된 출력 요구사항
- 증거 기반 주장 원칙
- 멀티 에이전트 협업 프로토콜

### 6.2 언어 설정 (`get_language_instruction()`)

| 설정 | 동작 |
|------|------|
| `auto` | 사용자 입력 언어에 맞춤 |
| `ko` | 한국어로 응답 |
| `en` | 영어로 응답 |

기술 용어는 항상 영어 유지.

### 6.3 에이전트별 프롬프트 전략

| 에이전트 | 프롬프트 유형 | 핵심 혁신 |
|---------|-------------|----------|
| query_analysis | Structured Output (`QueryResult`) | SemRank + CoQuest |
| paper_retrieval | Agent prompt + tool instruction | 멀티소스 오케스트레이션 |
| limitation_extract | System + user (2-track) | Track 1 vs Track 2 이중 추출 |
| limitation_eval | Call1 + Call2 system prompts | FActScore + Prometheus + LimAgents + Xu et al. |
| recency_check | System + context prompt | 도메인별 Tavily 검색 |
| gap_infer | 단계별 인라인 프롬프트 | 4단계 완전 동적 축 + 긴급도 점수 |
| critic_score | System prompt only | 판정 규칙 매트릭스 |
| final_response | System + data context | 새 메시지 (축적 없음) |

---

## 7. 평가 프레임워크

**파일:** `evaluation/evaluate.py`

### 7.1 평가 메트릭 (5개, 가중치 합계 1.0)

| 메트릭 | 가중치 | 측정 방법 |
|--------|--------|----------|
| **Groundedness** | 0.30 | `supporting_quotes` + `repeat_count` 기반 (1.0/0.6/0.3/0.0) |
| **Novelty** | 0.25 | LLM이 기존 연구 여부 판단 (0-1 척도) |
| **Specificity** | 0.20 | 방법 + 데이터셋 + 목표 정량화 존재 여부 |
| **Relevance** | 0.15 | 시맨틱 유사도 (SentenceTransformer 또는 TF-IDF) |
| **Diversity** | 0.10 | 코사인 유사도 쌍별 비교 |

### 7.2 베이스라인 비교

| 비교 대상 | 특징 |
|---------|------|
| **GAPAGO** | 실제 논문 기반 증거 근거 갭 |
| **Baseline LLM** | 제로샷 갭 제안 (논문 검색 없음) |

- 다중 프로바이더 비교 (azure, claude, gemini)
- 자동 베이스라인 생성
- GAP별 상세 점수
- 출력: JSON (점수) + Markdown (보고서)

---

## 8. 학술적 기반

코드에서 참조하는 연구 논문:

| 논문 | 적용 에이전트 | 핵심 기여 |
|------|-------------|----------|
| **SemRank** (Zhang et al., EMNLP 2025) | query_analysis | 쿼리 범위 평가 (broad/specific/narrow) |
| **CoQuest** (Liu et al., CHI 2024) | query_analysis | 인간-AI 공동 생성 (폭 우선 탐색) |
| **APA** (Kim et al., EMNLP 2024) | ~~query_refinement~~ | 인지된 모호성과의 정렬. 해당 노드는 그래프에 연결되지 않아 제거됨 (git 이력 참조) |
| **FActScore** | limitation_eval | 원자적 사실 검증 |
| **Prometheus** | limitation_eval | 루브릭 기반 평가 점수 |
| **LimAgents** | limitation_eval | 항목별 품질 판단 |
| **Xu et al.** | limitation_eval | 한계점 유형 분류 체계 (6 카테고리) |

---

## 9. 에이전트 유형 분류

| 유형 | 에이전트 수 | 역할 | 워크플로우 |
|------|-----------|------|----------|
| **쿼리 에이전트** | 2 | Human-in-loop 질문 정제 | 분석 → 명확화 루프 → 정제 |
| **검색 에이전트** | 1 | 멀티소스 논문 수집 | 의미 확장 → 도구 오케스트레이션 → BM25(동적k) → Full text 필터 → LLM 리랭크 |
| **추출 에이전트** | 2 | 비구조 → 구조 변환 | 전체 텍스트 로딩 → 배치 LLM → 중복 제거; 원자적 검증 + 총체적 판단 |
| **인텔리전스 에이전트** | 2 | 도메인 인지 신호 생성 | 동적 축 기반 한계점 분류; 최신성 웹 검증 |
| **합성 에이전트** | 2 | 창의적 지식 결합 | 완전 동적 축 추론 + 장벽 분석 + 방향 생성; 품질 평가 + 라우팅 |
| **생성 에이전트** | 1 | 사람이 읽을 수 있는 출력 | 데이터 컨텍스트 조립 + 마크다운 보고서 |
| **대화 에이전트** | 1 | 결과 검토 대화 | LLM 의도 분류 → GAP 상세 조회 / 자유 질문 답변 |

---

## 10. 파일 구조

```
agents/
├── __init__.py                      # 모든 노드 export
├── orchestrator_agent.py            # LLM 기반 동적 파이프라인 조율 (GAPAGO_ORCHESTRATOR=1)
├── query_agent/
│   └── query_analysis.py            # SemRank + CoQuest 기반 쿼리 분석
├── meaning_expand_agent.py          # 키워드 확장 (도구 없음)
├── retrieval_agent.py               # 멀티소스 논문 검색 오케스트레이터
├── limitation_agent.py              # 2-트랙 한계점 추출
├── limitation_eval_agent.py         # 이중 호출 한계점 평가
├── recency_agent.py                 # 최신성 웹 검증
├── gap_agent.py                     # 4단계 완전 동적 축 갭 추론
├── gap_chat_agent.py                # 결과 검토 대화 에이전트
├── critic_agent.py                  # 품질 게이트키퍼
└── response_agent.py                # 최종 보고서 생성

graphs/
├── graph.py                         # 메인 StateGraph + 라우팅 (GAPAGO_ORCHESTRATOR 조건부 분기)
├── orchestrator_graph.py            # 오케스트레이터 기반 동적 그래프
└── query_subgraph.py                # 쿼리 서브그래프 (인터럽트)

core/                                # 에이전트 · 그래프 · API 공용 모듈
├── config.py                        # 환경변수 기반 설정
├── llm.py                           # Provider 추상화
├── model_router.py                  # ModelRouter — 에이전트별 LLM 프로바이더 자동 라우팅
├── states.py                        # AgentState + Pydantic 모델
├── prompts.py                       # BASE_SYSTEM_PROMPT + 언어 설정
└── tools/                           # 검색 소스별 모듈 (arxiv · crossref · s2 · openalex · scienceon)

utils/
├── parse_json.py                    # 로버스트 JSON 추출
├── progress.py                      # 스레드 안전 진행률 큐 (SSE 중간 업데이트)
├── tavily.py                        # Tavily API 래퍼
├── logging.py                       # LangSmith 트레이싱
├── session_store.py                 # SQLite 세션 영속화 (서버 재시작 복구)
└── cancel.py                        # 파이프라인 취소 레지스트리

evaluation/evaluate.py                  # 베이스라인 비교 + 메트릭 점수
```
