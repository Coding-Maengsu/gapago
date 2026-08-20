# GAPAGO 에이전트 자율성 분석 및 개선안 설계 리포트

> 작성일: 2026-04-01
> 대상 코드: `main` branch (commit `a78438b`)
> 목적: 품질을 유지하면서 에이전트 자율성을 높이는 방안 설계

---

## 1. 현재 아키텍처 진단

### 1.1 파이프라인 흐름도

```
START
  │
  ▼
┌──────────────┐
│ query_subgraph│ ◄───────────────────────────────────┐
│  ├ query_analysis_node                              │
│  └ human_clarify (interrupt_before)                 │
└──────┬───────┘                                      │
       ▼                                              │
┌──────────────┐                                      │
│meaning_expand│ ◄────────────────────────┐           │
└──────┬───────┘                          │           │
       ▼                                  │           │
┌──────────────┐                          │           │
│paper_retrieval│                         │           │
└──────┬───────┘                          │           │
       ▼                                  │           │
┌──────────────────┐                      │           │
│limitation_extract│ ◄────────┐           │           │
└──────┬───────────┘          │           │           │
       ▼                      │           │           │
┌──────────────┐    RETRY     │           │           │
│limitation_eval├─────────────┘           │           │
└──────┬───────┘                          │           │
       │ PASS                             │           │
       ▼                                  │           │
┌──────────────┐                          │           │
│recency_check │                          │           │
└──────┬───────┘                          │           │
       ▼                                  │           │
┌──────────────┐                          │           │
│  gap_infer   │                          │           │
└──────┬───────┘                          │           │
       ▼                                  │           │
┌──────────────┐  REDO_RETRIEVAL          │           │
│ critic_score ├──────────────────────────┘           │
│              ├──────────────────────────────────────┘
│              │  REFINE_QUERY
└──────┬───────┘
       │ ACCEPT
       ▼
┌──────────────┐
│final_response│ (ReAct agent)
└──────┬───────┘
       ▼
      END
```

### 1.2 에이전트별 자율성 수준 평가

자율성 5단계 기준:
- **L1 (단순 실행)**: 고정 입출력, 분기 없음
- **L2 (조건 분기)**: 하드코딩된 규칙으로 경로 결정
- **L3 (파라미터 적응)**: LLM이 일부 파라미터를 동적 결정
- **L4 (전략 선택)**: LLM이 여러 전략 중 하나를 선택
- **L5 (자율 계획)**: 목표만 주어지면 스스로 단계를 설계

| 에이전트 | 현재 수준 | 근거 |
|----------|-----------|------|
| query_analysis | L3 | `scope_level` 판정은 LLM이 하지만, 분기(TOO_BROAD→후보 제시, SEARCHABLE→진행)는 하드코딩 |
| meaning_expand | L1 | 고정 프롬프트로 키워드 확장. critic 피드백 수신 경로 없음 |
| paper_retrieval | L2 | 8개 검색 도구 고정 호출, BM25→FAISS→CrossEncoder 순서 고정. 도메인별 전략 미분화 |
| limitation_extract | L1 | 논문별 동일 프롬프트, `retry_guidance` 미소비 |
| limitation_eval | L2 | Call1→Call2→후처리 고정, PASS/RETRY 임계값 하드코딩 |
| recency_check | L3 | LLM이 도메인 판단 + 검색 쿼리 생성. 도메인별 소스 매핑은 하드코딩 |
| gap_infer | **L4** | 동적 축 생성, 긴급도 점수화, 장벽 분석, 창의적 방향 제안 모두 LLM 주도. 추론 모델 별도 라우팅 |
| critic_score | L2 | 점수→결정 매핑 고정, ACCEPT 편향 fallback |
| final_response | **L4** | ReAct 패턴으로 도구 호출 자율 결정 |

### 1.3 "멀티 에이전트"가 아닌 이유

진정한 멀티 에이전트 시스템의 기준 (Wooldridge & Jennings, 1995; Park et al., 2023):

| 기준 | GAPAGO 현재 상태 | 충족 여부 |
|------|------------------|-----------|
| **자율적 목표 추구** | 각 노드는 호출되면 실행, 스스로 실행 시점 결정 불가 | ❌ |
| **에이전트 간 통신** | state dict를 통한 단방향 데이터 전달만 존재. 에이전트 간 직접 메시지 교환 없음 | ❌ |
| **동적 태스크 할당** | 그래프 구조가 컴파일 시점에 확정. 런타임에 노드 추가/제거 불가 | ❌ |
| **협상/투표** | critic이 단독 판정. 다수 에이전트의 합의 메커니즘 없음 | ❌ |
| **반응적 적응** | critic→meaning_expand 루프 존재하나, 피드백 내용이 meaning_expand에 전달되지 않음 | ⚠️ 부분 |

**결론**: 현재 GAPAGO는 **조건부 분기가 있는 순차 파이프라인**이다. 에이전트라 부를 수 있는 것은 `gap_infer`(L4, 동적 축 생성)와 `final_response`(L4, ReAct) 뿐이다.

---

## 2. 에이전트별 현황 분석

### 2.1 State 읽기/쓰기 맵

| 에이전트 | 읽는 state 필드 | 쓰는 state 필드 |
|----------|-----------------|-----------------|
| query_analysis | `messages`, `iteration`, `max_iterations` | `refined_query`, `keywords`, `negative_keywords`, `scope_level`, `scope_rationale`, `breadth_candidates`, `needs_user_input` |
| meaning_expand | `refined_query`, `keywords`, `negative_keywords`, `trace`, `llm_provider` | `messages` (payload), `trace`, `refined_query`, `keywords`, `negative_keywords` |
| paper_retrieval | `refined_query`, `user_question`, `messages`, `year_range`, `llm_provider`, `fast_mode`, `session_id` | `papers`, `total_candidates_count`, `backup_papers`, `web_results`, `research_domain` |
| limitation_extract | `papers`, `llm_provider`, `session_id`, `output_language` | `limitations`, `paper_extraction_status`, `errors` |
| limitation_eval | `limitations`, `refined_query`, `eval_retry_count`, `llm_provider`, `fast_mode` | `limitations` (필터링), `limitation_eval`, `eval_warnings`, `eval_retry_count` |
| recency_check | `limitations`, `refined_query`, `web_results`, `research_domain`, `llm_provider` | `limitations` (recency 필드 부착) |
| gap_infer | `limitations`, `refined_query`, `web_results`, `messages`, `session_id`, `output_language` | `gaps` |
| critic_score | `messages`, `limitations`, `gaps`, `critic_loop_count`, `llm_provider` | `messages`, `critic_loop_count` |
| final_response | `messages`, `papers`, `limitations`, `gaps` | `messages` |

### 2.2 LLM 판단 vs 하드코딩 비율

| 에이전트 | LLM 판단 | 하드코딩 | 비율 |
|----------|----------|----------|------|
| query_analysis | scope 판정, 키워드 생성, 후보 제시 | 분기 라우팅, 반복 제한 | 70:30 |
| meaning_expand | 키워드 확장, 쿼리 후보 생성 | 후보 수 제한 (4/4/3), fallback 로직 | 60:40 |
| paper_retrieval | — | 검색 도구 선택, BM25/FAISS/CE 파이프라인, 파라미터 | 0:100 |
| limitation_extract | 한계점 추출 | 프롬프트 고정, 섹션 키워드, full text 전략 | 50:50 |
| limitation_eval | Call1 채점, Call2 판정 | 임계값 (fact_score<0.4, groundedness<2, weak>50%), 배치 크기, 최대 재시도 | 40:60 |
| recency_check | 도메인 판단, 검색 쿼리 생성, recency 판정 | 도메인-소스 매핑 고정, 판정 기준 | 50:50 |
| gap_infer | 축 생성, 분류, 긴급도, 장벽 분석, 방향 제안 | recency 가중치, 최소/최대 축 수, 배치 크기 | **80:20** |
| critic_score | 점수 채점 | ACCEPT/REDO 임계값 (0.6/0.4), 최대 루프 (2), fallback ACCEPT | 40:60 |
| final_response | 보고서 작성, 도구 호출 결정 | 프롬프트 템플릿 고정 | 70:30 |

### 2.3 이미 계산되지만 활용되지 않는 데이터

| 데이터 | 생성 위치 | 미활용 위치 | 개선 가치 |
|--------|-----------|-------------|-----------|
| `retry_guidance` | `limitation_eval_agent.py:518` — Call2에서 RETRY 시 구체적 재추출 지침 생성 | `limitation_agent.py` — 재시도 시 동일 프롬프트 사용, guidance 참조 없음 | **높음** |
| `eval_improvement_hint` | `limitation_eval_agent.py:363` — weak 항목별 개선 힌트 | `gap_agent.py` — limitation의 eval 메타데이터 무시 | 중간 |
| `critic_score` 점수 상세 | `critic_agent.py:101` — query_specificity, paper_relevance, groundedness | `meaning_expand` — 재시도 시 어떤 점수가 낮았는지 전달 안됨 | **높음** |
| `research_domain` | `retrieval_agent.py` — 검색 시 자동 감지 | `limitation_extract` — 도메인별 프롬프트 분화 없음 | 낮음 |
| `web_results` | `retrieval_agent.py` — 웹 검색 결과 | `critic_score` — 최신 동향 대비 gap 검증에 미활용 | 낮음 |
| `eval_limitation_type` | `limitation_eval_agent.py:362` — methodology/data/scope 등 분류 | `gap_agent.py` — 자체 축 생성 사용, eval 분류 무시 | 낮음 (gap_agent가 자체 분류) |

---

## 3. 개선안 4가지 (우선순위순)

| # | 개선안 | 영향도 | 난이도 | 품질 리스크 |
|---|--------|--------|--------|------------|
| 1 | 품질 기반 적응형 재시도 | 높음 | 낮음 | 매우 낮음 |
| 2 | eval→limitation 피드백 전달 | 중간 | 낮음 | 낮음 |
| 3 | critic→에이전트 피드백 채널 | 높음 | 중간 | 낮음 |
| 4 | 검색 전략 적응형 선택 | 중간 | 중간 | 중간 |

---

### 3.1 개선안 #1: 품질 기반 적응형 재시도 (Quality-Adaptive Retry)

#### 문제 정의

현재 `critic_score`는 REDO_RETRIEVAL/REFINE_QUERY 결정만 내리고, 재시도 시 이전과 동일한 파이프라인을 반복한다. 어떤 품질 지표가 낮았는지에 따라 재시도 전략을 바꿀 수 있음에도 활용하지 않는다.

**현재 코드 위치:**
- `gapago/agents/critic_agent.py:46-118` — 점수 채점 + DECISION 출력
- `gapago/graphs/graph.py:47-71` — `route_after_critic()` 문자열 매칭으로 라우팅
- `gapago/agents/meaning_expand_agent.py:74-170` — `meaning_expand_node()` critic 피드백 미참조

#### 해결 방안

```python
# === states.py 추가 필드 ===
class AgentState(TypedDict):
    # ... 기존 필드 ...
    critic_feedback: dict  # {"low_dimension": str, "score": float, "suggestion": str}

# === critic_agent.py 수정 (L114 부근) ===
# 점수 파싱 후 가장 낮은 차원 식별
scores = {
    "query_specificity": parsed_query_specificity,
    "paper_relevance": parsed_paper_relevance,
    "groundedness": parsed_groundedness,
}
lowest_dim = min(scores, key=scores.get)
feedback = {
    "low_dimension": lowest_dim,
    "score": scores[lowest_dim],
    "suggestion": _generate_improvement_hint(lowest_dim, result_content),
    "loop": loop_count,
}
return {
    # ... 기존 반환 ...
    "critic_feedback": feedback,
}

# === meaning_expand_agent.py 수정 (L89 부근) ===
# critic_feedback가 있으면 프롬프트에 반영
critic_fb = state.get("critic_feedback", {})
retry_hint = ""
if critic_fb and critic_fb.get("low_dimension") == "paper_relevance":
    retry_hint = (
        "\n\nPREVIOUS ATTEMPT FEEDBACK: Paper relevance was low. "
        "Focus on more specific, narrower search terms. "
        "Add domain-specific jargon and method names."
    )
elif critic_fb and critic_fb.get("low_dimension") == "query_specificity":
    retry_hint = (
        "\n\nPREVIOUS ATTEMPT FEEDBACK: Query was too broad. "
        "Use more specific compound terms and exact method names."
    )
```

#### 품질 보호 장치

- **Fallback**: `critic_feedback`가 없거나 파싱 실패 시 기존 동작과 동일 (무변경 재시도)
- **루프 상한**: `critic_loop_count >= MAX_CRITIC_LOOPS` 시 강제 ACCEPT (기존 로직 유지, `critic_agent.py:51`)
- **점수 검증**: `scores[lowest_dim]` 이 0.0~1.0 범위 밖이면 feedback 생성 스킵

#### 관찰 가능성

```python
print(f"  [critic→retry] low_dimension={feedback['low_dimension']}, "
      f"score={feedback['score']:.2f}, loop={loop_count}")
# trace에 기록
trace.setdefault("critic_retries", []).append(feedback)
```

#### 롤백 방법

```bash
# 환경변수로 비활성화
GAPAGO_CRITIC_ADAPTIVE_RETRY=0  # 기본값 1 (활성화)
```

비활성화 시 `critic_feedback` 필드를 state에 쓰지 않으므로, 하위 에이전트는 기존 동작 유지.

---

### 3.2 개선안 #2: eval→limitation 피드백 전달 (Eval Guidance Injection)

#### 문제 정의

`limitation_eval`이 RETRY 결정 시 `retry_guidance`를 생성하지만 (`limitation_eval_agent.py:518-519`), `limitation_extract`는 재시도 시 이 guidance를 전혀 참조하지 않는다 (`limitation_agent.py` 전체에 `retry_guidance` 문자열 없음).

**현재 흐름:**
```
limitation_eval (RETRY) → retry_guidance 생성 → state["limitation_eval"]["retry_guidance"]에 저장
                        → limitation_extract 재호출 → 동일 프롬프트로 동일한 결과 생성 가능성 높음
```

#### 해결 방안

```python
# === limitation_agent.py 수정 (_build_prompt 함수 또는 노드 함수 내) ===

def limitation_extract_node(state: AgentState) -> AgentState:
    # ... 기존 코드 (L1204~) ...

    # eval retry guidance 반영
    eval_result = state.get("limitation_eval", {})
    retry_guidance = eval_result.get("retry_guidance", "")
    eval_retry_count = state.get("eval_retry_count", 0)

    guidance_prompt_suffix = ""
    if eval_retry_count > 0 and retry_guidance:
        guidance_prompt_suffix = (
            f"\n\n=== RETRY GUIDANCE (from quality evaluator) ===\n"
            f"{retry_guidance}\n"
            f"=== Please address the above feedback in your extraction. ==="
        )

    # _build_prompt 호출 시 또는 messages 구성 시 guidance 추가
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + lang_instruction},
        {"role": "user", "content": paper_prompt + guidance_prompt_suffix},
    ]
```

#### 품질 보호 장치

- **조건부 적용**: `eval_retry_count > 0`일 때만 guidance 추가 — 첫 실행에는 영향 없음
- **길이 제한**: `retry_guidance[:500]` 으로 프롬프트 비대화 방지
- **원본 프롬프트 보존**: guidance는 suffix로만 추가, 기존 SYSTEM_PROMPT 수정 없음

#### 관찰 가능성

```python
if guidance_prompt_suffix:
    print(f"  [limitation] retry_guidance 주입: {retry_guidance[:80]}...")
```

#### 롤백 방법

```bash
GAPAGO_EVAL_GUIDANCE_INJECTION=0  # 기본값 1
```

비활성화 시 `guidance_prompt_suffix`를 빈 문자열로 유지.

---

### 3.3 개선안 #3: critic→에이전트 피드백 채널 (Critic Feedback Channel)

#### 문제 정의

`critic_score`가 REDO_RETRIEVAL을 결정하면 `meaning_expand`부터 재실행되지만, critic의 평가 내용(어떤 점수가 낮았는지, 어떤 논문이 부적합했는지)이 meaning_expand에 전달되지 않는다.

**현재 코드:**
- `gapago/graphs/graph.py:64-65` — `REDO_RETRIEVAL → "meaning_expand"` 단순 라우팅
- `gapago/agents/meaning_expand_agent.py:74` — state에서 `critic_feedback` 읽는 코드 없음
- `gapago/agents/critic_agent.py:31-37` — Output Format에 점수만 있고 구체적 개선 방향 없음

**핵심 문제**: critic→meaning_expand 루프가 존재하지만, 루프를 통해 전달되는 **정보량이 0**이다.

#### 해결 방안

**Phase A: critic output 구조화** (`critic_agent.py`)

```python
# SYSTEM_PROMPT에 추가 출력 필드 요청
"""
## Additional Output (only if DECISION != ACCEPT)
improvement_target: <which dimension needs most improvement>
specific_feedback: <1 sentence: what specifically was wrong>
"""

# 파싱 후 state에 저장 (critic_agent.py:114 부근)
# critic_feedback dict 구성 (개선안 #1과 통합)
```

**Phase B: meaning_expand가 피드백 소비** (`meaning_expand_agent.py`)

```python
# meaning_expand_node() 내 (L89 부근)
critic_fb = state.get("critic_feedback", {})
if critic_fb:
    dim = critic_fb.get("low_dimension", "")
    specific = critic_fb.get("specific_feedback", "")
    prompt += f"""

RETRY CONTEXT:
The previous attempt scored low on '{dim}'.
Feedback: {specific}
Adjust your keyword expansion and query candidates accordingly.
"""
```

**Phase C: retrieval_agent 검색 전략 조정** (`retrieval_agent.py`)

```python
# paper_relevance가 낮았던 경우: 검색 소스 가중치 변경
critic_fb = state.get("critic_feedback", {})
if critic_fb.get("low_dimension") == "paper_relevance":
    # Semantic Scholar + OpenAlex 가중 (학술 소스 우선)
    # Tavily 웹 검색 비중 축소
    pass
```

#### 품질 보호 장치

- **Optional 소비**: `critic_feedback`가 없으면 기존 동작 유지
- **피드백 영향 제한**: 프롬프트 suffix로만 추가, 핵심 로직 미변경
- **루프 수렴 보장**: `MAX_CRITIC_LOOPS=2` 제한 유지 (`critic_agent.py:7`)
- **단계적 도입**: Phase A만 먼저 배포, 효과 관찰 후 Phase B/C 순차 적용

#### 관찰 가능성

```python
# trace에 루프별 피드백 이력 기록
trace.setdefault("critic_feedback_history", []).append({
    "loop": loop_count,
    "feedback": critic_feedback,
    "consumed_by": ["meaning_expand", "paper_retrieval"],  # 어떤 에이전트가 소비했는지
})
```

#### 롤백 방법

```bash
GAPAGO_CRITIC_FEEDBACK_CHANNEL=0  # 기본값 1
```

---

### 3.4 개선안 #4: 검색 전략 적응형 선택 (Adaptive Search Strategy)

#### 문제 정의

`retrieval_agent.py`는 모든 연구 도메인에 대해 동일한 8개 검색 도구를 동일 파라미터로 호출한다 (`_parallel_search`, L140-264).

- AI/CS 논문: arXiv 비중이 높아야 하지만, Crossref/OpenAlex와 동일 비중
- 의생명 논문: PubMed/PMC가 핵심이지만, 전용 검색 도구가 없음 (Crossref/S2로 간접 검색)
- ScienceON: `client_id` 유무로만 on/off, 국내 논문 비중 조절 불가

**현재 코드:**
- `gapago/agents/retrieval_agent.py:140-264` — `_parallel_search()` 고정 도구 목록
- `gapago/agents/retrieval_agent.py:167-181` — tasks 리스트 하드코딩
- `states.py:207` — `research_domain` 필드 존재하지만 retrieval에서 미활용

#### 해결 방안

```python
# === retrieval_agent.py 수정 ===

# 도메인별 검색 전략 정의
DOMAIN_SEARCH_STRATEGIES = {
    "ai_cs": {
        "arxiv": {"max_total": 150, "page_size": 100, "max_pages": 2},  # arXiv 강화
        "crossref": {"rows": 40},        # 축소
        "semantic_scholar": {"limit": 60},  # S2 강화 (CS 커버리지 높음)
        "openalex": {"per_page": 30},
        "web": {"max_queries": 3},
    },
    "biomedical": {
        "arxiv": {"max_total": 30, "page_size": 30, "max_pages": 1},  # arXiv 축소
        "crossref": {"rows": 80},         # Crossref 강화 (PubMed 색인)
        "semantic_scholar": {"limit": 60},  # S2 강화 (PubMed 커버리지)
        "openalex": {"per_page": 50},       # OpenAlex 강화
        "web": {"max_queries": 2},
    },
    "general": {
        # 현재와 동일한 기본 전략
        "arxiv": {"max_total": 100, "page_size": 100, "max_pages": 1},
        "crossref": {"rows": 60},
        "semantic_scholar": {"limit": 50},
        "openalex": {"per_page": 40},
        "web": {"max_queries": 3},
    },
}

def _parallel_search(state, resolved_year, cfg, session_id=""):
    domain = state.get("research_domain", "general")
    strategy = DOMAIN_SEARCH_STRATEGIES.get(domain, DOMAIN_SEARCH_STRATEGIES["general"])

    tasks = [
        ("arxiv", arxiv_api_call, {
            "search_query": arxiv_query,
            **strategy["arxiv"],
        }),
        ("crossref", crossref_search, {
            "query": general_q,
            "rows": strategy["crossref"]["rows"],
            "year": resolved_year,
        }),
        # ... 나머지 도구도 strategy 참조 ...
    ]
```

#### 품질 보호 장치

- **도메인 감지 의존**: `research_domain`이 비어있거나 `"auto"`면 `"general"` 전략 사용 (현재와 동일)
- **최소 결과 보장**: 어떤 전략이든 총 검색 결과가 20건 미만이면 general 전략으로 재시도
- **A/B 비교**: 동일 쿼리로 기존/적응형 전략 비교 로그 기록

#### 관찰 가능성

```python
print(f"  [retrieval] domain={domain}, strategy_key={strategy_key}")
print(f"  [retrieval] arxiv_max={strategy['arxiv']['max_total']}, "
      f"crossref_rows={strategy['crossref']['rows']}")
trace["retrieval"]["strategy"] = {
    "domain": domain,
    "strategy_applied": strategy_key,
    "results_per_source": {name: count for name, count in source_counts.items()},
}
```

#### 롤백 방법

```bash
GAPAGO_ADAPTIVE_SEARCH=0  # 기본값 1
# 비활성화 시 기존 하드코딩 파라미터 사용
```

#### 품질 리스크: 중간

- 도메인 오판 시 핵심 소스의 결과가 부족해질 수 있음
- 예: AI 논문인데 biomedical로 판정되면 arXiv 결과 30건으로 축소
- **대책**: 도메인 판정 로직에 confidence threshold 추가, 낮으면 general 사용

---

## 4. 구현 권장 순서 및 검증 방법

### 4.1 구현 순서

```
Phase 1 (즉시 가능, 1-2일)
├── 개선안 #1: 품질 기반 적응형 재시도
│   ├── states.py에 critic_feedback 필드 추가
│   ├── critic_agent.py에서 feedback dict 생성
│   └── meaning_expand_agent.py에서 feedback 소비
│
└── 개선안 #2: eval→limitation 피드백 전달
    └── limitation_agent.py에서 retry_guidance 소비

Phase 2 (Phase 1 검증 후, 3-5일)
└── 개선안 #3: critic→에이전트 피드백 채널
    ├── Phase A: critic output 구조화
    ├── Phase B: meaning_expand 피드백 소비
    └── Phase C: retrieval 전략 조정

Phase 3 (Phase 2 안정화 후, 3-5일)
└── 개선안 #4: 검색 전략 적응형 선택
    ├── DOMAIN_SEARCH_STRATEGIES 정의
    ├── _parallel_search() 수정
    └── 도메인 오판 시 fallback 로직
```

### 4.2 검증 방법

#### 개선안 #1, #2 검증

```python
# 테스트 쿼리 3개로 before/after 비교
test_queries = [
    "딥러닝 기반 단백질 구조 예측의 한계",           # biomedical
    "자율주행 시뮬레이션에서 도메인 갭 문제",          # ai_cs
    "리튬 이온 배터리 고속 충전 제약 요인",            # materials
]

# 비교 지표:
# 1. critic REDO 비율 변화 (낮아져야 함 — 같은 루프 수 내 품질 향상)
# 2. limitation eval RETRY 비율 변화 (guidance 반영으로 재추출 품질 향상)
# 3. 최종 gap 개수 및 novelty_score 분포
```

#### 개선안 #3 검증

```python
# critic_feedback_history 분석
# 1. 루프별 점수 변화 추적 (피드백 반영 시 점수가 단조 증가하는지)
# 2. REDO → 재시도 → ACCEPT까지의 루프 수 비교 (적어져야 함)
# 3. 총 LLM 호출 수 비교 (루프 수 감소로 비용 절감 확인)
```

#### 개선안 #4 검증

```python
# 도메인별 검색 결과 품질 비교
# 1. BM25 top-10 논문의 도메인 일치도 (수동 평가)
# 2. full text 접근 성공률 (도메인 전략이 접근 가능한 소스를 우선하는지)
# 3. 최종 리포트의 논문 다양성 (단일 소스 편중 방지)
```

### 4.3 LangGraph StateGraph 호환성

모든 개선안은 LangGraph의 기존 패턴을 유지한다:

| 검증 항목 | 호환성 |
|-----------|--------|
| `AgentState` TypedDict 필드 추가 | ✅ TypedDict는 추가 필드에 open (기존 필드 미영향) |
| 노드 반환값에 새 필드 추가 | ✅ LangGraph는 반환된 dict의 키만 state에 merge |
| 조건부 엣지 로직 변경 없음 | ✅ `route_after_critic`, `route_after_eval` 시그니처 유지 |
| 그래프 토폴로지 변경 없음 | ✅ 노드/엣지 추가 없이 노드 내부 로직만 수정 |
| 체크포인터 호환 | ✅ `critic_feedback: dict`는 JSON 직렬화 가능 |

### 4.4 state 추가 필드 요약

```python
# states.py에 추가할 필드 (전체 개선안 적용 시)
class AgentState(TypedDict):
    # ... 기존 필드 ...

    # 개선안 #1, #3
    critic_feedback: dict
    # 구조: {
    #   "low_dimension": str,     # "query_specificity" | "paper_relevance" | "groundedness"
    #   "score": float,           # 0.0~1.0
    #   "suggestion": str,        # 개선 방향 (선택)
    #   "specific_feedback": str, # 구체적 피드백 (Phase B)
    #   "loop": int,              # 현재 루프 번호
    # }

    # (개선안 #2는 기존 state["limitation_eval"]["retry_guidance"] 활용 — 추가 필드 불필요)
    # (개선안 #4는 기존 state["research_domain"] 활용 — 추가 필드 불필요)
```
