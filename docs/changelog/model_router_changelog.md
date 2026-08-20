# LLM Provider 선택 구조 개선 + 오케스트레이터 조건부 스킵 + 교차 검증

**작업 브랜치**: `worktree-typed-questing-llama`
**작업일**: 2026-04-02

---

## 배경

GAPAGO는 10개 이상의 에이전트가 동일한 LLM을 사용하는 "one-size-fits-all" 구조였다.
단순 분류 작업과 핵심 추론 작업이 같은 모델을 사용하여 비용/속도 최적화가 불가능했고,
일부 에이전트에는 provider 미전달 버그가 있었다.

---

## 변경 사항 요약

### 1. 버그 수정 (`e86bb7a`)

| 파일 | 문제 | 수정 |
|---|---|---|
| `gapago/agents/response_agent.py` | 모듈 레벨 `llm = get_llm()` → 서버 시작 시점 기본 provider로 고정, 사용자 선택 무시 | 모듈 레벨 LLM/agent 제거. `final_response_node()` 내부에서 `get_llm_for_agent(state, "response")` 사용 |
| `gapago/agents/gap_chat_agent.py` | `detect_user_intent()` 내 `get_llm()` → 기본 provider 사용 | 함수에 `provider` 파라미터 추가, 호출부에서 `state.get("llm_provider")` 전달 |

**채팅 컨텍스트 강화** (같은 커밋):
- 기존: 논문 수, limitation 수만 전달 → 구체적 질문 답변 불가
- 개선: 논문 목록(paper_id, 제목, 연도, 저자, abstract), limitation 상세(claim, evidence_quote), gap 전체 상세(elaboration, barriers, supporting_quotes 등)를 시스템 프롬프트에 포함

---

### 2. ModelRouter 도입 (`4220c47`)

**신규 파일**: `model_router.py`

에이전트별 최적 LLM provider를 프리셋 기반으로 자동 배정하는 라우팅 시스템.

```python
# 사용법
from model_router import ModelRouter
router = ModelRouter(default_provider="azure", profile="optimized")
llm = router.get_llm("limitation_extract")  # → claude
llm = router.get_llm("gap_classify")        # → gemini
```

#### 프리셋 정의

| 프로파일 | 설명 | 비용 | 품질 |
|---|---|---|---|
| `balanced` | 모든 에이전트가 기본 provider 사용 (기존 동작) | 기본 | 기본 |
| `optimized` | 단순→gemini, 핵심→claude, 추론→groq | 절감 | 향상 |
| `quality` | 핵심 작업 전부 claude + 추론 groq | 높음 | 최고 |
| `speed` | 대부분 gemini, 추론만 groq | 최저 | 낮음 |

#### 에이전트별 라우팅 매트릭스

| 에이전트 | 역할 | balanced | optimized | quality | speed |
|---|---|---|---|---|---|
| query_analysis | 쿼리 분석 | 기본 | gemini | 기본 | gemini |
| query_refine | 쿼리 정제 | 기본 | gemini | 기본 | gemini |
| meaning_expand | 의미 확장 | 기본 | gemini | 기본 | gemini |
| orchestrator | 파이프라인 조율 | 기본 | gemini | gemini | gemini |
| gap_classify | gap 분류 | 기본 | gemini | gemini | gemini |
| critic_score | 품질 평가 | 기본 | gemini | 기본 | gemini |
| **limitation_extract** | **핵심 추출** | 기본 | **claude** | **claude** | 기본 |
| **limitation_eval** | 추출 품질 평가 | 기본 | 기본 | **claude** | gemini |
| **recency_check** | 최신성 검증 | 기본 | 기본 | **claude** | gemini |
| **gap_reasoning** | **핵심 추론** | 기본 | **groq** | **groq** | groq |
| **response** | **최종 보고서** | 기본 | **claude** | **claude** | gemini |
| limitation_verify | 교차 검증 | _(없음)_ | gemini | **claude** | gemini |

#### 핵심 헬퍼 함수

```python
# llm.py에 추가
def get_llm_for_agent(state: dict, agent_name: str):
    """model_routing이 있으면 라우터 사용, 없으면 기존 llm_provider fallback"""
```

#### State 필드

```python
# states.py AgentState에 추가
model_routing: dict  # {"default_provider": "azure", "profile": "optimized"}
```

---

### 3. 전체 에이전트 마이그레이션 (`c96d5b9`)

11개 에이전트의 LLM 획득 방식을 `get_llm_for_agent(state, "agent_name")`으로 통일.

**변경 패턴**:
```python
# Before
llm = get_llm(provider=state.get("llm_provider"))

# After
llm = get_llm_for_agent(state, "agent_name")
```

**gap_agent 특수 처리**:
- `_llm_invoke(messages, use_reasoning, state)` — state 파라미터 추가
- `model_routing`이 있으면: `gap_reasoning` / `gap_classify`로 라우팅
- `model_routing`이 없으면: 기존 `GAP_REASONING_PROVIDER` env var fallback (하위 호환)
- 모든 내부 헬퍼 함수(`_generate_all_axes`, `_classify_limitations_batch`, `_score_axis_urgency`, `_analyze_barriers`, `_generate_creative_directions`)에 state 파라미터 전달

---

### 4. 오케스트레이터 fast_mode 유동 판단 (`c96d5b9`)

**기존**: fast_mode는 개별 에이전트에서 각자 체크, 오케스트레이터는 무시

**변경**: 오케스트레이터 프롬프트에 fast_mode 가이드라인 주입

```
⚡ FAST MODE 활성화됨
사용자가 빠른 분석을 요청했습니다. 속도와 품질의 균형을 판단하세요:
- optional 에이전트는 스킵 가능하나, 품질이 낮으면 실행해도 됩니다.
- 판단 기준: 스킵해도 최종 결과에 큰 영향이 없는가?
```

- 강제 스킵 아님 → LLM이 현재 state (논문 수, limitation 수, feedback)를 보고 유동적으로 판단
- limitation이 매우 적으면 → eval 실행 판단 가능
- 충분한 데이터가 있으면 → 스킵하고 다음 단계로

---

### 5. 교차 검증 파일럿 (`c96d5b9`)

**파일**: `gapago/agents/limitation_agent.py` — `_verify_limitations()` 함수 추가

limitation_extract 완료 후, 추출된 limitation을 **다른 provider**로 검증하는 단계.

**동작 조건**:
- `model_routing.profile`이 `optimized`, `quality`, `speed`일 때만 실행
- `balanced` 프로파일에서는 스킵 (기존 동작 유지)

**검증 방식**:
1. 논문별로 limitation 그룹핑
2. 해당 논문의 full text sections과 함께 검증 LLM에 전달
3. 각 limitation에 대해:
   - `evidence_quote`가 원문에 존재하는가? (FOUND/NOT_FOUND)
   - `claim`이 `evidence_quote`에서 도출 가능한가? (VALID/INVALID)
4. 결과를 limitation에 `verified: true/false` 플래그로 추가
5. 검증 실패 limitation은 제거하지 않고 플래그만 → 하류 에이전트가 판단

**검증 provider**: `limitation_verify` 에이전트 이름으로 라우팅
- optimized/speed: gemini
- quality: claude

---

### 6. UI/API 통합 (`57baeee`)

#### API 변경

| 엔드포인트 | 추가 파라미터 | 기본값 |
|---|---|---|
| `GET /api/analyze` | `routing_profile` | `"balanced"` |
| `GET /api/explore` | `routing_profile` | `"balanced"` |

내부적으로 `ModelRouter` 생성 → `model_routing` dict를 pipeline inputs에 주입.

#### 프론트엔드

LLM 프로바이더 드롭다운 아래에 라우팅 프로파일 선택기 추가:

```html
<select id="routingProfile">
    <option value="balanced">Balanced (단일 모델)</option>
    <option value="optimized">Optimized (에이전트별 최적화)</option>
    <option value="quality">Quality (최고 품질, Claude 활용)</option>
    <option value="speed">Speed (최대 속도)</option>
</select>
```

모든 API 호출(analyze, explore, explore fallback)에 `routing_profile` 파라미터 전달.

#### CLI (main.py)

대화형 메뉴에 라우팅 프로파일 선택 추가:

```
=== 라우팅 프로파일 선택 ===
  0) balanced  - 단일 모델 (기본값)
  1) optimized - 에이전트별 최적화
  2) quality   - 최고 품질 (Claude 활용)
  3) speed     - 최대 속도
```

---

## 변경 파일 목록

| 파일 | 변경 내용 |
|---|---|
| `model_router.py` | **신규** — ModelRouter + 프리셋 |
| `llm.py` | `get_llm_for_agent()` 헬퍼 추가 |
| `states.py` | `model_routing: dict` 필드 추가 |
| `gapago/agents/response_agent.py` | 버그 수정 (모듈 레벨 LLM 제거) |
| `gapago/agents/gap_chat_agent.py` | 버그 수정 (provider 전달) + 채팅 컨텍스트 강화 |
| `gapago/agents/gap_agent.py` | `_llm_invoke()` state 기반 마이그레이션 |
| `gapago/agents/meaning_expand_agent.py` | `get_llm_for_agent` 적용 |
| `gapago/agents/critic_agent.py` | `get_llm_for_agent` 적용 |
| `gapago/agents/query_agent/query_analysis.py` | `get_llm_for_agent` 적용 |
| `gapago/agents/query_agent/query_refine.py` | `get_llm_for_agent` 적용 |
| `gapago/agents/recency_agent.py` | `get_llm_for_agent` 적용 |
| `gapago/agents/limitation_eval_agent.py` | `get_llm_for_agent` 적용 |
| `gapago/agents/limitation_agent.py` | `get_llm_for_agent` 적용 + `_verify_limitations()` 추가 |
| `gapago/agents/retrieval_agent.py` | `get_llm_for_agent` 적용 |
| `gapago/agents/orchestrator_agent.py` | `get_llm_for_agent` 적용 + fast_mode 유동 판단 |
| `gapago/api/main.py` | `routing_profile` 파라미터 + `model_routing` state 주입 |
| `frontend/index.html` | 라우팅 프로파일 선택 UI |
| `main.py` | 라우팅 프로파일 CLI 메뉴 + `model_routing` state 주입 |

---

### 7. CLI/Web 기능 패리티 수정 (`0819bde`)

| 파일 | 문제 | 수정 |
|---|---|---|
| `gapago/agents/gap_chat_agent.py` | `gap_chat_respond()` 내 `get_llm()` → 기본 provider 사용 | `get_llm_for_agent(state, "gap_chat")` 로 마이그레이션 |
| `gapago/api/main.py` | explore 엔드포인트에 `fast_mode` 미전달, `_save_result`에 라우팅 정보 미저장 | `fast_mode` 파라미터 추가, `_save_result`에 `llm_provider`/`model_routing`/`output_language` 저장, `_build_chat_state`에 `model_routing`/`output_language` 전달 |
| `frontend/index.html` | explore 호출 시 `fast_mode` 파라미터 누락 | explore API 호출에 `fast_mode` 파라미터 추가 |

---

### 8. 비-balanced 프로파일 provider 자동 할당 (`3fd3c3c`)

비-balanced 프로파일(optimized, quality, speed) 선택 시 provider 드롭다운을 숨기고 프로파일 기본 provider를 자동 할당.

**변경 내용**:

- `model_router.py`: `PROFILE_DEFAULT_PROVIDERS` 딕셔너리 추가
  - `optimized` / `quality` → `azure`
  - `speed` → `gemini`
- `frontend/index.html`: 프로파일 변경 이벤트 리스너 추가 — `balanced`일 때만 provider 드롭다운 표시, 나머지는 숨기고 기본값 자동 설정
- `main.py`: CLI에서도 동일하게 프로파일 선택을 먼저 받고, `balanced`일 때만 provider 선택 프롬프트 표시

**의도**: 사용자가 "optimized" 선택 후 provider를 groq로 바꾸면 라우팅 의도와 충돌 → 혼란 방지

---

## 변경 파일 목록 (추가분)

| 파일 | 변경 내용 |
|---|---|
| `gapago/agents/gap_chat_agent.py` | `gap_chat_respond()` LLM 라우팅 마이그레이션 |
| `gapago/api/main.py` | explore `fast_mode`, `_save_result`/`_build_chat_state` 필드 보강 |
| `frontend/index.html` | explore `fast_mode` + 프로파일별 provider 드롭다운 조건부 표시 |
| `main.py` | 프로파일별 provider 선택 로직 분기 |
| `model_router.py` | `PROFILE_DEFAULT_PROVIDERS` 추가 |

---

## 하위 호환성

- `balanced` 프로파일 = 기존 동작과 100% 동일
- `model_routing`이 state에 없으면 `llm_provider` fallback → 기존 코드 호환
- gap_agent의 `GAP_REASONING_PROVIDER` env var는 `balanced` 프로파일에서 계속 동작
- 채팅 API(`/api/chat`)는 `model_routing` 없이도 정상 동작 (`llm_provider` fallback)

---

## 검증 방법

1. **balanced 프로파일**: 기존 동작과 동일한지 회귀 테스트
2. **optimized 프로파일**: 각 에이전트 로그에서 지정 provider 확인
3. **fast_mode + 오케스트레이터**: optional 에이전트 유동 판단 확인
4. **교차 검증**: optimized/quality 프로파일에서 limitation `verified` 플래그 확인
5. **동시 요청 테스트**: 다른 프로파일 2개 요청 동시 실행 → 결과 격리 확인
6. **채팅 테스트**: "3번 논문이 뭐야?", "GAP 2번 장벽 설명해줘" 등 구체적 질문 답변 확인
