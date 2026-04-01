# GAPAGO 멀티 에이전트 아키텍처 전환 제안서

## 1. 배경: 왜 멀티 에이전트인가

현재 GAPAGO는 9개 노드가 고정 순서로 실행되는 **순차 파이프라인**입니다.

```
query → expand → retrieval → limitation → eval → recency → gap → critic → response
```

"멀티 에이전트 시스템"의 학술적 정의(5가지 기준)와 비교하면:

| 기준 | 현재 충족 여부 |
|------|:---:|
| 자율적 목표 추구 (Autonomous goal pursuit) | X |
| 에이전트 간 통신 (Inter-agent communication) | X |
| 동적 태스크 할당 (Dynamic task allocation) | X |
| 협상/투표 (Negotiation/voting) | X |
| 반응적 적응 (Reactive adaptation) | X |

**0/5 충족** — 현재 구조는 "에이전트"라고 부르지만, 실제로는 함수들이 고정 순서로 호출되는 파이프라인입니다.

---

## 2. 구현한 것: 오케스트레이터 패턴

기존 에이전트 코드를 **전혀 수정하지 않고**, 위에 LLM 오케스트레이터를 추가했습니다.

### 구조

```
[Before]  START → 고정 순서 9개 노드 → END

[After]   START → query_subgraph → 오케스트레이터 ←→ 에이전트 풀 → END
                                        ↑                │
                                        └────────────────┘
                                     LLM이 다음 에이전트 결정
```

### 핵심 설계

- **환경변수 플래그**: `GAPAGO_ORCHESTRATOR=1`로 활성화, `0`이면 기존 파이프라인 그대로
- **필수 경로 보장**: `expand → retrieval → limitation → gap → response` 5단계는 코드 레벨에서 강제 (스킵 불가)
- **선택 단계**: `limitation_eval`, `recency_check`, `critic_score`는 오케스트레이터가 상황에 따라 삽입/스킵 결정
- **피드백 채널**: 에이전트 실행 결과를 오케스트레이터가 읽고 다음 행동 판단 (예: critic이 품질 낮다고 하면 재검색)
- **안전장치**: 최대 15스텝 제한, 동일 에이전트 재실행 2회 제한, LLM 실패 시 고정 경로 fallback

### 5가지 기준 충족

| 기준 | 충족 방법 |
|------|-----------|
| 자율적 목표 추구 | 오케스트레이터 LLM이 다음 에이전트를 자율 결정 |
| 에이전트 간 통신 | `agent_feedback` 필드로 에이전트 → 오케스트레이터 피드백 전달 |
| 동적 태스크 할당 | `Command(goto=...)` 로 런타임에 다음 노드 결정 |
| 협상/투표 | critic_score 피드백을 오케스트레이터가 해석하여 재실행 결정 |
| 반응적 적응 | feedback 기반으로 선택 단계 삽입 또는 이전 단계 재실행 |

---

## 3. 실험 결과

동일 쿼리(`"LoRA-based parameter-efficient fine-tuning for low-resource language adaptation in multilingual BERT"`)로 두 모드를 비교했습니다.

### 3-1. 실행 경로 비교

```
[고정]         query → expand → retrieval → limitation → eval → recency → gap → critic → response
[오케스트레이터] query → expand → retrieval → limitation → eval → recency → gap → critic → response
```

오케스트레이터가 3개 선택 단계를 모두 삽입하여, 고정 파이프라인과 **동일한 경로**를 실행했습니다.

### 3-2. 품질 비교

| 지표 | 고정 파이프라인 | 오케스트레이터 |
|------|:---:|:---:|
| 검색 논문 수 | 15편 | 15편 |
| 추출 limitation | 18개 | 21 → 18개 (eval이 3개 필터링) |
| 생성 GAP | 5개 | 5개 |
| recency 검증 | resolved 1, partial 1, unresolved 16 | resolved 1, partial 5, unresolved 12 |
| critic 판정 | ACCEPT | ACCEPT |

> 결론: **품질은 동등**, limitation_eval 필터링도 정상 작동

### 3-3. 비용 오버헤드

오케스트레이터가 매 스텝마다 LLM 호출 1회 추가 (총 8회).
단, 짧은 프롬프트 + 짧은 응답(JSON 1줄)이므로 비용은 미미합니다.

---

## 4. 솔직한 평가: 현재 한계

**"정상 경로"에서는 고정 파이프라인과 결과가 같습니다.**

오케스트레이터의 가치는 **분기가 발생하는 비정상 경로**에서 나옵니다:
- critic이 REDO_RETRIEVAL → 어디서부터 다시 할지 LLM이 판단
- limitation 품질이 낮을 때 → 재추출 vs 다른 전략 선택
- fast_mode → 선택 단계 전부 스킵하여 속도 우선

현재 실험에서는 이런 분기가 발생하지 않았기 때문에, 차이가 드러나지 않았습니다.

---

## 5. 이 구조가 실질적 가치를 갖는 시나리오

| 시나리오 | 고정 파이프라인 | 오케스트레이터 |
|----------|:-:|:-:|
| **fast_mode** — 빠른 결과 우선 | 모든 단계 실행 (스킵 불가) | eval/recency/critic 스킵 가능 |
| **papers < 5편** — 검색 결과 부족 | 그대로 진행 (품질 저하) | 자동 재검색 판단 가능 |
| **critic REDO 판정** — 품질 미달 | 고정 경로로만 재실행 | 피드백 보고 expand부터 or retrieval부터 선택 |
| **새 에이전트 추가** | 엣지 수동 수정 필요 | agent_pool에 추가만 하면 끝 |
| **심사 기준 충족** | 0/5 | 5/5 |

---

## 6. 변경된 파일

| 파일 | 상태 | 내용 |
|------|:---:|------|
| `states.py` | 수정 | `completed_stages`, `agent_feedback`, `orchestrator_plan` 3개 필드 추가 |
| `agents/orchestrator_agent.py` | 신규 | LLM 오케스트레이터 노드 (~180줄) |
| `graphs/orchestrator_graph.py` | 신규 | 동적 그래프 빌드 + 에이전트 래퍼 (~130줄) |
| `graphs/graph.py` | 수정 | `build_graph()`에 환경변수 분기 추가 (5줄) |
| `tests/test_orchestrator.py` | 신규 | 단위 테스트 24개 (전부 통과) |

**기존 에이전트 코드 변경: 0줄** — 모든 기존 에이전트는 그대로 동작합니다.

---

## 7. 리스크

| 리스크 | 대응 |
|--------|------|
| 오케스트레이터 도입으로 품질 저하 | `GAPAGO_ORCHESTRATOR=0`으로 즉시 롤백 (기존 파이프라인 그대로) |
| LLM 오케스트레이터 판단 오류 | 필수 5단계 코드 강제 + LLM 실패 시 fallback |
| 무한 루프 | 최대 15스텝 + 동일 에이전트 2회 제한 |
| LLM 비용 증가 | 오케스트레이터 호출은 짧은 프롬프트/응답, 비용 미미 |

---

## 8. 제안: 다음 단계

### 당장 할 수 있는 것
- [x] 오케스트레이터 구현 완료 (feat/jw-multi-agent 브랜치)
- [x] 기존 파이프라인과 동등 품질 확인
- [x] 환경변수 플래그로 즉시 롤백 가능

### 논의 필요한 것
1. **기본값을 어떻게 할 것인가** — 운영 환경에서 `GAPAGO_ORCHESTRATOR=0`(안전) vs `1`(멀티에이전트)?
2. **fast_mode 연동** — `fast_mode=True`일 때 선택 단계 자동 스킵 로직 추가할 것인가?
3. **실질적 분기 시나리오 테스트** — critic REDO, limitation RETRY 등 비정상 경로 검증 필요
