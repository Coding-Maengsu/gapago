# 웹 검색(Tavily) 결과 활용 현황 리포트

> 분석일: 2026-04-01
> 대상 코드: `agents/meaning_expand_agent.py`, `agents/retrieval_agent.py`, `agents/recency_agent.py`, `agents/gap_agent.py`, `api/main.py`

---

## 1. 파이프라인 내 웹 검색 호출 지점

### 1.1 1차 호출: 초기 병렬 검색 (`retrieval_agent.py:199-202`)

```python
tavily_tool = TavilySearch(max_results=cfg.tavily_max_results)  # default 5
web_query = web_qs[0] if web_qs else refined_query
tasks.append(("web", tavily_tool.search, {"query": web_query}))
```

- **시점**: `paper_retrieval` 노드 실행 시
- **쿼리**: `meaning_expand`가 생성한 `web_query_candidates[0]` (1개만 사용)
- **결과 수**: 최대 5건 (`tavily_max_results` 기본값)
- **결과 저장**: `state["web_results"]`에 `source: "web"` 태그로 저장

### 1.2 2차 호출: 최신성 검증 (`recency_agent.py:146-168`)

```python
tavily = TavilySearch(max_results=3)
for query in search_queries[:5]:  # LLM이 생성한 3~5개 쿼리
    results = tavily.search(query=query, include_domains=..., exclude_domains=..., max_results=3)
```

- **시점**: `recency_check` 노드 실행 시
- **쿼리**: LLM이 limitation 목록 기반으로 생성한 3~5개 맞춤 쿼리 (`QUERY_GEN_PROMPT`)
- **결과 수**: 쿼리당 최대 3건 × 최대 5개 쿼리 = 최대 15건
- **도메인 필터링**: LLM이 판단한 연구 도메인별 include/exclude 도메인 적용
- **결과 저장**: 기존 `web_results`에 `source: "recency_search"` 태그로 **추가**

---

## 2. 검색 쿼리 생성 방식

### 2.1 초기 웹 쿼리 (`meaning_expand_agent.py:108-109`)

```
"web_query_candidates": ["<=4 queries"]
```

- LLM이 `refined_query` + `keywords`를 기반으로 최대 4개 후보 생성
- **실제 사용: 1개만** (`web_qs[0]`, `retrieval_agent.py:201`)
- 나머지 3개는 **사용되지 않음** → 비효율

### 2.2 최신성 검증 쿼리 (`recency_agent.py:49-68`)

- LLM이 limitation 목록을 보고 도메인 판단 + 3~5개 타겟 쿼리 생성
- 쿼리에 "2024", "2025", "latest", "state-of-the-art" 같은 시간 마커 포함 유도
- limitation 클러스터별 쿼리 생성 → 더 정밀

---

## 3. 결과 전달/활용 경로

### 3.1 1차 소비: Recency Check (`recency_agent.py:175-287`)

```
web_results = state.get("web_results", [])  # 1차 검색 결과 포함
all_web = _search_for_recency(...)          # 기존 + 추가 검색 합산
```

**활용 방식**:
1. 기존 `web_results`(1차)를 시작점으로 사용
2. 추가 Tavily 검색 결과를 합산 → `all_web`
3. 최대 15건(`all_web[:15]`)을 LLM에 전달
4. LLM이 각 limitation의 최신성 판정 (unresolved/partial/resolved)
5. 갱신된 `web_results`(1차+2차 합산)를 state에 반환

### 3.2 2차 소비: Gap Inference (`gap_agent.py:670-678`)

```python
web_results = state.get("web_results", [])
recent = [r for r in web_results if r.get("source") == "recency_search"][:6]
```

**활용 방식**:
1. `source == "recency_search"`인 결과만 필터 (2차 검색 결과만 사용)
2. 최대 6건을 `_generate_creative_directions()` 컨텍스트로 전달
3. 프롬프트에 "Recent web developments (use as context, NOT as your answer)" 형태로 삽입
4. LLM이 창의적 연구 방향 제안 시 최신 동향 참고

### 3.3 저장 (`api/main.py:210`)

```python
"web_results": state_values.get("web_results", [])
```

- 최종 결과 JSON에 전체 `web_results` (1차+2차) 저장
- 프론트엔드에서 직접 표시하지는 않는 것으로 보임

---

## 4. 실제 데이터 분석

결과 파일 30건 분석:

| 지표 | 값 |
|------|-----|
| web_results가 있는 파일 | 19/30건 (63%) |
| 평균 web_results 수 | 12.2건 |
| source별 분포 | `web`: 평균 3건, `recency_search`: 평균 10.5건 |
| web_results가 0인 파일 | 11건 (초기 개발 기간 + Tavily API 미설정 추정) |

---

## 5. 미활용 / 비효율적 활용 지점

### 5.1 `web_query_candidates` 중 3개 미사용 (비효율)

- `meaning_expand`가 4개 후보를 생성하지만 `retrieval_agent`에서 **1개만** 사용
- 나머지 3개 후보는 LLM 비용만 발생하고 활용되지 않음

### 5.2 1차 웹 결과(`source: "web"`)가 Gap Agent에서 무시됨 (미활용)

- Gap Agent는 `source == "recency_search"`만 필터하여 1차 웹 검색 결과를 완전히 무시
- 1차 결과도 연구 동향을 포함할 수 있으므로 활용 가치 존재

### 5.3 웹 결과가 Limitation Agent에서 활용되지 않음 (미활용)

- Full text 실패 시 웹 검색으로 논문 요약/리뷰 페이지를 찾아 보충할 수 있으나, 현재 미구현
- Limitation Agent는 순수하게 논문 원문/abstract만 사용

### 5.4 프론트엔드에서 웹 결과 미표시 (미활용)

- 결과 JSON에 저장되지만 UI에서 표시/활용하지 않음
- 사용자가 최신 동향을 확인할 수 없음

### 5.5 Recency Check의 도메인 판단이 paper_retrieval과 분리됨 (비효율)

- `recency_agent`에서 LLM으로 도메인을 다시 판단
- `paper_retrieval`에서 이미 `research_domain` 정보가 state에 있지만 연계 부족

---

## 6. 추가 활용 방안 제안

### 제안 1: 1차 웹 쿼리 다중화 (난이도: 낮, 기대효과: 중)

**현재**: `web_qs[0]` 1개만 사용
**개선**: 2~3개 쿼리를 병렬 검색하여 다양한 관점의 웹 결과 확보

```python
for wq in web_qs[:3]:
    tasks.append(("web", tavily_tool.search, {"query": wq, "max_results": 3}))
```

- 추가 Tavily API 비용: 쿼리 2건 (약 0.01 USD)
- 기존 병렬 검색 프레임워크 재활용 가능

### 제안 2: Gap Agent에서 1차 웹 결과도 활용 (난이도: 매우 낮, 기대효과: 낮~중)

**현재**: `source == "recency_search"`만 필터
**개선**: `source in ("web", "recency_search")`로 확장

```python
recent = [r for r in web_results if r.get("source") in ("web", "recency_search")][:6]
```

- 코드 변경 1줄
- 1차 웹 결과에 일반적인 최신 동향이 포함되어 있을 수 있음

### 제안 3: Limitation Agent에서 웹 보충 검색 (난이도: 중, 기대효과: 중~높)

**목적**: Full text 실패 논문에 대해 웹 검색으로 논문 요약/리뷰 페이지 확보

```python
if not sections:  # full text 실패
    title_query = f"{paper.title} review summary limitations"
    web_supplement = tavily.search(query=title_query, max_results=2)
    # 보충 텍스트를 abstract와 합쳐서 LLM에 전달
```

- Full text 실패율을 낮추는 간접 대안
- Tavily API 추가 비용: 논문당 1건

### 제안 4: 프론트엔드 웹 결과 표시 (난이도: 중, 기대효과: 중)

- 결과 페이지에 "관련 최신 동향" 섹션 추가
- `web_results`에서 `recency_search` 소스를 카드 형태로 표시
- 사용자가 연구 최신성을 직접 판단 가능

### 제안 5: 웹 검색 결과를 Critic Agent에 전달 (난이도: 중, 기대효과: 중)

- Critic Score 단계에서 웹 결과를 참고하여 제안된 GAP의 현실성/최신성 검증
- 이미 해결된 문제를 GAP으로 제안하는 것을 방지

---

## 7. 요약 플로우 다이어그램

```
[meaning_expand]
    └─ web_query_candidates (4개 생성, 1개만 사용)

[paper_retrieval]
    └─ Tavily 1차 검색 (1개 쿼리, max 5건)
    └─ state["web_results"] = [{source: "web", ...}]

[recency_check]
    └─ 기존 web_results 수신
    └─ LLM → 도메인 판단 + 3~5개 맞춤 쿼리 생성
    └─ Tavily 2차 검색 (쿼리당 3건, 도메인 필터)
    └─ state["web_results"] += [{source: "recency_search", ...}]
    └─ LLM → limitation별 최신성 판정

[gap_infer]
    └─ web_results에서 source=="recency_search" 만 필터
    └─ 최대 6건 → 창의적 연구 방향 제안 컨텍스트

[_save_result]
    └─ 전체 web_results를 결과 JSON에 저장
```
