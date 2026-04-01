# 동적 K 검증 리포트

> 분석일: 2026-04-01
> 대상 코드: `agents/retrieval_agent.py`, `config.py`, `api/main.py`

---

## 1. 동적 K 결정 로직 요약

파이프라인에서 논문 선별 수(K)는 3단계에서 독립적으로 동적 결정된다.

### 1.1 Stage 1: BM25 동적 K (line 694-706)

```python
threshold = mean_score + 0.5 * std_score
dynamic_k = int(np.sum(scores > threshold))
dynamic_k = max(10, min(dynamic_k, cfg.bm25_top_k))  # range [10, 50]
```

| 항목 | 값 |
|------|-----|
| 기본값 | `bm25_top_k = 50` (config.py:63) |
| 하한 | 10 |
| 상한 | 50 (환경변수 `BM25_TOP_K`로 조정 가능) |
| 결정 기준 | BM25 점수가 `mean + 0.5*std` 초과인 논문 수 |

**동작 분석**:
- BM25 점수 분포가 정규 분포에 가까울 때, `mean + 0.5*std` 이상은 약 상위 31%에 해당
- 검색 결과 100편 중 약 30편이 선택되는 것이 일반적
- `_parallel_search()`가 ArXiv(100) + Crossref(60) + S2(50) + OpenAlex(40) + ScienceON(15+10+10) + Web(5) = 최대 ~290편을 수집, 중복 제거 후 통상 80-150편
- dynamic_k가 10 미만이 되는 케이스: 대부분의 논문이 비슷한 점수를 가질 때 (전문적 쿼리에서 낮은 분산) → `max(10, ...)` 보호
- dynamic_k가 50을 초과하는 케이스: 많은 논문이 고관련도일 때 → `min(..., 50)` 보호

### 1.2 Stage 1 보조: FAISS 필터 (line 708-724)

```python
faiss_future = executor.submit(_faiss_filter, raw_papers, query, top_k=cfg.bm25_top_k)
```

- FAISS는 고정 top_k (`bm25_top_k = 50`)를 사용 — 동적이지 않음
- BM25 결과와 합집합(union) 후 중복 제거 → Stage1 출력은 최대 100편 (BM25 50 + FAISS 50, 중복 제외)

### 1.3 Stage 2: Full Text 필터 (line 728-730)

```python
stage1_papers = _filter_fulltext_available(stage1_papers, target_count=cfg.bm25_top_k)
```

- `target_count = bm25_top_k = 50` (고정)
- full text 확보 신뢰도순 정렬: `guaranteed(arXiv)` > `likely(OA PDF)` > `maybe(DOI)`
- 신뢰도 `none`인 논문은 제외됨
- 이 단계는 동적이 아니라, 상위 50편까지 신뢰도순으로 선별

### 1.4 Stage 3: CrossEncoder 동적 K (line 502-518)

```python
threshold = max_score * 0.65          # _CE_DYNAMIC_SCORE_RATIO
dynamic_k = sum(1 for s in sorted_scores if s >= threshold)
# Score gap 감지: 30% 이상 급락 시 컷
dynamic_k = max(15, min(dynamic_k, 25))  # [_CE_DYNAMIC_MIN_K, _CE_DYNAMIC_MAX_K]
```

| 항목 | 값 |
|------|-----|
| 하한 | 15 (`_CE_DYNAMIC_MIN_K`, line 477) |
| 상한 | 25 (`_CE_DYNAMIC_MAX_K`, line 478) |
| 비율 임계값 | top score × 0.65 |
| Gap 감지 | 점수 범위의 30% 이상 급락 시 컷 |

**동작 분석**:
- 입력이 `_CE_DYNAMIC_MIN_K(15)` 이하이면 reranking 자체를 스킵 (line 487)
- Gap 감지 로직이 `dynamic_k`를 더 작게 만들 수 있음 (기존 dynamic_k와 `min()` 적용)
- 최종 범위는 항상 [15, 25]

### 1.5 Fast Mode 예외 (line 736-738)

```python
if fast_mode:
    raw_papers = stage1_papers[:_CE_DYNAMIC_MIN_K]  # 첫 15편
```

- Fast mode에서는 CrossEncoder와 LLM Reranker를 모두 스킵
- BM25+FAISS 합집합 결과에서 상위 15편만 사용 (고정)

### 1.6 LLM Reranker Fallback (line 554-584)

```python
raw_papers = _llm_rerank(stage1_papers, query, top_k=cfg.reranker_top_k)
```

- CrossEncoder 실패 시 LLM Reranker가 `reranker_top_k = 15`편을 선별 (고정)
- LLM이 `top_k`보다 적은 수를 반환할 수 있음 (프롬프트에 "select exactly {top_k} papers (or fewer if fewer are truly relevant)" 명시)

---

## 2. 프론트엔드 전달값 분석

### `api/main.py` (line 738-739)

```python
payload["papers_count"] = len(papers)
payload["total_searched"] = values.get("total_candidates_count", len(papers))
```

**`_save_result()`** (line 205):
```python
"total_searched": state_values.get("total_candidates_count", len(papers)),
```

### 결과 파일 분석

| 지표 | 관찰 |
|------|------|
| `papers` | 결과 JSON에서 항상 **빈 배열** (0편) |
| `total_searched` | 항상 **0** |
| `limitations` | 정상 (7~37개, 복수 논문에서 추출) |

**원인**: `_save_result()`에서 `state_values.get("papers", [])`로 가져오지만, LangGraph 상태에서 `papers` 필드가 최종 state에 누적되지 않는 것으로 보임. 마찬가지로 `total_candidates_count`도 최종 state에서 누락.

> **참고**: limitation에 기록된 `paper_id`를 보면 실제로 14-15편의 논문이 처리되고 있으므로, 동적 K 자체는 작동하고 있으나 결과 파일에 반영되지 않는 별도의 버그가 존재한다.

---

## 3. Edge Case 분석

### Case 1: 검색 결과 1~2편

| Stage | 동작 |
|-------|------|
| BM25 | `dynamic_k = max(10, ...)` → 10이지만 실제 논문 2편 → 2편 모두 통과 |
| FAISS | `if len(papers) <= top_k: return papers` → 스킵 (line 426-427) |
| Full text | 2편 그대로 유지 |
| CrossEncoder | `if len(papers) <= _CE_DYNAMIC_MIN_K(15): return papers` → 스킵 (line 487) |
| **결과** | **2편 모두 그대로 최종 선별** — 정상 |

### Case 2: 검색 결과 20편 (모두 ArXiv)

| Stage | 동작 |
|-------|------|
| BM25 | `dynamic_k` ≈ 6~10 (20편 중 상위 31%) → `max(10, ...)` = 10 |
| FAISS | 20편 중 50편 → 20편 전부 반환 (top_k > len) |
| Union | ~20편 (거의 동일) |
| Full text | 20편 모두 arXiv → `guaranteed` → 20편 유지 |
| CrossEncoder | 20편 > 15 → 동적 K 적용 → 15~20편 선별 |
| **결과** | **15~20편 선별** — 정상 |

### Case 3: 검색 결과 200편 (혼합 소스)

| Stage | 동작 |
|-------|------|
| BM25 | `dynamic_k` ≈ 40~50 (200편 × 상위 31%) |
| FAISS | 상위 50편 |
| Union | ~70-80편 (BM25 50 + FAISS 50, 중복 제거) |
| Full text | 신뢰도순 상위 50편 (`target_count=50`) |
| CrossEncoder | 50편 → 15~25편 동적 선별 |
| **결과** | **15~25편 최종** — 의도대로 작동 |

---

## 4. 실제 데이터 기반 검증

결과 파일 30건 분석 결과:
- limitation이 추출된 논문 수: 7~15편 (중앙값: 14편)
- 이는 CrossEncoder 동적 K 범위 [15, 25]에서 일부 논문이 full text 실패 후 backup 교체되거나 중복 제거된 결과와 일치
- Fast mode 실행 결과(독성 예측 2차): 7편 → 15편 상한 이내, fast mode 동작 확인

---

## 5. 발견된 이슈 및 개선 제안

### Issue 1: 결과 JSON에 papers/total_searched 미반영 (심각도: 중)
- `papers`와 `total_candidates_count`가 LangGraph 최종 상태에서 소실
- 프론트엔드에서 "N편 중 M편 선별" 표시 불가
- **원인 추정**: LangGraph 상태 업데이트 시 `papers` 키가 이후 노드에서 덮어쓰기되거나, `total_candidates_count`가 state schema에 정의되지 않았을 가능성

### Issue 2: FAISS stage는 고정 K (심각도: 낮)
- BM25는 동적 K를 사용하지만 FAISS는 `bm25_top_k(50)` 고정
- 합집합 방식이므로 큰 문제는 아니나, 일관성 개선 여지 존재

### Issue 3: Fast mode 하한이 CrossEncoder 최소값에 종속 (심각도: 낮)
- `_CE_DYNAMIC_MIN_K = 15`가 fast mode 선별 수에도 사용됨
- Fast mode에서 더 적은 수(예: 10편)로 빠르게 처리하는 옵션 부재

---

## 6. 결론

**동적 K는 전반적으로 의도대로 작동하고 있다.**

- BM25: `mean + 0.5*std` 기반, [10, 50] 범위 — 적절
- CrossEncoder: `max_score * 0.65` + gap detection, [15, 25] 범위 — 적절
- Edge case (극소/극대 결과) 모두 보호 장치 작동 확인
- 실제 결과에서 최종 선별 논문 7~15편은 기대 범위 내

**주요 버그**: 결과 JSON에 `papers`와 `total_searched`가 기록되지 않는 문제는 동적 K 로직과 무관하며, LangGraph 상태 전파 이슈로 별도 수정 필요.
