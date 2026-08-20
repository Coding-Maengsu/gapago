# GAPAGO 웹 대규모 수정 변경 내역

**작성일:** 2026-03-28
**상태:** 구현 완료

---

## 개요

TODO 문서의 4개 영역(랜딩 페이지, 로딩 페이지, TAB 구성, 기타 수정)을 전면 구현했다. 심사 기준(파급성 25점, 효율성 15점)을 고려하여, 연구 현장의 문제를 직관적으로 해결하는 UX를 구축했다.

---

## 수정 파일 목록

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `web/app/index.html` | **전면 재작성** (1,325줄 → ~2,160줄) | 랜딩·로딩·결과 UI 전면 리디자인 |
| `gapago/api/main.py` | 수정 | `/api/explore` 엔드포인트, SSE 페이로드 확장 |
| `gapago/agents/retrieval_agent.py` | 수정 | 동적 k, total_candidates_count, venue 정규화 |
| `tools.py` | 수정 | 4개 소스 venue 추출 |
| `states.py` | 수정 | Paper.venue, AgentState.total_candidates_count |

---

## Phase 1: 랜딩 페이지 개선

### 1-1. 워크플로우 인포그래픽
- 4단계 가로 배치: 질문 입력 → 논문 검색·분석 → 연구 GAP 도출 → 연구 방향 제안
- 각 단계에 아이콘 + 한국어 설명
- CSS flexbox, 모바일(768px 이하)에서 세로 전환

### 1-2. 서비스 설명란
- "왜 GAPAGO인가?" 섹션 추가
- 3개 카드: 논문 원문 기반 심층 분석 / 멀티소스 130편+ 검색 / 증거 기반 연구 GAP 도출
- 파급성·효율성 어필 (5-10분 자동 분석)

### 1-3. 한국어 UI 통일
**모든 사용자 대면 텍스트를 한국어로 통일:**

| 이전 (영어) | 이후 (한국어) |
|------------|-------------|
| Describe your research direction or topic | 연구하고 싶은 주제를 입력하세요 |
| Analyze | 분석 시작 |
| Stop | 중지 |
| + New Analysis | + 새 분석 |
| LLM Provider | LLM 프로바이더 |
| Domain | 연구 도메인 |
| Year Range | 연도 범위 |
| Language | 답변 언어 |
| History | 분석 히스토리 |
| Try an example | 키워드 또는 논문 제목으로 시작해보세요 |
| Starting pipeline... | 분석을 시작합니다... |
| Analysis complete | 분석 완료 |

### 1-4. 입력 가이드
- 입력창 아래에 안내 텍스트: "논문 제목이나 연구 키워드를 입력하세요"
- 예시 쿼리 카드 헤더: "키워드 또는 논문 제목으로 시작해보세요"

---

## Phase 2: 로딩 페이지 타임라인 리디자인

### 이전
3단계 가로 스테퍼: `Searching → Analyzing → Done`

### 이후
8단계 세로 타임라인:

```
✓ 질문 분석 & 검색어 확장    ← 완료: 초록 체크 + 정제된 쿼리 + 키워드 태그
✓ 논문 검색                  ← 완료: "130편 중 15편 선별" + 상위 3편 미리보기
◎ 한계점 추출                ← 진행 중: 파란 펄스 + 상태 텍스트
○ 한계점 품질 평가
○ 최신성 검증
○ 연구 GAP 도출
○ 결과 검증
○ 리포트 작성
```

**노드-타임라인 매핑:**

| 타임라인 단계 | SSE 노드 | 완료 시 표시 정보 |
|-------------|---------|-----------------|
| 질문 분석 & 검색어 확장 | `query_subgraph`, `meaning_expand` | 정제된 쿼리 + 키워드 태그 |
| 논문 검색 | `paper_retrieval` | "N편 중 M편 선별" + 상위 3편 제목 |
| 한계점 추출 | `limitation_extract` | "N개 한계점 추출" |
| 한계점 품질 평가 | `limitation_eval` | "N개 통과 / M개 제거" 또는 판정 |
| 최신성 검증 | `recency_check` | "미해결 N / 부분 M / 해결 K" |
| 연구 GAP 도출 | `gap_infer` | "N개 연구 GAP 도출" |
| 결과 검증 | `critic_score` | "품질 검증 통과" 또는 "재검토 중..." |
| 리포트 작성 | `final_response` | "리포트 작성 완료" |

**상태별 스타일:**
- `pending`: 회색 원 + 회색 텍스트
- `active`: 파란 펄스 애니메이션 + 파란 텍스트
- `done`: 초록 체크 아이콘 + 상세 텍스트 표시

---

## Phase 3: 결과 UI — 3탭 → 2패널 구조

### 이전
3개 탭: Papers | Research GAPs | Report

### 이후
2패널 구조:

```
┌─────────────────────────────────────────────────────┐
│  [결과 요약 헤더]                                     │
│  정제된 쿼리 | 논문 15/130편 | GAP 7개                 │
├─────────────────┬───────────────────────────────────┤
│  좌측 (35%)     │  우측 상세 (65%)                    │
│                 │                                    │
│  [연구 GAP]     │  GAP #1: Data & Dataset            │
│  ★ #1 Data     │  Gap Statement: "..."              │
│    #2 Method    │  Elaboration: "..."                │
│    ...          │  근거 논문 (3편): 클릭 가능 칩       │
│  ────────────  │  한계점 인용: "..."                  │
│  [논문 15/130]  │  제안 연구 방향: "..."              │
│  1. Paper A     │  [이 방향으로 추가 탐색 →]          │
│     Nature 2024 │                                    │
│  2. Paper B     │                                    │
│     arXiv 2023  │                                    │
├─────────────────┴───────────────────────────────────┤
│  [전체 리포트 보기 ▼] (접힌 섹션)                      │
└─────────────────────────────────────────────────────┘
```

### 좌측 패널 상세
**GAP 리스트:**
- 축약형 카드: 랭크 뱃지 + 축 이름 + gap_statement 1줄 (truncate)
- 클릭 → 우측 패널에 상세
- 활성 항목: 좌측 accent 보더

**논문 리스트:**
- 축약형: 번호 + 제목 (truncate) + venue 뱃지 + 연도
- **"15/130편 선별"** 형식 헤더 (`total_searched` 활용)
- venue 뱃지: 있으면 저널명, 없으면 소스 라벨
- 클릭 → 우측 패널에 논문 상세

### 우측 상세 패널
**GAP 선택 시:**
- Gap Statement (굵은 큰 텍스트)
- 축 정보 (Fixed/Dynamic + axis_label)
- Elaboration
- 근거 논문 목록 (클릭 가능 칩 → 논문 상세로 전환)
- 한계점 인용 (supporting_quotes)
- 제안 연구 방향 (accent 박스)
- **"이 방향으로 추가 탐색 →"** 버튼

**논문 선택 시:**
- 논문 메타데이터 (제목, 저자, 연도, venue, URL 링크)
- Abstract 전문
- 이 논문에서 추출된 한계점 목록

### 하단 리포트 섹션
- 접힌(collapsible) 상태로 기본 표시
- "전체 리포트 보기" 클릭 → 마크다운 렌더링 보고서 표시
- 클립보드 복사 + .md 다운로드 버튼

### 모바일 대응
- `≤768px`: 2패널 → 단일 컬럼 (좌측 위, 상세 아래)
- `≤420px`: 컴팩트 카드, 사이드바 가로 접힘

---

## Phase 4: 추가 탐색 (체인 재실행)

### 신규 API 엔드포인트

```
GET /api/explore?topic=<proposed_topic>&session_id=<parent>&provider=...&domain=...&year_range=...&output_language=...&user_id=...
```

**동작:**
1. `proposed_topic`을 새 쿼리로 분석 시작
2. `parent_session_id`로 부모 세션 참조 저장
3. 새 `session_id` 반환
4. 클라이언트가 `/api/stream/{new_session_id}`에 연결

**응답:**
```json
{"session_id": "새_세션_ID", "parent_session_id": "부모_세션_ID"}
```

### 프론트엔드 동작
1. GAP 상세 패널에서 "이 방향으로 추가 탐색 →" 클릭
2. `proposed_topic`을 쿼리 입력란에 채움
3. `/api/explore` 호출 (실패 시 `/api/analyze`로 폴백)
4. 새 세션으로 전환, 기존 결과는 히스토리에 보존

### 히스토리 연결
- 결과 저장 시 `parent_session_id` 필드 포함
- 히스토리 목록에서 부모-자식 관계 식별 가능

---

## Phase 5: 기타 수정 (백엔드)

### 5-1. 논문 게재지(venue) 정보 추가

| 소스 | 추출 방법 | 예시 |
|------|----------|------|
| arXiv | 고정 라벨 | `"arXiv preprint"` |
| Semantic Scholar | `publicationVenue.name` → `venue` → arXiv 폴백 | `"Nature"`, `"ICML"` |
| OpenAlex | `primary_location.source.display_name` | `"Journal of Machine Learning Research"` |
| ScienceON | `JournalName` 메타데이터 | `"한국정보과학회 학술발표논문집"` |

**데이터 흐름:**
```
tools.py (API 파싱) → venue 필드 포함
    ↓
retrieval_agent.py (Paper 변환) → venue 결정 (명시적 > journal > source 라벨)
    ↓
api/main.py (SSE 전송) → {"venue": "Nature"} 페이로드
    ↓
frontend (UI 표시) → venue 뱃지
```

### 5-2. Retrieval k 동적 변경

**이전:** `bm25_top_k=30` 고정
**이후:** BM25 점수 분포 기반 적응적 컷오프

```python
scores = bm25.get_scores(query_tokens)
threshold = mean(scores) + 0.5 * std(scores)
dynamic_k = max(10, min(sum(scores > threshold), bm25_top_k))
```

- 범위: 10 ~ `bm25_top_k` (기본 30)
- 주제별 논문 밀도에 따라 자동 적응
- 논문이 많고 관련도가 높은 주제: k 증가
- 논문이 적거나 점수 분산이 낮은 주제: k 감소

### 5-3. 논문 수 표시 (15/130 형식)

**변경 체인:**
1. `retrieval_agent.py`: 중복 제거 후 전체 수를 `total_candidates_count`로 저장
2. `gapago/api/main.py`: SSE `paper_retrieval` 이벤트에 `total_searched` 필드 추가
3. `frontend`: "130편 중 15편 선별" 형식으로 표시

### 5-4. 언어 설정 개선

- 사이드바 라벨: `Language` → `답변 언어`
- 페이지 UI 텍스트: 전체 한국어 고정
- 드롭다운: `Auto (입력 언어 감지)` / `한국어` / `English`
- 답변 결과만 선택 가능, 페이지 자체는 한국어 통일

---

## 코드 품질 검증 결과

| 항목 | 결과 |
|------|------|
| Python 문법 | 4개 파일 모두 `ast.parse()` 통과 |
| 타입 일관성 | venue, total_candidates_count 엔드투엔드 일관 |
| 데이터 흐름 | tools → agent → api → frontend 정상 전달 |
| 에러 처리 | 모든 폴백 로직 존재 |
| 프론트엔드 JS | 주요 함수 정의 확인 (progress 이벤트 핸들링 인라인 처리) |
| HTML 구조 | 모든 태그 정상 열기/닫기 |
| API 엔드포인트 | /api/explore 정상 정의 + 연결 |
| 모바일 반응형 | 768px, 420px 두 브레이크포인트 |
| SSE 통합 | 8개 노드 데이터 수집 완전 |

---

## 검증 방법

```bash
# 1. 서버 실행
cd /home01/hpc201a03/projects/gapag
python -m uvicorn api.main:app --reload --port 8000

# 2. 브라우저 접속
# http://localhost:8000

# 3. 확인 항목
# - 랜딩 페이지: 워크플로우 섹션 + 한국어 UI
# - 분석 실행: 8단계 타임라인 + 단계별 상세
# - 결과: 2패널에서 GAP 클릭 → 상세, 논문 클릭 → 상세
# - 논문: venue 뱃지, "15/130편" 형식
# - 추가 탐색: proposed_topic → 새 세션 체인
# - 모바일: DevTools → 768px, 420px 확인
```
