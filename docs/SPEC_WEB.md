# GAPAGO 웹 프론트엔드 스펙

## 1. 개요

GAPAGO의 웹 프론트엔드는 연구 갭 분석 파이프라인의 사용자 인터페이스를 제공한다. 세 가지 구현이 존재하며, 메인 프론트엔드는 Vanilla JS SPA(`frontend/index.html`)이고, Streamlit(`app.py`)과 Gradio(`app_gradio.py`)는 대안 구현이다.

---

## 2. 기술 스택

| 구현 | 프레임워크 | 파일 | 용도 |
|------|-----------|------|------|
| **메인 웹 UI** | HTML5 + CSS3 + Vanilla JavaScript | `frontend/index.html` | 프로덕션 SPA (1,325줄) |
| **대안 UI 1** | Streamlit (Python) | `app.py` | Python 네이티브 웹 UI |
| **대안 UI 2** | Gradio (Python) | `app_gradio.py` | 경량 인터페이스 |

- 외부 CSS/JS 프레임워크 없음 (React, Vue, Svelte 미사용)
- 실시간 통신: Server-Sent Events (SSE)
- 세션 저장: localStorage
- 백엔드 API: FastAPI (`api/main.py`)

---

## 3. 메인 웹 UI (`frontend/index.html`)

### 3.1 레이아웃 구조

```
+------------------+--------------------------------------------+
|    사이드바       |              메인 영역                      |
|   (272px 고정)   |                                            |
|                  |   +------------------------------------+   |
|  [로고]          |   |        입력 영역 (상단)              |   |
|  [새 분석]       |   |  [연구 질문 입력] [분석/중지 버튼]   |   |
|                  |   +------------------------------------+   |
|  [설정]          |   |                                    |   |
|  - LLM 프로바이더 |   |        결과 영역 (스크롤)           |   |
|  - 연구 도메인    |   |  (상태에 따라 동적 콘텐츠)          |   |
|  - 연도 범위     |   |                                    |   |
|  - 출력 언어     |   |                                    |   |
|                  |   +------------------------------------+   |
|  [분석 히스토리]  |                                            |
+------------------+--------------------------------------------+
```

### 3.2 사이드바 상세

#### 로고 및 브랜딩
- `new_logo.png` 표시
- "GAPAGO" 타이틀

#### 새 분석 버튼
- 전체 상태 초기화
- 결과 영역을 빈 상태로 복원

#### 설정 (4개 드롭다운)

| 설정 | 옵션 | 기본값 |
|------|------|--------|
| LLM Provider | `azure`, `claude`, `exaone` | `azure` |
| Research Domain | `auto`, `ai_cs`, `biomedical`, `materials_chemistry`, `physics`, `general` | `auto` |
| Year Range | `auto`, `1y`, `3y`, `5y` | `auto` |
| Output Language | `auto`, `ko`, `en` | `auto` |

#### 분석 히스토리
- 과거 분석 목록 (스크롤)
- 각 항목: 쿼리 미리보기, 타임스탬프, 상태 표시
- 클릭 시 저장된 결과 로드

### 3.3 페이지 상태 (5가지)

#### 상태 1: 초기 (Empty State)
- 히어로 이미지 (`middle_image.png`)
- 설명 텍스트 (한국어/영어)
- **예시 쿼리 카드** (6개 카테고리, 각 2개 쿼리)

| 카테고리 | 예시 |
|---------|------|
| AI/ML | 대규모 언어모델 관련 |
| Biomedical | 의생명 연구 관련 |
| Materials | 소재/화학 관련 |
| Robotics | 로봇공학 관련 |
| Environment | 환경/에너지 관련 |
| Quantum | 양자 컴퓨팅 관련 |

- 예시 카드 클릭 시 해당 쿼리가 입력란에 자동 채워짐

#### 상태 2: 파이프라인 실행 중

**3단계 스테퍼(Stepper) 표시:**

| 단계 | 이름 | 포함 노드 |
|------|------|----------|
| Stage 1 | Searching | `query_subgraph`, `meaning_expand`, `paper_retrieval` |
| Stage 2 | Analyzing | `limitation_extract`, `limitation_eval`, `recency_check`, `gap_infer`, `critic_score` |
| Stage 3 | Done | `final_response` |

- 실시간 상태 메시지 (애니메이션 씽킹 표시)
- 경과 시간 카운터
- 중지(Stop) 버튼

#### 상태 3: 인터럽트/명확화 (Clarification)
- 노란색 경고 카드 표시
- AI가 생성한 명확화 프롬프트 표시
- 사용자 응답 입력 필드
- 두 개 버튼:
  - **Continue**: 사용자 응답으로 파이프라인 재개
  - **Skip**: 명확화 생략, 강제 진행

#### 상태 4: 결과 표시 (탭 기반)

**Papers 탭:**
- 검색된 논문 테이블
- 컬럼: # | Title (링크) | Year | Authors
- GAPs에서 논문 참조 시 해당 행 하이라이트

**Research GAPs 탭 (기본 활성):**
- GAP 카드 구성:
  - 랭크 뱃지 (TOP GAP / GAP #N)
  - 축 유형 (Fixed/Dynamic) + 라벨
  - 논문 수
  - Gap Statement (볼드, 강조)
  - Elaboration 텍스트
  - Proposed Topic (컬러 박스 + 아이콘)
  - Supporting Papers (클릭 시 Papers 탭으로 이동)
- TOP GAP: 노란색 좌측 보더로 특별 스타일링

**Final Report 탭:**
- 마크다운 렌더링 (헤딩, 볼드/이탤릭, 코드 블록, 테이블, 리스트, 블록쿼트)
- **클립보드 복사** 버튼
- **마크다운 다운로드** (.md 파일) 버튼

#### 상태 5: 저장된 결과 보기
- 히스토리에서 로드 시 표시
- 쿼리, 정제된 쿼리, 타임스탬프 표시
- 모든 탭 기능 정상 작동

### 3.4 실시간 스트리밍 (SSE)

```
클라이언트                                    서버
   |                                           |
   |--- GET /api/analyze?query=...  --------->|
   |<-- { session_id: "abc123" }  ------------|
   |                                           |
   |--- GET /api/stream/abc123  ------------->|
   |<== SSE: {"event":"node","node":"query_subgraph",...} |
   |<== SSE: {"event":"node","node":"paper_retrieval",...} |
   |<== SSE: {"event":"interrupt","clarify_prompt":"..."} |
   |                                           |
   |--- GET /api/clarify?session_id=...&response=... -->|
   |                                           |
   |--- GET /api/stream/abc123?from_idx=5 --->|
   |<== SSE: {"event":"node","node":"gap_infer",...} |
   |<== SSE: {"event":"complete","filename":"..."} |
   |                                           |
```

- **Keepalive**: 30초 타임아웃 방지를 위한 heartbeat 이벤트
- **재연결**: `from_idx` 파라미터로 이벤트 재생 지원
- **페이지 새로고침 복구**: localStorage의 `gapago_active_session`으로 실행 중인 분석에 재연결

### 3.5 세션 관리

| 저장 키 | 용도 |
|---------|------|
| `gapago_active_session` | 현재 실행 중인 세션 ID |
| `gapago_user_id` | 사용자 식별자 |

- 페이지 새로고침 시 자동 복구
- 분석 완료 시 세션 정보 클리어
- 히스토리는 서버 API(`/api/history`)에서 조회

### 3.6 사용자 인터랙션 플로우

```
초기 상태
  → 연구 질문 입력 + 설정 구성
  → [분석] 클릭
  → SSE 스트림 시작
  → 스테퍼 + 실시간 진행 표시
  → [인터럽트?]
     ├─ 명확화 카드 표시 → 응답 입력 → [Continue/Skip]
     └─ (인터럽트 없음) → 계속 진행
  → 결과 탭 표시
  → Papers | Research GAPs | Final Report
  → [새 분석] 또는 히스토리에서 이전 결과 조회
```

---

## 4. Streamlit UI (`app.py`)

### 4.1 구조

**사이드바:**
- 앱 타이틀 + 브랜딩
- "New Analysis" 버튼
- Settings 확장 패널:
  - LLM Provider 드롭다운
  - Research Domain 드롭다운
- 분석 히스토리 섹션

**메인 콘텐츠:**
- 헤더: "GAPAGO — Research GAP Analyzer"
- 쿼리 입력 텍스트 필드
- 실행 버튼

### 4.2 파이프라인 시각화

9개 노드별 확장 가능한 섹션(Expander):

| 노드 | 표시 내용 |
|------|----------|
| Query Analysis | 정제된 쿼리, 키워드, 범위 수준 |
| Paper Retrieval | 논문 수 + DataFrame 테이블 |
| Limitation Extraction | 한계점 수 + 포맷된 카드 |
| Limitation Evaluation | 판정 뱃지, 품질 점수 바 차트, 유형 분포 차트 |
| Recency Check | Unresolved/Partial/Resolved 메트릭 |
| GAP Inference | GAP 카드 (별점 랭킹, 축 라벨) |
| Critic Score | 코드 블록 출력 |
| Final Response | 마크다운 보고서 |

### 4.3 Human-in-the-Loop UI
- AI 명확화 프롬프트 표시
- 보충 답변 입력 필드
- Resume / Skip 버튼

### 4.4 결과 저장
- 자동 저장: `outputs/gapago_result_YYYYMMDD_HHMMSS.json`
- 히스토리에서 과거 결과 로드 가능

---

## 5. Gradio UI (`app_gradio.py`)

### 5.1 구조

**입력 영역:**
- 쿼리 텍스트박스 (2줄)
- LLM Provider 드롭다운 (`azure`, `claude`, `gemini`)
- Research Domain 드롭다운
- Analyze 버튼

**탭 구성:**

| 탭 | 내용 |
|----|------|
| Progress | 실시간 파이프라인 상태 (마크다운) |
| Papers | 검색된 논문 테이블 (마크다운) |
| Research GAPs | 식별된 갭 상세 (마크다운) |
| Final Report | 최종 분석 보고서 (마크다운) |

### 5.2 특징
- `yield` 패턴으로 스트리밍 출력
- 실시간 진행 콜백
- 결과 자동 저장: `/tmp/gapago_outputs/`

---

## 6. 디자인 시스템

### 6.1 색상 팔레트 (CSS 변수)

| 변수 | 값 | 용도 |
|------|-----|------|
| Primary Accent | `#5469d4` | 버튼, 링크, 활성 상태 |
| Success | `#0fba81` | 완료, 성공 상태 |
| Warning | `#e8920b` | 경고, 인터럽트 |
| Error | `#e54e4e` | 오류 상태 |
| Background | `#f5f6fa` | 페이지 배경 |
| Surface | `#ffffff` | 카드, 패널 배경 |

### 6.2 반응형 디자인

| 브레이크포인트 | 변경 사항 |
|--------------|----------|
| `≤ 768px` | 사이드바 접힘, 모바일 레이아웃 |
| `≤ 420px` | 컴팩트 레이아웃, 폰트 축소 |
| `> 768px` | 데스크톱 전체 레이아웃 |

### 6.3 마크다운 렌더링
- HTML 이스케이핑 (XSS 방지)
- 지원 요소: 헤딩, 볼드/이탤릭, 코드 블록, 테이블, 리스트, 블록쿼트
- 테이블: 파이프(`|`)와 구분 행 사용

---

## 7. 정적 자산

| 파일 | 크기 | 용도 |
|------|------|------|
| `frontend/logo.png` | 96 KB | 브랜드 로고 |
| `frontend/new_logo.png` | 27 KB | 업데이트된 로고 |
| `frontend/middle_image.png` | 23 KB | 초기 화면 히어로 이미지 |
| `logo.png` (루트) | - | API 정적 서빙용 |

---

## 8. 구현별 비교

| 기능 | 메인 (Vanilla JS) | Streamlit | Gradio |
|------|-------------------|-----------|--------|
| 실시간 스트리밍 | SSE (네이티브) | 그래프 실행 콜백 | yield 스트리밍 |
| 세션 관리 | localStorage | st.session_state | 없음 |
| 히스토리 | API 기반 | 파일 기반 | 없음 |
| 반응형 | 3 브레이크포인트 | Streamlit 기본 | Gradio 기본 |
| Human-in-the-Loop | SSE interrupt | Interrupt UI | 미지원 |
| 결과 내보내기 | 복사/다운로드 | JSON 저장 | JSON 저장 |
| 배포 | FastAPI 정적 서빙 | `streamlit run` | `gradio launch` |
