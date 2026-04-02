# GAPAGO 웹 프론트엔드 스펙

> **최종 업데이트:** 2026-04-02 (웹 UI 개선사항 반영)

## 1. 개요

GAPAGO의 웹 프론트엔드는 연구 갭 분석 파이프라인의 사용자 인터페이스를 제공한다. 세 가지 구현이 존재하며, 메인 프론트엔드는 Vanilla JS SPA(`frontend/index.html`)이고, Streamlit(`app.py`)과 Gradio(`app_gradio.py`)는 대안 구현이다.

**주요 변경 (2026-04-02):**
- Cytoscape.js 제거 → 바닐라 JS/SVG 3-column 관계도 복원
- 라우팅 프로파일 선택 UI 추가 (optimized/quality), 비-balanced시 provider 자동 할당
- Fast Mode 툴팁 추가, provider 라벨 Groq→Qwen3-32B
- interrupt 체크포인트 버그 수정, 채팅 버블 여백 확대

**주요 변경 (2026-04-01):**
- 용어 통일: 쿼리→질문, Research GAP→연구 GAP, 검색어 확장→확장
- 클립보드 복사 제거 → .md/.docx 다운로드 버튼으로 교체
- 한계점 평가 차트, GAP Axis 클러스터링, GAP Axis 도넛 차트 제거 (불필요한 시각화 정리)

**주요 변경 (2026-03-28):**
- 전체 UI 텍스트 한국어 통일
- 3탭 → 2패널 결과 구조 (좌: GAP/논문 리스트, 우: 상세 패널)
- 3단계 스테퍼 → 8단계 세로 타임라인
- 랜딩 페이지 워크플로우 인포그래픽 + 서비스 설명 추가
- 추가 탐색 (체인 재실행) 기능
- 논문 게재지(venue) 표시, 15/130편 형식 표시

---

## 2. 기술 스택

| 구현 | 프레임워크 | 파일 | 용도 |
|------|-----------|------|------|
| **메인 웹 UI** | HTML5 + CSS3 + Vanilla JavaScript | `frontend/index.html` | 프로덕션 SPA (~2,700줄) |
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
- `new_logo.png` 표시 (클릭 시 홈 화면으로 이동)
- "GAPAGO" 타이틀

#### 새 분석 버튼
- 전체 상태 초기화
- 결과 영역을 빈 상태로 복원

#### 설정

| 설정 | 옵션 | 기본값 | 비고 |
|------|------|--------|------|
| 분석 모드 (Routing Profile) | `optimized`, `quality` | `optimized` | 에이전트별 LLM 자동 배정 |
| LLM Provider | `azure`, `claude`, `exaone` | `azure` | **비-balanced 프로파일에서는 숨김 (자동 할당)** |
| Year Range | `auto`, `1y`, `3y`, `5y` | `auto` | |
| Output Language | `auto`, `ko`, `en` | `auto` | |
| Fast Mode | 체크박스 (on/off) | off | 툴팁: "(빠른 분석, 품질 트레이드오프)" |

> **Note:** 연구 도메인(Domain) 드롭다운은 제거됨 (`auto` 고정)
> **Fast Mode:** CrossEncoder 리랭킹 스킵, 상위 3개 축만 분석 — 빠른 결과 제공
> **라우팅 프로파일:** `optimized`=에이전트별 최적화(경량→groq, 핵심→claude), `quality`=최고 품질(Claude 활용)

#### 분석 히스토리
- 과거 분석 목록 (스크롤)
- 각 항목: 쿼리 미리보기, 타임스탬프, 상태 표시
- 클릭 시 저장된 결과 로드

### 3.3 페이지 상태 (5가지)

#### 상태 1: 초기 (Empty State)
- 히어로 이미지 (`middle_image.png`)
- 설명 텍스트 (한국어/영어)
- **분석 소요 시간 안내 문구**: "🧠 단순 검색이 아닌 논문 원문 기반 심층 분석을 수행합니다. ⏱️ 정확한 응답을 위해 5~10분이 소요될 수 있습니다."
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

**8단계 세로 타임라인 표시** (`TIMELINE_STEPS`):

| 타임라인 단계 | SSE 노드 | 완료 시 표시 정보 |
|-------------|---------|-----------------|
| 질문 분석 & 검색어 확장 | `query_subgraph`, `meaning_expand` | 정제된 쿼리 + 키워드 태그 |
| 논문 검색 | `paper_retrieval` | "N편 중 M편 선별" + 상위 3편 제목 |
| 한계점 추출 | `limitation_extract` | "N개 한계점 추출" |
| 한계점 품질 평가 | `limitation_eval` | "N개 통과" 또는 판정 |
| 최신성 검증 | `recency_check` | "미해결 N / 부분 M / 해결 K" |
| 연구 GAP 도출 | `gap_infer` | "N개 연구 GAP 도출" |
| 결과 검증 | `critic_score` | "품질 검증 통과" 또는 "재검토 중..." |
| 리포트 작성 | `final_response` | "리포트 작성 완료" |

**상태별 스타일:**
- `pending`: 회색 원 + 회색 텍스트
- `active`: 파란 펄스 애니메이션 + 파란 텍스트
- `done`: 초록 체크 아이콘 + 상세 텍스트 표시

- 실시간 상태 메시지 (애니메이션 씽킹 표시 + `THINKING_MESSAGES` 순환)
- 경과 시간 카운터
- 중지(Stop) 버튼
- **타임라인 자동 접힘**: 분석 완료(`complete` 이벤트) 시 `.timeline.collapsing` → `.timeline.collapsed` 애니메이션
- **progress 이벤트 처리**: 노드 내 중간 진행률을 타임라인 detail에 실시간 반영

#### 상태 3: 인터럽트/명확화 (Clarification)
- 노란색 경고 카드 표시
- AI가 생성한 명확화 프롬프트 표시
- 사용자 응답 입력 필드
- 두 개 버튼:
  - **Continue**: 사용자 응답으로 파이프라인 재개
  - **Skip**: 명확화 생략, 강제 진행

#### 상태 4: 결과 표시 (2패널 구조)

```
┌─────────────────────────────────────────────────────┐
│  [분석 옵션 바] LLM: azure | 연도: auto | 언어: auto │
│  [결과 요약 헤더]                                     │
│  정제된 쿼리: ... | 논문 15/130편 | GAP 7개            │
├─────────────────┬───────────────────────────────────┤
│  좌측 (35%)     │  우측 상세 (65%)                    │
│                 │                                    │
│  [연구 GAP (7)] │  GAP #1 상세:                      │
│  ★ TOP Data    │  - 축 뱃지 + 논문 수               │
│    #2 Method    │  - Gap Statement (굵은 큰 텍스트)   │
│    ...          │  - Elaboration                     │
│  ────────────  │  - 근거 논문 (클릭 가능 칩)          │
│  [논문 15/130]  │  - 근거 인용                        │
│  1. Paper A     │  - 제안 연구 방향 (accent 박스)     │
│     Nature 2024 │  - [이 방향으로 추가 탐색 →]        │
│  2. Paper B     │                                    │
│     arXiv 2023  │                                    │
├─────────────────┴───────────────────────────────────┤
│  [전체 리포트 보기 ▼] (접힌 섹션)                      │
└─────────────────────────────────────────────────────┘
```

**좌측 패널 (`panel-left`, 35%):**
- **GAP 리스트** (`gap-item-compact`):
  - 랭크 뱃지 (TOP / #N) + 축 뱃지 (색상 자동 할당 `getAxisColor`)
  - gap_statement 2줄 truncate
  - 클릭 → 우측 패널에 상세 (`selectGap`)
  - 활성 항목: 좌측 accent 보더 (`.active`)
- **논문 리스트** (`paper-item-compact`):
  - 번호 + 제목 (truncate) + venue 뱃지 + 연도
  - **"N/M편 선별"** 형식 헤더 (`total_searched` 활용)
  - 클릭 → 우측 패널에 논문 상세 (`selectPaper`)

**우측 상세 패널 (`panel-right`, 65%):**
- **GAP 선택 시** (`getGapDetailHTML`):
  - 축 뱃지 + "N편 논문에서 도출"
  - Gap Statement (굵은 큰 텍스트)
  - Elaboration
  - 근거 논문 목록 (클릭 가능 칩 → 논문 상세로 전환)
  - 근거 인용 (supporting_quotes)
  - 제안 연구 방향 (accent 박스)
  - **"이 방향으로 추가 탐색 →"** 버튼 (`exploreDirection`)
- **논문 선택 시** (`getPaperDetailHTML`):
  - 논문 메타데이터 (제목, 저자, 연도, venue 뱃지, URL 링크)
  - Abstract 전문
  - 이 논문에서 추출된 한계점 목록

**하단 리포트 섹션** (`report-collapse`):
- 접힌(collapsible) 상태로 기본 표시
- "전체 리포트 보기" 클릭 → 마크다운 렌더링 보고서 표시 (`toggleReport`)
- **.md 다운로드** + **.docx 다운로드** 버튼 (타임스탬프 파일명: `GAPAGO_report_YYYYMMDD_HHMMSS.{md|docx}`)
  - Markdown: 분석 옵션, 정제된 질문, 확장 키워드, 요약 통계, AI 생성 리포트 포함
  - DOCX: Markdown → HTML 변환 후 Word 문서 생성

**관계도 (바닐라 JS/SVG 3-column 시각화):**
- 결과 요약 바 아래에 표시, 접힘/펼침(collapse/expand) 토글 지원
- **3-column 레이아웃:** 논문(Papers) | 한계점(Limitations) | 연구 GAP(GAPs)
- **노드 스타일:**
  - 논문: 파란 사각형
  - 한계점: 노란/주황 사각형
  - 연구 GAP: 축 색상별 사각형
- **엣지:** SVG path + 그라데이션 (논문→한계점: 파란→주황, 한계점→GAP: 파란→보라)
- **인터랙션:**
  - 호버: BFS 기반 이웃 하이라이트 (3-depth), 비관련 노드 dim 처리
  - 줌 3단계 지원, 진입 애니메이션, drop-shadow
- **Cytoscape.js 의존성 제거** — 순수 SVG로 교체

#### 상태 5: 결과 기반 채팅

분석 완료 후 결과에 대해 AI와 대화할 수 있는 기능:

- 결과 화면 하단 또는 별도 영역에 채팅 입력 UI 표시
- `POST /api/chat` 엔드포인트로 질문 전송
- GAP/limitation/papers 컨텍스트를 활용한 답변 생성
- 세션별 대화 히스토리 유지 (최대 100개 메시지)
- 저장된 결과에서도 채팅 가능 (`filename` 파라미터 사용)

#### 상태 6: 저장된 결과 보기
- 히스토리에서 로드 시 표시
- 쿼리, 정제된 쿼리, 타임스탬프 표시
- 2패널 구조 + 리포트 접힌 섹션 정상 작동

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
- **Progress**: 노드 내 중간 진행률 이벤트 (`event: "progress"`) — 타임라인 detail에 실시간 반영
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
  → 8단계 세로 타임라인 + 실시간 진행 표시
  → [인터럽트?]
     ├─ 명확화 카드 표시 → 응답 입력 → [Continue/Skip]
     └─ (인터럽트 없음) → 계속 진행
  → 2패널 결과 표시
  → 좌측: GAP/논문 리스트 | 우측: 상세 패널 | 하단: 접힌 리포트
  → [추가 탐색] 또는 [새 분석] 또는 히스토리에서 이전 결과 조회
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
- **테이블 separator 정규식**: 유니코드 box-drawing 문자 대응 (`/^[-─━═┄┅┈┉╌╍—–:\s]+$/`)
- **`richText()` 함수**: GAP 카드 등에서 볼드, 이탤릭, 줄바꿈을 HTML로 변환

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
| 결과 내보내기 | .md/.docx 다운로드 | JSON 저장 | JSON 저장 |
| 배포 | FastAPI 정적 서빙 | `streamlit run` | `gradio launch` |
