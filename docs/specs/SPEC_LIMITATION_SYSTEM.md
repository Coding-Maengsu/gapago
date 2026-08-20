# 한계점 추출·평가 시스템 기술 명세

## 1. 시스템 개요

한계점 시스템은 2개 에이전트로 구성됨:
- **Limitation Extraction Agent** (`gapago/agents/limitation_agent.py`, 1663줄): 논문 전문 획득 → 섹션 분할 → 한계점 추출
- **Limitation Eval Agent** (`gapago/agents/limitation_eval_agent.py`, 549줄): 2-Call 파이프라인으로 품질 검증 + PASS/RETRY 결정

핵심 원칙: **Full text 전용** — abstract fallback 없음. full text 획득 실패 논문은 제거 후 backup 논문으로 대체.

---

## 2. 전문(Full Text) 획득 시스템

### 2-1. 캐시 시스템

| 항목 | 값 |
|------|-----|
| 저장 경로 | `.cache/fulltext/` |
| 캐시 키 | SHA256(paper_id) → 앞 16자리 hex |
| 성공 TTL | 7일 (`_CACHE_TTL_DAYS = 7`) |
| 실패 TTL | 6시간 (`_CACHE_FAIL_TTL_HOURS = 6`) |
| 형식 | JSON (UTF-8) |

- 실패 시 빈 dict를 저장하여 "시도했으나 실패" 마킹
- 캐시 히트 시 전문 로딩 스킵

### 2-2. HTTP 세션

- `requests.Session()` 기반 커넥션 풀링
- User-Agent: `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36`
- Referer: `https://scholar.google.com/`

### 2-3. 8단계 폴백 체인

논문 유형별로 다른 폴백 경로를 따름:

#### arXiv 논문 (paper_id가 `arxiv:`로 시작)

| 순서 | 방법 | 엔드포인트 | 타임아웃 |
|------|------|-----------|---------|
| 1 | ar5iv HTML | `https://ar5iv.labs.arxiv.org/html/{id}` | 8s |
| 2 | arXiv PDF | LangChain `ArxivLoader` → pymupdf | - |

- ar5iv: BeautifulSoup로 파싱, script/style/nav/footer 제거
- 최소 텍스트 임계값: **200자**

#### DOI 기반 논문

| 순서 | 방법 | 상세 | 타임아웃 |
|------|------|------|---------|
| 1 | Direct PDF | API 응답의 `pdf_url` 직접 다운로드 | 15s |
| 2 | DOI 페이지 PDF 링크 | DOI 랜딩 페이지에서 정규식으로 PDF URL 탐색 | 10s |
| 3 | DOI 랜딩 HTML | 페이지 본문 직접 추출 | 10s |
| 4 | Unpaywall API | OA URL 탐색 (PMC > biorxiv > repository > publisher 순) | - |
| 5 | Semantic Scholar Batch API | alt ID 발견 (arxiv_id, pmcid, oa_pdf_url) | 10s |
| 6 | PMC BioC API | `https://ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode` | 15s |
| 7 | EuropePMC JATS XML | `https://ebi.ac.uk/europepmc/webservices/rest/search` → fullTextXML | 8-15s |
| 8 | Direct PDF (pymupdf4llm) | PDF 다운로드 → `pymupdf4llm.to_markdown()` | 15s |

**PDF 처리 제한:**
- 최대 PDF 크기: **10 MB** (`_MAX_PDF_BYTES`)
- PDF 차단 호스트: `api.elsevier.com`, `mdpi.com`, `tandfonline.com`, `igi-global.com`, `peerj.com`

**Unpaywall 우선순위:**
1. PMC (PubMed Central) — 최우선
2. biorxiv/medrxiv 프리프린트
3. Repository (호스트 필터링 적용)
4. Publisher (폴백)
- 스킵 호스트: `doaj.org`, `dx.doi.org`, `doi.org`
- 이메일: `gapago-research@university.edu`

**Semantic Scholar API:**
- Rate limiting: **1 req/s** (글로벌 락)
- 429 응답 시 **3초** 대기
- 발견된 alt ID로 추가 폴백: arxiv_id → ar5iv, pmcid → PMC BioC, oa_pdf_url → 다운로드

**NCBI ID Converter:**
- 엔드포인트: `https://ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/`
- DOI → PMCID 변환 (타임아웃 5s)

#### ScienceON 논문/특허/보고서
- DOI 있으면 DOI 폴백 체인으로 전달
- 특허/보고서: `FulltextURL` / `ContentURL`에서 PDF 직접 다운로드

### 2-4. HTML 본문 추출

- 제거 태그: script, style, nav, footer, header, aside, figure, table, sup, math
- 참고문헌 섹션 제거: id/class에 `ref|bib|citation` 매칭
- 본문 탐지 우선순위:
  1. `div.article-content` (PeerJ)
  2. `div.html-body` (MDPI)
  3. `article` 태그
  4. 정규식: `article|paper|fulltext|main-content`
- 최소 추출 텍스트: **500자**

---

## 3. 섹션 분할

### 3-1. 분할 정규식

```
(?m)^(?:#{1,4}\s+)?(?:\d+[\.\d]*\.?\s+)?(?:[^\n]{0,30}?\b)(KEYWORDS)s?\b[^\n]{0,60}$
```

- Markdown 헤딩 (`#` ~ `####`) 지원
- 번호 섹션 (`1.`, `1.1.`) 지원
- 키워드 앞 30자, 뒤 60자까지 허용
- 복수형 자동 매칭 (`s?`)
- 대소문자 무관
- 헤딩 최대 길이: **100자**
- 섹션 최소 내용: **100자**

### 3-2. 섹션 키워드 (7종)

| 섹션 | 키워드 | 트랙 |
|------|--------|------|
| conclusion | conclusion, concluding remarks, summary | Track 1 |
| limitations | limitation, limitations, weakness | Track 1 |
| future_work | future work, future research | Track 1 |
| introduction | introduction, background, motivation | Track 2 |
| method | method, methods, methodology, approach | Track 2 |
| experiment | experiment, experiments, evaluation | Track 2 |
| discussion | discussion, analysis, ablation | Track 2 |

### 3-3. 제한값

| 항목 | 값 |
|------|-----|
| 섹션당 최대 크기 | **3,000자** (`MAX_SECTION_CHARS`) |
| 최소 full text 합산 | **500자** (미만 시 실패 처리) |

---

## 4. Backup 논문 대체

- full text 획득 실패 시 retrieval에서 확보한 backup 풀에서 대체
- arXiv 논문 우선 정렬 (BM25 순)
- 1:1 대체: 실패 1편당 backup 1편 시도
- 대체 불가 시 해당 슬롯 제거
- 로그: `[fulltext_filter] {X}편 실패: Y편 대체, Z편 제거`

---

## 5. Dual-Track 추출

LLM에게 논문의 섹션 텍스트를 전달하여 한계점을 추출하는 단계. 논문 하나에 대해 Track 1과 Track 2 섹션을 **동시에** 하나의 프롬프트로 전달하며, LLM이 양쪽을 모두 읽고 한계점을 도출함.

### 5-1. 프롬프트 구성 (`_build_prompt`)

각 논문에 대해 다음 구조의 프롬프트를 생성:

```
paper_id: arxiv:2401.12345
title: Example Paper Title
year: 2024

## Track 1: Author-Stated Sections
### [CONCLUSION]
(conclusion 섹션 텍스트, 최대 3000자)

### [LIMITATIONS]
(limitations 섹션 텍스트, 최대 3000자)

### [FUTURE_WORK]
(future_work 섹션 텍스트, 최대 3000자)

## Track 2: Structural Analysis Sections
### [INTRODUCTION]
(introduction 텍스트)

### [METHOD]
(method 텍스트)

### [EXPERIMENT]
(experiment 텍스트)

### [DISCUSSION]
(discussion 텍스트)
```

- Track 1과 Track 2 중 하나라도 있으면 추출 진행
- 양쪽 모두 비어있으면 해당 논문 스킵 (`None` 반환)

### 5-2. Track 1 — 저자 명시 한계점

- **대상 섹션**: conclusion, limitations, future_work
- **LLM 지시**: "저자가 명시적으로 인정한 한계점을 추출하라"
- **메타 지시**: "저자는 방어적으로 작성하는 경향이 있으므로 비판적으로 읽을 것. 'future work'라고 쓰인 것이 실제로는 한계점 회피인 경우를 포착할 것"

### 5-3. Track 2 — 구조적 분석

- **대상 섹션**: introduction, method, experiment, discussion
- **LLM 지시**: 저자가 명시하지 않았지만 텍스트에서 드러나는 암묵적 한계점 식별
  - 좁은 가정 (예: "we assume i.i.d. data")
  - 제한된 데이터셋 또는 평가 범위 (단일 도메인, 소규모)
  - 누락된 baseline 또는 비교 실험
  - introduction에서 언급된 scope 제한

### 5-4. 추출 규칙

- 논문당 **1-3개** 한계점 추출
- **Track 2 구조적 발견 우선** (저자가 숨긴 한계점이 더 가치 있음)
- `claim`: 반드시 완전한 1-2문장. 절대 잘린 문장 불가. 입력 텍스트가 잘린 경우 맥락 기반으로 요약 완성
- `evidence_quote`: 원문에서 가져온 **정확하고 짧은** 인용문
- 갭(gap)을 추론하지 말 것 — 텍스트에 근거한 한계점만 추출

### 5-5. 출력 스키마

```json
[
  {
    "paper_id": "arxiv:2401.12345",
    "claim": "The model was only evaluated on English datasets, limiting its applicability to multilingual settings.",
    "evidence_quote": "We evaluate our approach on three English benchmarks",
    "track": "structural",
    "source_section": "experiment"
  }
]
```

- `track`: `"author_stated"` (Track 1) 또는 `"structural"` (Track 2)
- `source_section`: 실제 섹션명 (conclusion, method, experiment 등)
- 출력은 **JSON 리스트만** — 설명 텍스트 불가

### 5-6. 배치 LLM 처리

여러 논문을 하나의 LLM 호출로 묶어 처리:

| 항목 | 값 |
|------|-----|
| 배치 크기 | **3편/배치** (`BATCH_SIZE = 3`) |
| 배치 병렬 워커 | **min(2, 배치 수)** (ThreadPoolExecutor) |
| 논문당 프롬프트 최대 | **4,000자** (초과 시 잘림) |
| Full text 로딩 병렬 워커 | **min(3, 논문 수)** |

**배치 프롬프트 구조:**
```
=== PAPER: arxiv:2401.12345 ===
(paper 1 프롬프트, 4000자 이내)

=== PAPER: doi:10.1234/example ===
(paper 2 프롬프트, 4000자 이내)

=== PAPER: arxiv:2402.67890 ===
(paper 3 프롬프트, 4000자 이내)
```

- LLM 지시: "모든 논문에서 추출하여 단일 JSON 리스트로 반환. 각 항목에 올바른 paper_id 포함 필수"
- **배치 실패 시 폴백**: 배치 LLM 호출 실패 → 해당 배치의 논문을 **개별 처리**로 재시도
- **paper_id 검증**: 응답의 paper_id가 배치에 없으면 배치 첫 번째 논문으로 기본 배정
- **진행 상황 스트리밍**: 배치 완료마다 `partial_limitations` 이벤트로 프론트엔드에 부분 결과 전송

---

## 6. 중복 제거 (Deduplication)

다수 논문에서 추출된 한계점 간 중복을 제거하는 단계. 같은 주제의 논문은 유사한 한계점을 공유하는 경우가 많으므로 필수.

### 6-1. 알고리즘: Jaccard 유사도

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

- **임계값**: **0.55** (55% 이상 토큰 겹침 → 중복으로 판정)

### 6-2. 토큰화 과정

1. claim 텍스트를 **소문자** 변환
2. **구두점 제거** (`re.sub(r"[^\w\s]", "", text)`)
3. 공백 기준 분리 → 토큰 집합(set)
4. **영어 불용어 127개** 제거 (the, a, an, is, are, of, in, to, for, with, and, or, but, not, it, this, that, they, which, who 등)

### 6-3. 중복 제거 로직

1. 전체 한계점을 **evidence_quote 길이 내림차순** 정렬 (근거가 풍부한 것 우선)
2. 순회하며 기존 유지 목록과 Jaccard 비교
3. Jaccard ≥ 0.55인 기존 항목이 하나라도 있으면 → **중복으로 제거**
4. 없으면 → 유지 목록에 추가

**효과:**
- 동일 논문 내 유사 한계점 제거 (같은 한계를 conclusion과 discussion에서 각각 추출한 경우)
- 다른 논문 간 유사 한계점 제거 (같은 분야 논문이 공통적으로 가진 한계)
- evidence_quote가 긴 쪽(=근거가 더 풍부한 쪽)이 생존

---

## 7. 교차 검증 (Cross-Verification)

추출된 한계점의 **환각(hallucination)을 탐지**하기 위해, 추출에 사용된 LLM과 **다른 LLM**으로 검증하는 단계.

### 7-1. 실행 조건

- `model_routing` 프로파일이 `optimized` 또는 `speed`일 때만 실행
- 라우팅 없으면 스킵 (단일 모델 사용 시 교차 검증 의미 없음)
- 검증 LLM: `limitation_verify` 에이전트 설정에 따름 (추출과 다른 provider)

### 7-2. 검증 프롬프트

```
[논문 텍스트]
[CONCLUSION]
(conclusion 텍스트, 최대 1000자)
[METHOD]
(method 텍스트, 최대 1000자)
... (최대 5개 섹션, 합산 3000자 이내)

[추출된 Limitations]
[index=0]
  claim: The model was only evaluated on English datasets...
  evidence_quote: We evaluate our approach on three English benchmarks

[index=1]
  claim: ...
  evidence_quote: ...

각 limitation에 대해:
1. evidence_quote가 원문에 존재하는가? (FOUND/NOT_FOUND)
2. claim이 evidence_quote에서 도출 가능한가? (VALID/INVALID)
```

### 7-3. 검증 과정

1. 한계점을 **논문별로 그룹핑** (`paper_id` 기준)
2. 각 논문 그룹에 대해:
   - 해당 논문의 섹션 텍스트 조합 (최대 5섹션 × 1000자, 합산 3000자)
   - 해당 논문에서 추출된 한계점 목록
   - LLM에 검증 프롬프트 전달
3. 응답 파싱: 각 한계점의 `quote_check`와 `claim_check` 확인

### 7-4. 검증 결과 부착

| 조건 | 결과 |
|------|------|
| quote_check=`FOUND` AND claim_check=`VALID` | `verified: true` |
| 그 외 | `verified: false` + `verify_detail` 첨부 |

```json
{
  "verified": false,
  "verify_detail": {
    "quote_check": "NOT_FOUND",
    "claim_check": "INVALID"
  }
}
```

- 검증 실패한 한계점은 이 단계에서 제거되지 않음 — 플래그만 부착
- 이후 평가(Eval) 단계에서 종합적으로 판단

---

## 8. 한계점 평가 (Limitation Eval)

추출된 한계점의 품질을 **2단계 LLM 호출**로 검증하는 에이전트. Call 1(개별 평가)과 Call 2(전체 판단)를 순차 실행.

### 8-1. Call 1 — 원자적 검증 + 루브릭 채점 (Per-Item)

각 한계점을 개별적으로 평가. **두 가지 방법을 동시에** 적용:

#### Method A: FActScore 기반 원자적 팩트 검증

한계점의 claim을 **원자적 팩트(atomic fact)**로 분해하여 각각 검증:

1. claim을 분해 → 단순하고 분할 불가능한 진술 목록 (최소 **2개** 필수)
2. 각 팩트를 evidence_quote와 대조:
   - `SUPPORTED`: 증거가 이 팩트를 직접 지지
   - `NOT_SUPPORTED`: 증거가 지지하지 않거나 모순
   - `IRRELEVANT`: 증거로부터 검증 불가

3. 점수 계산: `fact_score = supported_count / total_count` (0.0 ~ 1.0)

**예시:**
```json
{
  "limitation_id": 0,
  "atomic_facts": [
    {"fact": "The model was only evaluated on English data", "verdict": "SUPPORTED"},
    {"fact": "This limits applicability to multilingual settings", "verdict": "NOT_SUPPORTED"}
  ],
  "fact_score": 0.5
}
```

#### Method B: Prometheus 기반 루브릭 채점

3개 차원, 각 1-5점:

**Groundedness (근거성):**

| 점수 | 판단 | 설명 |
|------|------|------|
| 1 | Fabricated | 증거 없음, 날조된 주장 |
| 2 | Weak | 증거가 존재하나 주장을 지지하지 않음 |
| 3 | Partial | 증거와 느슨하게 연관, 주장이 과장 |
| 4 | Solid | 증거가 주장을 명확히 지지 |
| 5 | Exact | 주장이 증거의 직접적 패러프레이즈 |

**Specificity (구체성):**

| 점수 | 판단 | 설명 |
|------|------|------|
| 1 | Generic | "모델에 한계가 있다" 수준 |
| 2 | Vague | 도메인 언급하나 구체적 디테일 없음 |
| 3 | Moderate | 조건을 서술하나 정량적/맥락적 디테일 부족 |
| 4 | Specific | 명확한 조건, 맥락, 범위 서술 |
| 5 | Precise | 정량적 범위 또는 정확한 실패 조건 포함 |

**Relevance (관련성 — 연구 쿼리 기준):**

| 점수 | 판단 | 설명 |
|------|------|------|
| 1 | Irrelevant | 연구 쿼리와 무관 |
| 2 | Tangential | 같은 넓은 분야이나 다른 초점 |
| 3 | Related | 같은 하위 분야, 간접적 연결 |
| 4 | Directly relevant | 쿼리의 핵심 측면 다룸 |
| 5 | Central | 쿼리가 정확히 조사하려는 한계점 |

#### Call 1 배치 처리

| 항목 | 값 |
|------|-----|
| 배치 크기 | **10개/배치** (`CALL1_BATCH_SIZE = 10`) |
| 최대 병렬 워커 | **min(3, 배치 수)** |
| 병렬 방식 | ThreadPoolExecutor |

- 워커 1개일 때는 순차 처리 (executor 오버헤드 방지)
- 배치 완료 후 **원래 limitation_id 순서**로 병합
- 배치 실패 시 해당 배치 결과 빈 리스트 반환 (다른 배치에 영향 없음)

#### Call 1 LLM 입력 형식

```
## Research Query
{refined_query}

## Limitations to Evaluate (10)
[limitation_id=0]
  paper_id: arxiv:2401.12345
  claim: The model was only evaluated on English datasets...
  evidence_quote: We evaluate our approach on three English benchmarks
  track: structural
  source_section: experiment

[limitation_id=1]
  ...
```

#### Fast Mode

- Call 1 **전체 스킵**
- 더미 점수 생성: `fact_score=0.8, groundedness=4, specificity=4, relevance=4`
- Call 2만 실행

---

### 8-2. Call 2 — 전체 판단 (Set-Level)

한계점 집합 전체를 조망하여 품질 판정 + 유형 분류 + 다양성 분석을 수행. Call 1 결과를 참조 입력으로 받음.

#### Method A: LimAgents 기반 품질 판정

각 한계점에 대해 3단계 판정:

| 판정 | 의미 | 후속 조치 |
|------|------|----------|
| `strong` | 잘 근거된, 구체적, 갭 분석에 유용 | 그대로 유지 |
| `weak` | 문제 있으나 개선 가능 | `improvement_hint` 제공 (예: "Add specific dataset size numbers") |
| `remove` | 날조, 무관, 너무 일반적 → 폐기 | 후처리에서 제거 |

- `remove` 판정은 **보수적**으로 — 명확하게 나쁜 경우만

#### Method B: Xu et al. 분류 체계 (6종)

각 한계점을 **하나의 primary type**으로 분류:

| 유형 | 대상 | 예시 |
|------|------|------|
| `methodology` | 모델 설계, 알고리즘, 접근법 | "Transformer의 quadratic attention 비용" |
| `data` | 데이터/표본 크기, 다양성, 품질 | "학습 데이터가 영어에 편중" |
| `scope` | 도메인, 전이가능성, 일반화 | "의료 도메인만 테스트" |
| `evaluation` | 메트릭, 벤치마크, 베이스라인 | "BLEU만 사용, human eval 미실시" |
| `theoretical` | 가정, 형식적 보증 | "수렴 보장 없이 경험적 결과만 제시" |
| `resource` | 컴퓨팅, 비용, 배포 | "A100 8장 필요, 실시간 추론 불가" |

#### Method C: Set-Level 분석

한계점 집합의 **전체적 균형**을 분석:

- `type_distribution`: 유형별 카운트 (예: `{"methodology": 5, "data": 3, "scope": 1}`)
- `coverage_warning`: 한 유형이 전체의 **>75%** 차지 시 경고 메시지 생성
  - 예: `"methodology type dominates at 80% — consider re-extracting with focus on data and evaluation limitations"`
- `diversity_score`: 1-5
  - 1 = 전부 같은 유형
  - 3 = 2-3개 유형에 걸쳐 분포
  - 5 = 6개 유형에 균형 있게 분포

#### Call 2 PASS/RETRY 결정

Call 1 점수와 Method A 판정을 종합하여 결정:

| RETRY 조건 | 임계값 |
|-----------|--------|
| weak/remove 비율 | > **50%** |
| 평균 groundedness | < **3.0** |
| 평균 fact_score | < **0.6** |
| diversity_score | ≤ **2** |

- **ANY** 조건 하나라도 충족 → `RETRY`
- RETRY 시 `retry_guidance` 필수 (실행 가능한 가이드)
  - 예: `"Focus on extracting evaluation and scope limitations from experiment sections"`

#### Call 2 LLM 입력 형식

```
## Research Query
{refined_query}

## Call 1 Scores (per-limitation)
  [id=0] fact_score=0.8, groundedness=4, specificity=3, relevance=5
  [id=1] fact_score=0.5, groundedness=2, specificity=2, relevance=3
  ...

## Limitations (15)
[limitation_id=0]
  paper_id: arxiv:2401.12345
  claim: The model was only evaluated on English datasets...
  evidence_quote: We evaluate our approach on three En...  (200자 잘림)
  track: structural

[limitation_id=1]
  ...
```

- evidence_quote는 **200자**로 잘림 (Call 2는 개별 증거보다 전체 판단이 목적)

---

### 8-3. 후처리 및 필터링

Call 1 + Call 2 결과를 바탕으로 한계점 필터링 및 메타데이터 부착.

#### 제거 조건 (OR — 하나라도 해당 시 제거)

| 조건 | 임계값 |
|------|--------|
| Call 2 quality | `"remove"` |
| Call 1 fact_score | < **0.4** |
| Call 1 groundedness | < **2** |

#### 각 생존 한계점에 부착되는 메타데이터

```json
{
  "eval_fact_score": 0.8,
  "eval_groundedness": 4,
  "eval_specificity": 3,
  "eval_relevance": 5,
  "eval_quality": "strong",
  "eval_limitation_type": "methodology",
  "eval_improvement_hint": null,
  "eval_flag": "weak"
}
```

- `eval_flag`: quality="weak"인 경우에만 `"weak"` 부착 (strong이면 없음)
- `eval_improvement_hint`: quality="weak"인 경우에만 개선 방향 제공

#### 자체 검증 RETRY 오버라이드

Call 2가 `PASS`를 반환해도, 후처리에서 아래 조건 발견 시 **RETRY로 강제 전환**:

- 평균 groundedness < 3.0
- 평균 fact_score < 0.6
- weak 비율 > 50%
- 필터 후 한계점 0개

#### 최대 재시도

- `MAX_EVAL_RETRIES = 1`
- 재시도 후에도 RETRY이면 → **강제 PASS** (무한 루프 방지)
- 경고 메시지: `"Forced PASS after {N} retries"`

---

## 10. 전체 흐름 요약

```
[논문 목록 (papers)]
        ↓
[1] 캐시 확인 (7일 TTL)
        ↓
[2] 소스별 전문 로딩 (8단계 폴백, 병렬 max 3 워커)
    - arXiv: ar5iv HTML → PDF
    - DOI: PDF → HTML → Unpaywall → S2 → NCBI → PMC BioC → EuropePMC → Direct PDF
    - ScienceON: DOI 폴백 / FulltextURL
        ↓
[3] 실패 논문 → backup 대체 (arXiv 우선, BM25 순)
        ↓
[4] 섹션 분할 (정규식, 7종, 3000자 제한)
        ↓
[5] Dual-Track LLM 추출 (3편/배치, 2 워커)
    - Track 1: 저자 명시 (conclusion, limitations, future_work)
    - Track 2: 구조적 분석 (method, experiment, discussion)
        ↓
[6] 중복 제거 (Jaccard ≥ 0.55)
        ↓
[7] 교차 검증 (optimized/speed 프로파일만)
        ↓
[8] Call 1: 원자적 팩트 검증 + 루브릭 채점 (10개/배치, 3 워커)
        ↓
[9] Call 2: 품질 판정 + 유형 분류 + 다양성 분석
        ↓
[10] 후처리: 필터링 (remove / fact<0.4 / ground<2 제거)
        ↓
[11] PASS/RETRY 결정 (최대 1회 재시도)
        ↓
[필터링된 한계점 + 평가 메타데이터]
```

---

## 11. 주요 상수 종합

| 상수 | 값 | 용도 |
|------|-----|------|
| `MAX_SECTION_CHARS` | 3,000 | 섹션당 최대 크기 |
| `_MIN_FULLTEXT_CHARS` | 500 | 최소 full text 합산 |
| `_CACHE_TTL_DAYS` | 7 | 캐시 만료 (성공) |
| `_CACHE_FAIL_TTL_HOURS` | 6 | 캐시 만료 (실패) |
| `_MAX_PDF_BYTES` | 10 MB | PDF 최대 크기 |
| `BATCH_SIZE` (추출) | 3 | 논문/배치 |
| `CALL1_BATCH_SIZE` (평가) | 10 | 한계점/배치 |
| `MAX_EVAL_RETRIES` | 1 | 최대 재시도 |
| Dedup 임계값 | 0.55 | Jaccard 유사도 |
| 제거: fact_score | < 0.4 | 하드 제거 |
| 제거: groundedness | < 2 | 하드 제거 |
| RETRY: avg groundedness | < 3.0 | 재추출 트리거 |
| RETRY: avg fact_score | < 0.6 | 재추출 트리거 |
| RETRY: weak 비율 | > 50% | 재추출 트리거 |
| RETRY: diversity_score | ≤ 2 | 재추출 트리거 |
| Coverage 경고 | > 75% | 단일 유형 편중 |
