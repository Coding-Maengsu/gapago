# 한계점 미추출 논문 — 원인 분석 리포트

> 분석일: 2026-04-01
> 대상 코드: `agents/limitation_agent.py`
> **주의**: 해결책 실행 전 사용자 확인 필요

---

## 1. 현재 파이프라인 개요

```
논문 입력 (15~25편)
  ├─ Step 1: Full text 병렬 로드 (3 workers)
  │    ├─ ArXiv: ar5iv HTML → ArXiv PDF fallback
  │    ├─ DOI 기반: PDF → HTML → DOI 랜딩 → Unpaywall → S2 → NCBI → EuropePMC (6단계 fallback)
  │    └─ ScienceON: DOI 경유 또는 FulltextURL/ContentURL
  ├─ Step 1.5: 실패 논문 → backup_papers로 대체
  ├─ Step 2: 배치 LLM 호출 (3편/배치, 2 workers)
  │    ├─ 2-Track 프롬프트 (Track1: 결론/한계/향후, Track2: 서론/방법/실험/토론)
  │    ├─ Abstract fallback (full text 없을 때)
  │    └─ Content filter 차단 시 abstract-only 재시도
  └─ Step 3: Jaccard 중복 제거 (threshold=0.55)
```

---

## 2. 원인 분류 및 코드 분석

### 원인 1: Full text 가져오기 실패

**경로**: `_load_full_text_sections()` (line 983-1015)

| 소스 | Fallback 체인 | 실패 원인 |
|------|--------------|-----------|
| ArXiv | ar5iv HTML → ArXiv PDF | ar5iv 서버 다운, PDF 파싱 실패 |
| CrossRef/S2/OpenAlex | PDF→HTML→DOI랜딩→Unpaywall→S2→NCBI→EuropePMC | 출판사 봇 차단, 유료 논문, PDF 없음 |
| ScienceON | DOI 경유 or FulltextURL | DOI 없음, URL 접근 불가 |

**실패 시 동작**:
- `sections = {}` 반환 → 빈 dict 캐시 저장 (`_cache_put`, line 1014)
- 캐시 TTL: 7일 → 7일간 같은 논문은 재시도하지 않음
- `fulltext_fail_count` 증가 → backup 논문 대체 트리거

**실패율 추정** (결과 파일 기반):
- 최종 추출 논문 수가 선별 논문 수보다 항상 적음 (14편 중 7~14편에서 limitation 추출)
- abstract에서만 추출된 limitation: `source_section: "abstract"` 비율 → 결과 파일에서 약 20~40%

**코드상 취약점**:
- `_load_arxiv_html()` (line 192): `timeout=8` — ar5iv 서버 응답이 느릴 때 실패
- `_load_doi_full_text()`: 6단계 fallback 중 각 단계에서 `try/except`로 에러를 삼킴 → 실패 원인 추적 어려움
- `_BLOCKED_PDF_HOSTS` (line 386-392): MDPI, Springer 등 주요 출판사 차단 → DOI 기반 논문의 full text 확보율 저하

### 원인 2: 섹션 분리 실패

**경로**: `_split_sections()` (line 74-114)

```python
heading_pattern = re.compile(
    r"(?m)^(?:#{1,4}\s+)?(?:\d+[\.\d]*\.?\s+)?(?:[^\n]{0,30}?\b)("
    + _kw_alt
    + r")s?\b[^\n]{0,60}$",
    re.IGNORECASE,
)
```

**실패 시나리오**:
1. 헤딩이 100자 초과 → 스킵 (line 98-99)
2. 키워드가 비영어 (한국어/중국어 논문) → 정규식 미매칭
3. 비표준 섹션 구조 (예: "Findings and Implications" → `SECTION_KEYWORDS`에 없음)
4. PDF에서 추출한 텍스트에 헤딩 형식이 깨짐
5. 같은 섹션 키워드가 여러 번 등장 → 첫 번째만 사용 (line 105: `if key in sections: continue`)

**실패 시 동작**:
- `sections = {}` → abstract fallback 사용 (Track1+Track2 모두 없을 때)
- Abstract fallback: "extract 1 limitation maximum" (프롬프트 규칙 5, line 1085)

**코드상 관찰**:
- "limitations" 키워드가 `SECTION_KEYWORDS`에 있지만 "limitation" (단수형)도 별도로 포함 → OK
- 결과 파일에서 `source_section: "abstract"` 비율이 높은 것은 섹션 분리 실패 증거

### 원인 3: LLM 빈 응답 / Content Filter

**경로**: `_process_single_paper()` (line 1235-1260), `_process_batch()` (line 1385-1398)

**실패 시나리오**:
1. **Content filter 차단** (line 1240): Azure OpenAI의 콘텐츠 필터가 학술 텍스트를 차단
   - 방어: abstract-only 재시도 (line 1242-1255)
   - 재시도도 실패 시: `content = "[]"` → 0개 limitation
2. **LLM 빈 JSON 반환**: 프롬프트에 "Output ONLY the JSON list" 지시했으나 LLM이 빈 배열 `[]` 반환
3. **JSON 파싱 실패**: `parse_json(content)`가 유효한 JSON 추출 실패 → `parsed = []`

**배치 실패 fallback** (line 1393-1398):
```python
# fallback: 개별 호출 시도
for paper in batch:
    single = _process_single_paper(paper)
```
- 배치(3편) LLM 호출 실패 시 개별 호출로 재시도 → 복원력 있음

### 원인 4: Context Window 초과로 Truncation

**경로**: `_process_batch()` (line 1368-1369)

```python
if len(paper_prompt) > 4000:
    paper_prompt = paper_prompt[:4000] + "\n... (truncated)"
```

- 배치 모드에서 논문당 4000자 제한
- 3편 배치: ~12000자 + 시스템 프롬프트 ~1500자 ≈ 13500자
- 대부분의 LLM 컨텍스트에는 충분하나, 섹션 텍스트가 이미 `MAX_SECTION_CHARS=3000`으로 제한되어 있어 truncation은 드묾

**개별 호출 시**:
- 논문당 자르기 없음 → 전체 섹션 텍스트 전달 (최대 7섹션 × 3000자 = 21000자)
- 큰 논문에서 LLM 입력이 과도할 수 있으나, 섹션당 3000자 제한이 방어

### 원인 5: 2-Track 중 특정 Track 실패

**경로**: `_build_prompt()` (line 1021-1057)

```python
track1 = {k: v for k, v in sections.items() if k in TRACK1_KEYS}  # conclusion, limitations, future_work
track2 = {k: v for k, v in sections.items() if k in TRACK2_KEYS}  # introduction, method, experiment, discussion
use_fallback = not track1 and not track2
```

**시나리오**:
- Track1만 있고 Track2 없음: 결론만 있는 짧은 논문 → Track1에서만 추출
- Track2만 있고 Track1 없음: 결론 섹션이 섹션 분리에서 누락 → Track2에서만 추출
- 둘 다 없음: abstract fallback → 1개 limitation만 추출

**관찰**: 결과 파일에서 `source_section` 분포 분석:
- `conclusion`: 가장 빈번 (Track1)
- `method`, `experiment`, `discussion`: Track2에서 추출
- `abstract`: fallback 사용 빈도 20~40%
- `introduction`: 의외로 높은 빈도 — introduction에서 관련 연구 한계를 지적하는 패턴

### 원인 6: 중복 제거에서 과도한 필터링

**경로**: `_dedupe_limitations()` (line 1132-1163)

```python
threshold = 0.55  # Jaccard similarity
```

- 불용어 제거 후 Jaccard 유사도 0.55 이상이면 중복 판정
- evidence_quote가 더 긴 쪽을 우선 유지

**과도 필터링 위험**:
- 같은 논문에서 유사한 표현으로 다른 한계를 기술한 경우 → 두 번째가 제거될 수 있음
- 예: "model lacks generalization to unseen domains" vs "model shows limited generalizability across domains" → Jaccard ≈ 0.6 → 하나 제거

**실제 결과**:
- 결과 파일의 limitation 수: 11~37개 (편차 큼)
- 중복 제거 비율: 로그 메시지 `[dedup] before → after (N개 중복 제거)`에서 확인 필요

---

## 3. 실패 경로별 영향도 분석

| 원인 | 발생 빈도 | 영향도 | 현재 방어 | 추가 대응 필요 |
|------|-----------|--------|-----------|---------------|
| Full text 실패 | 높음 (20~40%) | 중 | backup 교체 + abstract fallback | O |
| 섹션 분리 실패 | 중간 | 중 | abstract fallback | O |
| Content filter | 낮음 | 중 | abstract-only 재시도 | X |
| Context 초과 | 매우 낮음 | 낮 | 배치 4000자, 섹션 3000자 제한 | X |
| Track 부분 실패 | 중간 | 낮 | 다른 Track 사용 | X |
| 과도한 중복 제거 | 낮음 | 낮 | - | △ |

---

## 4. 로깅/추적 현황 분석

### 현재 로깅 수준

| 이벤트 | 로깅 | 추적 가능성 |
|--------|------|------------|
| Full text 소스별 성공/실패 | O (각 함수 내 `print`) | 서버 콘솔에서만 확인 |
| 섹션 분리 결과 | O (발견된 섹션 키 출력) | 서버 콘솔 |
| LLM 호출 실패 | O (`errors` 리스트에 추가) | 결과 JSON에는 미포함 |
| Content filter 차단 | O (print + errors) | 콘솔 + errors |
| 중복 제거 | O (개수만 출력) | 콘솔 |
| **논문별 성공/실패 요약** | **X** | **추적 불가** |
| **Full text 소스 태깅** | **X** | **어떤 경로로 성공했는지 결과에서 알 수 없음** |

### 개선 필요 지점

1. **논문별 추출 성공/실패 로그가 결과 JSON에 포함되지 않음**
   - `fulltext_fail_count`, `llm_fail_count`는 집계만 되고 결과에 기록되지 않음
   - 어떤 논문이 실패했는지 사후 분석 불가

2. **Full text 소스 태깅 없음**
   - limitation에 full text가 어떤 경로(ar5iv/PDF/DOI HTML/Unpaywall/...)로 확보되었는지 기록하지 않음
   - 성공률 분석에 필수

3. **빈 dict 캐시가 실패와 구분 불가**
   - `_cache_put(paper_id, {})` → 7일간 재시도 안 함
   - 일시적 네트워크 오류로 실패한 논문도 7일 대기 필요

---

## 5. 결과 파일 기반 분석

### 5.1 논문 대비 Limitation 비율

| 결과 파일 | 쿼리 | limitation 보유 논문 | 총 limitation | 논문당 평균 |
|-----------|-------|---------------------|---------------|------------|
| 20260323_095455 | 오디오-비주얼 딥페이크 | 14편 | 37개 | 2.6개 |
| 20260323_122225 | PINN fault detection | 13편 | 23개 | 1.8개 |
| 20260323_125242 | 적대적 공격 방어 | 15편 | 28개 | 1.9개 |
| 20260330_145636 | video detection | 13편 | 18개 | 1.4개 |
| 20260331_025515 | brain decoding | 14편 | 22개 | 1.6개 |

- 평균 논문당 1.4~2.6개 limitation → 프롬프트 지시("1-3 limitations per paper")와 일치
- 15편 선별 중 13~15편에서 추출 → 0~2편이 완전 실패

### 5.2 source_section 분포 (abstract fallback 비율)

- `abstract` 섹션에서 추출된 비율: 약 15~40% (파일마다 다름)
- `video detection` 결과: abstract 7/18 = 39% → non-arXiv 논문 비율이 높을수록 abstract fallback 증가
- `brain decoding` 결과 (0326): abstract 2/19 = 10% → arXiv 비율 높음
- `brain decoding` 결과 (0330): abstract 4/11 = 36% → 차이 큼 (당시 조건 차이)

### 5.3 한계점 0개 논문 추정

결과 파일에서 limitation `paper_id`에 나타나지 않는 논문:
- 직접 식별 불가 (결과 JSON에 `papers` 리스트가 비어 있는 별도 버그)
- limitation의 `paper_id` 종류 수와 선별 논문 수(15~25편)의 차이로 간접 추정: **0~10편이 완전 실패**

---

## 6. 원인별 해결책 제안 (사용자 확인 후 실행)

### 해결책 A: Full text 실패 논문 메타데이터 보강 (권장)

**목적**: 실패 원인 추적 및 사후 분석 지원

```python
# limitation_agent.py 수정
# 각 논문의 full text 로드 결과를 result JSON에 포함
paper_status = {
    "paper_id": paper.paper_id,
    "fulltext_source": "ar5iv" | "arxiv-pdf" | "doi-pdf" | "unpaywall" | "abstract" | "failed",
    "sections_found": list(sections.keys()),
    "fulltext_failed": bool,
}
```

### 해결책 B: 섹션 키워드 확장

**목적**: 비표준 섹션 이름 대응

```python
SECTION_KEYWORDS = {
    "conclusion": ["conclusion", "concluding remarks", "summary", "summary and conclusion",
                    "findings", "findings and implications"],
    "limitations": ["limitation", "limitations", "weakness", "weaknesses",
                     "threats to validity", "caveats"],
    # ...
}
```

### 해결책 C: 빈 캐시 TTL 분리

**목적**: 일시적 실패의 재시도 주기 단축

```python
_CACHE_TTL_DAYS = 7       # 성공 캐시
_CACHE_FAIL_TTL_HOURS = 6  # 실패 캐시 (빈 dict)

def _cache_get(paper_id):
    # ... 기존 로직 ...
    if isinstance(data, dict) and not data:  # 빈 dict = 실패 캐시
        if age_hours > _CACHE_FAIL_TTL_HOURS:
            path.unlink(missing_ok=True)
            return None
```

### 해결책 D: 논문별 추출 상태를 결과 JSON에 포함

**목적**: 프론트엔드에서 "한계점 미추출 논문" 표시 + 사후 디버깅

```python
# _save_result() 또는 state에 paper_extraction_status 추가
"paper_extraction_status": [
    {"paper_id": "arxiv:2401.12345", "status": "success", "limitations_count": 3, "source": "ar5iv"},
    {"paper_id": "crossref:10.xxx", "status": "abstract_fallback", "limitations_count": 1, "source": "abstract"},
    {"paper_id": "openalex:W123", "status": "failed", "limitations_count": 0, "source": "none"},
]
```

### 해결책 E: Jaccard threshold 미세 조정 (선택적)

현재 0.55 → 0.60으로 올려 과도한 필터링 방지. 단, 실제 중복 사례 분석 후 판단 필요.

---

## 7. 우선순위 권장

| 순서 | 해결책 | 이유 |
|------|--------|------|
| 1 | D (추출 상태 기록) | 진단 불가 → 진단 가능 전환, 모든 후속 개선의 기반 |
| 2 | A (메타데이터 보강) | Full text 소스 태깅으로 실패 패턴 정량화 |
| 3 | C (실패 캐시 TTL 분리) | 일시적 실패의 7일 대기 제거, 즉시 효과 |
| 4 | B (섹션 키워드 확장) | 섹션 분리 성공률 향상, 검증 용이 |
| 5 | E (Jaccard 조정) | 데이터 분석 후 판단 |
