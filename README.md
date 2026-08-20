<div align="center">

<img src="frontend/new_logo.png" alt="GAPAGO" width="88" />

# GAPAGO

**연구 논문에서 "아직 아무도 풀지 않은 문제"를 찾아내는 멀티 에이전트**

연구 주제를 입력하면 논문을 모으고, 각 논문 **본문에서** 한계점을 뽑고,
웹 검색으로 이미 해결된 것을 걸러낸 뒤, 남은 것들로 **연구 공백(research gap)** 과 후속 연구 방향을 제안합니다.

`LangGraph` · `FastAPI` · `Multi-LLM Routing`

</div>

---

## 빠른 시작

### Docker (권장)

```bash
git clone https://github.com/Coding-Maengsu/gapago.git
cd gapago
cp .env.example .env          # 키 입력 (LLM provider 1개 + TAVILY_API_KEY)

docker build -t gapago .
docker run --rm -p 8000:8000 --env-file .env gapago
# → http://localhost:8000
```

한 건만 분석해서 JSON으로 받고 싶다면:

```bash
docker run --rm --env-file .env \
  -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
  gapago analyze --input /app/data/input_sample.json --output /app/results/output.json
```

### 로컬 (Python 3.10+)

```bash
./run_agent.sh setup     # venv · 의존성 · 랜딩 빌드 · .env 점검을 한 번에
./run_agent.sh serve     # 웹 서버 → http://localhost:8000
```

실행 방법은 셋뿐입니다. `setup` 외의 인자는 그대로 `main.py` 로 전달되므로
`./run_agent.sh --help` 가 곧 전체 사용법입니다.

```bash
./run_agent.sh serve                        # 웹 서버
./run_agent.sh analyze "연구 주제"            # 1회 분석 → outputs/ 에 JSON
./run_agent.sh chat                         # 터미널 대화형

# 옵션은 --help 에 전부 나옵니다
./run_agent.sh analyze --input data/input_sample.json --output results/out.json --fast
```

### 입력 형식

```json
{
  "query": "Domain adaptation in clinical drug discovery",
  "routing_profile": "optimized",
  "fast_mode": true,
  "year_range": "auto",
  "output_language": "auto"
}
```

`query`만 필수입니다. 나머지는 생략 시 위 기본값이 적용됩니다.

---

## 무엇을 하는가

일반 LLM에게 "이 분야의 연구 공백을 알려줘"라고 물으면 그럴듯하지만 근거 없는 답이 돌아옵니다.
GAPAGO는 **실제 논문 원문에서 출발**합니다.

| 단계 | 하는 일 |
|---|---|
| **1. 쿼리 정제** | 모호한 주제를 검색 가능한 쿼리로 변환. 너무 넓거나 좁으면 사용자에게 되물음 |
| **2. 논문 검색** | arXiv · Crossref · Semantic Scholar · OpenAlex · ScienceON · Tavily 8개 소스 병렬 검색 |
| **3. 한계점 추출** | **초록이 아닌 전문(full text)** 에서 추출. 전문 확보 실패 논문은 후보로 교체 |
| **4. 근거 검증** | 추출된 한계점이 원문에 실제로 근거하는지 원자 단위로 검증. 미달 시 재추출 |
| **5. 최신성 확인** | 웹 검색으로 "이미 누가 풀었는지" 대조. 해결된 것은 후보에서 제외 |
| **6. GAP 추론** | 남은 한계점을 도메인 축으로 묶고, 기술적 장벽을 분석해 연구 방향 제안 |
| **7. 리포트** | 근거 논문 인용이 포함된 5개 섹션 마크다운 리포트 생성 |

각 단계는 품질 게이트를 통과하지 못하면 이전 단계로 되돌아갑니다.

## 파이프라인

```mermaid
flowchart TD
    START([시작]) --> Q[query_subgraph<br/>쿼리 분석 · 정제]
    Q --> ME[meaning_expand<br/>키워드 · 쿼리 확장]
    ME --> PR[paper_retrieval<br/>8개 소스 병렬 검색]
    PR --> LE[limitation_extract<br/>전문에서 한계점 추출]
    LE --> LV{limitation_eval<br/>근거 검증}
    LV -->|RETRY| LE
    LV -->|PASS| RC[recency_check<br/>최신성 대조]
    RC --> GI[gap_infer<br/>GAP 추론]
    GI --> CS{critic_score<br/>품질 채점}
    CS -->|ACCEPT| FR[final_response<br/>리포트 생성]
    CS -->|REDO_RETRIEVAL| ME
    CS -->|REFINE_QUERY| Q
    FR --> END([종료])
```

**두 가지 실행 모드** — 기본은 위 고정 파이프라인이고, `GAPAGO_ORCHESTRATOR=1` 이면
LLM이 매 스텝 상태를 보고 다음 에이전트를 정하는 오케스트레이터 모드로 동작합니다.

---

## 프로젝트 구조

```
.
├── run_agent.sh             입구 — setup + main.py 로 전달
├── main.py                  CLI 정의 (serve · analyze · chat)
├── Dockerfile               멀티스테이지 (랜딩 빌드 + 앱)
├── requirements.txt         직접 의존성 · requirements-lock.txt 는 재현용
│
├── gapago/                  애플리케이션 패키지
│   ├── paths.py               프로젝트 경로 단일 기준점
│   ├── api/main.py            FastAPI 서버 — SSE 스트리밍, 세션 관리
│   ├── graphs/                LangGraph 그래프 (고정 / 오케스트레이터 / 쿼리 서브그래프)
│   ├── agents/                에이전트 노드 10종
│   │   ├── query_agent/         쿼리 분석
│   │   ├── retrieval_agent.py   8소스 병렬 검색 + 3단계 리랭킹
│   │   ├── limitation_agent.py  전문 기반 한계점 추출
│   │   ├── gap_agent.py         동적 축 생성 + GAP 추론
│   │   └── ...                  eval · recency · critic · response · chat · orchestrator
│   ├── core/                  공용 모듈
│   │   ├── tools/               검색 소스별 (arxiv · crossref · s2 · openalex · scienceon)
│   │   ├── llm.py               Provider 추상화
│   │   ├── model_router.py      에이전트별 모델 배정
│   │   ├── states.py            AgentState 및 스키마
│   │   ├── config.py            환경 변수 설정
│   │   └── prompts.py           공통 시스템 프롬프트
│   └── utils/                 progress(SSE) · session_store · cancel · parse_json · tavily
│
├── frontend/index.html      분석 앱 UI
├── landing/                 랜딩 페이지 (React 19 + Vite + Tailwind)
│
├── data/input_sample.json   배치 입력 예시
├── evaluation/              평가 · 벤치마크 · 프로파일링 도구
├── tests/                   pytest
└── docs/                    설정 · API · 스펙 · 설계 문서
```

## 핵심 설계

**전문 기반 추출** — 한계점은 초록이 아닌 본문에서만 뽑습니다. 초록에는 한계가 거의 적히지 않기 때문입니다.
전문 확보는 8단계 폴백(ar5iv → arXiv PDF → DOI 랜딩 → Unpaywall → S2 Batch → PMC → EuropePMC → 직접 PDF)을 거치고,
그래도 실패하면 그 논문을 버리고 후보 풀에서 교체합니다. 결과는 `.cache/fulltext/` 에 캐싱됩니다.

**3단계 리랭킹** — `8소스 수집 → BM25(top-50) → 임베딩 유사도(FAISS) → CrossEncoder 정밀 리랭킹`.
CPU 환경은 MiniLM + ONNX Runtime, GPU 환경은 SPECTER2 + BGE Reranker v2-m3 로 자동 전환됩니다.

**근거 검증** — 한계점을 원자적 사실로 분해해 원문 근거를 개별 판정하고(FActScore),
Groundedness · Specificity · Relevance 를 1~5점으로 채점합니다(Prometheus).
기준 미달이면 추출 단계로 되돌립니다.

**Human-in-the-Loop** — 주제가 너무 넓거나 좁으면 파이프라인이 멈추고 사용자에게 되묻습니다.
웹은 `/api/clarify`, CLI는 프롬프트로 응답을 받아 재개합니다. 배치 모드에서는 첫 후보를 자동 선택합니다.

---

## 평가 · 개발 도구

```bash
python evaluation/evaluate.py --result-file outputs/gapago_result_*.json   # baseline LLM 과 품질 비교
python evaluation/compare_modes.py "연구 주제"                              # 고정 vs 오케스트레이터 모드
python evaluation/profile_timing.py                                        # 단계별 소요 시간

python evaluation/run_evaluation.py                                     # LitSearch · scope 벤치마크
pytest
```

## 배포

[`render.yaml`](render.yaml) 기반 Render 배포를 지원합니다. `main` 푸시 시 자동 배포되며,
빌드 단계에서 랜딩 페이지를 함께 빌드합니다. GPU가 없으므로 `RERANK_MODELS=light` 로 동작하고,
`--workers 1` 단일 프로세스로 기동해 임베딩/리랭커 모델을 한 번만 로드합니다.
API 키는 Render Dashboard의 Environment에서 주입합니다.

## 문서

**[`docs/README.md`](docs/README.md) 에서 시작하세요** — 목적별로 정리된 인덱스입니다.

| 자주 찾는 것 | 문서 |
|---|---|
| 환경 변수 · 모델 라우팅 · Fast Mode | [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) |
| HTTP API 레퍼런스 | [`docs/API.md`](docs/API.md) |
| 에이전트별 상세 동작 | [`docs/specs/SPEC_AGENT.md`](docs/specs/SPEC_AGENT.md) |
| 전체 파이프라인 요약 | [`docs/specs/GAPAGO_팩트시트.md`](docs/specs/GAPAGO_팩트시트.md) |

## 라이선스

[Apache License 2.0](LICENSE)

PDF 전문 추출에 쓰는 `PyMuPDF`/`pymupdf4llm` 은 **AGPL-3.0** 입니다.
소스를 내려받아 직접 설치해 쓰는 데는 제약이 없으나, **PyMuPDF 가 포함된 Docker 이미지를
배포하거나 웹 서비스로 운영하면 AGPL-3.0 제13조가 적용됩니다.** 자세한 내용은 [`NOTICE`](NOTICE) 참고.

---

<div align="center">
<sub>팀 코딩맹수 · AI Co-Scientist Challenge</sub>
</div>
