# 환경 변수 레퍼런스

`.env` 파일에 설정합니다. 템플릿은 저장소 루트의 [`.env.example`](../.env.example)을 복사해 쓰세요.

```bash
cp .env.example .env
```

**최소 요구사항**: LLM provider 1개 + `TAVILY_API_KEY`

---

## LLM Provider (1개 이상 필수)

| 변수 | 설명 |
|---|---|
| `LLM_PROVIDER` | 기본 provider — `azure` \| `claude` \| `gemini` \| `groq` \| `exaone` \| `qwq` |
| `AZURE_OPENAI_API_KEY` / `_ENDPOINT` / `_DEPLOYMENT` / `_API_VERSION` | Azure OpenAI (GPT) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | Claude via AWS Bedrock |
| `BEDROCK_CLAUDE_MODEL` | Bedrock 모델 ID (기본 `us.anthropic.claude-sonnet-4-20250514-v1:0`) |
| `GROQ_API_KEY` / `GROQ_MODEL` | Groq (기본 `qwen/qwen3-32b`) — GAP 추론 단계용 |
| `GROQ_REASONING_EFFORT` | Groq 추론 강도 (기본 `default`) |
| `GOOGLE_API_KEY` / `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | Google Gemini (Vertex AI) |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | 서비스 계정 JSON 문자열 (컨테이너 배포 시) |
| `GAP_REASONING_PROVIDER` | GAP 추론 단계만 다른 provider 로 분리하고 싶을 때 |
| `EXAONE_MODEL_PATH` / `QWQ_MODEL_PATH` | 로컬 GPU 모델 경로 |

> ⚠️ 키를 코드·문서·이미지에 하드코딩하지 마세요. 항상 환경변수로 주입합니다.

## 검색 소스

| 변수 | 필수 | 설명 |
|---|:--:|---|
| `TAVILY_API_KEY` | ✅ | 웹 검색 — 최신성 확인(recency_check)에 사용 |
| `SCIENCEON_CLIENT_ID` / `SCIENCEON_MAC_ADDRESS` / `SCIENCEON_KEY` | | 한국 ScienceON 논문·특허·보고서 |
| `SCIENCEON_DEFAULT_ROW_COUNT` | | ScienceON 기본 수집 건수 (기본 20) |

arXiv · Crossref · Semantic Scholar · OpenAlex 는 키가 필요 없습니다.

## 파이프라인 튜닝

| 변수 | 기본값 | 설명 |
|---|:--:|---|
| `ARXIV_MAX_RESULTS` | 20 | arXiv 최대 수집 편수 |
| `TAVILY_MAX_RESULTS` | 5 | Tavily 결과 수 |
| `FULLTEXT_TARGET_COUNT` | 30 | 전문 필터링 후 유지할 논문 수 |
| `BM25_TOP_K` | 50 | BM25 1차 필터 통과 수 |
| `RERANKER_TOP_K` | 15 | 리랭커 2차 선별 수 |
| `RERANK_MODELS` | `auto` | `light`(MiniLM, CPU) \| `full`(SPECTER2+BGE) \| `auto`(GPU 감지) |
| `GAPAGO_ORCHESTRATOR` | `0` | `1`이면 LLM 오케스트레이터 동적 라우팅 모드 |

배치 모드(`--input`)에서는 `ARXIV_MAX_RESULTS=3`, `TAVILY_MAX_RESULTS=3`,
`SCIENCEON_DEFAULT_ROW_COUNT=10` 이 기본으로 적용됩니다(환경변수로 덮어쓸 수 있음).

## 관측

| 변수 | 설명 |
|---|---|
| `LANGSMITH_TRACING` | `true` 로 두면 LangSmith 트레이싱 활성화 |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` / `LANGSMITH_ENDPOINT` | LangSmith 접속 정보 |

---

## 모델 라우팅 프로파일

에이전트마다 요구 능력이 달라 프로파일 단위로 모델을 배정합니다
(`core/model_router.py`).

| 프로파일 | 배정 |
|---|---|
| `optimized` (기본) | 정확성 필수 작업(쿼리 분석·평가·최신성) → Azure GPT · 추론(gap) → Groq · 핵심 추출/응답 → Claude |
| `quality` | 핵심 작업(한계점 추출·평가·응답·검증) → Claude · 추론 → Groq · 나머지 → Azure GPT |

## Fast Mode

`fast_mode=true` 이면 속도를 위해 다음을 생략합니다.

- CrossEncoder 리랭킹 (BM25 + FAISS 결과만 사용)
- `limitation_eval` 의 원자 검증(Call 1)
- GAP 추론 시 상위 3개 축만 분석
