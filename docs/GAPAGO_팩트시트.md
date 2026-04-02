# GAPAGO 팩트시트 — 코드 분석 기반

## 0. Orchestrator (`agents/orchestrator_agent.py`)
- LangGraph StateGraph 기반 전체 흐름 제어
- 필수 파이프라인: meaning_expand → paper_retrieval → limitation_extract → gap_infer → final_response
- 선택적 품질 게이트 (LLM이 동적 삽입 판단): limitation_eval, recency_check, critic_score
- 의존성 체인 코드레벨 검증
- 재실행 제한: optional agent당 최대 2회, 전체 최대 15 step
- 환경변수 `GAPAGO_ORCHESTRATOR=1`로 활성화

## 1. Query Analysis Agent (`agents/query_agent/`)
- **SemRank (Zhang et al., EMNLP 2025)**: multi-granular 과학 개념 인덱싱
  - `general_topic` (broad) vs `specific_phrases` (concrete) 추출
  - specific_phrases가 비어있으면 TOO_BROAD 판정
- **APA (Kim et al., EMNLP 2024)**: Alignment with Perceived Ambiguity
  - `perceived_ambiguity` 감지, `dominant_interpretation` 선택
  - CLAMBER 탐지 + INFOGAIN 임계값 기반 모호성 정량 평가
- **CoQuest (Liu et al., CHI 2024)**: Human-AI 공동 탐색
  - breadth-first 설계: TOO_BROAD 시 3개 하위 방향 후보 동시 제시
  - "AI Thoughts" 근거 설명 출력
- Scope 3분류:
  - TOO_BROAD: Type A (분야만) / Type B (아키텍처만) / Type C (방법론만)
  - SEARCHABLE: 구체적 task 또는 [Domain, Task, Modality, Problem] 중 2개 이상
  - TOO_NARROW: 지나치게 구체적 조합
- 구성요소 추출: Domain [D], Task [T], Modality [M], Problem [P]
- Human-in-the-Loop: TOO_BROAD/TOO_NARROW 시 사용자 인터럽트
- 최대 반복: max_iterations = 3

## 2. Meaning Expand Agent (`agents/meaning_expand_agent.py`)
- 키워드 확장: 약어 확장, 철자 변형, 동의어 매핑
- 플랫폼별 쿼리 후보 생성:
  - arxiv_query_candidates (최대 4개)
  - web_query_candidates (최대 4개)
  - scienceon_query_candidates (최대 3개, 한국어 특화)
- 최대 12개 expanded_terms 생성
- 사용자 메모리 통합 (선택적, 최대 1500자)

## 3. Retrieval Agent (`agents/retrieval_agent.py`)
- 8개 소스 병렬 검색 (ThreadPoolExecutor):
  - arXiv (100편), Crossref (60편), Semantic Scholar (50편), OpenAlex (40편)
  - ScienceON 논문(15편) + 특허(10편) + 보고서(10편)
  - Tavily 웹 검색 (3쿼리 × 3결과)
- 3단계 리랭킹 파이프라인:
  1. BM25 초벌 필터링 (top-50)
  2. SPECTER2 / MiniLM-L6-v2 임베딩 유사도 (FAISS)
  3. BGE Reranker v2-m3 / MiniLM CrossEncoder 정밀 리랭킹
- CrossEncoder 동적 k 선택: max score × 0.65 임계값, >30% 점수 갭 시 절단, 범위 15-25편
- LLM Reranker 폴백: CrossEncoder 불가 시 LLM 기반 리랭킹
- ONNX Runtime: CPU 환경 2-3× 추론 가속
- 디바이스 자동감지: CUDA > MPS > CPU
- Full-text 접근성 티어링: guaranteed(3, ar5iv) > likely(2, OA PDF) > maybe(1, DOI) > none(0)
- 최대 30편 full-text 접근 가능 논문 필터링
- 중복 제거: DOI + 제목 + 연도 기준

## 4. Limitation Extraction Agent (`agents/limitation_agent.py`)
- **Full text 전용**: abstract fallback 없음 — full text 획득 실패 논문은 제거 후 backup 논문으로 대체
- 8단계 전문 획득 폴백 체인:
  1. arXiv ar5iv HTML (최우선)
  2. arXiv PDF (LangChain 로더)
  3. DOI 랜딩 페이지 HTML
  4. Unpaywall OA 탐색
  5. S2 Batch API (ArXiv/PMC ID)
  6. PMC BioC XML
  7. EuropePMC JATS XML
  8. Direct PDF (pymupdf4llm, Markdown 보존)
- Full text 실패 시 backup 대체: retrieval에서 확보한 backup 풀(arXiv 우선, BM25 순)에서 full text 가능한 논문으로 교체, 대체 불가 시 해당 슬롯 제거
- 섹션 분할: 정규식 기반, 7종 (introduction, method, experiment, discussion, conclusion, limitations, future_work)
- 섹션당 최대 3000자 (토큰 예산 제한)
- 최소 full text 길이: 합산 500자 미만 시 실패 처리
- Dual-Track 병렬 추출:
  - Track 1 (저자 명시): conclusion, limitations, future_work 섹션
  - Track 2 (구조적 분석): method, experiment, discussion 마이닝
- 캐싱: SHA256 키, .cache/fulltext/ 저장, 성공 7일 TTL / 실패 6시간 TTL
- HTTP 커넥션 풀링: 세션 기반 requests + 커스텀 User-Agent

## 5. Limitation Eval Agent (`agents/limitation_eval_agent.py`)
- **Call 1 — 원자적 검증 + 루브릭 채점 (배치 병렬)**:
  - FActScore 기반: 주장 → 원자적 팩트 분해 → 개별 검증 (SUPPORTED/NOT_SUPPORTED/IRRELEVANT)
    - fact_score = supported_count / total_count
  - Prometheus 기반: 3차원 루브릭 (각 1-5점)
    - Groundedness (1=Fabricated ~ 5=Exact paraphrase)
    - Specificity (1=Generic ~ 5=Precise with quantitative bounds)
    - Relevance (1=Irrelevant ~ 5=Central to query)
  - 배치 처리: 10개/배치, ThreadPoolExecutor 최대 3 워커
- **Call 2 — 전체 판단**:
  - LimAgents 기반: strong / weak / remove 판정 + 개선 제안
  - Xu et al. 분류 체계: methodology / data / scope / evaluation / theoretical / resource
  - Set-level 분석:
    - type_distribution (유형별 카운트)
    - coverage_warning (한 유형 >75% 편중 시)
    - diversity_score (1-5, 1=전부 동일 ~ 5=균형)
- **RETRY 조건**: >50% weak/remove, 평균 groundedness <3.0, 평균 fact_score <0.6, diversity_score ≤2
- PASS 시 다음 단계 진행

## 6. Recency Check Agent (`agents/recency_agent.py`)
- 도메인 자동 감지 (LLM 기반, 5종): ai_cs, biomedical, materials_chemistry, physics, general
- 도메인별 특화 웹 검색 소스:
  - AI_CS: paperswithcode, github, huggingface, medium, towardsdatascience
  - Biomedical: PubMed, BioRxiv, MedRxiv, Nature, ScienceDirect
  - Materials/Chemistry: Nature, ScienceDirect, ACS, RSC, MaterialsProject
- Tavily 5개 쿼리, 시간 마커 포함 ("2024", "2025", "latest")
- 3단계 분류:
  - unresolved: 해결 증거 없음
  - partial: 부분적 진전
  - resolved: 명확한 해결 증거 (보수적 판정: 명확한 경우만)
- Tavily + LLM 교차 검증

## 7. Gap Inference Agent (`agents/gap_agent.py`)
- **Step 1 — 동적 축 생성** (귀납적, 사전 카테고리 없음):
  - LLM이 한계점 집합에서 도메인 특화 축 도출 (3-7개)
  - 축당 최소 2개 한계점, 상호 배타적 범위 검증
  - 실패 시 간소화 프롬프트로 폴백
- **Step 2 — 한계점 배치 분류**: BATCH_SIZE = 20, 각 한계점 → 단일 축 배정
- **Step 3 — Recency 가중치**: unresolved 1.0, partial 0.5, resolved 0.0 (resolved 제외)
- **Step 4a — 긴급도 채점**: (limitation_count × avg_recency_weight) × domain_specificity
- **Step 4b — 기술 장벽 분석**: Reasoning 모델 (QwQ-32B / Qwen3-32B)
  - 축별 barrier_type 동적 도출
  - "시도했으나 실패한 것" 추출
- **Step 4c — 창의적 방향 제안**:
  - 웹 검색 결과 맥락 주입 (최신 트렌드)
  - 장벽별 복수 방향 후보 생성
  - novelty_score (1-10) 채점
  - alt_topics (대안 연구 주제) 제안

## 8. Critic Agent (`agents/critic_agent.py`)
- LLM-as-a-Judge: 3개 메트릭 (0.0-1.0)
  - query_specificity: 정제된 쿼리의 학술 검색 적합성
  - paper_relevance: 검색 논문의 연구 질문 관련성
  - groundedness: 도출된 갭의 논문 근거 기반 여부
- 점수 가이드: 0.0-0.3 Poor / 0.4-0.6 Fair / 0.7-0.8 Good / 0.9-1.0 Excellent
- 결정 규칙:
  - 모두 ≥0.6 → ACCEPT → final_response
  - paper_relevance <0.4 → REDO_RETRIEVAL → meaning_expand
  - query_specificity <0.4 → REFINE_QUERY → query_subgraph
- MAX_CRITIC_LOOPS = 2 (초과 시 강제 ACCEPT)

## 9. Response Agent (`agents/response_agent.py`)
- 5-section 마크다운 리포트 생성:
  1. Related Papers (테이블: paper_id, Title, Year, Relevance)
  2. Key Limitations (축별 그룹핑)
  3. Research Gaps Overview (긴급도 순, 별점)
  4. Detailed Gap Analysis (근거 인용, 장벽 분석 포함)
  5. Critic Scores (평가 메트릭 테이블)
- 마크다운 전용 (유니코드 박스 드로잉 미사용)

## 10. 품질 제어 루프
- critic_score → meaning_expand: paper_relevance <0.4 시 재검색 (최대 2회)
- critic_score → query_subgraph: query_specificity <0.4 시 재정제
- limitation_eval → limitation_extract: RETRY 조건 충족 시 재추출
- Human-in-the-Loop: query_subgraph에서 scope=TOO_BROAD/TOO_NARROW 시 사용자 인터럽트

## 11. Model Routing (`model_router.py`)
- 프로파일 기반 에이전트별 LLM 배정
- **optimized 프로파일**:
  - Azure GPT (기본): query_analysis, query_refine, limitation_eval, recency_check (정확성 필수 작업)
  - Groq Qwen3-32B (light): meaning_expand, critic_score, orchestrator, gap_classify, limitation_verify
  - Claude (heavy): limitation_extract, response
  - Groq Qwen3-32B (heavy): gap_reasoning
- **quality 프로파일**: 핵심 작업(limitation, response, eval) → Claude, 추론 → Groq, 나머지 → Azure GPT

## 12. Fast Mode
- CrossEncoder 리랭킹 스킵
- limitation_eval Call 1 스킵
- 검색 논문 수 축소
- 경량 임베딩 모델 사용 (MiniLM-L6-v2)
- 전체 provider를 Groq (Qwen3-32B)로 전환 (speed 프로파일)
