# 2.5 핵심 알고리즘 및 기술적 특징

본 시스템은 Multi-Agent 구조를 기반으로 설계되며, 각 Agent는 특정 알고리즘 및 학술 프레임워크를 기반으로 동작함. 또한 Critic 기반 반복 개선 구조를 적용하여 전체 파이프라인의 품질을 지속적으로 보정하도록 설계함.

---

## 0. Orchestrator

- 목적 : 전체 Agent 실행 흐름 제어 및 단계 간 데이터 전달 관리  
- LangGraph StateGraph 기반으로 각 단계의 상태(State)를 구조화하여 관리함  
- 각 Agent 출력의 품질 및 입력 Query와의 정합성을 검증하여 다음 단계 진행 여부 결정함  
- 품질 미달 시 재실행 또는 보정 단계로 라우팅 수행함  

---

## 1. Query Analysis Agent

- 목적 : 사용자 입력 질문을 분석하여 논문 검색에 적합한 Query 생성  

- 적용 알고리즘  
  - SemRank : 질문 범위를 TOO_BROAD / SEARCHABLE / TOO_NARROW로 분류  
  - APA : CLAMBER 탐지 및 INFOGAIN 기준 기반 모호성 정량 평가  

- 주요 기능  
  - Domain, Task, Modality, Problem 구성요소 추출  
  - 동의어 확장, 조건 추가, negative keyword 적용을 통한 Query 보완  
  - Human-in-the-Loop 기반 사용자 검증 수행  

- 특징  
  - 단순 키워드 확장이 아닌 검색 가능성(Searchability) 기준으로 Query 정제  
  - 사용자 의도와 시스템 해석 간 불일치 최소화  

---

## 2. Research Intelligence Agent

### 2-1. Paper S-M (Searching & Matching)

- 목적 : 다중 학술 데이터 소스를 활용하여 관련 논문 탐색 및 선별  

- 적용 알고리즘  
  - BM25 기반 키워드 검색  
  - SPECTER2 및 Sentence Transformer 기반 임베딩 유사도  
  - CrossEncoder 기반 재순위화  

- 처리 과정  
  1) 8개 학술 API 병렬 검색 수행  
  2) DOI 및 제목 기반 중복 제거  
  3) BM25 + 임베딩 기반 후보 논문 생성  
  4) CrossEncoder 기반 정밀 재순위화  

- 특징  
  - Hybrid Retrieval 구조  
  - ONNX Runtime 백엔드 시도 (선택 의존성, 미설치 시 PyTorch 폴백)  
  - full-text 접근 가능 논문 중심 필터링  

---

### 2-2. Web Search Agent

- 목적 : 최신 연구 동향 및 논문 외 정보 보조 수집  

- 처리 과정  
  - Tavily 기반 웹 검색 수행  
  - 의미 유사도 기반 필터링  
  - 보조 정보로 활용  

---

## 3. Limitation Extraction Agent

- 목적 : 논문으로부터 연구 한계점 자동 추출  

- Dual-Track 구조  
  - Track 1 : 명시적 한계점  
  - Track 2 : 구조적 한계점  

---

## 4. Limitation Evaluation Agent

- 목적 : 한계점 품질 검증  

- Call 1  
  - FActScore  
  - Prometheus 평가  

- Call 2  
  - LimAgents 분류  
  - Xu et al. 유형 분류  

---

## 5. Recency Check Agent

- 목적 : 최신성 검증  

- 상태  
  - unresolved  
  - partial  
  - resolved  

---

## 6. GAP Inference Agent

- 목적 : Research GAP 도출  

- 주요 특징  
  - Dynamic Axis 생성  
  - 최신성 가중치 적용  
  - unresolved 중심 GAP 도출  

---

## 7. Critic Agent

- 목적 : 품질 평가 및 제어  

---

## 8. 품질 제어 루프

- Critic → 재검색  
- Critic → 재정제  
- Evaluation → 재추출  

---

## 9. Fast Mode

- 속도 최적화 모드 적용  

---

# 핵심 요약

Multi-Agent + 알고리즘 + 반복 개선 구조 기반 GAP 분석 시스템 구현
