# GAPAGO 문서

무엇을 알고 싶은지에 따라 골라 읽으세요.

## 쓰려는 사람

| 문서 | 내용 |
|---|---|
| [`CONFIGURATION.md`](CONFIGURATION.md) | 환경 변수 전체, 모델 라우팅 프로파일, Fast Mode |
| [`API.md`](API.md) | HTTP API 레퍼런스 — 엔드포인트 목록과 호출 예시 |

## 어떻게 동작하는지 알고 싶은 사람

현행 코드의 동작을 기술한 문서입니다.

| 문서 | 내용 |
|---|---|
| [`specs/SPEC_AGENT.md`](specs/SPEC_AGENT.md) | 에이전트 11종의 입출력·프롬프트·판정 규칙 (가장 상세) |
| [`specs/SPEC_API.md`](specs/SPEC_API.md) | FastAPI 서버 내부 동작 — 세션 관리, SSE, 공용 모듈 |
| [`specs/SPEC_LIMITATION_SYSTEM.md`](specs/SPEC_LIMITATION_SYSTEM.md) | 전문 확보 폴백 체인과 한계점 추출·검증 파이프라인 |
| [`specs/SPEC_WEB.md`](specs/SPEC_WEB.md) | 분석 앱 UI 구조와 디자인 시스템 |
| [`specs/GAPAGO_팩트시트.md`](specs/GAPAGO_팩트시트.md) | 전체 파이프라인 요약 — 한 파일로 훑어보기 |
| [`specs/GAPAGO_기술내용_2_5.md`](specs/GAPAGO_기술내용_2_5.md) | 핵심 알고리즘 요약 |

## 왜 그렇게 만들었는지 알고 싶은 사람

작성 시점의 판단을 기록한 문서입니다. **현행 코드와 다를 수 있습니다.**

| 문서 | 내용 |
|---|---|
| [`reports/technical_report_development.md`](reports/technical_report_development.md) | 개발 내용 기술보고서 |
| [`reports/agent_autonomy_analysis.md`](reports/agent_autonomy_analysis.md) | 에이전트 자율성 분석 및 개선안 |
| [`reports/limitation_extraction_failure_report.md`](reports/limitation_extraction_failure_report.md) | 한계점 추출 실패 원인 분석 |
| [`reports/dynamic_k_verification_report.md`](reports/dynamic_k_verification_report.md) | CrossEncoder 동적 k 선택 검증 |
| [`reports/web_search_usage_report.md`](reports/web_search_usage_report.md) | 웹 검색 활용 분석 |
| [`reports/multi_agent_proposal.md`](reports/multi_agent_proposal.md) | 멀티 에이전트 전환 제안 |
| [`design/`](design) | 리디자인 설계서·기능 요청서·메모리/비용 전략 |
| [`changelog/`](changelog) | 웹 메이저 업데이트, 모델 라우터 변경 이력 |
| [`assets/`](assets) | 파이프라인 아키텍처 다이어그램 |
