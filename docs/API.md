# HTTP API 레퍼런스

FastAPI 서버(`gapago/api/main.py`). 기본 베이스 URL: `http://localhost:8000`

더 깊은 내부 동작은 [`specs/SPEC_API.md`](specs/SPEC_API.md)를 참고하세요.

---

## 사용 흐름

```
POST/GET /api/analyze  →  session_id 수신
        ↓
GET /api/stream/{session_id}   (SSE 구독 — 노드별 진행 이벤트)
        ↓
쿼리가 모호하면 clarify 이벤트 → GET /api/clarify 로 응답 전달 → 파이프라인 재개
        ↓
완료 시 최종 리포트 이벤트 수신
```

분석은 백그라운드로 실행되므로 `/api/analyze` 응답 직후 바로 스트림을 구독해야 초기 이벤트를 놓치지 않습니다.

## 분석 실행

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/analyze` | 분석 시작 → `session_id` 반환 (백그라운드 실행) |
| `GET` | `/api/explore` | 이전 분석이 제안한 주제로 후속 분석. 부모 세션에 연결 |
| `GET` | `/api/stream/{session_id}` | **SSE** 진행 상황 스트리밍 |
| `GET` | `/api/status/{session_id}` | 세션 상태 조회 |
| `GET` | `/api/stop/{session_id}` | 실행 중단 |
| `GET` | `/api/clarify` | 쿼리가 모호할 때(Human-in-the-Loop) 사용자 응답 전달 |
| `POST` | `/api/chat` | 완료된 분석 결과에 대한 후속 질의응답 |

### `/api/analyze` 파라미터

| 이름 | 기본값 | 설명 |
|---|:--:|---|
| `query` | *(필수)* | 연구 주제 |
| `provider` | `azure` | 기본 LLM provider |
| `routing_profile` | `optimized` | `optimized` \| `quality` |
| `fast_mode` | `false` | 속도 우선 모드 |
| `domain` | `auto` | 연구 도메인 |
| `year_range` | `auto` | 논문 연도 범위 |
| `output_language` | `auto` | 리포트 출력 언어 |
| `user_id` | `""` | 히스토리 분리용 식별자 |

```bash
curl "http://localhost:8000/api/analyze?query=domain%20adaptation%20in%20drug%20discovery&fast_mode=true"
# {"session_id":"..."}

curl -N "http://localhost:8000/api/stream/<session_id>"
```

## 히스토리 · 기타

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/history` | 분석 히스토리 목록 (`user_id` 로 필터) |
| `GET` | `/api/history/{filename}` | 개별 결과 조회 |
| `DELETE` | `/api/history/{filename}` | 결과 삭제 |
| `GET` | `/api/providers` | 사용 가능한 provider 목록 |
| `GET` | `/api/health` | 헬스 체크 |

## 정적 페이지

| 경로 | 내용 |
|---|---|
| `/` | 랜딩 페이지 (`landing/dist`). 미빌드 시 분석 앱으로 fallback |
| `/app` | 분석 앱 (`frontend/index.html`) |
