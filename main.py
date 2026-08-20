"""
GAPAGO - Research GAP Analysis Multi-Agent System

실행 진입점. 두 가지 모드를 지원한다.
  대화형 : python main.py
  배치   : python main.py --input data/input_sample.json --output results/output.json
"""

# =====================================================================
# 0. 최상위 import 는 표준 라이브러리만
# =====================================================================
# 무거운 의존성(langgraph/langchain)을 최상위에서 import 하지 않는다.
# 덕분에 의존성 설치 전에도 `python main.py --help` 가 동작한다.
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


@lru_cache(maxsize=1)
def get_graph():
    """LangGraph 그래프를 최초 1회만 빌드해 재사용한다 (api/main.py 와 동일한 패턴)."""
    from graphs.graph import build_graph
    return build_graph()


def _load_env():
    """GAPAGO 모듈보다 먼저 .env 를 로드한다. 각 서브커맨드 진입 시 최초 1회."""
    from core import config  # noqa: F401

async def aenumerate(aiter, start=0):
    i = start
    async for item in aiter:
        yield i, item
        i += 1

# =====================================================================
# 2. 출력 유틸
# =====================================================================
def random_uuid():
    return str(uuid.uuid4())

def print_divider(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def print_message(msg):
    # ToolMessage인 경우에만 요약 출력 (디버깅 효율)
    if msg.type == "tool":
        try:
            import json

            data = json.loads(msg.content)
            print(
                f"🛠️ [Tool: {msg.name}] {len(data)} results retrieved. (Top 1: {data[0].get('title', 'No Title')})"
            )
        except:
            print(f"🛠️ [Tool: {msg.name}] (Content too long to display)")
        return

    # 내부 파이프라인 전달용 메시지 — 터미널에 raw JSON 전체 출력 불필요
    # 최종 리포트(final_response)에서 정제된 형태로 출력됨
    _INTERNAL_AGENTS = {
        "gap_infer",
        "limitation_extract",
        "paper_retrieval",
        "meaning_expand",
        "recency_check",
        "limitation_eval",
    }
    msg_name = getattr(msg, "name", "") or ""
    if msg_name in _INTERNAL_AGENTS:
        content = getattr(msg, "content", "") or ""
        preview = content.replace("\n", " ")[:80]
        print(f"  [{msg_name}] {preview}{'...' if len(content) > 80 else ''}")
        return

    # Human, AI Message (query_analysis, clarify_prompt, critic_score, final_response 등)
    msg.pretty_print()


async def print_stream_events_and_capture_interrupt(app, stream_input, config_dict):
    """
    subgraphs=True로 이벤트를 출력하면서
    - clarify_prompt
    - interrupt 발생 여부
    를 함께 수집
    """
    interrupted = False
    latest_clarify_prompt = None

    async for i, event in aenumerate(app.astream(stream_input, config_dict, subgraphs=True)):
        path, update = event

        # subgraph 내부 이벤트는 건너뛰고 root 이벤트만 출력
        if path:
            # interrupt 체크는 subgraph에서도 필요
            for node, values in update.items():
                if node == "__interrupt__":
                    interrupted = True
                # clarify_prompt도 subgraph에서 발생
                if isinstance(values, dict):
                    for msg in values.get("messages", []):
                        if getattr(msg, "name", None) == "clarify_prompt":
                            latest_clarify_prompt = msg.content
            continue

        for node, values in update.items():
            if node == "__interrupt__":
                interrupted = True
                print("\n*** INTERRUPT ***")
                continue

            print(f"\n--- {node} ---")

            if not isinstance(values, dict):
                print(values)
                continue

            # 상태값 일부 출력
            for key in ["iteration", "is_ambiguous", "forced_proceed", "refined_query"]:
                if key in values:
                    print(f"{key} = {values[key]}")

            if "clarify_questions" in values:
                print("clarify_questions =", values["clarify_questions"])

            # 메시지 출력
            for msg in values.get("messages", []):
                print_message(msg)

                if getattr(msg, "name", None) == "clarify_prompt":
                    latest_clarify_prompt = msg.content

    return interrupted, latest_clarify_prompt

# =====================================================================
# 결과 저장 - evaluate.py가 이 파일을 읽습니다
# =====================================================================
def save_result(query: str, state_values: dict, output_path: str | None = None) -> Path:
    """
    파이프라인 완료 후 결과를 JSON 으로 저장.
    웹 API(_save_result)와 동일한 필드를 포함하여 evaluate.py 호환성 보장.

    output_path 가 주어지면 그 경로에 그대로 쓰고(배치 모드),
    없으면 outputs/gapago_result_YYYYMMDD_HHMMSS.json 으로 저장한다.
    """
    # messages 직렬화
    messages_out = []
    for msg in state_values.get("messages", []):
        messages_out.append({
            "type":    msg.type,
            "name":    getattr(msg, "name", None),
            "content": getattr(msg, "content", ""),
        })

    # papers 구조화 (웹과 동일)
    papers = state_values.get("papers", [])
    papers_out = []
    for p in papers:
        d = p if isinstance(p, dict) else (p.model_dump() if hasattr(p, "model_dump") else p.__dict__)
        papers_out.append({
            "paper_id": d.get("paper_id", ""),
            "title":    d.get("title", ""),
            "year":     d.get("year", 0),
            "authors":  d.get("authors", []),
            "url":      d.get("url", ""),
            "venue":    d.get("venue", ""),
        })

    result = {
        "query":         query,
        "timestamp":     datetime.now().isoformat(),
        "refined_query": state_values.get("refined_query", ""),
        "keywords":      state_values.get("keywords", []),
        "papers":        papers_out,
        "total_searched": state_values.get("total_candidates_count", len(papers)),
        "limitations":   state_values.get("limitations", []),
        "limitation_eval": state_values.get("limitation_eval", {}),
        "eval_warnings": state_values.get("eval_warnings", []),
        "gaps":          state_values.get("gaps", []),
        "web_results":   state_values.get("web_results", []),
        "paper_extraction_status": state_values.get("paper_extraction_status", []),
        "messages":      messages_out,
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        fname = f"gapago_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path  = OUTPUT_DIR / fname
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  ✅ 결과 저장 완료 → {path}")
    print(f"  평가 실행: python scripts/evaluate.py --result-file {path}")
    return path


# =====================================================================
# 3. 배치 실행 — 입력 JSON 하나로 무인 실행 (Docker / CI 용)
# =====================================================================
async def run_batch(graph, input_path: str | None = None, output_path: str | None = None,
                    query: str | None = None, overrides: dict | None = None):
    """
    사용자 입력 없이 파이프라인을 끝까지 실행한다.

    입력은 두 방식 중 하나:
      - query   : 주제 문자열을 직접 전달
      - input_path: 아래 형식의 JSON 파일
    CLI 플래그(overrides)는 JSON 값보다 우선한다.

    입력 JSON 형식:
      {
        "query":           "연구 질문 (필수)",
        "routing_profile": "optimized",   # optimized | quality
        "fast_mode":       false,
        "year_range":      "auto",        # auto | 1y | 3y | 5y
        "output_language": "auto"         # auto | ko | en
      }
    """
    import os
    from langchain_core.messages import HumanMessage
    from core.llm import get_llm
    from core.model_router import ModelRouter, PROFILE_DEFAULT_PROVIDERS

    # 무인 실행이므로 수집량을 보수적으로 고정 (환경변수로 덮어쓸 수 있음)
    os.environ.setdefault("ARXIV_MAX_RESULTS", "3")
    os.environ.setdefault("TAVILY_MAX_RESULTS", "3")
    os.environ.setdefault("SCIENCEON_DEFAULT_ROW_COUNT", "10")

    input_data = {}
    if input_path:
        input_data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if query:
        input_data["query"] = query          # 위치 인자가 JSON 의 query 를 덮어쓴다
    input_data.update(overrides or {})       # CLI 플래그가 최우선

    user_input = input_data.get("query", "")
    if not user_input:
        src = input_path or "(입력 없음)"
        raise ValueError(f"{src}: 'query' 가 비어 있습니다.")
    routing_profile = input_data.get("routing_profile", "optimized")
    fast_mode       = input_data.get("fast_mode", False)
    year_range      = input_data.get("year_range", "auto")
    output_language = input_data.get("output_language", "auto")

    # provider: 환경변수 우선, 비어 있으면 프로파일 기본값
    selected_provider = os.environ.get("LLM_PROVIDER") or PROFILE_DEFAULT_PROVIDERS.get(routing_profile, "azure")
    os.environ["LLM_PROVIDER"] = selected_provider

    print_divider("[배치 모드] 설정")
    print(f"  Query           : {user_input}")
    print(f"  Routing Profile : {routing_profile}")
    print(f"  LLM Provider    : {selected_provider}")
    print(f"  Fast Mode       : {fast_mode}")
    print(f"  Year Range      : {year_range}")
    print(f"  Output Language : {output_language}")
    print(f"  Output Path     : {output_path or f'{OUTPUT_DIR}/ (자동 생성)'}")

    get_llm.cache_clear()
    print(f"\n  [warmup] LLM ({selected_provider}) 초기화 중...")
    get_llm(provider=selected_provider)

    reasoning_provider = os.getenv("GAP_REASONING_PROVIDER", "")
    if reasoning_provider:
        print(f"  [warmup] GAP 추론 LLM ({reasoning_provider}) 초기화 중...")
        get_llm(provider=reasoning_provider)
    print("  [warmup] 완료")

    router = ModelRouter(default_provider=selected_provider, profile=routing_profile)
    config_dict = {"configurable": {"thread_id": random_uuid()}, "recursion_limit": 30}

    inputs = {
        "messages":        [HumanMessage(content=user_input)],
        "max_iterations":  3,
        "research_domain": "auto",
        "llm_provider":    selected_provider,
        "year_range":      year_range,
        "output_language": output_language,
        "fast_mode":       fast_mode,
        "model_routing":   router.to_dict(),
    }

    print_divider("[STEP 1] 파이프라인 시작")
    interrupted, latest_clarify_prompt = await print_stream_events_and_capture_interrupt(
        graph, inputs, config_dict
    )

    # Human-in-the-Loop 인터럽트: 배치 모드에서는 첫 번째 후보를 자동 선택
    while interrupted:
        print_divider("[INFO] Interrupt 감지 — 자동 진행 (배치 모드)")

        auto_response = user_input
        if latest_clarify_prompt:
            candidates = {
                m.group(1): m.group(2).strip()
                for m in re.finditer(r"^\s{0,4}(\d)\.\s+(.+)$", latest_clarify_prompt, re.MULTILINE)
            }
            if candidates:
                auto_response = candidates[min(candidates)]
                print(f"  → 자동 선택: {auto_response}")

        graph.update_state(config_dict, {"messages": [HumanMessage(content=auto_response)]})

        print_divider("[STEP 2] 파이프라인 재개")
        interrupted, latest_clarify_prompt = await print_stream_events_and_capture_interrupt(
            graph, None, config_dict
        )

    print_divider("[STEP 3] 결과 저장")
    final_state = graph.get_state(config_dict)
    values = final_state.values if final_state else {}
    save_result(user_input, values, output_path)


# =====================================================================
# 4. 대화형 실행
# =====================================================================
async def run(graph):
    config_dict = {"configurable": {"thread_id": random_uuid()}, "recursion_limit": 30} # 최대 노드 실행 개수 지정 (순환 로직에 빠지지 않기 위함)

    # --- 라우팅 프로파일 선택 (provider 선택보다 먼저) ---
    print("\n=== 라우팅 프로파일 선택 ===")
    print("  0) optimized - 에이전트별 최적화 (단순→groq, 핵심→claude)")
    print("  1) quality   - 최고 품질 (핵심 작업 claude 활용)")
    profile_map = {"0": "optimized", "1": "quality"}
    profile_choice = input("\n선택 (기본값: optimized) > ").strip()
    routing_profile = profile_map.get(profile_choice, "optimized")
    print(f"  → {routing_profile} 선택됨")

    # --- LLM Provider 선택 ---
    import os
    from langchain_core.messages import HumanMessage
    from core.model_router import ModelRouter, PROFILE_DEFAULT_PROVIDERS

    selected_provider = PROFILE_DEFAULT_PROVIDERS.get(routing_profile, "azure")
    print(f"\n  → 프로파일 '{routing_profile}'의 기본 provider: {selected_provider} (자동 선택)")

    os.environ["LLM_PROVIDER"] = selected_provider

    # lru_cache 초기화 (provider 변경 반영)
    from core.llm import get_llm
    get_llm.cache_clear()

    # --- LLM 사전 초기화 (warmup) ---
    # 기본 LLM 미리 캐시에 올림
    print(f"  [warmup] 기본 LLM ({selected_provider}) 초기화 중...")
    get_llm(provider=selected_provider)
    print(f"  [warmup] 기본 LLM 초기화 완료")

    # GAP 추론 LLM도 미리 캐시 (API 클라이언트 객체 생성, 수십 ms)
    reasoning_provider = os.getenv("GAP_REASONING_PROVIDER", "")
    if reasoning_provider:
        print(f"  [warmup] GAP 추론 LLM ({reasoning_provider}) 초기화 중...")
        get_llm(provider=reasoning_provider)
        print(f"  [warmup] GAP 추론 LLM 초기화 완료")

    # --- 분석 모드 선택 ---
    print("\n=== 분석 모드 선택 ===")
    print("  0) 일반 모드 - 정밀 분석 (기본값)")
    print("  1) Fast 모드 - 빠른 분석 (CrossEncoder 스킵, 상위 3개 축만 분석)")
    mode_choice = input("\n선택 (기본값: 일반) > ").strip()
    fast_mode = mode_choice == "1"
    print(f"  → {'⚡ Fast 모드' if fast_mode else '일반 모드'} 선택됨")

    # --- 연도 필터 선택 ---
    print("\n=== 검색 연도 범위 선택 ===")
    print("  0) auto - LLM이 자동 판단 (기본값)")
    print("  1) 1y   - 최근 1년")
    print("  2) 3y   - 최근 3년")
    print("  3) 5y   - 최근 5년")
    year_map = {"0": "auto", "1": "1y", "2": "3y", "3": "5y"}
    year_choice = input("\n선택 (기본값: auto) > ").strip()
    year_range = year_map.get(year_choice, "auto")
    print(f"  → {year_range} 선택됨")

    # --- 출력 언어 선택 ---
    print("\n=== 출력 언어 선택 ===")
    print("  0) auto - 입력 언어에 맞춤 (기본값)")
    print("  1) ko   - 한국어")
    print("  2) en   - English")
    lang_map = {"0": "auto", "1": "ko", "2": "en"}
    lang_choice = input("\n선택 (기본값: auto) > ").strip()
    output_language = lang_map.get(lang_choice, "auto")
    print(f"  → {output_language} 선택됨")

    # --- 사용자 입력 ---
    default_query = "Domain adaptation"
    user_input = input("\n연구 질문을 입력하세요: ").strip() or default_query
    if not user_input:
        user_input = "Domain adaptation in clinical drug"

    router = ModelRouter(default_provider=selected_provider, profile=routing_profile)

    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "max_iterations": 3,
        "research_domain": "auto",
        "llm_provider": selected_provider,
        "year_range": year_range,
        "output_language": output_language,
        "fast_mode": fast_mode,
        "model_routing": router.to_dict(),
    }

    print_divider("[STEP 1] 초기 실행")

    # 첫 실행은 inputs 사용
    interrupted, latest_clarify_prompt = await print_stream_events_and_capture_interrupt(
        graph, inputs, config_dict
    )

    # -----------------------------------------------------------------
    # Human-in-the-loop clarification loop
    # -----------------------------------------------------------------
    while interrupted:
        print_divider("[STEP 2] HUMAN CLARIFICATION 필요")

        if latest_clarify_prompt:
            print("\nAI 질문:")
            print(latest_clarify_prompt)
        else:
            print("\n질문을 더 구체화할 필요가 있습니다. 추가 정보를 입력해주세요.")

        # clarify_prompt 텍스트에서 번호별 direction 파싱 (TOO_BROAD 숫자 선택 지원)
        # 예: "  1. CNN-based image classification\n     └ ..."
        import re
        prompt_candidates = {}
        if latest_clarify_prompt:
            for m in re.finditer(r"^\s{0,4}(\d)\.\s+(.+)$", latest_clarify_prompt, re.MULTILINE):
                num, direction = m.group(1), m.group(2).strip()
                prompt_candidates[num] = direction

        user_response = ""
        while not user_response:
            user_response = input(
                "\n보완 답변 입력 > "
            ).strip()  ## ex. domain adaptation for fault detection in smart factory
            if not user_response:
                print("보완 답변을 입력해야 다음 단계로 진행할 수 있습니다.")
                continue

            # 숫자 입력 시 해당 번호의 direction으로 자동 치환
            if user_response in prompt_candidates:
                selected_direction = prompt_candidates[user_response]
                print(f"  → [{selected_direction}] 선택됨")
                user_response = selected_direction

        # 사용자 답변을 messages에 추가
        graph.update_state(
            config_dict,
            {
                "messages": [HumanMessage(content=user_response)],
            },
        )

        print_divider("[STEP 3] 파이프라인 재개")

        # resume 시에는 stream_input = None
        interrupted, latest_clarify_prompt = await print_stream_events_and_capture_interrupt(
            graph, None, config_dict
        )

    # -----------------------------------------------------------------
    # 최종 결과 출력
    # -----------------------------------------------------------------
    print_divider("[STEP 4] 최종 상태")

    final_state = graph.get_state(config_dict)
    values = final_state.values if final_state else {}

    print("next =", final_state.next if final_state else None)
    print("iteration =", values.get("iteration"))
    print("is_ambiguous =", values.get("is_ambiguous"))
    print("refined_query =", values.get("refined_query"))

    save_result(user_input, values)

    # -----------------------------------------------------------------
    # Gap Chat — 결과 검토 대화
    # -----------------------------------------------------------------
    if values.get("gaps"):
        print("\n결과에 대해 질문하시겠습니까?")
        chat_answer = input("(y/n, 기본값: n) > ").strip().lower()
        if chat_answer in ("y", "yes", "ㅛ"):
            from agents.gap_chat_agent import interactive_chat_loop
            interactive_chat_loop(values)

# =====================================================================
# 5. 진입점 — serve / analyze / chat
# =====================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="GAPAGO — 논문 한계점에서 미해결 연구 공백을 찾는 멀티 에이전트",
        epilog="예) python main.py serve\n"
               "    python main.py analyze \"domain adaptation in drug discovery\"\n"
               "    python main.py analyze --input data/input_sample.json --output results/out.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="{serve,analyze,chat}")

    # ── serve ──
    p_serve = sub.add_parser("serve", help="FastAPI 웹 서버 실행")
    p_serve.add_argument("--host", default="0.0.0.0", help="바인드 주소 (기본 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=None,
                         help="포트. 미지정 시 $PORT 환경변수, 그것도 없으면 8000")
    p_serve.add_argument("--reload", action="store_true", help="코드 변경 시 자동 재시작 (개발용)")

    # ── analyze ──
    p_an = sub.add_parser("analyze", help="무인 1회 분석 → JSON 저장")
    p_an.add_argument("query", nargs="?", help="연구 주제. --input 대신 문자열로 바로 전달")
    p_an.add_argument("--input", help="입력 JSON 경로 (query 외 옵션까지 담고 싶을 때)")
    p_an.add_argument("--output", help="결과 JSON 경로. 미지정 시 outputs/ 에 자동 생성")
    p_an.add_argument("--profile", choices=["optimized", "quality"], default=None,
                      help="모델 라우팅 프로파일 (기본 optimized)")
    p_an.add_argument("--fast", action="store_true", help="Fast 모드 — 속도 우선")
    p_an.add_argument("--year", choices=["auto", "1y", "3y", "5y"], default=None,
                      help="논문 연도 범위 (기본 auto)")
    p_an.add_argument("--lang", choices=["auto", "ko", "en"], default=None,
                      help="리포트 출력 언어 (기본 auto)")

    # ── chat ──
    sub.add_parser("chat", help="터미널 대화형 실행 (질문 되묻기 · 결과 후속 대화)")
    return parser


def _cmd_serve(args):
    import os
    _load_env()
    import uvicorn

    port = args.port or int(os.getenv("PORT", "8000"))
    if not (BASE_DIR / "landing" / "dist" / "index.html").exists():
        print("[안내] landing/dist 가 없어 / 는 분석 앱으로 대체됩니다.")
        print("       랜딩 페이지까지 보려면: cd landing && npm install && npm run build")
    # import 문자열로 넘겨야 --reload 가 동작하고, 이 프로세스가 FastAPI 를 직접 import 하지 않는다
    uvicorn.run("api.main:app", host=args.host, port=port, reload=args.reload, workers=1)


def _cmd_analyze(args, parser):
    if not args.query and not args.input:
        parser.error(
            "분석할 주제가 필요합니다.\n"
            '  python main.py analyze "domain adaptation in drug discovery"\n'
            "  python main.py analyze --input data/input_sample.json"
        )
    _load_env()
    overrides = {
        "routing_profile": args.profile,
        "fast_mode": True if args.fast else None,
        "year_range": args.year,
        "output_language": args.lang,
    }
    asyncio.run(run_batch(get_graph(), args.input, args.output,
                          query=args.query,
                          overrides={k: v for k, v in overrides.items() if v is not None}))


def main():
    parser = _build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args()

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "analyze":
        _cmd_analyze(args, parser)
    elif args.command == "chat":
        _load_env()
        asyncio.run(run(get_graph()))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
