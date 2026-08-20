"""
GAPAGO - Research GAP Analysis Multi-Agent System
기존 모듈(agents/, states.py, graph.py, llm.py, utils/)을 활용한 실행 진입점
"""

# =====================================================================
# 0. 환경 설정
# =====================================================================
import asyncio
import json
from core import config  # noqa: F401
import uuid
from pathlib import Path
from datetime import datetime
# =====================================================================
# 1. 그래프 빌드
# =====================================================================
from graphs.graph import build_graph
from langchain_core.messages import HumanMessage

app = build_graph()
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

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
def save_result(query: str, state_values: dict) -> Path:
    """
    파이프라인 완료 후 결과를 outputs/gapago_result_YYYYMMDD_HHMMSS.json 으로 저장.
    웹 API(_save_result)와 동일한 필드를 포함하여 evaluate.py 호환성 보장.
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

    fname = f"gapago_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path  = OUTPUT_DIR / fname
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  ✅ 결과 저장 완료 → {path}")
    print(f"  평가 실행: python evaluate.py --result-file {path}")
    return path


# =====================================================================
# 3. 실행 로직
# =====================================================================
async def run():
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
        app, inputs, config_dict
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
        app.update_state(
            config_dict,
            {
                "messages": [HumanMessage(content=user_response)],
            },
        )

        print_divider("[STEP 3] 파이프라인 재개")

        # resume 시에는 stream_input = None
        interrupted, latest_clarify_prompt = await print_stream_events_and_capture_interrupt(
            app, None, config_dict
        )

    # -----------------------------------------------------------------
    # 최종 결과 출력
    # -----------------------------------------------------------------
    print_divider("[STEP 4] 최종 상태")

    final_state = app.get_state(config_dict)
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

if __name__ == "__main__":
    asyncio.run(run())