"""
GAPAGO - Research GAP Analysis Multi-Agent System
기존 모듈(agents/, states.py, graph.py, llm.py, utils/)을 활용한 실행 진입점
"""

# =====================================================================
# 0. 환경 설정
# =====================================================================
import json
import config  # noqa: F401
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

    # 그 외 Human, AI Message는 깔끔하게 출력
    msg.pretty_print()


def print_stream_events_and_capture_interrupt(app, stream_input, config_dict):
    """
    subgraphs=True로 이벤트를 출력하면서
    - clarify_prompt
    - interrupt 발생 여부
    를 함께 수집
    """
    interrupted = False
    latest_clarify_prompt = None

    for i, event in enumerate(app.stream(stream_input, config_dict, subgraphs=True)):
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
 
    저장 내용:
      - query / refined_query / keywords
      - gaps     : gap_infer_node가 state["gaps"]에 저장한 구조화 데이터
                   (repeat_count 내림차순 정렬 = 가장 시급한 GAP 순서)
      - messages : 각 agent의 AIMessage 원문 (name 포함)
    """
    # messages 직렬화
    messages_out = []
    for msg in state_values.get("messages", []):
        messages_out.append({
            "type":    msg.type,
            "name":    getattr(msg, "name", None),
            "content": getattr(msg, "content", ""),
        })
 
    result = {
        "query":         query,
        "timestamp":     datetime.now().isoformat(),
        "refined_query": state_values.get("refined_query", ""),
        "keywords":      state_values.get("keywords", []),
 
        # ★ 핵심: gap_infer_node가 state["gaps"]에 저장한 구조화 데이터
        #   repeat_count 내림차순 정렬 상태 그대로 저장
        "limitations": state_values.get("limitations", []),
        "gaps": state_values.get("gaps", []),
        "web_results": state_values.get("web_results", []),

        "messages": messages_out,
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
def run():
    config_dict = {"configurable": {"thread_id": random_uuid()}, "recursion_limit": 30} # 최대 노드 실행 개수 지정 (순환 로직에 빠지지 않기 위함)

    # --- LLM Provider 선택 ---
    from llm import select_provider_interactive
    import os
    selected_provider = select_provider_interactive()
    os.environ["LLM_PROVIDER"] = selected_provider

    # lru_cache 초기화 (provider 변경 반영)
    from llm import get_llm
    get_llm.cache_clear()

    # --- 연구 도메인 선택 (recency check용) ---
    print("\n=== 연구 도메인 선택 (최신성 검증 소스 결정) ===")
    print("  0) auto - LLM이 자동 판단 (기본값)")
    print("  1) ai_cs - AI / Computer Science")
    print("  2) biomedical - 바이오 / 의학")
    print("  3) materials_chemistry - 재료 / 화학")
    print("  4) physics - 물리")
    print("  5) general - 범용 (전체 웹)")
    domain_map = {"0": "auto", "1": "ai_cs", "2": "biomedical",
                  "3": "materials_chemistry", "4": "physics", "5": "general"}
    domain_choice = input("\n선택 (기본값: auto) > ").strip()
    research_domain = domain_map.get(domain_choice, "auto")
    print(f"  → {research_domain} 선택됨")

    # --- 사용자 입력 ---
    default_query = "Domain adaptation"
    user_input = input("\n연구 질문을 입력하세요: ").strip() or default_query
    if not user_input:
        user_input = "Domain adaptation in clinical drug"

    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "max_iterations": 3,
        "research_domain": research_domain,
    }

    print_divider("[STEP 1] 초기 실행")

    # 첫 실행은 inputs 사용
    interrupted, latest_clarify_prompt = print_stream_events_and_capture_interrupt(
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
        interrupted, latest_clarify_prompt = print_stream_events_and_capture_interrupt(
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

if __name__ == "__main__":
    run()