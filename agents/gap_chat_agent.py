"""
GAP Chat Agent — 최종 결과 검토 후 사용자와 대화

기능:
  - GAP 분석 결과에 대한 질문 답변
  - 특정 축이나 방향에 대한 추가 설명
  - 재분석 요청 처리
  - 대안 제안 요청
  - 자연어 기반 의도 파악 (종료 의도, 질문 의도 등)
"""

from states import AgentState
from llm import get_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from utils.parse_json import parse_json


def detect_user_intent(user_input: str, num_gaps: int) -> dict:
    """
    사용자 입력의 의도를 파악

    Returns:
        {
            "intent": "question" | "exit" | "help" | "show_gap_detail",
            "gap_index": int | None,  # show_gap_detail일 때만
            "confidence": float
        }
    """
    llm = get_llm()

    prompt = f"""사용자 입력의 의도를 파악하세요.

사용자 입력: "{user_input}"

현재 컨텍스트:
- GAP 분석 결과 검토 대화 중
- 총 {num_gaps}개의 GAP이 발견됨

가능한 의도:
1. "exit": 대화 종료 의도 (예: "끝", "종료", "그만", "나가기", "됐어", "괜찮아")
2. "help": 도움말 요청 (예: "도움말", "help", "뭘 할 수 있어?", "사용법")
3. "show_gap_detail": 특정 GAP 상세 조회 (예: "1번 보여줘", "첫 번째 GAP", "2번 자세히")
4. "question": 일반 질문 (분석 결과에 대한 질문, 추가 설명 요청 등)

JSON만 출력:
{{
  "intent": "<exit|help|show_gap_detail|question>",
  "gap_index": <1-based index or null>,
  "reasoning": "<1-2 문장으로 판단 근거>"
}}"""

    messages = [
        SystemMessage(content="You are an intent classifier. Always respond with valid JSON only."),
        HumanMessage(content=prompt)
    ]

    try:
        response = llm.invoke(messages)
        result = parse_json(response.content)
        return {
            "intent": result.get("intent", "question"),
            "gap_index": result.get("gap_index"),
            "reasoning": result.get("reasoning", "")
        }
    except Exception:
        # 파싱 실패 시 기본값
        return {"intent": "question", "gap_index": None, "reasoning": ""}


def gap_chat_respond(state: AgentState, user_question: str) -> str:
    """
    사용자 질문에 대해 GAP 분석 결과를 기반으로 답변 생성

    Args:
        state: 현재 AgentState (gaps, limitations, papers 등 포함)
        user_question: 사용자 질문

    Returns:
        AI 답변 텍스트
    """
    # 컨텍스트 수집
    gaps = state.get("gaps", [])
    limitations = state.get("limitations", [])
    refined_query = state.get("refined_query", "")
    papers = state.get("papers", [])

    # GAP 요약
    gaps_summary = ""
    if gaps:
        gaps_summary = "\n\n".join([
            f"**{i+1}. [{g.get('axis_label', g.get('axis', ''))}]** {g.get('gap_statement', '')}\n"
            f"   제안 주제: {g.get('proposed_topic', '')}\n"
            f"   지지 논문: {len(g.get('supporting_papers', []))}개"
            for i, g in enumerate(gaps[:5])  # 상위 5개만
        ])

    # Limitation 요약
    lim_summary = f"{len(limitations)}개 limitation 추출됨 ({len(papers)}개 논문 분석)"

    # 대화 히스토리 (최근 5개만)
    recent_messages = state.get("messages", [])[-10:]

    # 시스템 프롬프트
    system_prompt = f"""당신은 연구 GAP 분석 전문가입니다.

다음 분석 결과를 바탕으로 사용자의 질문에 답변하세요:

**연구 질문**: {refined_query}

**발견된 주요 GAP** (상위 5개):
{gaps_summary if gaps_summary else "(GAP 없음)"}

**분석 통계**:
- {lim_summary}
- GAP 후보: {len(gaps)}개

**답변 지침**:
1. 분석 결과를 기반으로 정확하게 답변
2. 구체적인 논문 ID나 축(axis) 이름 등 근거 제시
3. 추가 분석이 필요한 경우 솔직하게 언급
4. 사용자가 "재분석" 요청 시 어떤 부분을 어떻게 조정할지 제안
5. 간결하고 명확하게 답변 (불필요한 반복 제거)

**가능한 질문 유형**:
- 특정 GAP에 대한 상세 설명 요청
- 특정 축(axis)에 대한 질문
- 다른 연구 방향 제안 요청
- 특정 논문 관련 질문
- 결과 해석에 대한 질문
"""

    # 메시지 구성
    messages = [SystemMessage(content=system_prompt)]

    # 최근 대화 히스토리 추가
    for msg in recent_me좋ssages:
        if isinstance(msg, HumanMessage):
            messages.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage) and getattr(msg, "name", None) == "gap_chat":
            messages.append(AIMessage(content=msg.content))

    # 현재 질문 추가
    messages.append(HumanMessage(content=user_question))

    # LLM 호출
    llm = get_llm(provider=state.get("llm_provider"))
    response = llm.invoke(messages)

    return response.content


def format_gap_details(gap: dict) -> str:
    """GAP 상세 정보를 보기 좋게 포맷팅"""
    details = f"""
**축(Axis)**: {gap.get('axis_label', gap.get('axis', 'N/A'))} ({gap.get('axis_type', 'fixed')})

**GAP 진술**: {gap.get('gap_statement', 'N/A')}

**제안 연구 주제**: {gap.get('proposed_topic', 'N/A')}

**상세 설명**:
{gap.get('elaboration', 'N/A')}

**지지 근거**:
- 관련 논문: {len(gap.get('supporting_papers', []))}개
- 반복 언급 횟수: {gap.get('repeat_count', 0)}회

**지지 논문들**: {', '.join(gap.get('supporting_papers', [])[:5])}

**주요 인용구**:
"""
    quotes = gap.get('supporting_quotes', [])
    if quotes:
        for i, quote in enumerate(quotes[:3], 1):
            details += f"\n  {i}. \"{quote[:150]}...\""
    else:
        details += "\n  (인용구 없음)"

    return details


def interactive_chat_loop(state: AgentState):
    """
    결과 검토 후 대화형 루프 실행

    사용자 의도를 LLM이 자동으로 파악하여 처리
    """
    print("\n" + "="*70)
    print("💬 GAP 분석 결과 대화 모드")
    print("="*70)
    print("\n분석이 완료되었습니다. 결과에 대해 자유롭게 질문하거나 대화하세요.\n")

    gaps = state.get("gaps", [])

    # 간단한 결과 요약 출력
    print(f"📊 발견된 GAP: {len(gaps)}개")
    if gaps:
        print("\n주요 GAP 목록:")
        for i, gap in enumerate(gaps[:5], 1):
            print(f"  {i}. [{gap.get('axis_label', 'N/A')}] {gap.get('gap_statement', '')[:80]}...")
    print()

    # 대화 루프
    while True:
        try:
            user_input = input("\n💬 > ").strip()

            if not user_input:
                continue

            # 사용자 의도 파악
            intent_info = detect_user_intent(user_input, len(gaps))
            intent = intent_info.get("intent", "question")

            # 의도에 따라 처리
            if intent == "exit":
                print("\n대화를 종료합니다.")
                break

            elif intent == "help":
                print("""
💡 이 대화 모드에서 할 수 있는 것:

  📋 GAP 상세 조회:
     "1번 보여줘", "첫 번째 GAP 자세히", "2번은 뭐야?"

  ❓ 자유로운 질문:
     "가장 시급한 GAP은 무엇인가요?"
     "Data 축에서 다른 연구 방향은 없나요?"
     "이 결과를 어떻게 활용할 수 있나요?"
     "특정 논문에 대해 설명해줘"

  👋 종료:
     "끝", "종료", "그만", "됐어" 등 자연스럽게 말하면 됩니다

  💬 자연스럽게 대화하듯이 입력하세요!
""")
                continue

            elif intent == "show_gap_detail":
                gap_idx = intent_info.get("gap_index")
                if gap_idx and 1 <= gap_idx <= len(gaps):
                    print("\n" + format_gap_details(gaps[gap_idx - 1]))
                else:
                    # 인덱스가 불명확하면 LLM에게 질문으로 처리
                    print("\n🤖 답변 생성 중...", end="", flush=True)
                    response = gap_chat_respond(state, user_input)
                    print("\r" + " "*30 + "\r", end="")
                    print(f"\n🤖 {response}\n")
                    state["messages"].append(HumanMessage(content=user_input))
                    state["messages"].append(AIMessage(content=response, name="gap_chat"))
                continue

            else:  # question
                # AI 응답 생성
                print("\n🤖 답변 생성 중...", end="", flush=True)
                response = gap_chat_respond(state, user_input)
                print("\r" + " "*30 + "\r", end="")  # 진행 메시지 지우기

                # 응답 출력
                print(f"\n🤖 {response}\n")

                # 대화 히스토리에 추가
                state["messages"].append(HumanMessage(content=user_input))
                state["messages"].append(AIMessage(content=response, name="gap_chat"))

        except KeyboardInterrupt:
            print("\n\n대화를 종료합니다.")
            break
        except Exception as e:
            print(f"\n⚠️ 오류 발생: {e}")
            print("다시 시도해주세요.")
