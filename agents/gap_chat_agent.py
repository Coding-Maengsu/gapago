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
from llm import get_llm, get_llm_for_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from utils.parse_json import parse_json


def detect_user_intent(user_input: str, num_gaps: int, provider: str = None) -> dict:
    """
    사용자 입력의 의도를 파악

    Returns:
        {
            "intent": "question" | "exit" | "help" | "show_gap_detail",
            "gap_index": int | None,  # show_gap_detail일 때만
            "confidence": float
        }
    """
    llm = get_llm(provider=provider)

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


def _build_papers_context(papers: list) -> str:
    """논문 목록을 컨텍스트 문자열로 변환"""
    if not papers:
        return "(논문 없음)"
    lines = []
    for i, p in enumerate(papers, 1):
        if isinstance(p, dict):
            pid, title, year, authors = p.get("paper_id", ""), p.get("title", ""), p.get("year", 0), p.get("authors", [])
            abstract = p.get("abstract", "")
        else:
            pid, title, year, authors = p.paper_id, p.title, p.year, p.authors
            abstract = getattr(p, "abstract", "")
        author_str = ", ".join(authors[:3]) if authors else "N/A"
        abstract_preview = (abstract[:200] + "...") if len(abstract) > 200 else abstract
        lines.append(f"{i}. [{pid}] {title} ({year}) — {author_str}\n   Abstract: {abstract_preview}")
    return "\n".join(lines)


def _build_limitations_context(limitations: list) -> str:
    """limitation 목록을 컨텍스트 문자열로 변환"""
    if not limitations:
        return "(limitation 없음)"
    lines = []
    for i, lim in enumerate(limitations, 1):
        pid = lim.get("paper_id", "")
        claim = lim.get("claim", "")
        track = lim.get("track", "")
        section = lim.get("source_section", "")
        evidence = lim.get("evidence_quote", "")
        evidence_preview = (evidence[:150] + "...") if len(evidence) > 150 else evidence
        lines.append(f"{i}. [{pid}][{track}/{section}] {claim}\n   근거: \"{evidence_preview}\"")
    return "\n".join(lines)


def _build_gaps_context(gaps: list) -> str:
    """gap 목록을 상세 컨텍스트 문자열로 변환"""
    if not gaps:
        return "(GAP 없음)"
    lines = []
    for i, g in enumerate(gaps, 1):
        axis_label = g.get("axis_label", g.get("axis", ""))
        gap_statement = g.get("gap_statement", "")
        elaboration = g.get("elaboration", "")
        proposed_topic = g.get("proposed_topic", "")
        barriers = g.get("barriers", [])
        what_was_tried = g.get("what_was_tried", [])
        supporting_papers = g.get("supporting_papers", [])
        supporting_quotes = g.get("supporting_quotes", [])
        urgency = g.get("urgency_score", 0)
        novelty = g.get("novelty_score", 0)
        repeat_count = g.get("repeat_count", 0)

        block = (
            f"### GAP #{i} — [{axis_label}] ({repeat_count}개 논문)\n"
            f"**Gap Statement**: {gap_statement}\n"
            f"**상세 설명**: {elaboration}\n"
            f"**제안 연구 주제**: {proposed_topic}\n"
            f"**기술적 장벽**: {'; '.join(barriers[:3]) if barriers else 'N/A'}\n"
            f"**기존 시도**: {'; '.join(what_was_tried[:3]) if what_was_tried else 'N/A'}\n"
            f"**지지 논문**: {', '.join(supporting_papers[:5])}\n"
            f"**주요 인용**: {'; '.join(q[:100] for q in supporting_quotes[:2]) if supporting_quotes else 'N/A'}\n"
            f"**긴급도**: {urgency}/10 | **참신성**: {novelty}/10"
        )
        lines.append(block)
    return "\n\n".join(lines)


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

    # 상세 컨텍스트 구성
    papers_context = _build_papers_context(papers)
    limitations_context = _build_limitations_context(limitations[:30])  # 최대 30개
    gaps_context = _build_gaps_context(gaps)

    # 대화 히스토리 (최근 10개)
    recent_messages = state.get("messages", [])[-10:]

    # 시스템 프롬프트
    system_prompt = f"""당신은 연구 GAP 분석 전문가입니다.

아래의 분석 결과 전체를 숙지하고, 사용자의 질문에 정확하게 답변하세요.

---

## 연구 질문
{refined_query}

---

## 분석된 논문 ({len(papers)}편)
{papers_context}

---

## 추출된 Limitation ({len(limitations)}개)
{limitations_context}

---

## 발견된 Research GAP ({len(gaps)}개)
{gaps_context}

---

## 답변 지침
1. 위 데이터를 근거로 정확하게 답변. 데이터에 없는 내용을 지어내지 마세요.
2. 논문 언급 시 paper_id와 제목을 함께 제시.
3. GAP 언급 시 번호와 축(axis) 이름을 함께 제시.
4. limitation 언급 시 어떤 논문에서 나온 것인지 명시.
5. 사용자가 특정 번호를 물으면 해당 항목의 상세 정보를 제공.
6. 추가 분석이 필요한 경우 솔직하게 언급.
7. 간결하고 명확하게 답변 (불필요한 반복 제거).
"""

    # 메시지 구성
    messages = [SystemMessage(content=system_prompt)]

    # 최근 대화 히스토리 추가
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            messages.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage) and getattr(msg, "name", None) == "gap_chat":
            messages.append(AIMessage(content=msg.content))

    # 현재 질문 추가
    messages.append(HumanMessage(content=user_question))

    # LLM 호출
    llm = get_llm_for_agent(state, "gap_chat")
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
            intent_info = detect_user_intent(user_input, len(gaps), provider=state.get("llm_provider"))
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
