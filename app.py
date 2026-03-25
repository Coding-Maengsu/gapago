"""
GAPAGO - Streamlit Demo UI
Research GAP Analysis Multi-Agent System
"""

import os
import json
import uuid
import streamlit as st
from datetime import datetime
from pathlib import Path

# ── 페이지 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="GAPAGO - Research GAP Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 환경 초기화 (최초 1회) ────────────────────────────────────────
import config  # noqa: F401  (.env 로드, LangSmith 등)

from graphs.graph import build_graph
from langchain_core.messages import HumanMessage
from llm import AVAILABLE_PROVIDERS, get_llm

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =====================================================================
# 히스토리 관리
# =====================================================================
def _load_history() -> list[dict]:
    """outputs/ 폴더의 결과 파일들을 시간순으로 로드 (최신 먼저)."""
    files = sorted(OUTPUT_DIR.glob("gapago_result_*.json"), reverse=True)
    history = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            history.append({
                "file": f,
                "query": data.get("query", "(no query)"),
                "timestamp": data.get("timestamp", ""),
                "refined_query": data.get("refined_query", ""),
                "gaps_count": len(data.get("gaps", [])),
                "data": data,
            })
        except Exception:
            continue
    return history


def _format_timestamp(ts: str) -> str:
    """ISO timestamp → 간결한 표시 형식."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%m/%d %H:%M")
    except Exception:
        return ts[:16] if ts else ""


# =====================================================================
# 유틸
# =====================================================================
def _clean_report_markdown(text: str) -> str:
    """
    response_agent 출력의 마크다운을 정리.
    - 유니코드 구분선(────)을 마크다운 hr(---)로 교체
    - 마크다운 테이블 앞뒤 공백 확보
    """
    import re
    # 유니코드 수평선 → markdown hr
    text = re.sub(r'[─━═]{4,}', '---', text)
    # FINAL ANSWER 태그 제거
    text = re.sub(r'FINAL ANSWER\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _get_node_label(name: str) -> tuple[str, str]:
    """노드 이름 → (아이콘, 한글 라벨)"""
    labels = {
        "query_subgraph":     ("🔍", "Query Analysis"),
        "meaning_expand":     ("📖", "Meaning Expansion"),
        "paper_retrieval":    ("📄", "Paper Retrieval"),
        "limitation_extract": ("⚠️", "Limitation Extraction"),
        "limitation_eval":    ("✅", "Limitation Evaluation"),
        "recency_check":      ("🕐", "Recency Check"),
        "gap_infer":          ("💡", "GAP Inference"),
        "critic_score":       ("📊", "Critic Scoring"),
        "final_response":     ("📝", "Final Report"),
    }
    icon, label = labels.get(name, ("⚙️", name))
    return icon, label


PIPELINE_NODES = [
    "query_subgraph", "meaning_expand", "paper_retrieval",
    "limitation_extract", "limitation_eval", "recency_check",
    "gap_infer", "critic_score", "final_response",
]


# =====================================================================
# 사이드바
# =====================================================================
with st.sidebar:
    st.title("🔬 GAPAGO")
    st.caption("Research GAP Analysis System")

    # 새 분석 버튼
    if st.button("➕ 새 분석", use_container_width=True):
        st.session_state.pop("selected_history", None)
        st.session_state.pop("loaded_result", None)
        st.session_state.pop("show_loaded", None)
        st.rerun()

    st.divider()

    # ── 설정 (접을 수 있는 영역) ──
    provider_options = {v[1]: v[0] for k, v in AVAILABLE_PROVIDERS.items()}
    domain_options = {
        "auto (자동 판단)": "auto",
        "AI / Computer Science": "ai_cs",
        "Biomedical / 의학": "biomedical",
        "Materials / Chemistry": "materials_chemistry",
        "Physics": "physics",
        "General (범용)": "general",
    }

    with st.expander("⚙️ 설정", expanded=False):
        provider_display = st.selectbox(
            "LLM Provider",
            list(provider_options.keys()),
            index=0,
        )

        domain_display = st.selectbox(
            "Research Domain",
            list(domain_options.keys()),
            index=0,
        )

    selected_provider = provider_options[provider_display]
    selected_domain = domain_options[domain_display]

    st.divider()

    # ── 분석 히스토리 ──
    st.subheader("📂 분석 기록")
    history = _load_history()

    if history:
        for i, item in enumerate(history):
            ts = _format_timestamp(item["timestamp"])
            query_preview = item["query"][:35] + ("..." if len(item["query"]) > 35 else "")
            gaps_count = item["gaps_count"]

            # 각 기록을 클릭 가능한 버튼으로 표시
            btn_label = f"🔹 {query_preview}\n{ts} · GAP {gaps_count}개"
            if st.button(
                btn_label,
                key=f"history_{i}",
                use_container_width=True,
            ):
                st.session_state["loaded_result"] = item["data"]
                st.session_state["show_loaded"] = True
                st.session_state["selected_history"] = i
                st.rerun()

        st.divider()
        st.caption(f"총 {len(history)}개 기록")
    else:
        st.caption("아직 분석 기록이 없습니다.")


# =====================================================================
# 메인 화면
# =====================================================================
st.header("🔬 GAPAGO — Research GAP Analyzer")

# 연구 질문 입력
query = st.text_input(
    "연구 질문을 입력하세요",
    placeholder="예: Domain adaptation for fault detection in smart manufacturing",
    key="query_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("🚀 분석 시작", type="primary", use_container_width=True)


# =====================================================================
# 파이프라인 실행
# =====================================================================
def _stream_and_render(app, stream_input, config_dict, progress_bar,
                       node_containers, completed_nodes, node_results):
    """
    app.stream()을 돌면서 노드 결과를 렌더링.
    interrupt 발생 시 clarify_prompt를 반환.
    Returns: (interrupted: bool, clarify_prompt: str|None)
    """
    interrupted = False
    clarify_prompt = None

    for event in app.stream(stream_input, config_dict, subgraphs=True):
        path, update = event

        # subgraph 내부 이벤트
        if path:
            for node, values in update.items():
                if node == "__interrupt__":
                    interrupted = True
                if isinstance(values, dict):
                    for msg in values.get("messages", []):
                        if getattr(msg, "name", None) == "scope_prompt":
                            clarify_prompt = msg.content
            continue

        for node, values in update.items():
            if node == "__interrupt__":
                interrupted = True
                continue
            if node.startswith("__"):
                continue

            completed_nodes.append(node)
            node_results[node] = values

            # 진행률 업데이트
            idx = PIPELINE_NODES.index(node) + 1 if node in PIPELINE_NODES else len(completed_nodes)
            pct = min(idx / len(PIPELINE_NODES), 1.0)
            icon, label = _get_node_label(node)
            progress_bar.progress(pct, text=f"{icon} {label} 완료")

            # 노드별 결과 표시
            if node in node_containers:
                _render_node_result(node, values, node_containers[node])

    return interrupted, clarify_prompt


def run_pipeline(query: str, provider: str, domain: str):
    """파이프라인을 실행하며 각 노드 결과를 실시간으로 표시."""
    # LLM provider 설정
    os.environ["LLM_PROVIDER"] = provider
    get_llm.cache_clear()

    app = build_graph()
    config_dict = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 30,
    }

    inputs = {
        "messages": [HumanMessage(content=query)],
        "max_iterations": 3,
        "research_domain": domain,
    }

    # 진행률 표시
    progress_bar = st.progress(0, text="파이프라인 시작...")

    # 노드별 결과 저장
    node_results = {}
    completed_nodes = []

    # 각 노드별 expander 미리 생성
    node_containers = {}
    for node_name in PIPELINE_NODES:
        icon, label = _get_node_label(node_name)
        node_containers[node_name] = st.expander(f"{icon} {label}", expanded=False)

    # 스트림 실행
    try:
        interrupted, clarify_prompt = _stream_and_render(
            app, inputs, config_dict, progress_bar,
            node_containers, completed_nodes, node_results,
        )

        # ── Human-in-the-loop: interrupt 처리 ──
        # interrupt 발생 시 세션에 저장하고 rerun으로 입력 받기
        if interrupted:
            st.session_state["pipeline_app"] = app
            st.session_state["pipeline_config"] = config_dict
            st.session_state["pipeline_progress"] = progress_bar
            st.session_state["pipeline_node_containers"] = node_containers
            st.session_state["pipeline_completed"] = completed_nodes
            st.session_state["pipeline_results"] = node_results
            st.session_state["pipeline_interrupted"] = True
            st.session_state["pipeline_clarify_prompt"] = clarify_prompt
            return None  # 메인 로직에서 interrupt UI 처리

    except Exception as e:
        st.error(f"파이프라인 실행 중 오류: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

    progress_bar.progress(1.0, text="✅ 파이프라인 완료!")

    # 최종 state 가져오기
    final_state = app.get_state(config_dict)
    state_values = final_state.values if final_state else {}

    # 결과 저장
    _save_result(query, state_values)

    return state_values


def resume_pipeline(user_response: str):
    """interrupt 후 사용자 입력을 받아 파이프라인을 재개."""
    app = st.session_state["pipeline_app"]
    config_dict = st.session_state["pipeline_config"]
    node_containers = st.session_state["pipeline_node_containers"]
    completed_nodes = st.session_state["pipeline_completed"]
    node_results = st.session_state["pipeline_results"]

    # 사용자 응답을 state에 주입
    app.update_state(
        config_dict,
        {"messages": [HumanMessage(content=user_response)]},
    )

    progress_bar = st.progress(
        len(completed_nodes) / len(PIPELINE_NODES),
        text="파이프라인 재개...",
    )

    try:
        interrupted, clarify_prompt = _stream_and_render(
            app, None, config_dict, progress_bar,
            node_containers, completed_nodes, node_results,
        )

        if interrupted:
            # 또 interrupt (반복 clarification)
            st.session_state["pipeline_clarify_prompt"] = clarify_prompt
            st.session_state["pipeline_interrupted"] = True
            return None

    except Exception as e:
        st.error(f"파이프라인 재개 중 오류: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

    # interrupt 상태 정리
    for key in ["pipeline_app", "pipeline_config", "pipeline_progress",
                "pipeline_node_containers", "pipeline_completed",
                "pipeline_results", "pipeline_interrupted",
                "pipeline_clarify_prompt"]:
        st.session_state.pop(key, None)

    progress_bar.progress(1.0, text="✅ 파이프라인 완료!")

    final_state = app.get_state(config_dict)
    state_values = final_state.values if final_state else {}
    _save_result(st.session_state.get("query_input", ""), state_values)

    return state_values


def _render_node_result(node: str, values: dict, container):
    """노드별 결과를 expander 안에 렌더링."""
    if not isinstance(values, dict):
        return

    with container:
        # ── Query Analysis ──
        if node == "query_subgraph":
            if values.get("refined_query"):
                st.success(f"**Refined Query:** {values['refined_query']}")
            if values.get("keywords"):
                st.write("**Keywords:**", ", ".join(values["keywords"]))
            if values.get("scope_level"):
                st.info(f"Scope: {values['scope_level']}")

        # ── Meaning Expansion ──
        elif node == "meaning_expand":
            msgs = values.get("messages", [])
            for msg in msgs:
                content = getattr(msg, "content", "")
                if content:
                    st.text(content[:500])

        # ── Paper Retrieval ──
        elif node == "paper_retrieval":
            papers = values.get("papers", [])
            web_results = values.get("web_results", [])
            st.metric("논문 수", len(papers))
            if papers:
                import pandas as pd
                rows = []
                for p in papers:
                    if isinstance(p, dict):
                        rows.append({
                            "ID": p.get("paper_id", "")[:20],
                            "Title": p.get("title", "")[:60],
                            "Year": p.get("year", ""),
                            "Source": p.get("paper_id", "").split(":")[0] if ":" in p.get("paper_id", "") else "",
                        })
                    else:
                        rows.append({
                            "ID": p.paper_id[:20],
                            "Title": p.title[:60],
                            "Year": p.year,
                            "Source": p.paper_id.split(":")[0] if ":" in p.paper_id else "",
                        })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if web_results:
                st.caption(f"웹 검색 결과: {len(web_results)}건")

        # ── Limitation Extraction ──
        elif node == "limitation_extract":
            limitations = values.get("limitations", [])
            st.metric("추출된 Limitations", len(limitations))
            for i, lim in enumerate(limitations):
                track_badge = "🟢 Author" if lim.get("track") == "author_stated" else "🔵 Structural"
                st.markdown(
                    f"**{i+1}.** {track_badge} `{lim.get('source_section', '')}`\n\n"
                    f"> {lim.get('claim', '')}\n\n"
                    f"*Evidence:* _{lim.get('evidence_quote', '')[:150]}_"
                )
                st.divider()

        # ── Limitation Evaluation ──
        elif node == "limitation_eval":
            eval_data = values.get("limitation_eval", {})
            limitations = values.get("limitations", [])
            warnings = values.get("eval_warnings", [])

            # 판정 결과
            decision = eval_data.get("decision", "N/A")
            if decision == "PASS":
                st.success(f"🎯 Decision: **{decision}**")
            else:
                st.warning(f"🔄 Decision: **{decision}**")

            # 경고
            for w in warnings:
                st.warning(w)

            # 점수 차트
            call1 = eval_data.get("call1_results", [])
            if call1:
                import pandas as pd
                scores_data = []
                for r in call1:
                    lid = r.get("limitation_id", "?")
                    scores_data.append({
                        "ID": f"Lim {lid}",
                        "Fact Score": r.get("fact_score", 0),
                        "Groundedness": r.get("groundedness", 0) / 5.0,
                        "Specificity": r.get("specificity", 0) / 5.0,
                        "Relevance": r.get("relevance", 0) / 5.0,
                    })
                df = pd.DataFrame(scores_data)
                st.bar_chart(df.set_index("ID"), height=250)

            # 유형 분포
            call2 = eval_data.get("call2_result", {})
            type_dist = call2.get("type_distribution", {})
            if type_dist:
                st.subheader("Limitation Type Distribution")
                import pandas as pd
                dist_df = pd.DataFrame([
                    {"Type": k, "Count": v}
                    for k, v in type_dist.items() if v
                ])
                if not dist_df.empty:
                    st.bar_chart(dist_df.set_index("Type"), height=200)

            # 필터링된 limitations
            st.metric("평가 통과 Limitations", len(limitations))

        # ── Recency Check ──
        elif node == "recency_check":
            limitations = values.get("limitations", [])
            msgs = values.get("messages", [])
            for msg in msgs:
                content = getattr(msg, "content", "")
                if content:
                    st.info(content)

            # recency 상태 요약
            status_counts = {"unresolved": 0, "partial": 0, "resolved": 0}
            for lim in limitations:
                s = lim.get("recency_status", "unresolved")
                status_counts[s] = status_counts.get(s, 0) + 1

            c1, c2, c3 = st.columns(3)
            c1.metric("Unresolved", status_counts["unresolved"])
            c2.metric("Partial", status_counts["partial"])
            c3.metric("Resolved", status_counts["resolved"])

        # ── GAP Inference ──
        elif node == "gap_infer":
            gaps = values.get("gaps", [])
            st.metric("Research GAPs", len(gaps))

            for i, gap in enumerate(gaps):
                count = gap.get("repeat_count", 0)
                stars = "⭐⭐⭐" if i == 0 else ("⭐⭐" if i <= 2 else "⭐")
                axis_type = "🔵" if gap.get("axis_type") == "fixed" else "🟢"

                st.markdown(f"""
### {stars} GAP #{i+1} — {axis_type} {gap.get('axis_label', '')} ({count}개 논문)

**{gap.get('gap_statement', '')}**

{gap.get('elaboration', '')}

📌 **Proposed Topic:** _{gap.get('proposed_topic', '')}_

_Supporting papers: {', '.join(gap.get('supporting_papers', [])[:5])}_
""")
                st.divider()

        # ── Critic Score ──
        elif node == "critic_score":
            msgs = values.get("messages", [])
            for msg in msgs:
                content = getattr(msg, "content", "")
                if content:
                    st.code(content[:1000], language=None)

        # ── Final Response ──
        elif node == "final_response":
            msgs = values.get("messages", [])
            for msg in msgs:
                content = getattr(msg, "content", "")
                if content:
                    st.markdown(_clean_report_markdown(content))


def _save_result(query: str, state_values: dict):
    """결과를 JSON 파일로 저장."""
    messages_out = []
    for msg in state_values.get("messages", []):
        messages_out.append({
            "type": msg.type,
            "name": getattr(msg, "name", None),
            "content": getattr(msg, "content", ""),
        })

    # papers 직렬화 (dataclass/Pydantic 객체도 dict로 변환)
    papers_out = []
    for p in state_values.get("papers", []):
        if isinstance(p, dict):
            papers_out.append(p)
        else:
            papers_out.append({
                "paper_id": getattr(p, "paper_id", ""),
                "title": getattr(p, "title", ""),
                "year": getattr(p, "year", ""),
                "authors": getattr(p, "authors", []),
            })

    result = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "refined_query": state_values.get("refined_query", ""),
        "keywords": state_values.get("keywords", []),
        "papers": papers_out,
        "limitations": state_values.get("limitations", []),
        "gaps": state_values.get("gaps", []),
        "web_results": state_values.get("web_results", []),
        "limitation_eval": state_values.get("limitation_eval", {}),
        "eval_warnings": state_values.get("eval_warnings", []),
        "messages": messages_out,
    }

    fname = f"gapago_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = OUTPUT_DIR / fname
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    st.sidebar.success(f"결과 저장: {fname}")


# =====================================================================
# 저장된 결과 표시
# =====================================================================
def _show_loaded_result(data: dict):
    """저장된 결과 파일을 화면에 표시."""
    st.subheader(f"📋 Query: {data.get('query', '')}")
    if data.get("refined_query"):
        st.caption(f"Refined: {data['refined_query']}")
    st.caption(f"Timestamp: {data.get('timestamp', '')}")

    # Papers — state에 저장된 구조화 데이터 우선, 없으면 messages에서 추출
    papers = data.get("papers", [])
    with st.expander(f"📄 Paper Retrieval ({len(papers)}편)", expanded=False):
        if papers:
            import pandas as pd
            rows = []
            for p in papers:
                if isinstance(p, dict):
                    rows.append({
                        "ID": p.get("paper_id", "")[:25],
                        "Title": p.get("title", "")[:70],
                        "Year": p.get("year", ""),
                        "Source": p.get("paper_id", "").split(":")[0] if ":" in p.get("paper_id", "") else "",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            # fallback: messages에서 텍스트 추출
            paper_msgs = [m for m in data.get("messages", []) if m.get("name") == "paper_retrieval"]
            if paper_msgs:
                st.markdown(paper_msgs[0].get("content", "")[:2000])

    # Limitations
    limitations = data.get("limitations", [])
    with st.expander(f"⚠️ Limitations ({len(limitations)})", expanded=False):
        for i, lim in enumerate(limitations):
            track_badge = "🟢 Author" if lim.get("track") == "author_stated" else "🔵 Structural"
            quality = lim.get("eval_quality", "")
            quality_badge = f" | {'💪 Strong' if quality == 'strong' else '⚡ Weak'}" if quality else ""
            st.markdown(
                f"**{i+1}.** {track_badge}{quality_badge} `{lim.get('source_section', '')}`\n\n"
                f"> {lim.get('claim', '')}"
            )

    # Limitation Eval
    eval_data = data.get("limitation_eval", {})
    if eval_data and not eval_data.get("skipped"):
        with st.expander("✅ Limitation Evaluation", expanded=False):
            decision = eval_data.get("decision", "N/A")
            if decision == "PASS":
                st.success(f"Decision: **{decision}**")
            else:
                st.warning(f"Decision: **{decision}**")
            for w in data.get("eval_warnings", []):
                st.warning(w)

            call1 = eval_data.get("call1_results", [])
            if call1:
                import pandas as pd
                scores_data = []
                for r in call1:
                    scores_data.append({
                        "ID": f"Lim {r.get('limitation_id', '?')}",
                        "Fact Score": r.get("fact_score", 0),
                        "Groundedness": r.get("groundedness", 0) / 5.0,
                        "Specificity": r.get("specificity", 0) / 5.0,
                        "Relevance": r.get("relevance", 0) / 5.0,
                    })
                df = pd.DataFrame(scores_data)
                st.bar_chart(df.set_index("ID"), height=250)

    # GAPs
    gaps = data.get("gaps", [])
    with st.expander(f"💡 Research GAPs ({len(gaps)})", expanded=True):
        for i, gap in enumerate(gaps):
            count = gap.get("repeat_count", 0)
            stars = "⭐⭐⭐" if i == 0 else ("⭐⭐" if i <= 2 else "⭐")
            axis_type = "🔵" if gap.get("axis_type") == "fixed" else "🟢"

            st.markdown(f"""
### {stars} GAP #{i+1} — {axis_type} {gap.get('axis_label', '')} ({count}개 논문)

**{gap.get('gap_statement', '')}**

{gap.get('elaboration', '')}

📌 **Proposed Topic:** _{gap.get('proposed_topic', '')}_
""")
            st.divider()

    # Final Report
    final_msgs = [m for m in data.get("messages", []) if m.get("name") == "final_response"]
    if final_msgs:
        with st.expander("📝 Final Report", expanded=True):
            cleaned = _clean_report_markdown(final_msgs[0].get("content", ""))
            st.markdown(cleaned)


# =====================================================================
# 메인 로직
# =====================================================================

# ── 1) interrupt 중이면 clarification UI 표시 ──
if st.session_state.get("pipeline_interrupted"):
    st.divider()
    clarify_prompt = st.session_state.get("pipeline_clarify_prompt", "")

    st.warning("🔍 쿼리를 더 구체화할 필요가 있습니다.")
    if clarify_prompt:
        st.info(clarify_prompt)

    clarify_input = st.text_input(
        "보완 답변을 입력하세요",
        placeholder="예: 스마트 팩토리 환경에서의 고장 감지를 위한 도메인 적응",
        key="clarify_input",
    )
    col_resume, col_skip = st.columns([1, 1])
    with col_resume:
        resume_btn = st.button("▶️ 계속 진행", type="primary", use_container_width=True)
    with col_skip:
        skip_btn = st.button("⏭️ 현재 쿼리로 강제 진행", use_container_width=True)

    if resume_btn and clarify_input:
        st.session_state["pipeline_interrupted"] = False
        result = resume_pipeline(clarify_input)
        if result:
            st.balloons()
    elif skip_btn:
        # 강제 진행: 빈 응답으로 resume (query_analysis가 iteration 초과로 SEARCHABLE 처리)
        st.session_state["pipeline_interrupted"] = False
        result = resume_pipeline("proceed as is")
        if result:
            st.balloons()
    elif resume_btn and not clarify_input:
        st.warning("보완 답변을 입력해주세요.")

# ── 2) 새 분석 실행 ──
elif run_btn and query:
    st.divider()
    result = run_pipeline(query, selected_provider, selected_domain)
    if result:
        st.balloons()
    elif st.session_state.get("pipeline_interrupted"):
        st.rerun()  # interrupt UI 표시를 위해 rerun

# ── 3) 저장된 결과 불러오기 ──
elif st.session_state.get("show_loaded") and st.session_state.get("loaded_result"):
    st.divider()
    _show_loaded_result(st.session_state["loaded_result"])
    st.session_state["show_loaded"] = False

elif not query and run_btn:
    st.warning("연구 질문을 입력해주세요.")