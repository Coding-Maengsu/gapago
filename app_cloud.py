"""
GAPAGO - Streamlit Cloud Frontend
Lightweight UI that calls the FastAPI backend via HTTP.
"""

import os
import time
import streamlit as st
import requests
from datetime import datetime

# ── Configuration ──
API_BASE = os.environ.get("GAPAGO_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="GAPAGO - Research GAP Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PIPELINE_NODES = [
    "query_subgraph", "meaning_expand", "paper_retrieval",
    "limitation_extract", "limitation_eval", "recency_check",
    "gap_infer", "critic_score", "final_response",
]

NODE_LABELS = {
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

PROVIDER_OPTIONS = {
    "Azure OpenAI (GPT)": "azure",
    "Claude (AWS Bedrock)": "claude",
    "Google Gemini": "gemini",
    "LG EXAONE (Local GPU)": "exaone",
}

DOMAIN_OPTIONS = {
    "auto (자동 판단)": "auto",
    "AI / Computer Science": "ai_cs",
    "Biomedical / 의학": "biomedical",
    "Materials / Chemistry": "materials_chemistry",
    "Physics": "physics",
    "General (범용)": "general",
}


# =====================================================================
# API helpers
# =====================================================================
def api_call(method: str, path: str, **kwargs):
    """Make API call with error handling."""
    url = f"{API_BASE}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=10, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"백엔드 서버에 연결할 수 없습니다: {API_BASE}")
        st.info("백엔드 서버가 실행 중인지 확인하세요: `python api.py`")
        return None
    except requests.HTTPError as e:
        st.error(f"API 오류: {e}")
        return None


def check_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.title("🔬 GAPAGO")
    st.caption("Research GAP Analysis System")

    # Health check
    if check_health():
        st.success("Backend: Connected", icon="✅")
    else:
        st.error("Backend: Disconnected", icon="❌")
        st.caption(f"URL: {API_BASE}")

    # New analysis button
    if st.button("➕ 새 분석", use_container_width=True):
        for key in ["session_id", "loaded_result", "show_loaded", "selected_history"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()

    # Settings
    with st.expander("⚙️ 설정", expanded=False):
        provider_display = st.selectbox("LLM Provider", list(PROVIDER_OPTIONS.keys()), index=0)
        domain_display = st.selectbox("Research Domain", list(DOMAIN_OPTIONS.keys()), index=0)

    selected_provider = PROVIDER_OPTIONS[provider_display]
    selected_domain = DOMAIN_OPTIONS[domain_display]

    st.divider()

    # History
    st.subheader("📂 분석 기록")
    history_data = api_call("get", "/api/history")
    if history_data and history_data.get("history"):
        for i, item in enumerate(history_data["history"]):
            ts = item.get("timestamp", "")[:16].replace("T", " ")
            query_preview = item["query"][:35] + ("..." if len(item["query"]) > 35 else "")
            gaps_count = item.get("gaps_count", 0)

            btn_label = f"🔹 {query_preview}\n{ts} · GAP {gaps_count}개"
            if st.button(btn_label, key=f"history_{i}", use_container_width=True):
                detail = api_call("get", f"/api/history/{item['file']}")
                if detail:
                    st.session_state["loaded_result"] = detail
                    st.session_state["show_loaded"] = True
                    st.session_state.pop("session_id", None)
                    st.rerun()

        st.divider()
        st.caption(f"총 {len(history_data['history'])}개 기록")
    else:
        st.caption("아직 분석 기록이 없습니다.")


# =====================================================================
# Main
# =====================================================================
st.header("🔬 GAPAGO — Research GAP Analyzer")

query = st.text_input(
    "연구 질문을 입력하세요",
    placeholder="예: Domain adaptation for fault detection in smart manufacturing",
    key="query_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("🚀 분석 시작", type="primary", use_container_width=True)


# =====================================================================
# Pipeline polling & rendering
# =====================================================================
def poll_and_render(session_id: str):
    """Poll the backend and render progress until completion or interrupt."""
    progress_bar = st.progress(0, text="파이프라인 시작...")

    # Pre-create expanders
    node_containers = {}
    for node_name in PIPELINE_NODES:
        icon, label = NODE_LABELS.get(node_name, ("⚙️", node_name))
        node_containers[node_name] = st.expander(f"{icon} {label}", expanded=False)

    rendered_nodes = set()
    prev_status = None

    while True:
        data = api_call("get", f"/api/sessions/{session_id}")
        if not data:
            break

        status = data["status"]
        completed = data["completed_nodes"]
        summaries = data["node_summaries"]

        # Update progress
        pct = min(len(completed) / len(PIPELINE_NODES), 1.0) if completed else 0
        if completed:
            last_node = completed[-1]
            icon, label = NODE_LABELS.get(last_node, ("⚙️", last_node))
            progress_bar.progress(pct, text=f"{icon} {label} 완료")
        elif status == "running":
            progress_bar.progress(0, text="파이프라인 실행 중...")

        # Render newly completed nodes
        for node in completed:
            if node not in rendered_nodes and node in summaries:
                rendered_nodes.add(node)
                render_node_summary(node, summaries[node], node_containers.get(node))

        if status == "completed":
            progress_bar.progress(1.0, text="✅ 파이프라인 완료!")
            return data.get("result")

        if status == "interrupted":
            progress_bar.progress(pct, text="⏸️ 사용자 입력 대기...")
            st.session_state["session_id"] = session_id
            st.session_state["clarify_prompt"] = data.get("clarify_prompt", "")
            return None

        if status == "error":
            progress_bar.progress(pct, text="❌ 오류 발생")
            st.error(f"파이프라인 오류: {data.get('error', 'Unknown')}")
            return None

        if status == prev_status and status != "starting":
            time.sleep(2)
        else:
            time.sleep(1)
        prev_status = status

    return None


def render_node_summary(node: str, summary: dict, container):
    """Render node summary in its expander."""
    if not container:
        return

    with container:
        if node == "query_subgraph":
            if summary.get("refined_query"):
                st.success(f"**Refined Query:** {summary['refined_query']}")
            if summary.get("keywords"):
                st.write("**Keywords:**", ", ".join(summary["keywords"]))
            if summary.get("scope_level"):
                st.info(f"Scope: {summary['scope_level']}")

        elif node == "meaning_expand":
            if summary.get("message"):
                st.text(summary["message"])

        elif node == "paper_retrieval":
            import pandas as pd
            st.metric("논문 수", summary.get("paper_count", 0))
            papers = summary.get("papers", [])
            if papers:
                df = pd.DataFrame([{
                    "ID": p.get("paper_id", "")[:20],
                    "Title": p.get("title", "")[:60],
                    "Year": p.get("year", ""),
                    "Source": p.get("source", ""),
                } for p in papers])
                st.dataframe(df, use_container_width=True, hide_index=True)

        elif node == "limitation_extract":
            limitations = summary.get("limitations", [])
            st.metric("추출된 Limitations", summary.get("limitation_count", 0))
            for i, lim in enumerate(limitations):
                track_badge = "🟢 Author" if lim.get("track") == "author_stated" else "🔵 Structural"
                st.markdown(
                    f"**{i+1}.** {track_badge} `{lim.get('source_section', '')}`\n\n"
                    f"> {lim.get('claim', '')}\n\n"
                    f"*Evidence:* _{lim.get('evidence_quote', '')[:150]}_"
                )
                st.divider()

        elif node == "limitation_eval":
            decision = summary.get("decision", "N/A")
            if decision == "PASS":
                st.success(f"🎯 Decision: **{decision}**")
            else:
                st.warning(f"🔄 Decision: **{decision}**")
            for w in summary.get("eval_warnings", []):
                st.warning(w)
            call1 = summary.get("call1_results", [])
            if call1:
                import pandas as pd
                scores_data = [{
                    "ID": f"Lim {r.get('limitation_id', '?')}",
                    "Fact Score": r.get("fact_score", 0),
                    "Groundedness": r.get("groundedness", 0) / 5.0,
                    "Specificity": r.get("specificity", 0) / 5.0,
                    "Relevance": r.get("relevance", 0) / 5.0,
                } for r in call1]
                df = pd.DataFrame(scores_data)
                st.bar_chart(df.set_index("ID"), height=250)
            type_dist = summary.get("type_distribution", {})
            if type_dist:
                import pandas as pd
                st.subheader("Limitation Type Distribution")
                dist_df = pd.DataFrame([
                    {"Type": k, "Count": v} for k, v in type_dist.items() if v
                ])
                if not dist_df.empty:
                    st.bar_chart(dist_df.set_index("Type"), height=200)
            st.metric("평가 통과 Limitations", summary.get("limitation_count", 0))

        elif node == "recency_check":
            if summary.get("message"):
                st.info(summary["message"])
            counts = summary.get("recency_counts", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Unresolved", counts.get("unresolved", 0))
            c2.metric("Partial", counts.get("partial", 0))
            c3.metric("Resolved", counts.get("resolved", 0))

        elif node == "gap_infer":
            gaps = summary.get("gaps", [])
            st.metric("Research GAPs", summary.get("gap_count", 0))
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

        elif node == "critic_score":
            if summary.get("message"):
                st.code(summary["message"], language=None)

        elif node == "final_response":
            if summary.get("report"):
                st.markdown(summary["report"])


# =====================================================================
# Loaded result display
# =====================================================================
def show_loaded_result(data: dict):
    """Display a saved result."""
    st.subheader(f"📋 Query: {data.get('query', '')}")
    if data.get("refined_query"):
        st.caption(f"Refined: {data['refined_query']}")
    st.caption(f"Timestamp: {data.get('timestamp', '')}")

    with st.expander("📄 Paper Retrieval", expanded=False):
        paper_msgs = [m for m in data.get("messages", []) if m.get("name") == "paper_retrieval"]
        if paper_msgs:
            st.text(paper_msgs[0].get("content", "")[:500])

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

    final_msgs = [m for m in data.get("messages", []) if m.get("name") == "final_response"]
    if final_msgs:
        with st.expander("📝 Final Report", expanded=False):
            st.markdown(final_msgs[0].get("content", ""))


# =====================================================================
# Main logic
# =====================================================================

# 1) Interrupted — show clarification UI
if st.session_state.get("session_id") and st.session_state.get("clarify_prompt") is not None:
    session_id = st.session_state["session_id"]

    # Check if still interrupted
    data = api_call("get", f"/api/sessions/{session_id}")
    if data and data["status"] == "interrupted":
        st.divider()
        st.warning("🔍 쿼리를 더 구체화할 필요가 있습니다.")
        clarify_prompt = data.get("clarify_prompt", "")
        if clarify_prompt:
            st.info(clarify_prompt)

        clarify_input = st.text_input(
            "보완 답변을 입력하세요",
            placeholder="예: 스마트 팩토리 환경에서의 고장 감지를 위한 도메인 적응",
            key="clarify_input",
        )
        col_resume, col_skip = st.columns(2)
        with col_resume:
            resume_btn = st.button("▶️ 계속 진행", type="primary", use_container_width=True)
        with col_skip:
            skip_btn = st.button("⏭️ 현재 쿼리로 강제 진행", use_container_width=True)

        if resume_btn and clarify_input:
            resp = api_call("post", f"/api/sessions/{session_id}/clarify", json={"response": clarify_input})
            if resp:
                st.session_state["clarify_prompt"] = None
                result = poll_and_render(session_id)
                if result:
                    st.balloons()
        elif skip_btn:
            resp = api_call("post", f"/api/sessions/{session_id}/clarify", json={"response": "proceed as is"})
            if resp:
                st.session_state["clarify_prompt"] = None
                result = poll_and_render(session_id)
                if result:
                    st.balloons()
        elif resume_btn and not clarify_input:
            st.warning("보완 답변을 입력해주세요.")

    elif data and data["status"] in ("completed", "running"):
        st.session_state.pop("clarify_prompt", None)
        if data["status"] == "completed" and data.get("result"):
            show_loaded_result(data["result"])

# 2) New analysis
elif run_btn and query:
    st.divider()
    resp = api_call("post", "/api/analyze", json={
        "query": query,
        "provider": selected_provider,
        "domain": selected_domain,
    })
    if resp:
        session_id = resp["session_id"]
        st.session_state["session_id"] = session_id
        result = poll_and_render(session_id)
        if result:
            st.balloons()
        elif st.session_state.get("clarify_prompt") is not None:
            st.rerun()

# 3) Show loaded result
elif st.session_state.get("show_loaded") and st.session_state.get("loaded_result"):
    st.divider()
    show_loaded_result(st.session_state["loaded_result"])
    st.session_state["show_loaded"] = False

elif not query and run_btn:
    st.warning("연구 질문을 입력해주세요.")
