"""
GAP Inference Agent (고도화 버전)

처리 흐름:
  Step 1. 고정 축 5개 로드
  Step 2. 수집된 limitations 전체 → LLM → 도메인 특화 동적 축 생성 (최대 2개)
  Step 3. 고정 + 동적 = 최종 축 확정
  Step 4. 각 limitation을 최종 축으로 분류 (배치 처리 — LLM 호출 최소화)
  Step 5. 축별 GAP statement + 상세 설명 + 제안 연구 주제 생성
"""

import re
from collections import defaultdict
from states import AgentState, GapCandidate, LimitationItem
from llm import get_llm
from utils.parse_json import parse_json
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


# ── 고정 축 5개 정의 ─────────────────────────────────────────────────────────

GAP_AXES_FIXED = {
    "data": {
        "label": "Data & Dataset",
        "description": "Limitations related to dataset size, diversity, quality, or availability",
    },
    "methodology": {
        "label": "Methodology",
        "description": "Limitations in experimental design, model architecture, or algorithmic approach",
    },
    "generalizability": {
        "label": "Generalizability",
        "description": "Limitations in applying results to other domains, tasks, or populations",
    },
    "evaluation": {
        "label": "Evaluation & Metrics",
        "description": "Limitations in evaluation protocols, benchmark coverage, or metric adequacy",
    },
    "scalability": {
        "label": "Scalability & Efficiency",
        "description": "Limitations related to computational cost, real-world deployment, or scaling",
    },
}

GAP_AXES_DYNAMIC_MIN_PAPERS = 2
GAP_AXES_DYNAMIC_MAX = 2


# ── LLM 헬퍼 ─────────────────────────────────────────────────────────────────

def _llm_invoke(messages: list[dict], provider: str = None) -> str:
    """dict 형태의 messages를 LangChain 메시지로 변환 후 LLM 호출."""
    llm = get_llm(provider=provider)
    lc_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return llm.invoke(lc_messages).content


# ── limitation_extract 메시지 텍스트 파싱 ────────────────────────────────────

def _parse_limitations_from_messages(messages) -> list:
    """
    state["messages"]에서 name="limitation_extract" 메시지를 찾아
    텍스트를 파싱해 List[LimitationItem]으로 반환한다.

    limitation_extract_node 출력 포맷:
        paper_id: <id>
          - claim: <...>
            evidence_quote: <...>
    """
    raw_text = ""
    for msg in reversed(messages):
        name = getattr(msg, "name", None) or (msg.get("name") if isinstance(msg, dict) else None)
        if name == "limitation_extract":
            raw_text = msg.content if hasattr(msg, "content") else msg.get("content", "")
            break

    if not raw_text:
        print("  ⚠️ limitation_extract 메시지를 찾을 수 없음")
        return []

    limitations = []
    blocks = re.split(r"(?m)^paper_id:\s*", raw_text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        current_paper_id = lines[0].strip()
        claim = None
        evidence_quote = ""

        for line in lines[1:]:
            line_stripped = line.strip()

            claim_match = re.match(r"^-\s*claim:\s*(.+)", line_stripped)
            if claim_match:
                if claim:
                    limitations.append(LimitationItem(
                        paper_id=current_paper_id,
                        claim=claim,
                        evidence_quote=evidence_quote.strip().strip('"'),
                    ))
                claim = claim_match.group(1).strip()
                evidence_quote = ""
                continue

            evidence_match = re.match(r"^evidence_quote:\s*(.+)", line_stripped)
            if evidence_match:
                evidence_quote = evidence_match.group(1).strip()
                continue

        if claim:
            limitations.append(LimitationItem(
                paper_id=current_paper_id,
                claim=claim,
                evidence_quote=evidence_quote.strip().strip('"'),
            ))

    return limitations


# ── Step 1. 고정 축 로드 ──────────────────────────────────────────────────────

def _load_fixed_axes() -> dict:
    return dict(GAP_AXES_FIXED)


# ── Step 2. 동적 축 생성 ──────────────────────────────────────────────────────

def _generate_dynamic_axes(limitations_text: str, fixed_axes: dict, research_question: str, provider: str = None) -> list:
    fixed_summary = "\n".join(
        f"  - {k}: {v['description']}" for k, v in fixed_axes.items()
    )

    prompt = f"""You are a research gap analyst covering ALL scientific domains.

Research Question: "{research_question}"

Below are limitation/future_work statements extracted from related papers:
{limitations_text}

We already have these 5 universal axes that cover most limitations:
{fixed_summary}

TASK:
Find domain-specific patterns in the limitations above that are NOT well covered
by the 5 fixed axes. These should be recurring themes appearing in {GAP_AXES_DYNAMIC_MIN_PAPERS}+ papers.

Rules:
- Propose {GAP_AXES_DYNAMIC_MAX} extra axes at most. If no clear pattern exists, return empty list.
- Each axis must be distinct from the 5 fixed axes.
- Name must be snake_case English (e.g. "reproducibility", "ethical_concern").
- Reason must cite which limitations triggered this axis.

Output JSON only:
{{
  "dynamic_axes": [
    {{
      "name": "axis_key",
      "label": "Short readable label",
      "description": "What kind of limitation this axis captures",
      "reason": "This pattern appeared in N papers: [brief evidence]"
    }}
  ]
}}
"""
    messages = [
        {"role": "system", "content": (
            "You are an expert research analyst. "
            "Identify domain-specific recurring limitation patterns. "
            "Always respond in valid JSON."
        )},
        {"role": "user", "content": prompt},
    ]

    try:
        response = _llm_invoke(messages, provider=provider)
        result = parse_json(response)
        axes = result.get("dynamic_axes", [])

        fixed_keys = set(fixed_axes.keys())
        valid = []
        for ax in axes:
            name = ax.get("name", "").strip()
            if not name or name in fixed_keys:
                continue
            valid.append({
                "name":        name,
                "label":       ax.get("label", name),
                "description": ax.get("description", ""),
                "reason":      ax.get("reason", ""),
            })
        return valid[:GAP_AXES_DYNAMIC_MAX]

    except Exception as e:
        print(f"  ⚠️ Dynamic axis generation failed: {e}")
        return []


# ── Step 3. 최종 축 확정 ─────────────────────────────────────────────────────

def _build_final_axes(fixed_axes: dict, dynamic_axes: list) -> dict:
    final = {}
    for k, v in fixed_axes.items():
        final[k] = {"label": v["label"], "description": v["description"], "type": "fixed"}
    for ax in dynamic_axes:
        final[ax["name"]] = {"label": ax["label"], "description": ax["description"], "type": "dynamic"}
    return final


# ── Step 4. 배치 분류 ────────────────────────────────────────────────────────

def _classify_limitations_batch(limitations: list, final_axes: dict, provider: str = None) -> dict:
    BATCH_SIZE = 20
    axis_mapping = {}
    fallback = "methodology"

    axes_block = "\n".join(f"  {k}: {v['description']}" for k, v in final_axes.items())
    axis_keys = list(final_axes.keys())

    batches = [limitations[i: i + BATCH_SIZE] for i in range(0, len(limitations), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        items_block = "\n".join(
            f'  [{i}] "{lim.claim}"' for i, lim in enumerate(batch)
        )

        prompt = f"""Classify each limitation into ONE axis from the list below.

Available axes:
{axes_block}

Limitations to classify:
{items_block}

Output JSON only — map each index to its axis key:
{{
  "classifications": {{
    "0": "axis_key",
    "1": "axis_key"
  }}
}}
"""
        messages = [
            {"role": "system", "content": "You are a research limitation classifier. Always respond in valid JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = _llm_invoke(messages, provider=provider)
            result = parse_json(response)
            cls_map = result.get("classifications", {})

            offset = batch_idx * BATCH_SIZE
            for idx_str, axis in cls_map.items():
                global_idx = offset + int(idx_str)
                axis_mapping[global_idx] = axis if axis in axis_keys else fallback

        except Exception as e:
            offset = batch_idx * BATCH_SIZE
            for i in range(len(batch)):
                axis_mapping[offset + i] = fallback
            print(f"  ⚠️ Batch {batch_idx} classification failed: {e}")

    for i in range(len(limitations)):
        if i not in axis_mapping:
            axis_mapping[i] = fallback

    return axis_mapping


# ── Step 5. 축별 GAP 생성 ────────────────────────────────────────────────────

def _generate_gap_for_axis(ax_key: str, ax_info: dict, ax_lims: list, research_question: str, provider: str = None, output_language: str = "auto"):
    claims_block = "\n".join(
        f"  - [Paper {lim.paper_id}] {lim.claim}" for lim in ax_lims[:8]
    )
    quotes_block = "\n".join(
        f'  - "{lim.evidence_quote}"' for lim in ax_lims[:5] if lim.evidence_quote
    )

    prompt = f"""You are a research gap analyst.

Research Question: "{research_question}"

Axis: {ax_info['label']} ({ax_key})
Axis description: {ax_info['description']}

Limitations from {len(ax_lims)} papers under this axis:
{claims_block}

Supporting evidence quotes:
{quotes_block if quotes_block else "  (none)"}

TASK: Based on the recurring limitations above, generate:
1. gap_statement  : One concise sentence describing the core research gap.
2. elaboration    : 2-3 sentences explaining WHY this gap matters and WHAT is missing.
3. proposed_topic : One concrete research topic title that directly addresses this gap.

Output JSON only:
{{
  "gap_statement":  "...",
  "elaboration":    "...",
  "proposed_topic": "..."
}}
"""
    from prompts.system import get_language_instruction
    lang_instruction = get_language_instruction(output_language)

    messages = [
        {"role": "system", "content": (
            "You are an expert at identifying and articulating research gaps "
            "across all scientific disciplines. Always respond in valid JSON."
            + lang_instruction
        )},
        {"role": "user", "content": prompt},
    ]

    try:
        response = _llm_invoke(messages, provider=provider)
        result = parse_json(response)
        return {
            "gap_statement":  result.get("gap_statement",  f"Gap in {ax_key}"),
            "elaboration":    result.get("elaboration",    ""),
            "proposed_topic": result.get("proposed_topic", ""),
        }
    except Exception as e:
        print(f"  ⚠️ GAP generation failed for axis '{ax_key}': {e}")
        return None


# ── 메인 노드 ────────────────────────────────────────────────────────────────

def gap_infer_node(state: AgentState) -> AgentState:
    """
    GAP Inference Node

    limitation_extract_node 출력은 state["messages"]에 텍스트로만 쌓임.
    → messages에서 name="limitation_extract" 메시지를 파싱해 LimitationItem 리스트 구성.
    → 이후 축 분류 및 GAP 생성 수행.
    """
    print("\n💡 GAP Inference Node")

    provider = state.get("llm_provider")

    # ── limitations 획득: state["limitations"] 우선, 없으면 messages 파싱 ──
    raw_limitations = state.get("limitations", [])

    if raw_limitations:
        limitations = []
        for lim in raw_limitations:
            if isinstance(lim, dict):
                limitations.append(LimitationItem(
                    paper_id=lim.get("paper_id", "unknown"),
                    claim=lim.get("claim", ""),
                    evidence_quote=lim.get("evidence_quote", ""),
                ))
            else:
                limitations.append(lim)
    else:
        # limitation_extract_node가 messages에만 텍스트로 남긴 경우 파싱
        limitations = _parse_limitations_from_messages(state.get("messages", []))

    if not limitations:
        print("  ⚠️ No limitations to analyze")
        return {**state, "gaps": []}

    print(f"  ✓ {len(limitations)}개 limitation 로드 완료")

    # ── research_question ───────────────────────────────────────────────────
    research_question = state.get("refined_query", "")
    if not research_question:
        for msg in state.get("messages", []):
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            role = msg.type if hasattr(msg, "type") else msg.get("role", "")
            if role == "human" and content:
                research_question = content
                break

    # ── Step 1 ──────────────────────────────────────────────────────────────
    fixed_axes = _load_fixed_axes()
    print(f"  ✓ 고정 축 {len(fixed_axes)}개 로드: {list(fixed_axes.keys())}")

    # ── Step 2 ──────────────────────────────────────────────────────────────
    all_claims_text = "\n".join(f"[{lim.paper_id}] {lim.claim}" for lim in limitations)
    print(f"  🔄 동적 축 생성 중 (limitations {len(limitations)}개 분석)...")
    dynamic_axes = _generate_dynamic_axes(all_claims_text, fixed_axes, research_question, provider=provider)

    if dynamic_axes:
        for ax in dynamic_axes:
            print(f"  🟢 동적 축 추가: [{ax['name']}] {ax['label']}")
            print(f"     근거: {ax['reason'][:80]}")
    else:
        print("  ℹ️  도메인 특화 동적 축 없음 (고정 5개로 충분)")

    # ── Step 3 ──────────────────────────────────────────────────────────────
    final_axes = _build_final_axes(fixed_axes, dynamic_axes)
    print(f"  ✓ 최종 축 {len(final_axes)}개 확정")

    # ── Step 4 ──────────────────────────────────────────────────────────────
    print(f"  🔄 {len(limitations)}개 limitation 배치 분류 중...")
    axis_mapping = _classify_limitations_batch(limitations, final_axes, provider=provider)

    axis_groups = defaultdict(list)
    for idx, lim in enumerate(limitations):
        axis = axis_mapping.get(idx, "methodology")
        axis_groups[axis].append(lim)

    print(f"\n  {'축':<28} {'유형':>6}  {'논문 수':>6}")
    print(f"  {'-'*45}")
    for ax_key, lims in sorted(axis_groups.items(), key=lambda x: -len(x[1])):
        ax_type = final_axes.get(ax_key, {}).get("type", "fixed")
        label = final_axes.get(ax_key, {}).get("label", ax_key)
        tag = "🔵 고정" if ax_type == "fixed" else "🟢 동적"
        print(f"  {label:<28} {tag:>6}  {len(lims):>5}개")

    # ── Step 5 ──────────────────────────────────────────────────────────────
    gaps = []
    active_axes = {k: v for k, v in axis_groups.items() if len(v) >= 1}
    print(f"\n  🔄 {len(active_axes)}개 축 GAP 생성 중...")

    for ax_key, ax_lims in sorted(active_axes.items(), key=lambda x: -len(x[1])):
        ax_info = final_axes.get(ax_key, {"label": ax_key, "description": "", "type": "fixed"})
        output_language = state.get("output_language", "auto")
        result = _generate_gap_for_axis(ax_key, ax_info, ax_lims, research_question, provider=provider, output_language=output_language)
        if result is None:
            continue

        gaps.append(GapCandidate(
            axis=ax_key,
            axis_label=ax_info["label"],
            axis_type=ax_info["type"],
            gap_statement=result["gap_statement"],
            elaboration=result["elaboration"],
            proposed_topic=result["proposed_topic"],
            repeat_count=len(ax_lims),
            supporting_papers=list({lim.paper_id for lim in ax_lims}),
            supporting_quotes=[lim.evidence_quote for lim in ax_lims if lim.evidence_quote][:5],
        ))
        print(f"  ✓ [{ax_key}] {result['gap_statement'][:65]}...")

    gaps.sort(key=lambda g: g.repeat_count, reverse=True)
    gaps_as_dict = [g.model_dump() for g in gaps]

    print(f"\n  ✅ GAP {len(gaps)}개 생성 완료")

    trace = dict(state.get("trace", {}))
    trace["gaps_generated"] = len(gaps)
    trace["axes_used"] = list(final_axes.keys())
    trace["dynamic_axes"] = [ax["name"] for ax in dynamic_axes]
    trace["axis_distribution"] = {k: len(v) for k, v in axis_groups.items()}

    return {
        **state,
        "gaps": gaps_as_dict,
        "trace": trace,
    }