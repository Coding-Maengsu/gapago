"""
GAP Inference Agent — 성능 최우선 버전

핵심 설계 원칙:
  1. limitation의 단순 반전(반사적 반전) 금지
  2. recency_status 활용 → "아직 아무도 안 푼" limitation만 GAP 후보로 승격
  3. web_results 맥락 주입 → 창의적 방향 제안 시 최신 동향 반영
  4. 3단계 추론 파이프라인:
       Step 5a. 가장 시급한 축 선정 (urgency scoring)
       Step 5b. 왜 아무도 못 풀었는가 (기술적 장벽 분석)
       Step 5c. 장벽을 우회하는 창의적 연구 방향 제안
  5. 동일 축 내에서도 여러 방향 후보 생성 후 가장 참신한 것을 채택

처리 흐름:
  Step 1. 고정 축 5개 로드
  Step 2. limitations 전체 → LLM → 도메인 특화 동적 축 생성 (최대 2개)
  Step 3. 고정 + 동적 = 최종 축 확정
  Step 4. 각 limitation 배치 분류 + recency 가중치 적용
  Step 5a. 축별 긴급도(urgency) 점수화 → 우선순위 결정
  Step 5b. 상위 N개 축에 대해 기술적 장벽 분석
  Step 5c. 장벽 기반 창의적 방향 제안 (web_results 맥락 활용)
"""

import re
from collections import defaultdict
from states import AgentState, GapCandidate, LimitationItem
from llm import get_llm
from utils.parse_json import parse_json
from utils.progress import report_progress
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

# recency 가중치: unresolved limitation이 더 많이 GAP에 기여
RECENCY_WEIGHT = {
    "unresolved": 1.0,
    "partial":    0.5,
    "resolved":   0.0,   # 이미 해결된 것은 GAP 후보에서 제외
}


# ── LLM 헬퍼 ─────────────────────────────────────────────────────────────────

def _llm_invoke(messages: list[dict], provider: str = None) -> str:
    """dict 형태의 messages를 LangChain 메시지로 변환 후 LLM 호출."""
    llm = get_llm(provider=provider)
    lc_messages = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return llm.invoke(lc_messages).content


# ── limitation_extract 메시지 텍스트 파싱 ────────────────────────────────────

def _parse_limitations_from_messages(messages) -> list:
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
        track = "author_stated"
        source_section = ""

        for line in lines[1:]:
            line = line.strip()
            if line.startswith("- claim:"):
                claim = line[len("- claim:"):].strip()
            elif line.startswith("evidence_quote:"):
                evidence_quote = line[len("evidence_quote:"):].strip()
            elif line.startswith("track:"):
                track = line[len("track:"):].strip()
            elif line.startswith("source_section:"):
                source_section = line[len("source_section:"):].strip()
            elif claim and line and not line.startswith("-"):
                claim += " " + line

        if current_paper_id and claim:
            limitations.append(LimitationItem(
                paper_id=current_paper_id,
                claim=claim,
                evidence_quote=evidence_quote,
                track=track,
                source_section=source_section,
            ))

    return limitations


# ── Step 1. 고정 축 로드 ─────────────────────────────────────────────────────

def _load_fixed_axes() -> dict:
    return dict(GAP_AXES_FIXED)


# ── Step 2. 동적 축 생성 ─────────────────────────────────────────────────────

def _generate_dynamic_axes(all_claims_text: str, fixed_axes: dict, research_question: str) -> list:
    fixed_desc = "\n".join(f"  {k}: {v['description']}" for k, v in fixed_axes.items())

    prompt = f"""You are a research domain expert.

Research Question: "{research_question}"

Existing fixed axes (do NOT duplicate):
{fixed_desc}

All collected limitations:
{all_claims_text[:3000]}

TASK: Identify up to {GAP_AXES_DYNAMIC_MAX} domain-specific axes that appear
repeatedly but are NOT covered by the fixed axes.
Only include an axis if at least {GAP_AXES_DYNAMIC_MIN_PAPERS} limitations clearly belong to it.

Output JSON only:
{{
  "dynamic_axes": [
    {{
      "name": "short_snake_case_key",
      "label": "Human-readable Label",
      "description": "One sentence describing what limitations belong here",
      "reason": "Why this is distinct from the fixed axes"
    }}
  ]
}}
"""
    messages = [
        {"role": "system", "content": "You are a research gap analyst. Always respond in valid JSON."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = _llm_invoke(messages, provider=provider)
        result = parse_json(response)
        axes = result.get("dynamic_axes", [])
        valid = []
        fixed_keys = set(fixed_axes.keys())
        for ax in axes:
            name = ax.get("name", "").strip().lower().replace(" ", "_")
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


# ── Step 4. 배치 분류 + recency 가중치 적용 ──────────────────────────────────

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

Output JSON only:
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


def _build_axis_groups_with_recency(
    limitations: list,
    axis_mapping: dict,
) -> dict[str, dict]:
    """
    축별로 limitations를 그룹화하고 recency 가중치를 반영한 유효 점수 계산.
    resolved된 limitation은 GAP 후보에서 제외하되, 카운트는 유지(근거 문장용).
    반환: {ax_key: {"lims": [...], "weighted_count": float, "unresolved_lims": [...]}}
    """
    raw_groups: dict[str, list] = defaultdict(list)
    for idx, lim in enumerate(limitations):
        axis = axis_mapping.get(idx, "methodology")
        raw_groups[axis].append(lim)

    groups = {}
    for ax_key, lims in raw_groups.items():
        weighted = sum(
            RECENCY_WEIGHT.get(getattr(lim, "recency_status", None) or "unresolved", 1.0)
            for lim in lims
        )
        unresolved = [
            lim for lim in lims
            if RECENCY_WEIGHT.get(getattr(lim, "recency_status", None) or "unresolved", 1.0) > 0
        ]
        groups[ax_key] = {
            "lims": lims,
            "unresolved_lims": unresolved,
            "weighted_count": weighted,
            "total_count": len(lims),
        }
    return groups


# ── Step 5a. 긴급도(urgency) 점수화 ─────────────────────────────────────────

def _score_axis_urgency(
    axis_groups: dict,
    final_axes: dict,
    research_question: str,
) -> list[tuple[str, float]]:
    """
    각 축의 긴급도를 LLM으로 점수화하여 우선순위를 결정한다.

    점수 기준:
      - 미해결 limitation 수 (recency-weighted)
      - 연구 질문과의 직접적 관련성
      - 해당 축이 다른 축의 병목이 되는 정도 (cascade impact)
    """
    axes_summary = []
    for ax_key, grp in axis_groups.items():
        ax_info = final_axes.get(ax_key, {})
        unresolved_claims = "\n".join(
            f"    - {lim.claim[:100]}" for lim in grp["unresolved_lims"][:5]
        )
        axes_summary.append(
            f"[{ax_key}] {ax_info.get('label', ax_key)} "
            f"(weighted={grp['weighted_count']:.1f}, total={grp['total_count']})\n"
            f"  Unresolved limitations:\n{unresolved_claims}"
        )

    axes_block = "\n\n".join(axes_summary)

    prompt = f"""You are a research strategy expert.

Research Question: "{research_question}"

Below are the limitation axes, each with their unresolved limitations:

{axes_block}

TASK: Score each axis by URGENCY (0-10) for the given research question.

Urgency is determined by:
1. How many unresolved limitations cluster in this axis (weighted count)
2. How directly this axis blocks progress on the research question
3. Whether solving this axis would unlock breakthroughs in other axes (cascade impact)
4. How feasible it is to make meaningful progress NOW (not someday)

Output JSON only:
{{
  "urgency_scores": {{
    "axis_key": {{
      "score": <0-10>,
      "rationale": "<one sentence>",
      "cascade_impact": "<axes that would benefit if this is solved, or 'none'>"
    }}
  }}
}}
"""
    messages = [
        {"role": "system", "content": (
            "You are a research prioritization expert. "
            "Be critical and differentiate scores meaningfully — avoid giving everything the same score. "
            "Always respond in valid JSON."
        )},
        {"role": "user", "content": prompt},
    ]

    try:
        response = _llm_invoke(messages)
        result = parse_json(response)
        scores_raw = result.get("urgency_scores", {})

        scored = []
        for ax_key, grp in axis_groups.items():
            base_score = grp["weighted_count"]
            llm_score = scores_raw.get(ax_key, {}).get("score", 5)
            # 최종 점수: LLM 긴급도(60%) + recency 가중 카운트 정규화(40%)
            max_weighted = max((g["weighted_count"] for g in axis_groups.values()), default=1)
            normalized = grp["weighted_count"] / max_weighted * 10
            final_score = 0.6 * llm_score + 0.4 * normalized
            cascade = scores_raw.get(ax_key, {}).get("cascade_impact", "none")
            rationale = scores_raw.get(ax_key, {}).get("rationale", "")
            scored.append((ax_key, final_score, cascade, rationale))
            print(f"  [urgency] {ax_key}: LLM={llm_score}, weighted={grp['weighted_count']:.1f}, final={final_score:.2f} | {rationale[:60]}")

        scored.sort(key=lambda x: -x[1])
        return scored

    except Exception as e:
        print(f"  ⚠️ Urgency scoring failed: {e} → weighted_count 기준으로 정렬")
        return sorted(
            [(k, g["weighted_count"], "unknown", "") for k, g in axis_groups.items()],
            key=lambda x: -x[1],
        )


# ── Step 5b. 기술적 장벽 분석 ────────────────────────────────────────────────

def _analyze_barriers(
    ax_key: str,
    ax_info: dict,
    unresolved_lims: list,
    all_lims: list,
    research_question: str,
) -> dict:
    """
    왜 N편의 논문이 이 문제를 인정하면서도 해결하지 못했는지를 분석한다.
    단순히 "어렵다"가 아니라 구체적인 기술적/구조적 장벽을 명시한다.
    """
    claims_block = "\n".join(
        f"  [{i+1}] [{lim.paper_id}] (recency={getattr(lim, 'recency_status', 'unresolved')}) {lim.claim}"
        for i, lim in enumerate(unresolved_lims[:12])
    )
    quotes_block = "\n".join(
        f'  - "{lim.evidence_quote}"'
        for lim in unresolved_lims[:6] if lim.evidence_quote
    )
    resolved_examples = [
        lim for lim in all_lims
        if getattr(lim, "recency_status", "unresolved") == "resolved"
    ]
    resolved_block = ""
    if resolved_examples:
        resolved_block = "\nPartially/fully resolved limitations (for contrast):\n" + "\n".join(
            f"  - [{lim.paper_id}] {lim.claim[:100]}" for lim in resolved_examples[:3]
        )

    prompt = f"""You are a critical research analyst.

Research Question: "{research_question}"
Axis: {ax_info['label']} — {ax_info['description']}

{len(unresolved_lims)} papers acknowledge this limitation as UNRESOLVED:
{claims_block}

Direct evidence quotes:
{quotes_block if quotes_block else "  (none)"}
{resolved_block}

=== ANALYSIS TASK ===

STEP A — gap_statement
One precise sentence (≤25 words) naming the exact unsolved problem.
Focus on WHAT capability or knowledge is missing, not on what papers did wrong.

STEP B — barriers
List exactly 3 concrete barriers that explain why this gap persists.
Each barrier must be:
  - Technical or structural (not "more research needed")
  - Specific enough that a researcher knows what to tackle
  - Distinct from each other (different root causes)

Examples of GOOD barriers:
  - "No standardized benchmark exists that combines class imbalance + OOD conditions simultaneously"
  - "Computing ECE on minority classes requires >10k samples per class, which most datasets lack"
  - "Existing uncertainty calibration methods assume i.i.d. test distributions, failing under domain shift"

STEP C — barrier_type
Classify the PRIMARY barrier:
  "data_scarcity" | "benchmark_absence" | "computational_cost" |
  "evaluation_mismatch" | "methodological_gap" | "domain_shift" |
  "conflicting_objectives" | "other"

STEP D — what_was_tried
List 2-3 approaches the existing papers already tried that did NOT work.
This is critical to avoid proposing the same thing again.

Output JSON only:
{{
  "gap_statement":  "...",
  "barriers":       ["barrier 1", "barrier 2", "barrier 3"],
  "barrier_type":   "...",
  "what_was_tried": ["approach 1", "approach 2"]
}}
"""
    messages = [
        {"role": "system", "content": (
            "You are a rigorous research analyst who identifies root causes, not symptoms. "
            "Be specific and honest about what has already failed. "
            "Always respond in valid JSON."
        )},
        {"role": "user", "content": prompt},
    ]

    try:
        response = _llm_invoke(messages)
        result = parse_json(response)
        return {
            "gap_statement":  result.get("gap_statement",  f"Unsolved gap in {ax_key}"),
            "barriers":       result.get("barriers",       []),
            "barrier_type":   result.get("barrier_type",   "methodological_gap"),
            "what_was_tried": result.get("what_was_tried", []),
        }
    except Exception as e:
        print(f"  ⚠️ Barrier analysis failed for '{ax_key}': {e}")
        return {
            "gap_statement":  f"Unsolved gap in {ax_key}",
            "barriers":       [],
            "barrier_type":   "methodological_gap",
            "what_was_tried": [],
        }


# ── Step 5c. 창의적 연구 방향 제안 ──────────────────────────────────────────

def _generate_creative_directions(
    ax_key: str,
    ax_info: dict,
    unresolved_lims: list,
    research_question: str,
    gap_statement: str,
    barriers: list,
    barrier_type: str,
    what_was_tried: list,
    web_results: list,
    cascade_impact: str,
    output_language: str = "auto",
    provider: str = None,
) -> dict:
    """
    Step 5b에서 도출한 장벽과 "이미 시도된 것들"을 출발점으로,
    기존 논문이 시도하지 않은 창의적 연구 방향을 3개 후보 생성 후
    LLM 스스로 가장 참신하고 실행 가능한 것을 선택한다.

    핵심 원칙:
      - limitation의 반대를 제안하는 단순 반전 절대 금지
      - 이미 시도된 접근 재사용 금지
      - 인접 분야 기법 조합, 문제 재정의, 평가 프레임워크 변경 등 창의적 각도 탐색
      - web_results의 최신 동향을 활용하되, 거기서 이미 다룬 것은 제외
    """
    barriers_block  = "\n".join(f"  - {b}" for b in barriers)
    tried_block     = "\n".join(f"  - {t}" for t in what_was_tried)
    lim_block       = "\n".join(
        f"  [{lim.paper_id}] {lim.claim[:110]}" for lim in unresolved_lims[:8]
    )

    # 최신 웹 동향 요약 (최신성 맥락 제공)
    web_context = ""
    if web_results:
        recent = [r for r in web_results if r.get("source") == "recency_search"][:6]
        if recent:
            web_context = "\nRecent web developments (use as context, NOT as your answer):\n" + "\n".join(
                f"  [{r.get('title', 'N/A')}] {r.get('content', '')[:200]}"
                for r in recent
            )

    cascade_note = ""
    if cascade_impact and cascade_impact.lower() not in ("none", "unknown", ""):
        cascade_note = f"\nNote: Solving this axis would also benefit: {cascade_impact}"

    prompt = f"""You are a world-class research director known for surprising yet rigorous proposals.

Research Question: "{research_question}"
Axis: {ax_info['label']}

Core unsolved gap:
  "{gap_statement}"

Why it has persisted (technical barriers):
{barriers_block if barriers_block else "  (not specified)"}

Primary barrier type: {barrier_type}

What previous papers already tried (DO NOT propose these again):
{tried_block if tried_block else "  (not documented)"}

Unresolved limitations to address:
{lim_block}
{web_context}
{cascade_note}

=== YOUR TASK ===

Generate 3 DISTINCT candidate research directions. For each:
  - It must NOT be a simple reversal of the limitation ("data is small → use more data")
  - It must NOT repeat what was already tried
  - It should route AROUND the identified barriers through an unexpected angle:
      e.g., reframing the problem, borrowing a technique from an adjacent field,
      changing what gets measured, decomposing the task differently,
      exploiting an under-used data source or signal, or combining two existing
      partial solutions in a novel way
  - It must be specific enough to start experiments next week

Then, choose the BEST candidate: the one that is most (a) novel, (b) feasible, and (c) impactful.

Output JSON only:
{{
  "candidates": [
    {{
      "direction_id": 1,
      "core_insight": "<one sentence: the surprising angle that makes this work>",
      "proposed_topic": "<paper-style title: method + dataset + baseline + goal>",
      "methodology_hint": "<2-3 sentences: what to implement first, which baseline to beat, what result validates it>",
      "novelty_score": <1-10>
    }},
    {{
      "direction_id": 2,
      ...
    }},
    {{
      "direction_id": 3,
      ...
    }}
  ],
  "best_candidate_id": <1, 2, or 3>,
  "selection_rationale": "<why this is the best: novelty + feasibility + impact>"
}}
"""
    from prompts.system import get_language_instruction
    lang_instruction = get_language_instruction(output_language)

    messages = [
        {"role": "system", "content": (
            "You are a creative yet rigorous research mentor. "
            "Your proposals must feel surprising to someone who has read all the papers, "
            "yet be grounded enough to implement. "
            "Avoid generic benchmark-expansion proposals. "
            "Always respond in valid JSON with no extra text."
        )},
        {"role": "user", "content": prompt},
    ]

    try:
        response = _llm_invoke(messages, provider=provider)
        result = parse_json(response)

        candidates   = result.get("candidates", [])
        best_id      = result.get("best_candidate_id", 1)
        rationale    = result.get("selection_rationale", "")

        # best candidate 선택
        best = next((c for c in candidates if c.get("direction_id") == best_id), None)
        if not best and candidates:
            best = max(candidates, key=lambda c: c.get("novelty_score", 0))

        if not best:
            return None

        core_insight     = best.get("core_insight", "")
        proposed_topic   = best.get("proposed_topic", "")
        methodology_hint = best.get("methodology_hint", "")
        novelty_score    = best.get("novelty_score", 0)

        # 다른 후보들 요약 (UI에서 대안으로 표시 가능)
        alt_topics = [
            c.get("proposed_topic", "")
            for c in candidates
            if c.get("direction_id") != best_id and c.get("proposed_topic")
        ]

        # elaboration 조립
        elaboration_parts = []
        if core_insight:
            elaboration_parts.append(core_insight)
        if rationale:
            elaboration_parts.append(f"**Why this direction:** {rationale}")
        if methodology_hint:
            elaboration_parts.append(f"\n💡 **First experiment:** {methodology_hint}")
        if alt_topics:
            alt_str = " / ".join(f"_{t}_" for t in alt_topics[:2])
            elaboration_parts.append(f"\n🔀 **Alternative directions considered:** {alt_str}")

        elaboration = "\n\n".join(elaboration_parts)

        print(f"     best candidate (novelty={novelty_score}): {proposed_topic[:70]}...")

        return {
            "elaboration":    elaboration,
            "proposed_topic": proposed_topic,
        }

    except Exception as e:
        print(f"  ⚠️ Creative direction generation failed for '{ax_key}': {e}")
        return None


# ── 메인 노드 ────────────────────────────────────────────────────────────────

def gap_infer_node(state: AgentState) -> AgentState:
    """
    GAP Inference Node — 3단계 추론 파이프라인
    """
    print("\n💡 GAP Inference Node (성능 최우선 버전)")

    # ── limitations 획득 ────────────────────────────────────────────────────
    raw_limitations = state.get("limitations", [])
    if raw_limitations:
        limitations = []
        for lim in raw_limitations:
            if isinstance(lim, dict):
                item = LimitationItem(
                    paper_id=lim.get("paper_id", "unknown"),
                    claim=lim.get("claim", ""),
                    evidence_quote=lim.get("evidence_quote", ""),
                    track=lim.get("track", "author_stated"),
                    source_section=lim.get("source_section", ""),
                )
                item = item.model_copy(update={"recency_status": lim.get("recency_status", "unresolved"), "recency_evidence": lim.get("recency_evidence", "")})
                limitations.append(item)
            else:
                limitations.append(lim)
    else:
        limitations = _parse_limitations_from_messages(state.get("messages", []))

    output_language = state.get("output_language", "auto")
    session_id = state.get("session_id", "")

    if not limitations:
        print("  ⚠️ No limitations to analyze")
        return {**state, "gaps": []}

    print(f"  ✓ {len(limitations)}개 limitation 로드")
    unresolved_count = sum(
        1 for lim in limitations
        if getattr(lim, "recency_status", "unresolved") == "unresolved"
    )
    print(f"  ✓ recency: unresolved={unresolved_count}, "
          f"partial={sum(1 for l in limitations if getattr(l,'recency_status','unresolved')=='partial')}, "
          f"resolved={sum(1 for l in limitations if getattr(l,'recency_status','unresolved')=='resolved')}")

    # ── web_results ─────────────────────────────────────────────────────────
    web_results = state.get("web_results", [])
    print(f"  ✓ web_results: {len(web_results)}개 (창의적 방향 제안 시 활용)")

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
    print(f"  ✓ 고정 축 {len(fixed_axes)}개 로드")

    # ── Step 2 ──────────────────────────────────────────────────────────────
    all_claims_text = "\n".join(f"[{lim.paper_id}] {lim.claim}" for lim in limitations)
    print(f"  🔄 동적 축 생성 중...")
    dynamic_axes = _generate_dynamic_axes(all_claims_text, fixed_axes, research_question)
    for ax in dynamic_axes:
        print(f"  🟢 동적 축: [{ax['name']}] {ax['label']}")
    if not dynamic_axes:
        print("  ℹ️  동적 축 없음 (고정 5개로 충분)")

    # ── Step 3 ──────────────────────────────────────────────────────────────
    final_axes = _build_final_axes(fixed_axes, dynamic_axes)
    print(f"  ✓ 최종 축 {len(final_axes)}개 확정")

    # ── Step 4 ──────────────────────────────────────────────────────────────
    print(f"  🔄 배치 분류 중...")
    axis_mapping = _classify_limitations_batch(limitations, final_axes)
    axis_groups  = _build_axis_groups_with_recency(limitations, axis_mapping)

    print(f"\n  {'축':<28} {'유형':>6}  {'가중':>6}  {'전체':>6}")
    print(f"  {'-'*52}")
    for ax_key, grp in sorted(axis_groups.items(), key=lambda x: -x[1]["weighted_count"]):
        ax_type = final_axes.get(ax_key, {}).get("type", "fixed")
        label   = final_axes.get(ax_key, {}).get("label", ax_key)
        tag     = "🔵 고정" if ax_type == "fixed" else "🟢 동적"
        print(f"  {label:<28} {tag:>6}  {grp['weighted_count']:>5.1f}  {grp['total_count']:>5}개")

    # ── Step 5a. 긴급도 점수화 ──────────────────────────────────────────────
    active_groups = {k: v for k, v in axis_groups.items() if v["weighted_count"] > 0}
    if not active_groups:
        print("  ⚠️ 모든 limitation이 resolved → gaps 없음")
        return {**state, "gaps": []}

    print(f"\n  🔄 Step 5a: {len(active_groups)}개 축 긴급도 점수화...")
    scored_axes = _score_axis_urgency(active_groups, final_axes, research_question)

    # ── Step 5b + 5c. 축별 장벽 분석 → 창의적 방향 제안 ────────────────────
    gaps = []
    total_axes = len(scored_axes)
    print(f"\n  🔄 Step 5b+5c: 장벽 분석 → 창의적 방향 제안...")

    for ax_idx, (ax_key, urgency_score, cascade_impact, urgency_rationale) in enumerate(scored_axes):
        grp     = active_groups[ax_key]
        ax_info = final_axes.get(ax_key, {"label": ax_key, "description": "", "type": "fixed"})
        unresolved_lims = grp["unresolved_lims"]

        if not unresolved_lims:
            continue

        report_progress(
            session_id, "gap_infer",
            f"Analyzing research axis {ax_idx + 1}/{total_axes}: {ax_info.get('label', ax_key)}",
            current=ax_idx + 1, total=total_axes,
        )
        print(f"\n  ── [{ax_key}] urgency={urgency_score:.2f} ──")

        # Step 5b
        print(f"  🔍 5b 장벽 분석...")
        barrier = _analyze_barriers(
            ax_key, ax_info, unresolved_lims, grp["lims"], research_question
        )
        print(f"     gap: {barrier['gap_statement'][:70]}...")
        print(f"     barrier_type: {barrier['barrier_type']}")
        for b in barrier["barriers"]:
            print(f"     - {b[:80]}")

        # Step 5c
        print(f"  💡 5c 창의적 방향 제안 (web_results={len(web_results)}개 활용)...")
        direction = _generate_creative_directions(
            ax_key=ax_key,
            ax_info=ax_info,
            unresolved_lims=unresolved_lims,
            research_question=research_question,
            gap_statement=barrier["gap_statement"],
            barriers=barrier["barriers"],
            barrier_type=barrier["barrier_type"],
            what_was_tried=barrier["what_was_tried"],
            web_results=web_results,
            cascade_impact=cascade_impact,
            output_language=output_language,
        )

        if direction is None:
            continue

        gaps.append(GapCandidate(
            axis=ax_key,
            axis_label=ax_info["label"],
            axis_type=ax_info["type"],
            gap_statement=barrier["gap_statement"],
            elaboration=direction["elaboration"],
            proposed_topic=direction["proposed_topic"],
            repeat_count=grp["total_count"],
            supporting_papers=list({lim.paper_id for lim in grp["lims"]}),
            supporting_quotes=[lim.evidence_quote for lim in unresolved_lims if lim.evidence_quote][:5],
        ))

    # urgency 점수 기준 정렬 (repeat_count 아닌 urgency 우선)
    urgency_map = {ax: score for ax, score, _, _ in scored_axes}
    gaps.sort(key=lambda g: urgency_map.get(g.axis, 0), reverse=True)
    gaps_as_dict = [g.model_dump() for g in gaps]

    print(f"\n  ✅ GAP {len(gaps)}개 생성 완료")

    trace = dict(state.get("trace", {}))
    trace["gaps_generated"]    = len(gaps)
    trace["axes_used"]         = list(final_axes.keys())
    trace["dynamic_axes"]      = [ax["name"] for ax in dynamic_axes]
    trace["axis_distribution"] = {k: v["total_count"] for k, v in axis_groups.items()}
    trace["urgency_scores"]    = {ax: score for ax, score, _, _ in scored_axes}

    return {
        **state,
        "gaps":  gaps_as_dict,
        "trace": trace,
    }