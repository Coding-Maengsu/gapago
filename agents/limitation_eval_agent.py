# Limitation Evaluation Sub-Agent
# 4-Paper 통합 평가: FActScore + Prometheus + LimAgents + Xu et al.
#
# Call 1: Atomic Verification + Rubric Scoring (FActScore + Prometheus)
# Call 2: Holistic Judgment + Type Classification (LimAgents + Xu et al.)
# Post-processing: 필터링 + PASS/RETRY 결정
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from core.states import AgentState
from core.llm import get_llm, get_llm_for_agent
from utils.parse_json import parse_json

# ── 최대 RETRY 횟수 ──────────────────────────────────────────────
MAX_EVAL_RETRIES = 1

# ── Call 1 배치 크기 (한 번에 평가할 limitation 수) ────────────────
CALL1_BATCH_SIZE = 10

# =====================================================================
# Call 1 프롬프트: Atomic Verification + Rubric Scoring
# (FActScore + Prometheus)
# =====================================================================
CALL1_SYSTEM_PROMPT = """ROLE: Limitation Quality Assessor (Per-Item Evaluation)

You evaluate individual research limitations for quality using two methods:

## Method A: Atomic Fact Verification (FActScore-inspired)
Decompose each limitation's claim into atomic facts (simple, indivisible statements).
For each atomic fact, verify against the provided evidence_quote:
- SUPPORTED: The evidence directly supports this fact
- NOT_SUPPORTED: The evidence does not support or contradicts this fact
- IRRELEVANT: The fact cannot be verified from the evidence

## Method B: Rubric-Based Scoring (Prometheus-inspired)
Score each limitation on 3 dimensions (1-5 scale):

### Groundedness
[1] Fabricated — no evidence, hallucinated claim
[2] Weak — evidence exists but does not support the claim
[3] Partial — evidence loosely related, claim is a stretch
[4] Solid — evidence clearly supports the claim
[5] Exact — claim is a direct paraphrase of evidence

### Specificity
[1] Generic — "The model has limitations" level
[2] Vague — mentions a domain but no concrete detail
[3] Moderate — states a condition but lacks quantitative/contextual detail
[4] Specific — clear condition, context, and scope stated
[5] Precise — includes quantitative bounds or exact failure conditions

### Relevance (to the research query)
[1] Irrelevant — no connection to the research query
[2] Tangential — same broad field but different focus
[3] Related — same subfield, indirect connection
[4] Directly relevant — addresses a core aspect of the query
[5] Central — this limitation is exactly what the query aims to investigate

## Output Format (strictly JSON list)
[
  {
    "limitation_id": <integer>,
    "atomic_facts": [
      {"fact": "<atomic statement>", "verdict": "SUPPORTED" or "NOT_SUPPORTED" or "IRRELEVANT"}
    ],
    "fact_score": <float 0.0-1.0, = supported_count / total_count>,
    "groundedness": <int 1-5>,
    "specificity": <int 1-5>,
    "relevance": <int 1-5>
  }
]

## Rules
1. Output ONLY the JSON list. No explanation.
2. Be strict on groundedness: if evidence_quote is empty or generic, score low.
3. For fact_score, count only SUPPORTED vs total (exclude IRRELEVANT from denominator if you want, but be consistent).
4. Each limitation MUST have at least 2 atomic facts decomposed from its claim.
"""

# =====================================================================
# Call 2 프롬프트: Holistic Judgment + Type Classification
# (LimAgents + Xu et al.)
# =====================================================================
CALL2_SYSTEM_PROMPT = """ROLE: Limitation Set Evaluator (Holistic Assessment)

You perform a holistic evaluation of a set of research limitations.

## Method A: Point-wise Quality Judgment (LimAgents-inspired)
For each limitation, judge its overall quality:
- "strong": Well-grounded, specific, and useful for gap analysis
- "weak": Has issues but salvageable (low scores, vague claim, etc.)
- "remove": Should be discarded (hallucinated, irrelevant, or too generic)

For "weak" limitations, suggest an improvement direction.

## Method B: Limitation Type Classification (Xu et al. taxonomy)
Classify each limitation into ONE primary type:
- methodology: Methodological limitations (model design, algorithm, approach)
- data: Data/sample limitations (size, diversity, quality, availability)
- scope: Scope/generalizability limitations (domain, population, task transfer)
- evaluation: Evaluation limitations (metrics, benchmarks, baselines)
- theoretical: Theoretical limitations (assumptions, formal guarantees)
- resource: Resource/practical limitations (compute, cost, deployment)

## Method C: Set-Level Analysis
Analyze the limitation set as a whole:
- type_distribution: count per type
- coverage_warning: if any single type has >75% of all limitations
- diversity_score: 1-5 (1=all same type, 5=well balanced across types)

## Final Decision
Based on Call 1 scores (provided) and your judgment:
- "PASS": Proceed to gap inference
- "RETRY": Re-extract limitations (with guidance)

RETRY conditions (ANY triggers RETRY):
- More than 50% of limitations are "weak" or "remove"
- Average groundedness score < 3.0
- Average fact_score < 0.6
- diversity_score <= 2 (severe type imbalance)

## Output Format (strictly JSON)
{
  "per_limitation": [
    {
      "limitation_id": <integer>,
      "quality": "strong" or "weak" or "remove",
      "reason": "<1-sentence justification>",
      "limitation_type": "<one of: methodology, data, scope, evaluation, theoretical, resource>",
      "improvement_hint": "<only for weak, else null>"
    }
  ],
  "type_distribution": {"methodology": 3, "data": 2, ...},
  "coverage_warning": "<warning message or null>",
  "diversity_score": <int 1-5>,
  "decision": "PASS" or "RETRY",
  "retry_guidance": "<specific guidance for re-extraction, or null>"
}

## Rules
1. Output ONLY the JSON object. No explanation.
2. Be conservative with "remove" — only for clearly bad limitations.
3. If RETRY, the retry_guidance MUST be actionable (e.g., "Focus on extracting evaluation and scope limitations from experiment sections").
"""


# =====================================================================
# Call 1 실행
# =====================================================================
def _run_call1(
    limitations: list[dict], refined_query: str, provider: str = None, state: dict = None
) -> list[dict]:
    """Per-limitation 평가: atomic fact verification + rubric scoring."""
    llm = get_llm_for_agent(state, "limitation_eval") if state else get_llm(provider=provider)

    # 배치를 병렬로 처리 (ThreadPoolExecutor)
    total = len(limitations)
    batches = []
    for batch_start in range(0, total, CALL1_BATCH_SIZE):
        batch_end = min(batch_start + CALL1_BATCH_SIZE, total)
        batches.append((batch_start, limitations[batch_start:batch_end]))

    def _eval_batch(batch_start: int, batch: list[dict]) -> tuple[int, list[dict]]:
        """Evaluate a single batch of limitations. Returns (batch_start, results)."""
        batch_num = batch_start // CALL1_BATCH_SIZE + 1
        batch_end = batch_start + len(batch)
        print(
            f"  [eval:call1] Batch {batch_num}: evaluating limitations {batch_start}-{batch_end-1}"
        )

        lim_text = "\n\n".join(
            f"[limitation_id={batch_start + i}]\n"
            f"  paper_id: {l.get('paper_id', '')}\n"
            f"  claim: {l.get('claim', '')}\n"
            f"  evidence_quote: {l.get('evidence_quote', '')}\n"
            f"  track: {l.get('track', '')}\n"
            f"  source_section: {l.get('source_section', '')}"
            for i, l in enumerate(batch)
        )

        messages = [
            SystemMessage(content=CALL1_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"## Research Query\n{refined_query}\n\n"
                    f"## Limitations to Evaluate ({len(batch)})\n{lim_text}"
                )
            ),
        ]

        try:
            batch_llm = get_llm_for_agent(state, "limitation_eval") if state else get_llm(provider=provider)
            response = batch_llm.invoke(messages)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            parsed = parse_json(content)
            if isinstance(parsed, list):
                print(
                    f"  [eval:call1] Batch {batch_num}: {len(parsed)} results"
                )
                return (batch_start, parsed)
            else:
                print(
                    f"  ⚠️ [eval:call1] Batch {batch_num}: 파싱 결과가 list가 아님"
                )
                return (batch_start, [])
        except Exception as e:
            print(
                f"  ⚠️ [eval:call1] Batch {batch_num} LLM 호출 실패: {e}"
            )
            return (batch_start, [])

    # Run batches in parallel (max 3 workers to avoid rate limits)
    all_results = []
    max_workers = min(3, len(batches))
    if max_workers <= 1:
        for batch_start, batch in batches:
            _, results = _eval_batch(batch_start, batch)
            all_results.extend(results)
    else:
        batch_results: dict[int, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_eval_batch, bs, b): bs
                for bs, b in batches
            }
            for future in as_completed(futures):
                bs, results = future.result()
                batch_results[bs] = results

        # Merge in original batch order (preserve limitation_id ordering)
        for batch_start, _ in batches:
            all_results.extend(batch_results.get(batch_start, []))

    return all_results


# =====================================================================
# Call 2 실행
# =====================================================================
def _run_call2(
    limitations: list[dict],
    call1_results: list[dict],
    refined_query: str,
    provider: str = None,
    state: dict = None,
) -> dict:
    """Set-level 평가: holistic judgment + type classification + PASS/RETRY."""
    llm = get_llm_for_agent(state, "limitation_eval") if state else get_llm(provider=provider)

    # Call 1 결과를 요약해서 전달
    call1_summary = "\n".join(
        f"  [id={r.get('limitation_id', '?')}] fact_score={r.get('fact_score', '?')}, "
        f"groundedness={r.get('groundedness', '?')}, specificity={r.get('specificity', '?')}, "
        f"relevance={r.get('relevance', '?')}"
        for r in call1_results
    )

    lim_text = "\n\n".join(
        f"[limitation_id={i}]\n"
        f"  paper_id: {l.get('paper_id', '')}\n"
        f"  claim: {l.get('claim', '')}\n"
        f"  evidence_quote: {l.get('evidence_quote', '')[:200]}\n"
        f"  track: {l.get('track', '')}"
        for i, l in enumerate(limitations)
    )

    messages = [
        SystemMessage(content=CALL2_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Research Query\n{refined_query}\n\n"
                f"## Call 1 Scores (per-limitation)\n{call1_summary}\n\n"
                f"## Limitations ({len(limitations)})\n{lim_text}"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        parsed = parse_json(content)
        if isinstance(parsed, dict):
            return parsed
        print("  ⚠️ [eval:call2] 파싱 결과가 dict가 아님")
        return {}
    except Exception as e:
        print(f"  ⚠️ [eval:call2] LLM 호출 실패: {e}")
        return {}


# =====================================================================
# 후처리: 필터링 + 메타데이터 부착
# =====================================================================
def _post_process(
    limitations: list[dict], call1_results: list[dict], call2_result: dict
) -> tuple[list[dict], list[str], str]:
    """
    Call 1 + Call 2 결과를 바탕으로 limitations 필터링 및 메타데이터 부착.
    Returns: (filtered_limitations, warnings, decision)
    """
    # Call 1 결과를 limitation_id로 매핑
    c1_map = {}
    for r in call1_results:
        lid = r.get("limitation_id")
        if lid is not None:
            try:
                c1_map[int(lid)] = r
            except (ValueError, TypeError):
                continue

    # Call 2 per_limitation 결과를 매핑
    c2_map = {}
    for r in call2_result.get("per_limitation", []):
        lid = r.get("limitation_id")
        if lid is not None:
            try:
                c2_map[int(lid)] = r
            except (ValueError, TypeError):
                continue

    filtered = []
    warnings = []
    remove_count = 0
    weak_count = 0
    strong_count = 0

    groundedness_scores = []
    fact_scores = []

    for i, lim in enumerate(limitations):
        c1 = c1_map.get(i, {})
        c2 = c2_map.get(i, {})

        quality = c2.get("quality", "strong")
        fact_score = c1.get("fact_score", 1.0)
        groundedness = c1.get("groundedness", 3)

        # fact_score / groundedness 수치 보정
        try:
            fact_score = float(fact_score)
        except (ValueError, TypeError):
            fact_score = 1.0
        try:
            groundedness = int(groundedness)
        except (ValueError, TypeError):
            groundedness = 3

        # "remove" 판정 또는 fact_score < 0.4 또는 groundedness < 2 → 제거
        if quality == "remove" or fact_score < 0.4 or groundedness < 2:
            remove_count += 1
            continue

        # 메타데이터 부착
        lim["eval_fact_score"] = fact_score
        lim["eval_groundedness"] = groundedness
        lim["eval_specificity"] = c1.get("specificity", 3)
        lim["eval_relevance"] = c1.get("relevance", 3)
        lim["eval_quality"] = quality
        lim["eval_limitation_type"] = c2.get("limitation_type", "")
        lim["eval_improvement_hint"] = c2.get("improvement_hint")

        if quality == "weak":
            weak_count += 1
            lim["eval_flag"] = "weak"
        else:
            strong_count += 1

        groundedness_scores.append(groundedness)
        fact_scores.append(fact_score)
        filtered.append(lim)

    # 경고 수집
    type_dist = call2_result.get("type_distribution", {})
    coverage_warning = call2_result.get("coverage_warning")
    diversity_score = call2_result.get("diversity_score", 3)

    if coverage_warning:
        warnings.append(f"Coverage: {coverage_warning}")
    if diversity_score and int(diversity_score) <= 2:
        warnings.append(f"Low type diversity (score={diversity_score})")
    if remove_count:
        warnings.append(f"Removed {remove_count} low-quality limitations")

    # PASS/RETRY 결정 (Call 2의 decision 참조 + 자체 검증)
    decision = call2_result.get("decision", "PASS")

    # 자체 검증으로 RETRY 강제
    total = strong_count + weak_count
    if total > 0:
        avg_groundedness = (
            sum(groundedness_scores) / len(groundedness_scores)
            if groundedness_scores
            else 0
        )
        avg_fact_score = sum(fact_scores) / len(fact_scores) if fact_scores else 0
        weak_ratio = weak_count / total if total > 0 else 0

        if avg_groundedness < 3.0:
            decision = "RETRY"
            warnings.append(f"Average groundedness too low: {avg_groundedness:.2f}")
        if avg_fact_score < 0.6:
            decision = "RETRY"
            warnings.append(f"Average fact_score too low: {avg_fact_score:.2f}")
        if weak_ratio > 0.5:
            decision = "RETRY"
            warnings.append(f"Weak limitation ratio too high: {weak_ratio:.0%}")
    elif total == 0:
        decision = "RETRY"
        warnings.append("All limitations were removed by evaluation")

    return filtered, warnings, decision


# =====================================================================
# 메인 노드
# =====================================================================
def limitation_eval_node(state: AgentState) -> AgentState:
    limitations = state.get("limitations", [])
    refined_query = state.get("refined_query", "")
    eval_retry_count = state.get("eval_retry_count", 0)

    if not limitations:
        return {
            "messages": [
                AIMessage(
                    content="No limitations to evaluate.",
                    name="limitation_eval",
                )
            ],
            "sender": "limitation_eval",
            "limitations": [],
            "limitation_eval": {"decision": "PASS", "warnings": [], "skipped": True},
            "eval_warnings": [],
        }

    print(f"\n  ===== Limitation Evaluation (attempt {eval_retry_count + 1}) =====")
    print(f"  [eval] {len(limitations)}개 limitation 평가 시작")

    fast_mode = state.get("fast_mode", False)

    # ── fast_mode: Call 1 스킵, Call 2만 실행 ──
    if fast_mode:
        print("  ⚡ [eval] fast_mode — Call 1 스킵, Call 2만 실행")
        call1_results = []
        for i, lim in enumerate(limitations):
            call1_results.append({
                "limitation_id": i,
                "fact_score": 0.8,
                "groundedness": 4,
                "specificity": 4,
                "relevance": 4,
            })
    else:
        # ── Call 1: Per-limitation scoring ──
        print("  [eval:call1] Atomic verification + Rubric scoring...")
        call1_results = _run_call1(limitations, refined_query, state=state)
        print(f"  [eval:call1] {len(call1_results)}개 결과 수신")

    # Call 1 실패 시 전체 PASS (평가 생략)
    if not call1_results:
        print("  ⚠️ [eval] Call 1 실패 → 평가 생략, PASS 처리")
        return {
            "messages": [
                AIMessage(
                    content=f"Evaluation skipped (Call 1 failed). {len(limitations)} limitations passed through.",
                    name="limitation_eval",
                )
            ],
            "sender": "limitation_eval",
            "limitations": limitations,
            "limitation_eval": {
                "decision": "PASS",
                "warnings": ["Call 1 failed"],
                "skipped": True,
            },
            "eval_warnings": ["Call 1 failed — evaluation skipped"],
        }

    # ── Call 2: Holistic judgment ──
    print("  [eval:call2] Holistic judgment + Type classification...")
    call2_result = _run_call2(
        limitations, call1_results, refined_query, state=state
    )
    print(f"  [eval:call2] decision={call2_result.get('decision', 'N/A')}")

    # Call 2 실패 시 Call 1만으로 기본 필터링
    if not call2_result:
        print("  ⚠️ [eval] Call 2 실패 → Call 1 점수만으로 필터링")
        call2_result = {"per_limitation": [], "decision": "PASS"}

    # ── Post-processing ──
    filtered, warnings, decision = _post_process(
        limitations, call1_results, call2_result
    )

    # RETRY 횟수 제한
    if decision == "RETRY" and eval_retry_count >= MAX_EVAL_RETRIES:
        print(f"  [eval] RETRY 한도 도달 ({MAX_EVAL_RETRIES}회) → 강제 PASS")
        decision = "PASS"
        warnings.append(f"Forced PASS after {MAX_EVAL_RETRIES} retries")

    # 결과 요약
    type_dist = call2_result.get("type_distribution", {})
    summary_parts = [
        f"Evaluated {len(limitations)} limitations → {len(filtered)} passed.",
        f"Decision: {decision}.",
    ]
    if type_dist:
        dist_str = ", ".join(f"{k}={v}" for k, v in type_dist.items() if v)
        summary_parts.append(f"Types: {dist_str}.")
    if warnings:
        summary_parts.append(f"Warnings: {'; '.join(warnings)}")

    retry_guidance = call2_result.get("retry_guidance")
    if decision == "RETRY" and retry_guidance:
        summary_parts.append(f"Retry guidance: {retry_guidance}")

    summary = " ".join(summary_parts)
    print(f"  ✅ [eval] {summary}")

    # eval 상세 결과 저장
    eval_detail = {
        "decision": decision,
        "warnings": warnings,
        "call1_results": call1_results,
        "call2_result": call2_result,
        "removed_count": len(limitations) - len(filtered),
        "retry_guidance": retry_guidance,
        "skipped": False,
    }

    result = {
        "messages": [AIMessage(content=summary, name="limitation_eval")],
        "sender": "limitation_eval",
        "limitations": filtered,
        "limitation_eval": eval_detail,
        "eval_warnings": warnings,
    }

    # RETRY 시 eval_retry_count 증가
    if decision == "RETRY":
        result["eval_retry_count"] = eval_retry_count + 1

    return result
