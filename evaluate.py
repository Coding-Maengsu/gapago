"""
evaluate.py — GAPAGO vs Baseline LLM 비교 평가 스크립트
=========================================================

사용법:
  # 단일 provider와 비교 (기본: azure)
  python evaluate.py --result-file outputs/gapago_result_20260324_095705.json

  # 특정 provider 지정
  python evaluate.py --result-file outputs/gapago_result_*.json --baseline-provider gemini

  # 모든 provider 동시 비교 (azure / claude / gemini)
  python evaluate.py --result-file outputs/gapago_result_*.json --compare-all

  # 이미 생성한 baseline JSON 재사용
  python evaluate.py --result-file outputs/gapago_result_*.json --baseline-file outputs/baseline_azure_*.json

출력:
  outputs/eval_report_YYYYMMDD_HHMMSS.json  — 상세 수치 결과
  outputs/eval_report_YYYYMMDD_HHMMSS.md    — 비교 리포트

평가 항목 (5가지):
  1. Groundedness  : 실제 논문 근거 인용의 존재 여부 + 품질
  2. Novelty       : 제안 주제가 기존 연구에서 이미 해결됐는지 (LLM 판정)
  3. Specificity   : 방법론 / 데이터셋 / 목표가 명시됐는지
  4. Relevance     : 원래 질문과 제안 주제의 의미적 연관성
  5. Diversity     : 제안된 주제들이 서로 다른 관점을 커버하는지
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── numpy ───────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    print("[ERROR] numpy 없음: pip install numpy")
    sys.exit(1)

# ── arxiv (optional) ────────────────────────────────────────────────────────
try:
    import arxiv
except ImportError:
    arxiv = None

# ── SentenceTransformer / TF-IDF 폴백 ───────────────────────────────────────
_USE_TFIDF_FALLBACK = False
try:
    from sentence_transformers import SentenceTransformer
    _CACHE_DIR = Path.home() / ".cache" / "torch" / "sentence_transformers"
    _HAS_CACHE = any(_CACHE_DIR.rglob("config.json")) if _CACHE_DIR.exists() else False
    if not _HAS_CACHE:
        _USE_TFIDF_FALLBACK = True
        print("  ℹ️  SentenceTransformer 캐시 없음 → TF-IDF 폴백 모드로 동작합니다.")
except ImportError:
    _USE_TFIDF_FALLBACK = True

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ════════════════════════════════════════════════════════════════════════
# 0. 상수 / 설정
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

METRIC_WEIGHTS = {
    "groundedness": 0.30,
    "novelty":      0.25,
    "specificity":  0.20,
    "relevance":    0.15,
    "diversity":    0.10,
}

# repeat_count 임계값: 이 미만이면 "약한 근거" GAP으로 분류
REPEAT_COUNT_THRESHOLD = 2


# ════════════════════════════════════════════════════════════════════════
# 0-1. Embedding / 유사도 헬퍼
# ════════════════════════════════════════════════════════════════════════

_EMBED_MODEL = None


def _cosine_matrix(texts: list) -> np.ndarray:
    global _EMBED_MODEL
    if not _USE_TFIDF_FALLBACK:
        if _EMBED_MODEL is None:
            print("  📦 SentenceTransformer 로딩 중...")
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        embeds = _EMBED_MODEL.encode(texts)
    elif _HAS_SKLEARN:
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        embeds = vec.fit_transform(texts).toarray()
    else:
        vocab = {}
        for t in texts:
            for i in range(len(t) - 2):
                vocab[t[i:i+3]] = vocab.get(t[i:i+3], 0) + 1
        keys = list(vocab.keys())
        embeds = np.zeros((len(texts), max(len(keys), 1)))
        for i, t in enumerate(texts):
            for j, k in enumerate(keys):
                embeds[i][j] = t.count(k)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    normed = embeds / (norms + 1e-9)
    return normed @ normed.T


def _embed_query_topics(query_texts: list, topic_texts: list) -> np.ndarray:
    all_texts = query_texts + topic_texts
    sim_mat = _cosine_matrix(all_texts)
    nq = len(query_texts)
    return sim_mat[:nq, nq:].mean(axis=0)


# ════════════════════════════════════════════════════════════════════════
# 1. Baseline 생성 — llm.py의 get_llm() 재활용
# ════════════════════════════════════════════════════════════════════════

BASELINE_SYSTEM_PROMPT = """You are a research assistant.
Given a research query, suggest 5 novel research topics that address unexplored gaps.

For each topic, provide:
- proposed_topic: a specific, actionable research title
- gap_statement: one sentence describing the gap this topic addresses
- elaboration: 2-3 sentences explaining why this gap matters
- axis: one of [data, methodology, generalizability, evaluation, scalability]
- axis_label: human-readable label for the axis

Respond ONLY in valid JSON:
{
  "gaps": [
    {
      "proposed_topic": "...",
      "gap_statement": "...",
      "elaboration": "...",
      "axis": "...",
      "axis_label": "..."
    }
  ]
}"""


def generate_baseline_response(query: str, provider: str = "azure") -> dict:
    """
    llm.py의 get_llm()을 재활용해 GAPAGO와 동일한 API 인증으로 baseline 생성.
    별도 API 키 설정 불필요 — GAPAGO가 동작하는 환경이면 그대로 작동.

    provider:
      azure  — Azure OpenAI GPT  (기본)
      claude — AWS Bedrock Claude
      gemini — Google Gemini
      exaone — 로컬 GPU EXAONE
    """
    sys.path.insert(0, str(Path(__file__).parent))
    print(f"\n  🤖 Baseline 생성 중 (provider={provider})")
    print(f"     query: '{query[:70]}'")

    try:
        from llm import get_llm
        os.environ["LLM_PROVIDER"] = provider
        get_llm.cache_clear()
        llm = get_llm(provider=provider)
    except Exception as e:
        print(f"  [ERROR] LLM 초기화 실패: {e}")
        print("  → .env 파일 확인:")
        print("    azure  : AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT")
        print("    claude : AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION")
        print("    gemini : GOOGLE_API_KEY")
        raise

    from langchain_core.messages import SystemMessage, HumanMessage
    response = llm.invoke([
        SystemMessage(content=BASELINE_SYSTEM_PROMPT),
        HumanMessage(content=f"Research query: {query}"),
    ])
    raw = response.content if hasattr(response, "content") else str(response)

    # JSON 파싱
    gaps = []
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(m.group() if m else raw)
        gaps = parsed.get("gaps", [])
    except json.JSONDecodeError:
        print(f"  ⚠️ JSON 파싱 실패:\n{raw[:400]}")

    # GAPAGO 스키마에 맞게 보강
    # Baseline은 실제 논문 검색 없이 생성 → supporting_quotes/papers = []
    for g in gaps:
        g.setdefault("repeat_count", 0)
        g.setdefault("supporting_papers", [])
        g.setdefault("supporting_quotes", [])
        g.setdefault("axis", "unknown")
        g.setdefault("axis_label", "Unknown")
        g.setdefault("axis_type", "baseline")

    result = {
        "query":         query,
        "timestamp":     datetime.now().isoformat(),
        "refined_query": query,
        "keywords":      [],
        "limitations":   [],
        "gaps":          gaps,
        "web_results":   [],
        "messages":      [],
        "source":        provider,
    }
    fname = f"baseline_{provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    fpath = OUTPUT_DIR / fname
    fpath.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✅ Baseline 저장 → {fpath}  (gaps={len(gaps)}개)")
    return result


# ════════════════════════════════════════════════════════════════════════
# 2. 평가 항목별 측정 함수
# ════════════════════════════════════════════════════════════════════════

# ── 2-1. Groundedness ────────────────────────────────────────────────────────
#
# 개선된 설계:
#   A) 단순 유무(0/1) → 단계별 점수로 세분화
#      - supporting_quotes 있음 + repeat_count >= THRESHOLD → 1.0  (강한 근거)
#      - supporting_quotes 있음 + repeat_count <  THRESHOLD → 0.6  (약한 근거: 논문 1편 지지)
#      - supporting_papers만 있음                           → 0.3  (논문 연결은 됐으나 인용 없음)
#      - 아무것도 없음                                      → 0.0  (hallucination 위험)
#
#   B) 왜 이렇게 하는가?
#      - repeat_count=1 인 GAP은 논문 1편에서만 나온 한계점 → 연구 전반의 공통 GAP이라 보기 어려움
#      - Baseline LLM도 "데이터 의존성이 높다"는 자체 논리가 있지만 실제 논문 인용이 없으므로
#        partial credit 없이 0점이 맞음. 인용 없으면 근거없는 주장.

def score_groundedness(gaps: list) -> dict:
    """
    각 GAP의 근거 품질을 단계적으로 측정.

    점수 체계:
      1.0 — supporting_quotes 있음 AND repeat_count >= REPEAT_COUNT_THRESHOLD (강한 근거)
      0.6 — supporting_quotes 있음 AND repeat_count < REPEAT_COUNT_THRESHOLD  (약한 근거)
      0.3 — supporting_papers만 있음 (논문 연결은 됐으나 직접 인용 없음)
      0.0 — 아무 근거 없음 (baseline LLM의 전형적 패턴)
    """
    if not gaps:
        return {"score": 0.0, "per_gap": [], "strong_count": 0, "weak_count": 0, "no_evidence_count": 0}

    per_gap = []
    for g in gaps:
        n_quotes = len(g.get("supporting_quotes", []))
        n_papers = len(g.get("supporting_papers", []))
        repeat   = g.get("repeat_count", 0)

        if n_quotes > 0 and repeat >= REPEAT_COUNT_THRESHOLD:
            grade, label = 1.0, "strong"
        elif n_quotes > 0:
            grade, label = 0.6, "weak"   # 논문 1편짜리 → partial credit
        elif n_papers > 0:
            grade, label = 0.3, "paper_only"
        else:
            grade, label = 0.0, "none"

        per_gap.append({
            "topic":        g.get("proposed_topic", "")[:60],
            "n_quotes":     n_quotes,
            "n_papers":     n_papers,
            "repeat_count": repeat,
            "grade":        grade,
            "label":        label,
        })

    score         = float(np.mean([p["grade"] for p in per_gap]))
    strong_count  = sum(1 for p in per_gap if p["label"] == "strong")
    weak_count    = sum(1 for p in per_gap if p["label"] == "weak")
    paper_count   = sum(1 for p in per_gap if p["label"] == "paper_only")
    no_ev_count   = sum(1 for p in per_gap if p["label"] == "none")

    return {
        "score":             round(score, 4),
        "per_gap":           per_gap,
        "strong_count":      strong_count,    # 강한 근거 (복수 논문 반복 인용)
        "weak_count":        weak_count,      # 약한 근거 (논문 1편 인용)
        "paper_only_count":  paper_count,     # 논문 연결만 있음
        "no_evidence_count": no_ev_count,     # 근거 없음
        "repeat_threshold":  REPEAT_COUNT_THRESHOLD,
    }


# ── 2-2. Novelty ─────────────────────────────────────────────────────────────
#
# 핵심: "proposed_topic이 기존 연구에서 이미 해결됐는가?"
#
# 방식 A — sklearn 있고 arXiv 접근 가능: TF-IDF cosine (arXiv abstract vs 아이디어)
# 방식 B — sklearn 없거나 arXiv 불가   : LLM에게 직접 판정 요청
#
# 방식 B (LLM-as-Judge) 설계:
#   - LLM에게 각 proposed_topic + gap_statement + elaboration 을 주고
#   - "이 연구 주제가 2024년 이전에 이미 활발히 연구됐는지" 0~1 점수로 판정
#   - 1.0 = 완전히 새로운 주제, 0.0 = 이미 넘쳐나는 주제
#   - GAPAGO 기대값: 실제 논문 한계점 기반 → 구체적 + 새로움 → 높은 점수
#   - Baseline 기대값: 포괄적 흔한 주제 → 낮은 점수

def _extract_search_query(idea_text: str) -> str:
    text   = re.sub(r"[^a-zA-Z0-9 ]", " ", idea_text.lower())
    tokens = text.split()
    stops  = {
        "a","an","the","is","are","was","were","be","been","have","has","had",
        "do","does","did","will","would","should","could","may","might","in",
        "on","at","to","for","of","with","by","from","this","that","and","or",
        "but","not","research","study","paper","work","approach","method","based",
        "using","via","lack","limited","existing","current","proposed","novel","new",
    }
    filtered = [t for t in tokens if t not in stops and len(t) > 2]
    return " ".join(filtered[:5]) or " ".join(tokens[:4])


def _score_novelty_llm(gaps: list) -> dict:
    """
    [방식 B] sklearn/arXiv 없을 때 LLM-as-Judge로 Novelty 판정.

    LLM에게 각 GAP의 아이디어 텍스트를 주고:
      - 이 주제가 2024년 이전 연구에서 이미 광범위하게 다뤄졌는가?
      - 0.0 (완전히 기존 연구) ~ 1.0 (매우 새로운 주제) 점수 반환

    단일 LLM 호출로 모든 GAP을 한번에 판정 (비용 절약).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from llm import get_llm
        llm = get_llm()
    except Exception as e:
        print(f"    ⚠️ LLM 로드 실패 ({e}) → Novelty 스킵")
        return {"score": None, "per_gap": [], "skipped": True, "method": "llm_failed"}

    from langchain_core.messages import SystemMessage, HumanMessage

    # 모든 GAP을 한 번의 LLM 호출로 판정
    items_block = "\n\n".join(
        f"[{i}]\n"
        f"proposed_topic: {g.get('proposed_topic','')}\n"
        f"gap_statement: {g.get('gap_statement','')}\n"
        f"elaboration: {g.get('elaboration','')}"
        for i, g in enumerate(gaps)
    )

    prompt = f"""You are a research novelty assessor.

For each research proposal below, assess how NOVEL it is — meaning how likely it is that this EXACT combination of problem + approach + scope has NOT been thoroughly addressed in existing research (up to 2024).

Score: 0.0 = already well-researched topic  |  1.0 = genuinely new unexplored gap

Key distinctions:
- "Improving deepfake detection with deep learning" → 0.1  (extremely common topic)
- "Cross-domain compression-robust deepfake detection for non-facial media" → 0.8  (specific + novel combination)
- "Transfer learning for medical imaging" → 0.2  (very common)
- "Few-shot adaptation for rare disease subtype classification under label scarcity" → 0.85  (specific niche)

Proposals:
{items_block}

Respond ONLY in valid JSON:
{{
  "novelty_scores": [
    {{"index": 0, "score": 0.0, "reason": "one sentence explanation"}},
    ...
  ]
}}"""

    try:
        resp = llm.invoke([
            SystemMessage(content="You are a research novelty assessor. Always respond in valid JSON."),
            HumanMessage(content=prompt),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(m.group() if m else raw)
        scores_raw = parsed.get("novelty_scores", [])
    except Exception as e:
        print(f"    ⚠️ LLM novelty 판정 실패: {e}")
        return {"score": None, "per_gap": [], "skipped": True, "method": "llm_failed"}

    # index 기반 매핑
    score_map = {item["index"]: item for item in scores_raw}
    per_gap   = []
    for i, g in enumerate(gaps):
        item    = score_map.get(i, {})
        novelty = float(item.get("score", 0.5))
        per_gap.append({
            "topic":   g.get("proposed_topic", "")[:80],
            "novelty": round(novelty, 4),
            "reason":  item.get("reason", ""),
            "method":  "llm_judge",
        })
        print(f"      [{g.get('axis','?'):20}] novelty={novelty:.3f}  {item.get('reason','')[:60]}")

    avg = float(np.mean([p["novelty"] for p in per_gap])) if per_gap else 0.5
    return {
        "score":   round(avg, 4),
        "per_gap": per_gap,
        "skipped": False,
        "method":  "llm_judge",
    }


def _score_novelty_tfidf(gaps: list, max_results: int = 10) -> dict:
    """
    [방식 A] arXiv abstract와 TF-IDF cosine 유사도 기반 Novelty 판정.
    novelty = 1 - max_sim (가장 유사한 기존 논문과의 의미적 거리)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    client  = arxiv.Client(num_retries=2, delay_seconds=1.5)
    per_gap = []

    for g in gaps:
        idea_text = " ".join(filter(None, [
            g.get("proposed_topic", ""),
            g.get("gap_statement", ""),
            g.get("elaboration", ""),
        ]))
        if not idea_text.strip():
            per_gap.append({"topic": "", "max_sim": 0.0, "novelty": 1.0, "method": "tfidf"})
            continue

        search_query = _extract_search_query(idea_text)
        abstracts, titles = [], []
        try:
            search  = arxiv.Search(query=search_query, max_results=max_results,
                                   sort_by=arxiv.SortCriterion.Relevance)
            results = list(client.results(search))
            abstracts = [r.summary for r in results]
            titles    = [r.title   for r in results]
            time.sleep(1.0)
        except Exception as e:
            print(f"    ⚠️ arXiv 검색 실패 ({search_query[:40]}): {e}")

        if not abstracts:
            per_gap.append({
                "topic": g.get("proposed_topic","")[:80], "max_sim": None,
                "novelty": 0.5, "method": "tfidf", "note": "검색 실패 → 중립값"
            })
            continue

        corpus = [idea_text] + abstracts
        try:
            vec          = TfidfVectorizer(stop_words="english", max_features=3000)
            tfidf_matrix = vec.fit_transform(corpus)
            sims         = sk_cosine(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        except Exception as e:
            print(f"    ⚠️ TF-IDF 실패: {e}")
            per_gap.append({"topic": g.get("proposed_topic","")[:80], "max_sim": None,
                            "novelty": 0.5, "method": "tfidf"})
            continue

        max_sim    = float(np.max(sims))
        best_idx   = int(np.argmax(sims))
        best_title = titles[best_idx] if best_idx < len(titles) else ""
        novelty    = round(1.0 - max_sim, 4)
        per_gap.append({
            "topic":              g.get("proposed_topic","")[:80],
            "search_query":       search_query,
            "max_sim":            round(max_sim, 4),
            "most_similar_paper": best_title[:100],
            "novelty":            novelty,
            "method":             "tfidf",
        })
        print(f"      [{g.get('axis','?'):20}] max_sim={max_sim:.3f} → novelty={novelty:.3f}"
              f"  ('{best_title[:45]}...')")

    valid = [p["novelty"] for p in per_gap if p.get("novelty") is not None]
    return {
        "score":   round(float(np.mean(valid)), 4) if valid else 0.5,
        "per_gap": per_gap,
        "skipped": False,
        "method":  "tfidf",
    }


def score_novelty(gaps: list) -> dict:
    """
    Novelty 측정 — 방식 자동 선택:
      방식 A (TF-IDF + arXiv) : sklearn 있고 arXiv 검색 실제로 성공 시
      방식 B (LLM-as-Judge)   : sklearn 없거나 arXiv 접근 불가 시 자동 전환

    방식 A → B 전환 조건:
      - sklearn 미설치
      - arxiv 패키지 미설치
      - arXiv 검색이 모든 GAP에서 실패한 경우 (네트워크 차단, 프록시 오류 등)
        → per_gap 전체의 max_sim이 None이면 실제 검색이 0건임을 감지해 B로 전환
    """
    if not gaps:
        return {"score": 0.0, "per_gap": [], "skipped": False}

    # ── 방식 A 시도 ──────────────────────────────────────────────────
    if _HAS_SKLEARN and arxiv is not None:
        print("    → Novelty 방식 A 시도: TF-IDF + arXiv")
        try:
            result = _score_novelty_tfidf(gaps)
            # 실제 유사도 계산이 한 건도 없으면(모두 max_sim=None) 방식 B로 전환
            all_failed = all(p.get("max_sim") is None for p in result.get("per_gap", []))
            if all_failed:
                print("    ⚠️ arXiv 검색 전체 실패 (네트워크 차단 등) → 방식 B(LLM)로 전환")
                raise RuntimeError("arXiv 전체 실패")
            return result
        except Exception as e:
            print(f"    ⚠️ 방식 A 실패: {e} → 방식 B(LLM)로 전환")

    # ── 방식 B: LLM-as-Judge ─────────────────────────────────────────
    print("    → Novelty 방식 B: LLM-as-Judge (sklearn/arXiv 불필요)")
    return _score_novelty_llm(gaps)



# ── 2-3. Specificity ─────────────────────────────────────────────────────────

def score_specificity(gaps: list) -> dict:
    """
    방법론 / 데이터셋 / 측정목표 키워드 3범주 충족률.
    """
    if not gaps:
        return {"score": 0.0, "avg_keywords_found": 0.0, "per_gap": []}

    methodology_kw = {"learning","network","model","transformer","GAN","diffusion",
                      "attention","BERT","GPT","ViT","LSTM","CNN","federated",
                      "contrastive","self-supervised","fine-tuning","prompt"}
    dataset_kw     = {"dataset","benchmark","corpus","TCGA","MIMIC","ImageNet","COCO",
                      "GloVe","GDSC","CCLE","FaceForensics","DFD","sample",
                      "annotation","labeled","unlabeled","database","collection"}
    metric_kw      = {"accuracy","AUROC","F1","AUC","precision","recall","BLEU",
                      "ROUGE","mAP","IoU","FID","perplexity","latency","throughput",
                      "performance","improvement","superiority","score","rate"}

    per_gap = []
    for g in gaps:
        text = f"{g.get('proposed_topic','')} {g.get('elaboration','')} {g.get('gap_statement','')}".lower()
        has_method  = any(k.lower() in text for k in methodology_kw)
        has_dataset = any(k.lower() in text for k in dataset_kw)
        has_metric  = any(k.lower() in text for k in metric_kw)
        found = sum([has_method, has_dataset, has_metric])
        per_gap.append({
            "topic":       g.get("proposed_topic","")[:60],
            "has_method":  has_method,
            "has_dataset": has_dataset,
            "has_metric":  has_metric,
            "found":       found,
            "spec_score":  round(found / 3, 4),
        })

    avg_found = float(np.mean([p["found"] for p in per_gap]))
    score     = float(np.mean([p["spec_score"] for p in per_gap]))
    return {
        "score":              round(score, 4),
        "avg_keywords_found": round(avg_found, 4),
        "per_gap":            per_gap,
    }


# ── 2-4. Relevance ───────────────────────────────────────────────────────────

def score_relevance(gaps: list, original_query: str, refined_query: str = "") -> dict:
    """원래 질문 ↔ proposed_topic 의미적 유사도."""
    if not gaps:
        return {"score": 0.0, "per_gap": []}
    queries = [q for q in [original_query, refined_query] if q]
    topics  = [g.get("proposed_topic","") or g.get("gap_statement","") for g in gaps]
    sims    = _embed_query_topics(queries, topics)
    per_gap = [{"topic": t[:70], "cosine_sim": round(float(s), 4)} for t, s in zip(topics, sims)]
    return {"score": round(float(np.mean(sims)), 4), "per_gap": per_gap}


# ── 2-5. Diversity ───────────────────────────────────────────────────────────

def score_diversity(gaps: list) -> dict:
    """제안 주제들 간 pairwise cosine + axis entropy."""
    if len(gaps) < 2:
        return {"score": 1.0, "avg_pairwise_sim": 0.0, "axis_entropy": 0.0, "axis_distribution": {}}

    import math
    topics     = [g.get("proposed_topic","") or g.get("gap_statement","") for g in gaps]
    sim_matrix = _cosine_matrix(topics)
    n          = len(topics)
    pairs      = [sim_matrix[i][j] for i in range(n) for j in range(i+1, n)]
    avg_sim    = float(np.mean(pairs))

    axes        = [g.get("axis","unknown") for g in gaps]
    axis_counts = Counter(axes)
    total       = sum(axis_counts.values())
    probs       = [c / total for c in axis_counts.values()]
    entropy     = -sum(p * math.log2(p + 1e-9) for p in probs)
    max_entropy = math.log2(len(gaps))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    score = 0.7 * (1.0 - avg_sim) + 0.3 * norm_entropy
    return {
        "score":             round(float(score), 4),
        "avg_pairwise_sim":  round(avg_sim, 4),
        "axis_entropy":      round(norm_entropy, 4),
        "axis_distribution": dict(axis_counts),
    }


# ════════════════════════════════════════════════════════════════════════
# 3. 종합 평가 실행
# ════════════════════════════════════════════════════════════════════════

def evaluate_result(result: dict, skip_novelty: bool = False) -> dict:
    gaps          = result.get("gaps", [])
    query         = result.get("query", "")
    refined_query = result.get("refined_query", "")
    source        = result.get("source", "gapago")

    print(f"\n  📊 평가 시작 — source={source}, gaps={len(gaps)}개")

    scores = {}

    print("    [1/5] Groundedness...")
    scores["groundedness"] = score_groundedness(gaps)

    if skip_novelty:
        print("    [2/5] Novelty — SKIP (--skip-novelty 플래그)")
        scores["novelty"] = {"score": None, "skipped": True}
    else:
        print("    [2/5] Novelty...")
        scores["novelty"] = score_novelty(gaps)

    print("    [3/5] Specificity...")
    scores["specificity"] = score_specificity(gaps)

    print("    [4/5] Relevance...")
    scores["relevance"] = score_relevance(gaps, query, refined_query)

    print("    [5/5] Diversity...")
    scores["diversity"] = score_diversity(gaps)

    # 가중 합산 (Novelty 스킵 시 나머지에 재분배)
    if skip_novelty or scores["novelty"].get("score") is None:
        weights = {k: v for k, v in METRIC_WEIGHTS.items() if k != "novelty"}
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}
    else:
        weights = METRIC_WEIGHTS

    total_score = sum(
        weights[m] * scores[m]["score"]
        for m in weights
        if scores[m].get("score") is not None
    )

    return {
        "source":       source,
        "query":        query,
        "gaps_count":   len(gaps),
        "scores":       scores,
        "total_score":  round(total_score, 4),
        "weights_used": weights,
    }


# ════════════════════════════════════════════════════════════════════════
# 4. 리포트 생성
# ════════════════════════════════════════════════════════════════════════

def build_markdown_report(gapago_eval: dict, baseline_evals: list, gapago_gaps_raw: list = None, baseline_gaps_raw: list = None) -> str:
    """
    GAPAGO vs 1개 이상의 Baseline 비교 Markdown 리포트.
    baseline_evals: 리스트 (--compare-all 시 여러 개)
    gapago_gaps_raw: GAPAGO 원본 gaps 리스트 (실제 답변 내용 표시용)
    baseline_gaps_raw: Baseline 원본 gaps 리스트들 (시스템 수만큼)
    """
    if gapago_gaps_raw is None:
        gapago_gaps_raw = []
    if baseline_gaps_raw is None:
        baseline_gaps_raw = [[] for _ in baseline_evals]
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = gapago_eval["query"]
    bl_names = [e["source"].upper() for e in baseline_evals]
    title_vs = " vs ".join(["GAPAGO"] + bl_names)

    lines = [
        f"# {title_vs} — 연구 GAP 분석 품질 비교 리포트",
        f"",
        f"> **쿼리**: {query}",
        f"> **평가 시각**: {now}",
        f"",
        f"---",
        f"",
        f"## 1. 종합 점수",
        f"",
    ]

    # 종합 점수 테이블
    header_cols = "| 시스템 | 총점 (0~1) | GAP 개수 |"
    sep_cols    = "|--------|------------|---------|"
    lines += [header_cols, sep_cols]
    lines.append(f"| **GAPAGO** | {gapago_eval['total_score']:.4f} | {gapago_eval['gaps_count']} |")
    for be in baseline_evals:
        lines.append(f"| **{be['source'].upper()}** | {be['total_score']:.4f} | {be['gaps_count']} |")
    lines += ["", "---", "", "## 2. 항목별 점수 비교", ""]

    # 항목별 점수 테이블
    bl_header = " | ".join(f"{n:>8}" for n in bl_names)
    lines.append(f"| 항목 | 가중치 | GAPAGO | {bl_header} | 우위 |")
    lines.append(f"|------|--------|--------|" + "-|" * len(bl_names) + "------|")

    metrics_info = {
        "groundedness": ("Groundedness (근거성)", "★★★"),
        "novelty":      ("Novelty (참신성)",      "★★★"),
        "specificity":  ("Specificity (구체성)",  "★★"),
        "relevance":    ("Relevance (연관성)",    "★"),
        "diversity":    ("Diversity (다양성)",    "★★"),
    }

    for metric, (label, stars) in metrics_info.items():
        w  = gapago_eval["weights_used"].get(metric, METRIC_WEIGHTS.get(metric, 0))
        gs = gapago_eval["scores"][metric].get("score")

        if gs is None:
            bl_scores_str = " | ".join("skipped" for _ in baseline_evals)
            lines.append(f"| {label} ({stars}) | {w:.2f} | skipped | {bl_scores_str} | — |")
            continue

        bl_scores = [be["scores"][metric].get("score", 0.0) for be in baseline_evals]
        bl_scores_str = " | ".join(f"{s:.4f}" for s in bl_scores)

        # 우위 판정: GAPAGO vs 모든 baseline 중 최고점
        best_bl = max(bl_scores) if bl_scores else 0.0
        if gs > best_bl + 0.02:
            winner = "✅ GAPAGO"
        elif best_bl > gs + 0.02:
            best_name = bl_names[bl_scores.index(best_bl)]
            winner = f"✅ {best_name}"
        else:
            winner = "≈"

        lines.append(f"| {label} ({stars}) | {w:.2f} | {gs:.4f} | {bl_scores_str} | {winner} |")

    lines += ["", "---", "", "## 3. 항목별 세부 분석", ""]

    # ── 3-1 Groundedness ──────────────────────────────────────────────
    g_gr = gapago_eval["scores"]["groundedness"]
    lines += [
        "### 3-1. Groundedness (근거성)",
        "",
        f"> 판정 기준: repeat_count ≥ {REPEAT_COUNT_THRESHOLD}이면 '강한 근거', 미만이면 '약한 근거'",
        "",
        "**GAPAGO**",
        f"- 강한 근거 (복수 논문, repeat ≥ {REPEAT_COUNT_THRESHOLD}): {g_gr['strong_count']}개",
        f"- 약한 근거 (논문 1편, repeat < {REPEAT_COUNT_THRESHOLD}): {g_gr['weak_count']}개",
        f"- 논문 연결만 있음: {g_gr['paper_only_count']}개",
        f"- 근거 없음: {g_gr['no_evidence_count']}개",
        f"- 종합 점수: {g_gr['score']:.4f}",
        "",
    ]
    for be in baseline_evals:
        b_gr = be["scores"]["groundedness"]
        lines += [
            f"**{be['source'].upper()}**",
            f"- 강한 근거: {b_gr['strong_count']}개  /  약한 근거: {b_gr['weak_count']}개  /  근거 없음: {b_gr['no_evidence_count']}개",
            f"- 종합 점수: {b_gr['score']:.4f}",
            "",
        ]
    lines += [
        "> 💡 GAPAGO는 실제 논문 Limitation 섹션의 evidence_quote를 직접 추출합니다.",
        ">    Baseline LLM은 자체 추론만으로 GAP을 생성하므로 supporting_quotes = 0.",
        "",
    ]

    # ── 3-2 Novelty ───────────────────────────────────────────────────
    g_nv  = gapago_eval["scores"]["novelty"]
    lines += ["### 3-2. Novelty (참신성)", ""]
    if g_nv.get("skipped"):
        lines += ["> ⏭ 스킵됨 (--skip-novelty)", ""]
    else:
        method = g_nv.get("method", "unknown")
        lines += [f"> 측정 방식: {'LLM-as-Judge' if 'llm' in method else 'TF-IDF + arXiv'}", ""]
        lines.append(f"| 시스템 | Novelty 점수 |")
        lines.append(f"|--------|-------------|")
        lines.append(f"| GAPAGO | {g_nv['score']:.4f} |")
        for be in baseline_evals:
            b_nv = be["scores"]["novelty"]
            s = b_nv.get("score")
            s_str = f"{s:.4f}" if s is not None else "N/A"
            lines.append(f"| {be['source'].upper()} | {s_str} |")
        lines.append("")

        # per_gap 상세
        lines.append("**GAPAGO GAP별 Novelty:**")
        for p in g_nv.get("per_gap", []):
            reason = f"  ← {p.get('reason','')[:60]}" if p.get("reason") else ""
            lines.append(f"- {p['topic'][:65]}: **{p['novelty']:.3f}**{reason}")
        lines.append("")

    # ── 3-3 Specificity ───────────────────────────────────────────────
    g_sp = gapago_eval["scores"]["specificity"]
    lines += [
        "### 3-3. Specificity (구체성)",
        "",
        f"- **GAPAGO**  평균 {g_sp['avg_keywords_found']:.2f}/3 → {g_sp['score']:.4f}",
    ]
    for be in baseline_evals:
        b_sp = be["scores"]["specificity"]
        lines.append(f"- **{be['source'].upper()}** 평균 {b_sp['avg_keywords_found']:.2f}/3 → {b_sp['score']:.4f}")

    lines += ["", "| 시스템 | 주제 (일부) | 방법론 | 데이터 | 지표 |",
              "|--------|-----------|--------|--------|------|"]
    for p in g_sp["per_gap"]:
        m = "✅" if p["has_method"] else "❌"
        d = "✅" if p["has_dataset"] else "❌"
        me = "✅" if p["has_metric"] else "❌"
        lines.append(f"| GAPAGO | {p['topic']} | {m} | {d} | {me} |")
    for be in baseline_evals:
        for p in be["scores"]["specificity"]["per_gap"]:
            m = "✅" if p["has_method"] else "❌"
            d = "✅" if p["has_dataset"] else "❌"
            me = "✅" if p["has_metric"] else "❌"
            lines.append(f"| {be['source'].upper()} | {p['topic']} | {m} | {d} | {me} |")
    lines.append("")

    # ── 3-4 Relevance ─────────────────────────────────────────────────
    g_rl = gapago_eval["scores"]["relevance"]
    lines += ["### 3-4. Relevance (연관성)", ""]
    lines.append(f"- **GAPAGO**  cosine similarity: {g_rl['score']:.4f}")
    for be in baseline_evals:
        lines.append(f"- **{be['source'].upper()}** cosine similarity: {be['scores']['relevance']['score']:.4f}")
    lines += ["", "> 참고: Relevance는 학습 데이터 전체를 활용하는 LLM이 유리할 수 있습니다.", ""]

    # ── 3-5 Diversity ─────────────────────────────────────────────────
    g_dv = gapago_eval["scores"]["diversity"]
    lines += ["### 3-5. Diversity (다양성)", ""]
    lines += [
        f"- **GAPAGO**  pairwise sim={g_dv['avg_pairwise_sim']:.4f} / axis entropy={g_dv['axis_entropy']:.4f}",
        f"  axis 분포: {g_dv['axis_distribution']}",
    ]
    for be in baseline_evals:
        b_dv = be["scores"]["diversity"]
        lines += [
            f"- **{be['source'].upper()}** pairwise sim={b_dv['avg_pairwise_sim']:.4f} / axis entropy={b_dv['axis_entropy']:.4f}",
            f"  axis 분포: {b_dv['axis_distribution']}",
        ]

    # ── 4. 실제 답변 내용 비교 ────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 4. 실제 답변 내용 비교",
        "",
        "> 동일한 쿼리에 대해 GAPAGO와 각 Baseline LLM이 제안한 연구 주제를 나란히 표시합니다.",
        "",
    ]

    # GAPAGO 답변
    lines += [
        "### GAPAGO 제안 연구 주제",
        "",
        "| # | 축(Axis) | 제안 연구 주제 | GAP statement | 근거 논문 수 | repeat |",
        "|---|----------|--------------|--------------|------------|--------|",
    ]
    for i, gd in enumerate(gapago_gaps_raw, 1):
        axis     = gd.get("axis_label") or gd.get("axis", "")
        topic    = gd.get("proposed_topic", "")[:70]
        gap_stmt = gd.get("gap_statement", "")[:60]
        n_papers = len(gd.get("supporting_papers", []))
        repeat   = gd.get("repeat_count", 0)
        lines.append(f"| {i} | {axis} | {topic} | {gap_stmt} | {n_papers} | {repeat} |")
    lines.append("")

    lines += ["**GAPAGO GAP 상세:**", ""]
    for i, gd in enumerate(gapago_gaps_raw, 1):
        axis   = gd.get("axis_label") or gd.get("axis", "")
        topic  = gd.get("proposed_topic", "")
        elab   = gd.get("elaboration", "")
        quotes = gd.get("supporting_quotes", [])
        lines += [
            f"#### GAP {i}. [{axis}] {topic}",
            "",
            f"**Gap statement**: {gd.get('gap_statement', '')}",
            "",
            f"**Elaboration**: {elab}",
            "",
        ]
        if quotes:
            lines.append("**Evidence quotes (실제 논문 인용):**")
            for q in quotes[:3]:
                lines.append(f'> "{q}"')
            lines.append("")
        else:
            lines.append("*근거 인용 없음*\n")

    # Baseline 답변
    for be_idx, be in enumerate(baseline_evals):
        bl_name  = be["source"].upper()
        bl_gaps  = baseline_gaps_raw[be_idx] if be_idx < len(baseline_gaps_raw) else []
        lines += [
            f"### {bl_name} 제안 연구 주제",
            "",
            "| # | 축(Axis) | 제안 연구 주제 | GAP statement |",
            "|---|----------|--------------|--------------|",
        ]
        for i, gd in enumerate(bl_gaps, 1):
            axis     = gd.get("axis_label") or gd.get("axis", "")
            topic    = gd.get("proposed_topic", "")[:70]
            gap_stmt = gd.get("gap_statement", "")[:60]
            lines.append(f"| {i} | {axis} | {topic} | {gap_stmt} |")
        lines.append("")
        lines += [f"**{bl_name} GAP 상세:**", ""]
        for i, gd in enumerate(bl_gaps, 1):
            axis  = gd.get("axis_label") or gd.get("axis", "")
            topic = gd.get("proposed_topic", "")
            elab  = gd.get("elaboration", "")
            lines += [
                f"#### GAP {i}. [{axis}] {topic}",
                "",
                f"**Gap statement**: {gd.get('gap_statement', '')}",
                "",
                f"**Elaboration**: {elab}",
                "",
                "*⚠️ 근거 논문 인용 없음 (LLM 자체 생성)*",
                "",
            ]

    lines += [
        "",
        "---",
        "",
        "## 5. 평가 방법론",
        "",
        "| 항목 | 측정 방법 |",
        "|------|-----------|",
        "| Groundedness | quotes 보유 + repeat_count 기반 4단계 채점 (1.0/0.6/0.3/0.0) |",
        "| Novelty      | TF-IDF+arXiv (방식A) 또는 LLM-as-Judge (방식B, sklearn 없을 때) |",
        "| Specificity  | 방법론·데이터셋·측정목표 키워드 3범주 충족률 |",
        "| Relevance    | TF-IDF 또는 SentenceTransformer cosine similarity |",
        "| Diversity    | 1 - pairwise cosine (70%) + axis entropy (30%) |",
        "",
        "---",
        "*Generated by evaluate.py — GAPAGO Evaluation Framework*",
    ]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# 5. CLI
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="GAPAGO vs Baseline LLM 비교 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--result-file", required=True,
        help="main.py가 저장한 GAPAGO 결과 JSON 경로",
    )
    parser.add_argument(
        "--baseline-file", default=None,
        help="이미 생성된 baseline JSON 경로 (단일). 없으면 새로 생성.",
    )
    parser.add_argument(
        "--baseline-provider", default="azure",
        choices=["azure", "claude", "gemini", "exaone"],
        help="단일 Baseline provider (기본: azure). --compare-all 사용 시 무시됨.",
    )
    parser.add_argument(
        "--compare-all", action="store_true",
        help="azure / claude / gemini 세 provider를 동시에 비교.",
    )
    parser.add_argument(
        "--skip-novelty", action="store_true",
        help="Novelty 평가 생략 (빠른 실행).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── GAPAGO 결과 로드 ─────────────────────────────────────────────
    result_path = Path(args.result_file)
    if not result_path.exists():
        print(f"[ERROR] 파일 없음: {result_path}")
        sys.exit(1)

    print(f"\n📂 GAPAGO 결과 로드: {result_path}")
    gapago_result = json.loads(result_path.read_text(encoding="utf-8"))
    gapago_result.setdefault("source", "gapago")
    query = gapago_result.get("query", "")

    # ── Baseline 결과 수집 ────────────────────────────────────────────
    baseline_results = []

    if args.baseline_file:
        # 단일 파일 지정
        bl_path = Path(args.baseline_file)
        print(f"📂 Baseline 로드: {bl_path}")
        br = json.loads(bl_path.read_text(encoding="utf-8"))
        br.setdefault("source", args.baseline_provider)
        baseline_results.append(br)

    elif args.compare_all:
        # 여러 provider 동시 생성
        providers = ["azure", "claude", "gemini"]
        print(f"\n🔄 --compare-all: {providers} 순서로 Baseline 생성")
        for prov in providers:
            try:
                br = generate_baseline_response(query, provider=prov)
                baseline_results.append(br)
            except Exception as e:
                print(f"  ⚠️ {prov} baseline 생성 실패: {e} — 스킵")

    else:
        # 단일 provider
        br = generate_baseline_response(query, provider=args.baseline_provider)
        baseline_results.append(br)

    if not baseline_results:
        print("[ERROR] Baseline을 하나도 생성하지 못했습니다.")
        sys.exit(1)

    # ── 평가 실행 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" GAPAGO 평가")
    print("=" * 60)
    gapago_eval = evaluate_result(gapago_result, skip_novelty=args.skip_novelty)

    baseline_evals = []
    for br in baseline_results:
        print("\n" + "=" * 60)
        print(f" {br['source'].upper()} (Baseline) 평가")
        print("=" * 60)
        baseline_evals.append(evaluate_result(br, skip_novelty=args.skip_novelty))

    # ── 콘솔 요약 ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" 최종 비교 결과")
    print("=" * 60)

    metrics_order = ["groundedness", "novelty", "specificity", "relevance", "diversity"]
    bl_header_str = "  ".join(f"{be['source'].upper():>8}" for be in baseline_evals)
    print(f"  {'항목':<20} {'GAPAGO':>8}  {bl_header_str}  우위")
    print("  " + "-" * (50 + 10 * len(baseline_evals)))

    for m in metrics_order:
        gs = gapago_eval["scores"][m].get("score")
        if gs is None:
            print(f"  {m:<20} {'SKIP':>8}  " + "  ".join(f"{'SKIP':>8}" for _ in baseline_evals))
            continue
        bl_scores = [be["scores"][m].get("score", 0.0) for be in baseline_evals]
        bl_str    = "  ".join(f"{s:>8.4f}" for s in bl_scores)
        best_bl   = max(bl_scores) if bl_scores else 0.0
        if gs > best_bl + 0.02:
            winner = "← GAPAGO ✅"
        elif best_bl > gs + 0.02:
            best_idx  = bl_scores.index(best_bl)
            winner    = f"← {baseline_evals[best_idx]['source'].upper()} ✅"
        else:
            winner = "≈"
        print(f"  {m:<20} {gs:>8.4f}  {bl_str}  {winner}")

    print("  " + "-" * (50 + 10 * len(baseline_evals)))
    bl_totals = "  ".join(f"{be['total_score']:>8.4f}" for be in baseline_evals)
    print(f"  {'TOTAL':<20} {gapago_eval['total_score']:>8.4f}  {bl_totals}")

    # ── 저장 ─────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_report = {
        "timestamp": datetime.now().isoformat(),
        "query":     query,
        "gapago":    gapago_eval,
        "baselines": baseline_evals,
    }
    json_path = OUTPUT_DIR / f"eval_report_{ts}.json"
    json_path.write_text(json.dumps(json_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 원본 gaps 리스트 수집 (실제 답변 내용 표시용)
    gapago_gaps_raw   = gapago_result.get("gaps", [])
    bl_gaps_raw_list  = [br.get("gaps", []) for br in baseline_results]
    md_text  = build_markdown_report(
        gapago_eval, baseline_evals,
        gapago_gaps_raw=gapago_gaps_raw,
        baseline_gaps_raw=bl_gaps_raw_list,
    )
    md_path  = OUTPUT_DIR / f"eval_report_{ts}.md"
    md_path.write_text(md_text, encoding="utf-8")

    print(f"\n✅ 평가 완료!")
    print(f"   JSON → {json_path}")
    print(f"   MD   → {md_path}")


if __name__ == "__main__":
    main()