# 3-3) Paper Retrieval Agent (parallel search, no ReAct)
from __future__ import annotations

import json
import re
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from states import AgentState, Paper
from langchain_core.messages import AIMessage, HumanMessage
from tools import (
    arxiv_api_call, crossref_search, semantic_scholar_search,
    openalex_search, scienceon_search, scienceon_patent_search,
    scienceon_report_search, bm25_rank, _safe_json_loads, _tokenize, _norm,
)
from rank_bm25 import BM25Okapi
from llm import get_llm
from config import Configuration
from utils.parse_json import parse_json
from utils.tavily import TavilySearch
from utils.progress import report_progress

# ── Embedding + CrossEncoder models (lazy load) ─────────────────────────────
# 환경별 모델 선택: RERANK_MODELS 환경변수 또는 auto-detect
_EMBEDDING_MODEL_FULL  = "allenai/specter2_base"            # GPU/고사양 CPU
_EMBEDDING_MODEL_LIGHT = "all-MiniLM-L6-v2"                 # Render CPU
_CE_MODEL_FULL  = "BAAI/bge-reranker-v2-m3"                 # GPU/고사양 CPU
_CE_MODEL_LIGHT = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # Render CPU

_specter_model = None
_cross_encoder  = None
_loaded_embedding_tier = None  # 현재 로드된 모델 tier 추적
_loaded_ce_tier = None


def _get_device() -> str:
    """사용 가능한 디바이스 자동 감지 (CUDA > MPS > CPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            device  = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  [Device] GPU 감지: {gpu_name} ({vram_gb:.1f}GB VRAM) → cuda 사용")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("  [Device] Apple MPS 감지 → mps 사용")
        else:
            device = "cpu"
            print("  [Device] GPU 없음 → cpu 사용")
        return device
    except ImportError:
        print("  [Device] torch 미설치 → cpu 사용")
        return "cpu"


def _get_specter_model(model_tier: str = "light"):
    """임베딩 모델 lazy load — tier에 따라 모델 선택, ONNX Runtime 우선"""
    global _specter_model, _loaded_embedding_tier
    if _specter_model is not None and _loaded_embedding_tier == model_tier:
        return _specter_model
    # tier가 바뀌었으면 재로딩
    _specter_model = None
    try:
        from sentence_transformers import SentenceTransformer
        device = _get_device()
        model_name = _EMBEDDING_MODEL_LIGHT if model_tier == "light" else _EMBEDDING_MODEL_FULL

        # ONNX 백엔드 시도 (CPU에서 2~3배 빠름)
        if device == "cpu":
            try:
                _specter_model = SentenceTransformer(
                    model_name,
                    backend="onnx",
                    model_kwargs={"provider": "CPUExecutionProvider"},
                )
                print(f"  [Embedding] {model_name} ONNX Runtime 로딩 완료 (CPU 최적화)")
            except Exception as onnx_err:
                print(f"  [Embedding] ONNX 실패({onnx_err}), PyTorch fallback")
                _specter_model = SentenceTransformer(model_name, device=device)
                print(f"  [Embedding] {model_name} PyTorch 로딩 완료")
        else:
            _specter_model = SentenceTransformer(model_name, device=device)
            print(f"  [Embedding] {model_name} PyTorch 로딩 완료 (device={device})")
        _loaded_embedding_tier = model_tier
    except Exception as e:
        print(f"  [WARN] Embedding 모델 로딩 실패: {e} → FAISS 단계 스킵")
        _specter_model = None
    return _specter_model


def _get_cross_encoder(model_tier: str = "light"):
    """CrossEncoder 모델 lazy load — tier에 따라 모델 선택, ONNX Runtime 우선"""
    global _cross_encoder, _loaded_ce_tier
    if _cross_encoder is not None and _loaded_ce_tier == model_tier:
        return _cross_encoder
    # tier가 바뀌었으면 재로딩
    _cross_encoder = None
    try:
        from sentence_transformers import CrossEncoder
        device = _get_device()
        model_name = _CE_MODEL_LIGHT if model_tier == "light" else _CE_MODEL_FULL

        # ONNX 백엔드 시도 (CPU에서 2~3배 빠름)
        if device == "cpu":
            try:
                _cross_encoder = CrossEncoder(
                    model_name,
                    backend="onnx",
                    model_kwargs={"provider": "CPUExecutionProvider"},
                )
                print(f"  [CrossEncoder] {model_name} ONNX Runtime 로딩 완료 (CPU 최적화)")
            except Exception as onnx_err:
                print(f"  [CrossEncoder] ONNX 실패({onnx_err}), PyTorch fallback")
                _cross_encoder = CrossEncoder(model_name, device=device)
                print(f"  [CrossEncoder] {model_name} PyTorch 로딩 완료")
        else:
            _cross_encoder = CrossEncoder(model_name, device=device)
            print(f"  [CrossEncoder] {model_name} PyTorch 로딩 완료 (device={device})")
        _loaded_ce_tier = model_tier
    except Exception as e:
        print(f"  [WARN] CrossEncoder 로딩 실패: {e} → LLM Reranker fallback")
        _cross_encoder = None
    return _cross_encoder


def _extract_meaning_expand_data(state: AgentState) -> dict:
    """meaning_expand 메시지에서 쿼리 후보 데이터 추출."""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "name", None) == "meaning_expand":
            data = _safe_json_loads(getattr(m, "content", ""))
            if isinstance(data, dict):
                return data
    return {}


def _parallel_search(state: AgentState, resolved_year: str, cfg: Configuration) -> tuple[list, list]:
    """8개 검색 도구를 ThreadPoolExecutor로 병렬 호출."""

    me_data = _extract_meaning_expand_data(state)
    refined_query = state.get("refined_query") or state.get("user_question", "")
    arxiv_qs = me_data.get("arxiv_query_candidates", [refined_query])
    web_qs = me_data.get("web_query_candidates", [refined_query])
    scienceon_qs = me_data.get("scienceon_query_candidates", [refined_query])
    general_q = arxiv_qs[0] if arxiv_qs else refined_query

    # arXiv 쿼리에 연도 필터 삽입
    arxiv_query = arxiv_qs[0] if arxiv_qs else refined_query
    if resolved_year and "-" in resolved_year:
        parts = resolved_year.split("-")
        from_year = parts[0].strip()
        to_year = parts[1].strip() if parts[1].strip() else "2026"
        arxiv_query = arxiv_query + f" AND submittedDate:[{from_year}01010000 TO {to_year}12312359]"

    # ScienceON 인증 정보
    scienceon_kwargs = {
        "client_id": cfg.scienceon_client_id or "",
        "mac_address": cfg.scienceon_mac_address,
        "key": cfg.scienceon_key,
        "year": resolved_year,
    }

    # (name, callable, kwargs) 리스트
    tasks: list[tuple[str, callable, dict]] = [
        ("arxiv", arxiv_api_call, {
            "search_query": arxiv_query,
            "max_total": 100, "page_size": 100, "max_pages": 1,
        }),
        ("crossref", crossref_search, {
            "query": general_q, "rows": 60, "year": resolved_year,
        }),
        ("semantic_scholar", semantic_scholar_search, {
            "query": general_q, "limit": 50, "year": resolved_year,
        }),
        ("openalex", openalex_search, {
            "query": general_q, "per_page": 40, "year": resolved_year,
        }),
    ]

    # ScienceON (client_id 필요)
    if cfg.scienceon_client_id:
        scienceon_q = scienceon_qs[0] if scienceon_qs else refined_query
        tasks.append(("scienceon", scienceon_search, {
            "query": scienceon_q, "target": "ARTI", "row_count": 15,
            **scienceon_kwargs,
        }))
        tasks.append(("scienceon_patent", scienceon_patent_search, {
            "query": scienceon_q, "row_count": 10,
            **scienceon_kwargs,
        }))
        tasks.append(("scienceon_report", scienceon_report_search, {
            "query": scienceon_q, "row_count": 10,
            **scienceon_kwargs,
        }))

    # Tavily 웹 검색
    tavily_tool = TavilySearch(max_results=cfg.tavily_max_results)
    web_query = web_qs[0] if web_qs else refined_query
    tasks.append(("web", tavily_tool.search, {"query": web_query}))

    all_papers: list[dict] = []
    web_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {}
        for name, fn, kw in tasks:
            futures[executor.submit(fn, **kw)] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result(timeout=60)
            except Exception as e:
                print(f"  [parallel_search] {name} 실패: {e}")
                continue

            if name == "web":
                # Tavily는 list[dict] 반환
                if isinstance(result, list):
                    for r in result:
                        web_results.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "content": r.get("content") or r.get("snippet", ""),
                            "source": "web",
                        })
                print(f"  [parallel_search] {name}: {len(web_results)} results")
            elif name in ("scienceon_patent", "scienceon_report"):
                # dict with "results" list containing patent/report items
                if isinstance(result, dict):
                    for r in result.get("results", []):
                        paper_id = r.get("patent_id") or r.get("report_id") or ""
                        all_papers.append({
                            "paper_id": paper_id,
                            "title": r.get("title", ""),
                            "abstract": r.get("abstract", ""),
                            "url": r.get("url", ""),
                            "year": r.get("year", 0),
                            "authors": r.get("authors", []) if "authors" in r else [r.get("applicants", "")],
                            "score_bm25": 0.0,
                            "source": name,
                            "full_text_sections": r.get("raw", {}),
                        })
                print(f"  [parallel_search] {name}: {len(result.get('results', []) if isinstance(result, dict) else [])} results")
            elif name == "scienceon":
                # scienceon_search returns dict with "results"
                if isinstance(result, dict):
                    all_papers.extend(result.get("results", []))
                print(f"  [parallel_search] {name}: {len(result.get('results', []) if isinstance(result, dict) else [])} results")
            else:
                # arxiv, crossref, semantic_scholar, openalex return list[dict]
                if isinstance(result, list):
                    all_papers.extend(result)
                print(f"  [parallel_search] {name}: {len(result) if isinstance(result, list) else 0} results")

    return all_papers, web_results


def _dedupe_papers(raw_papers: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for p in raw_papers:
        key = (
            (p.get("doi") or "").lower().strip(),
            (p.get("title") or "").lower().strip(),
            int(p.get("year", 0) or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped



# =====================================================================
# Full text 접근 가능 여부 체크 (빠른 테스트)
# =====================================================================
_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
}


def _extract_arxiv_id(paper: dict) -> str:
    """paper_id에서 arXiv ID 추출."""
    pid = paper.get("paper_id", "")
    if pid.lower().startswith("arxiv:"):
        pid = pid[len("arxiv:"):]
    pid = pid.strip()
    if not pid or not re.match(r"^(\d{4}\.\d{4,5}|[\w\-]+\.?[\w\-]*/\d{7})", pid):
        return ""
    pid = re.sub(r"v\d+$", "", pid)
    return pid


def _find_pdf_url_from_doi(doi: str) -> str:
    """DOI 페이지에서 PDF URL 찾기 (다운로드 없음)."""
    doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    try:
        resp = requests.get(doi_url, headers=_REQUEST_HEADERS, timeout=10, allow_redirects=True)
        resp.raise_for_status()

        if resp.headers.get("content-type", "").startswith("application/pdf"):
            return resp.url

        pdf_patterns = [
            r'href=["\'][^"\']*\.pdf[^"\']*["\']',
            r'href=["\'][^"\']*\/pdf[^"\']*["\']',
        ]
        for pattern in pdf_patterns:
            matches = re.findall(pattern, resp.text, re.IGNORECASE)
            for url in matches:
                if not url.startswith("http"):
                    from urllib.parse import urljoin
                    url = urljoin(resp.url, url)
                return url
    except Exception:
        pass
    return ""


# full text 확보 신뢰도 등급
_FULLTEXT_CONFIDENCE = {
    "guaranteed": 3,   # arXiv (ar5iv HTML 거의 100%)
    "likely": 2,       # OA PDF URL 확보됨
    "maybe": 1,        # DOI만 있음 (출판사 차단 가능성)
    "none": 0,
}


def _fulltext_confidence(paper: dict) -> int:
    """논문의 full text 확보 신뢰도를 반환."""
    pid = paper.get("paper_id", "").lower()

    # arXiv: ar5iv HTML이 거의 항상 존재
    if pid.startswith("arxiv:"):
        if _extract_arxiv_id(paper):
            return _FULLTEXT_CONFIDENCE["guaranteed"]
        return _FULLTEXT_CONFIDENCE["none"]

    # ScienceON 특허/보고서: URL 존재 여부
    if pid.startswith(("scienceon_patent:", "scienceon_report:")):
        fts = paper.get("full_text_sections", {})
        for key in ("fulltext_url", "FulltextURL", "content_url", "ContentURL"):
            if fts.get(key):
                return _FULLTEXT_CONFIDENCE["likely"]
        return _FULLTEXT_CONFIDENCE["maybe"] if paper.get("url") else _FULLTEXT_CONFIDENCE["none"]

    # API가 OA PDF URL을 제공한 경우
    pdf_url = paper.get("pdf_url") or paper.get("full_text_sections", {}).get("pdf_url", "")
    if pdf_url:
        return _FULLTEXT_CONFIDENCE["likely"]

    # DOI 기반 소스: DOI가 있으면 가능성은 있지만 확실하지 않음
    if pid.startswith(("crossref:", "s2:", "openalex:", "scienceon:")):
        doi = paper.get("doi", "") or paper.get("full_text_sections", {}).get("doi", "")
        if not doi and paper.get("url") and "doi.org" in paper.get("url", ""):
            doi = paper.get("url")
        if doi:
            return _FULLTEXT_CONFIDENCE["maybe"]

    return _FULLTEXT_CONFIDENCE["none"]


def _can_load_full_text_fast(paper: dict) -> bool:
    """Full text 접근 가능 여부를 메타데이터만으로 빠르게 판별."""
    return _fulltext_confidence(paper) > _FULLTEXT_CONFIDENCE["none"]


def _filter_fulltext_available(papers: list[dict], target_count: int = 30) -> list[dict]:
    """
    논문 리스트에서 full text 접근 가능한 논문만 필터링.
    arXiv(guaranteed) → OA PDF(likely) → DOI(maybe) 순으로 우선 배치.
    각 신뢰도 그룹 내에서는 BM25 순서 유지.
    """
    if not papers:
        return []

    print(f"  [fulltext_filter] {len(papers)}편 중 full text 접근 가능 여부 확인 중...")

    # 각 논문의 신뢰도 계산
    scored = []
    for i, paper in enumerate(papers):
        conf = _fulltext_confidence(paper)
        if conf > 0:
            scored.append((conf, i, paper))

    # 신뢰도 내림차순, 동일 신뢰도 내에서는 원래 순서(BM25) 유지
    scored.sort(key=lambda x: (-x[0], x[1]))

    fulltext_available = [paper for _, _, paper in scored[:target_count]]

    # 소스별 통계 출력
    stats = {"guaranteed": 0, "likely": 0, "maybe": 0}
    for conf, _, _ in scored[:target_count]:
        if conf == 3:
            stats["guaranteed"] += 1
        elif conf == 2:
            stats["likely"] += 1
        else:
            stats["maybe"] += 1

    print(f"  [fulltext_filter] {len(fulltext_available)}/{len(papers)}편 선별 "
          f"(guaranteed={stats['guaranteed']}, likely={stats['likely']}, maybe={stats['maybe']})")
    return fulltext_available



def _faiss_filter(papers: list[dict], query: str, top_k: int, model_tier: str = "light") -> list[dict]:
    """
    임베딩 + FAISS 코사인 유사도 기반 2차 필터.
    BM25 통과 논문 중 의미적으로 가장 유사한 top_k 편을 선별.
    모델 로딩 실패 시 원본 리스트 그대로 반환.
    """
    if len(papers) <= top_k:
        return papers

    model = _get_specter_model(model_tier)
    if model is None:
        print(f"  [FAISS] 모델 없음 → 스킵 (BM25 결과 {len(papers)}편 유지)")
        return papers

    try:
        import faiss

        # 논문 텍스트: title + abstract 결합
        corpus_texts = [
            f"{p.get('title', '')} {p.get('abstract', '')}".strip()
            for p in papers
        ]

        # 쿼리 + 논문 임베딩 (배치 처리, GPU 자동 활용)
        all_texts  = [query] + corpus_texts
        embeddings = model.encode(
            all_texts,
            batch_size=64,           # GPU면 배치 크기 키워서 속도 향상
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_vec  = embeddings[0:1].astype("float32")   # (1, dim)
        paper_vecs = embeddings[1:].astype("float32")    # (N, dim)

        # FAISS CPU 인덱스 (코사인 유사도 = 정규화된 벡터의 내적)
        dim   = paper_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(paper_vecs)

        actual_k        = min(top_k, len(papers))
        scores, indices = index.search(query_vec, actual_k)  # (1, k)

        selected = [papers[i] for i in indices[0] if 0 <= i < len(papers)]
        print(f"  [FAISS] {len(papers)}편 → {len(selected)}편 선별 (SPECTER2 코사인 유사도)")
        return selected

    except ImportError:
        print("  [WARN] faiss 미설치 → FAISS 단계 스킵. pip install faiss-cpu")
        return papers
    except Exception as e:
        print(f"  [WARN] FAISS 필터 실패: {e} → BM25 결과 유지")
        return papers


def _cross_encoder_rerank(papers: list[dict], query: str, top_k: int, model_tier: str = "light") -> list[dict]:
    """
    CrossEncoder 기반 3차 reranking.
    로딩 실패 시 LLM Reranker로 fallback (None 반환).
    Returns top_k papers sorted by cross-encoder score (desc).
    """
    if len(papers) <= top_k:
        return papers

    reranker = _get_cross_encoder(model_tier)
    if reranker is None:
        return None  # fallback 신호: None 반환 시 LLM Reranker 사용

    try:
        pairs  = [(query, f"{p.get('title', '')} {p.get('abstract', '')}") for p in papers]
        scores = reranker.predict(pairs, show_progress_bar=False)

        # 점수 내림차순 정렬 후 top_k 선택
        ranked = sorted(zip(scores, papers), key=lambda x: x[0], reverse=True)
        selected = [p for _, p in ranked[:top_k]]
        print(f"  [CrossEncoder] {len(papers)}편 → {len(selected)}편 reranking 완료")
        return selected

    except Exception as e:
        print(f"  [WARN] CrossEncoder reranking 실패: {e} → LLM Reranker fallback")
        return None  # fallback 신호


RERANKER_PROMPT = """You are a research paper relevance assessor for GAP analysis.

Research Query: "{query}"

Below are {n} candidate papers. Select the {top_k} most relevant papers for identifying research gaps, limitations, and future directions.

Prioritize:
1. Papers directly addressing the research topic (methodology, dataset, application domain match)
2. Papers with substantive abstracts that likely contain discussable limitations
3. Diversity of perspectives (different approaches/subfields within the topic)
4. Recent papers (more likely to have current limitations)

Papers:
{papers_text}

Return a JSON object:
{{
  "selected_indices": [1, 3, 5, ...],
  "rationale": "Brief explanation of selection strategy"
}}

IMPORTANT: Return ONLY the JSON object. Select exactly {top_k} papers (or fewer if fewer are truly relevant)."""


def _llm_rerank(papers: list[dict], query: str, top_k: int, provider: str = None) -> list[dict]:
    """LLM Reranker: BM25 필터링된 논문에서 GAP 분석에 가장 적합한 논문 선별."""
    if len(papers) <= top_k:
        return papers

    papers_text = "\n".join(
        f"[{i+1}] Title: {p.get('title', '')}\n"
        f"    Year: {p.get('year', 'N/A')}\n"
        f"    Abstract: {(p.get('abstract', '') or '')[:300]}\n"
        for i, p in enumerate(papers)
    )

    prompt = RERANKER_PROMPT.format(
        query=query,
        n=len(papers),
        top_k=top_k,
        papers_text=papers_text,
    )

    try:
        llm = get_llm(provider=provider)
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        parsed = parse_json(content)

        if not isinstance(parsed, dict) or "selected_indices" not in parsed:
            print(f"  [WARN] LLM Reranker 파싱 실패 → BM25 top-{top_k} fallback")
            return papers[:top_k]

        indices = parsed["selected_indices"]
        # 유효한 인덱스만 필터 (1-indexed → 0-indexed)
        seen = set()
        selected = []
        for idx in indices:
            try:
                idx = int(idx)
            except (ValueError, TypeError):
                continue
            if idx < 1 or idx > len(papers) or idx in seen:
                continue
            seen.add(idx)
            selected.append(papers[idx - 1])
            if len(selected) >= top_k:
                break

        if not selected:
            print(f"  [WARN] LLM Reranker 유효 결과 없음 → BM25 top-{top_k} fallback")
            return papers[:top_k]

        rationale = parsed.get("rationale", "")
        print(f"  [reranker] {len(papers)}편 → {len(selected)}편 선별 ({rationale[:80]})")
        return selected

    except Exception as e:
        print(f"  [WARN] LLM Reranker 실패: {e} → BM25 top-{top_k} fallback")
        return papers[:top_k]


def _resolve_year_range(year_range: str) -> str:
    """Convert year_range setting to 'YYYY-YYYY' format for tools."""
    from datetime import datetime
    current_year = datetime.now().year
    if year_range == "1y":
        return f"{current_year - 1}-{current_year}"
    elif year_range == "3y":
        return f"{current_year - 3}-{current_year}"
    elif year_range == "5y":
        return f"{current_year - 5}-{current_year}"
    # "auto" or unknown → return empty (agent decides)
    return ""


def paper_retrieval_node(state: AgentState) -> AgentState:
    provider = state.get("llm_provider")
    session_id = state.get("session_id", "")

    # state에서 연도 필터 읽기
    year_range = state.get("year_range", "auto")
    resolved_year = _resolve_year_range(year_range)
    print(f"  [retrieval] year_range={year_range}, resolved_year={resolved_year}")

    cfg = Configuration()

    # 모델 tier 결정: auto → CPU이면 light, GPU이면 full
    model_tier = cfg.rerank_models
    if model_tier == "auto":
        model_tier = "light" if _get_device() == "cpu" else "full"
    print(f"  [retrieval] rerank_models tier={model_tier}")

    # ✅ 병렬 검색 (ReAct 에이전트 대신 ThreadPoolExecutor)
    raw_papers, web_results = _parallel_search(state, resolved_year, cfg)

    raw_papers = _dedupe_papers(raw_papers)
    total_candidates_count = len(raw_papers)  # BM25 전 전체 후보 수
    print(f"  [DEBUG] raw_papers count (after dedup): {total_candidates_count}")

    # ✅ 연도 필터 (resolved_year가 있으면 범위 밖 논문 제거)
    if resolved_year and "-" in resolved_year:
        parts = resolved_year.split("-")
        from_year = int(parts[0].strip())
        to_year = int(parts[1].strip()) if parts[1].strip() else 9999
        before_filter = len(raw_papers)
        raw_papers = [
            p for p in raw_papers
            if not p.get("year") or from_year <= int(p.get("year", 0) or 0) <= to_year
        ]
        filtered_count = before_filter - len(raw_papers)
        if filtered_count:
            print(f"  [year_filter] {before_filter} → {len(raw_papers)} ({filtered_count}편 범위 밖 제거, {from_year}-{to_year})")

    # ✅ 1단계: BM25 + SPECTER2/FAISS 병렬 필터 → union → 중복 제거
    total_found = len(raw_papers)
    query = state.get("refined_query") or state.get("user_question", "")

    # BM25와 FAISS 각각 bm25_top_k 편씩 선별 후 합집합
    # → CrossEncoder 입력 후보 풀: 최대 bm25_top_k * 2편 (중복 제거 후)
    stage1_papers = []
    if raw_papers and query:
        # BM25 dynamic_k 계산 (BM25 스레드에서 사용)
        corpus = [_tokenize(f"{p.get('title', '')} {p.get('abstract', '')}") for p in raw_papers]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))
        if len(scores) > 0:
            mean_score = float(np.mean(scores))
            std_score  = float(np.std(scores))
            threshold  = mean_score + 0.5 * std_score
            dynamic_k  = int(np.sum(scores > threshold))
            dynamic_k  = max(10, min(dynamic_k, cfg.bm25_top_k))
            print(f"  [BM25] dynamic_k={dynamic_k} (mean={mean_score:.2f}, std={std_score:.2f}, threshold={threshold:.2f})")
        else:
            dynamic_k = cfg.bm25_top_k

        # ── BM25 + FAISS 병렬 실행 ──────────────────────────────────────
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(bm25_rank, raw_papers, query, top_k=dynamic_k)
            faiss_future = executor.submit(_faiss_filter, raw_papers, query, top_k=cfg.bm25_top_k, model_tier=model_tier)
            bm25_pool = bm25_future.result().get("selected", raw_papers[:dynamic_k])
            faiss_pool = faiss_future.result()
        print(f"  [BM25] 1st pool: {len(bm25_pool)} papers")
        print(f"  [FAISS] 1st pool: {len(faiss_pool)} papers")

        # ── Union + 중복 제거 ────────────────────────────────────────────
        seen_ids = set()
        for p in bm25_pool + faiss_pool:
            pid = (p.get("paper_id") or p.get("title", "")).strip().lower()
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                stage1_papers.append(p)
        print(f"  [Stage1] BM25 ∪ FAISS = {len(stage1_papers)} papers (중복 제거 후)")
    else:
        stage1_papers = raw_papers

    # ✅ 2단계: Full text 접근 가능 여부 필터링 (arXiv 우선 정렬)
    stage1_papers = _filter_fulltext_available(stage1_papers, target_count=cfg.bm25_top_k)
    print(f"  [fulltext_filter] after filter: {len(stage1_papers)} papers")

    # ✅ 3단계: CrossEncoder reranking → 실패 시 LLM Reranker fallback
    # 선별되지 않은 논문은 backup으로 보관 (full text 실패 시 대체용)
    backup_raw = []
    if stage1_papers and query and len(stage1_papers) > cfg.reranker_top_k:
        ce_result = _cross_encoder_rerank(stage1_papers, query, top_k=cfg.reranker_top_k, model_tier=model_tier)
        if ce_result is not None:
            raw_papers = ce_result
            print(f"  [CrossEncoder] 2nd stage: {len(raw_papers)} papers")
        else:
            print("  [CrossEncoder] 실패 → LLM Reranker fallback")
            raw_papers = _llm_rerank(stage1_papers, query, top_k=cfg.reranker_top_k, provider=provider)
            print(f"  [LLM Reranker] fallback: {len(raw_papers)} papers")
        # backup: 선별되지 않은 논문 (arXiv 우선, BM25 높은 순)
        reranked_ids = {p.get("paper_id") for p in raw_papers}
        backup_raw = [p for p in stage1_papers if p.get("paper_id") not in reranked_ids]
        backup_raw = sorted(
            backup_raw,
            key=lambda p: (
                0 if p.get("paper_id", "").lower().startswith("arxiv:") else 1,
                -p.get("score_bm25", 0.0),
            ),
        )
    else:
        raw_papers = stage1_papers
    print(f"  [Reranker] final: {len(raw_papers)} papers, backup: {len(backup_raw)} papers")
    report_progress(
        session_id, "paper_retrieval",
        f"{total_found}편 중 가장 관련도 높은 {len(raw_papers)}편을 선별했습니다",
    )

    # ✅ Paper 객체로 변환
    papers = []
    for p in raw_papers:
        if not p.get("title") or not p.get("paper_id"):
            continue
        try:
            # ScienceON 등 외부 소스의 doi를 full_text_sections에 보존
            fts = dict(p.get("full_text_sections") or {})
            if p.get("doi") and "doi" not in fts:
                fts["doi"] = p["doi"]
            if p.get("pdf_url") and "pdf_url" not in fts:
                fts["pdf_url"] = p["pdf_url"]
            # venue 결정: 명시적 venue > journal > source 라벨
            venue = p.get("venue", "")
            if not venue:
                venue = p.get("journal", "")
            if not venue:
                source = p.get("source", "")
                venue_map = {
                    "arxiv": "arXiv preprint",
                    "semantic_scholar": "Semantic Scholar",
                    "openalex": "OpenAlex",
                    "scienceon": "ScienceON",
                    "scienceon_patent": "Patent",
                    "scienceon_report": "R&D Report",
                }
                venue = venue_map.get(source, source)
            papers.append(Paper(
                paper_id=p.get("paper_id", ""),
                title=p.get("title", ""),
                abstract=p.get("abstract", ""),
                url=p.get("url", ""),
                year=p.get("year", 0) or 0,
                authors=p.get("authors") or [],
                score_bm25=p.get("score_bm25", 0.0),
                venue=venue,
                full_text_sections=fts,
            ))
        except Exception as e:
            print(f"  ⚠️ Paper 변환 실패: {e}")
            continue

    # backup_papers를 dict 형태로 변환 (state에 저장)
    backup_papers = []
    for p in backup_raw:
        try:
            fts = dict(p.get("full_text_sections") or {})
            if p.get("doi") and "doi" not in fts:
                fts["doi"] = p["doi"]
            if p.get("pdf_url") and "pdf_url" not in fts:
                fts["pdf_url"] = p["pdf_url"]
            backup_papers.append({
                "paper_id": p.get("paper_id", ""),
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "url": p.get("url", ""),
                "year": p.get("year", 0) or 0,
                "authors": p.get("authors") or [],
                "score_bm25": p.get("score_bm25", 0.0),
                "venue": p.get("venue", ""),
                "full_text_sections": fts,
            })
        except Exception:
            continue
    arxiv_backup = sum(1 for bp in backup_papers if bp.get("paper_id", "").lower().startswith("arxiv:"))
    print(f"  ✓ Retrieved {len(papers)}/{total_candidates_count} papers + {len(web_results)} web results")
    print(f"  ✓ Backup pool: {len(backup_papers)} papers ({arxiv_backup} arXiv)")

    last_content = json.dumps({
        "total_candidates": total_candidates_count,
        "selected": len(papers),
        "web_results": len(web_results),
        "backup": len(backup_papers),
    }, ensure_ascii=False)
    last = AIMessage(content=last_content, name="paper_retrieval")
    return {
        "messages": [last],
        "sender": "paper_retrieval",
        "papers": papers,
        "backup_papers": backup_papers,
        "total_candidates_count": total_candidates_count,
        "web_results": web_results,
    }