# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GAPAGO is a multi-agent Research GAP Analysis system built with LangGraph and LangChain. It takes a user's research question, retrieves academic papers from multiple sources, extracts limitations, infers research gaps, and produces a structured report.

## Running

```bash
python main.py
```

Requires a `.env` file in the project root with API keys (see Environment section below).

## Environment Variables

Required in `.env`:
- `TAVILY_API_KEY` — for web search
- Azure OpenAI keys (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`)
- `LANGCHAIN_API_KEY` or `LANGSMITH_API_KEY` — for LangSmith tracing

Optional:
- `LLM_PROVIDER` (default: `azure`) — supports `azure`, `claude`/`anthropic`, `exaone`
- `LLM_MODEL` — model/deployment name
- `GROQ_API_KEY`, `GROQ_MODEL` — for Groq (Qwen3-32B), used by ModelRouter routing
- `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `BEDROCK_CLAUDE_MODEL` — for Claude (Bedrock)
- `SCIENCEON_CLIENT_ID`, `SCIENCEON_MAC_ADDRESS`, `SCIENCEON_KEY` — for ScienceON paper search
- `RERANK_MODELS` (`auto`/`light`/`full`) — embedding/CrossEncoder model tier
- `TAVILY_MAX_RESULTS`, `ARXIV_MAX_RESULTS`, `BM25_TOP_K`, `RERANKER_TOP_K`
- `GAPAGO_ORCHESTRATOR` — set to `1` to enable LLM-based dynamic orchestrator pipeline

## Architecture

### Pipeline Flow (LangGraph)

The system is a LangGraph `StateGraph` with a shared `AgentState` (defined in `states.py`). The main graph is built in `graphs/graph.py`:

```
START → query_subgraph → meaning_expand → paper_retrieval → limitation_extract → limitation_eval → [PASS/RETRY] → recency_check → gap_infer → critic_score → [conditional] → final_response → END
```

**Critic routing** (`route_after_critic`): After critic evaluation, the graph can loop back:
- `ACCEPT` → `final_response`
- `REDO_RETRIEVAL` → `meaning_expand` (re-retrieval loop)
- `REFINE_QUERY` → `query_subgraph` (query refinement loop)
- Max 2 critic loops before forced ACCEPT

**Orchestrator mode** (`GAPAGO_ORCHESTRATOR=1`): Alternative pipeline in `graphs/orchestrator_graph.py`. LLM-based dynamic routing where an orchestrator agent decides the next node based on current state. Optional agents (limitation_eval, recency_check, critic_score) can be inserted or skipped dynamically.

### Query Subgraph (Human-in-the-Loop)

`graphs/query_subgraph.py` builds a nested subgraph with `interrupt_before=["human_clarify"]`:

```
START → query_analysis → [ambiguous?] → human_clarify → query_analysis (loop) | → END
```

The query analysis node scores user questions on 5 weighted criteria (domain, task, methodology, data, temporal clarity). If ambiguous and under `max_iterations`, it interrupts for human clarification via `app.update_state()`.

### LLM Routing (ModelRouter)

All agents use `get_llm_for_agent(state, agent_name)` (in `llm.py`) to obtain their LLM instance. This delegates to `ModelRouter` (`model_router.py`) when `model_routing` exists in state, otherwise falls back to `llm_provider`.

**Profiles** (set via `routing_profile` API parameter):
- `optimized` (default): most agents use default provider (azure), orchestrator → Groq, core extraction/report → Claude, reasoning → Groq. Light tasks (query_analysis, meaning_expand, critic_score etc.) stay on azure for accuracy.
- `quality`: core tasks all Claude + reasoning Groq

**Key pattern** — every agent does:
```python
llm = get_llm_for_agent(state, "agent_name")
```

### Agent Nodes

All agent nodes are in `agents/` and exported via `agents/__init__.py`. Each node function takes `AgentState` and returns a partial state update. Key patterns:

- **query_analysis** (`agents/query_agent/query_analysis.py`): Uses `llm.with_structured_output(QueryAnalysis)` for structured scoring. SemRank-based scope classification (TOO_BROAD / SEARCHABLE / TOO_NARROW).
- **meaning_expand** (`agents/meaning_expand_agent.py`): Expands keywords into search candidates (arxiv, web, scienceon queries). No tool calls — preparation only.
- **paper_retrieval** (`agents/retrieval_agent.py`): Direct parallel search (`_parallel_search` + `ThreadPoolExecutor`) across 8 sources. 6-stage ranking: parallel search → dedup → year filter → BM25+FAISS union → fulltext confidence filter → CrossEncoder reranking (fallback: LLM reranker). Async node with `run_in_executor`.
- **limitation_extract** (`agents/limitation_agent.py`): Iterates over papers, loads full text via 8-stage fallback chain, uses 2-track extraction (author-stated + structural). **Full-text only** — no abstract fallback; papers without valid full-text sections are skipped. Includes garbage data detection (`_MIN_FULLTEXT_CHARS=500`) with automatic backup paper replacement. Cross-validation (`_verify_limitations()`) runs on optimized/quality profiles.
- **limitation_eval** (`agents/limitation_eval_agent.py`): Dual-call evaluation — Call 1: FActScore atomic fact verification + Prometheus rubric scoring; Call 2: LimAgents set-level quality judgment + Xu et al. type classification. PASS/RETRY routing.
- **gap_infer** (`agents/gap_agent.py`): 4-step process — fully dynamic axes (3-7, LLM-generated inductively from limitations, no predefined categories) → batch classification + recency weighting → per-axis barrier analysis → creative direction generation. Uses `_llm_invoke(messages, use_reasoning, state)` for routing between `gap_reasoning` (Groq) and `gap_classify` agents.
- **critic_score** (`agents/critic_agent.py`): Scores on 3 dimensions (query_specificity, paper_relevance, groundedness), outputs a `DECISION:` tag that drives routing.
- **final_response** (`agents/response_agent.py`): Creates LLM at runtime via `get_llm_for_agent(state, "response")` — no module-level LLM. Writes final markdown report ending with `FINAL ANSWER`.
- **orchestrator** (`agents/orchestrator_agent.py`): LLM-based dynamic pipeline orchestrator. Decides next agent based on `completed_stages` and current state. Max 15 steps, max 2 re-runs per agent. Fast mode hint injected into prompt.
- **gap_chat** (`agents/gap_chat_agent.py`): Post-pipeline interactive chat agent. LLM-based intent classification + context-aware Q&A over GAP results (papers, limitations, gaps detail in system prompt). Called via `POST /api/chat` endpoint (web) and `main.py` (CLI). Not part of the StateGraph.

### Key Modules

- `llm.py`: LLM factory via `get_llm(provider, model)`. Cached with `lru_cache`. Supports providers: azure, claude (Bedrock), gemini (VertexAI, kept but removed from user selection), exaone, groq (Qwen3-32B), qwq (local GPU). `get_llm_for_agent(state, agent_name)` helper routes via ModelRouter.
- `model_router.py`: `ModelRouter` class with `optimized`/`quality` profiles. Maps agent names to optimal providers. `PROFILE_DEFAULT_PROVIDERS` auto-assigns base provider per profile.
- `tools.py`: 8 search functions (`arxiv_api_call`, `crossref_search`, `semantic_scholar_search`, `openalex_search`, `scienceon_search`, `scienceon_patent_search`, `scienceon_report_search`) and BM25 ranking. Called directly by `_parallel_search()`, not via LangChain tool wrappers.
- `config.py`: `Configuration` dataclass loaded from env vars, also supports LangGraph `RunnableConfig` overrides. Initializes LangSmith tracing on import.
- `states.py`: All Pydantic data models (schemas) and the `AgentState` TypedDict. Key state fields include `model_routing` (dict), `fast_mode` (bool), `completed_stages` (list, orchestrator), `agent_feedback` (dict), `paper_extraction_status`, `session_id`.
- `prompts/system.py`: Base system prompt shared across agents via `make_system_prompt(suffix)`.
- `utils/session_store.py`: SQLite-based session persistence for server restart recovery.
- `utils/cancel.py`: Pipeline cancellation registry (per-session cancel signals).

### API (`api/main.py`)

FastAPI server with SSE streaming. Key endpoints:
- `GET /api/analyze?query=...&routing_profile=optimized&fast_mode=false` — starts analysis, returns session_id. Creates `ModelRouter` and injects `model_routing` into pipeline state.
- `GET /api/explore?topic=...&routing_profile=...&fast_mode=...` — chain re-execution on proposed topic.
- `GET /api/stream/{session_id}` — SSE stream with reconnection support (`from_idx`).
- `GET /api/clarify?session_id=...&response=...` — resume after human clarification. Returns 410 if session lost to server restart.
- `POST /api/chat` — Q&A on analysis results.
- `GET /` — landing page (React SPA, built during Render deploy via `npm run build`), `GET /app` — main app (`frontend/index.html`).

### Data Sources

Papers are retrieved from six academic sources + web search, each returning normalized dicts with `paper_id`, `title`, `abstract`, `url`, `year`, `authors`, `venue`, `source`:
- **arXiv** — direct API calls with XML (Atom) parsing, `threading.Lock` + 5초 간격
- **Crossref** — 1.5억+ 메타데이터, PDF URL 추출, venue 포함
- **Semantic Scholar** — Graph API (200M+ papers, citation data)
- **OpenAlex** — public API (200M+ works, inverted index abstract reconstruction)
- **ScienceON** (KISTI) — Korean academic database with AES-encrypted token auth (articles, patents, R&D reports)
- **Tavily** — web search via `utils/tavily.py` wrapper (trends only, not papers)

### State Communication

Agents communicate through both `AgentState` fields (e.g., `papers`, `limitations`, `gaps`, `model_routing`) and message history (`messages` with named `AIMessage`s like `name="query_analysis"`, `name="meaning_expand"`). Downstream agents often parse upstream messages by name as a fallback.
