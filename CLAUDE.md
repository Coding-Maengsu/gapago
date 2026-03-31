# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GAPAGO** - A multi-agent research GAP analysis system built with LangGraph. It takes a user's research question, retrieves academic papers, extracts limitations, identifies research gaps, and produces a structured report. The project is Korean-authored; comments and UI strings are often in Korean.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main pipeline (interactive CLI)
python main.py
```

The pipeline prompts for a research question, runs through the agent graph, and may pause for human-in-the-loop clarification via `interrupt_before` on the query subgraph.

## Architecture

The system is a **LangGraph StateGraph** with a shared `AgentState` (defined in `states.py`). The main graph is built in `graphs/graph.py` and contains a nested subgraph for query analysis.

### Pipeline Flow

```
START -> query_subgraph -> meaning_expand -> paper_retrieval -> limitation_extract -> gap_infer -> critic_score -> [conditional]
```

**Conditional routing after critic_score** (`route_after_critic`):
- `ACCEPT` -> `final_response` -> END
- `REDO_RETRIEVAL` -> `meaning_expand` (re-enters retrieval loop)
- `REFINE_QUERY` -> `query_subgraph` (re-enters query analysis)
- `FINAL ANSWER` -> END
- Fallback -> `final_response` (loop prevention)

**Query Subgraph** (`graphs/query_subgraph.py`):
```
START -> query_analysis -> [ambiguous?] -> human_clarify (interrupt) -> query_analysis (loop)
                        -> [clear?] -> END
```
Uses `interrupt_before=["human_clarify"]` for human-in-the-loop. The main loop in `main.py` handles the interrupt by calling `app.update_state()` with user input, then resuming with `app.stream(None, config)`.

### Agent Nodes (all in `agents/`)

| Node | File | Role |
|------|------|------|
| `query_analysis_node` | `agents/query_agent/query_analysis.py` | Scores query ambiguity on 5 axes, decides if clarification needed |
| `human_clarify_node` | `agents/query_agent/query_analysis.py` | No-op interrupt node for human input |
| `query_refinement_node` | `agents/query_agent/query_refine.py` | Refines query for retrieval (currently commented out in subgraph) |
| `meaning_expand_node` | `agents/meaning_expand_agent.py` | Expands keywords/synonyms, prepares search candidates |
| `paper_retrieval_node` | `agents/retrieval_agent.py` | Uses tools (arXiv API, Tavily web search) to retrieve papers, BM25-ranks them |
| `limitation_extract_node` | `agents/limitation_agent.py` | Loads full text via ArxivLoader, extracts limitations per paper (2-track: author-stated + structural) |
| `gap_infer_node` | `agents/gap_agent.py` | Classifies limitations into 5 fixed + up to 2 dynamic axes, generates gap statements |
| `critic_score_node` | `agents/critic_agent.py` | Evaluates pipeline quality (query_specificity, paper_relevance, groundedness), max 2 retry loops |
| `final_response_node` | `agents/response_agent.py` | Generates final structured report |

### Key Modules

- **`states.py`**: All Pydantic data models (QueryAnalysis, Paper, LimitationItem, GapCandidate, CriticScores, etc.) and the shared `AgentState` TypedDict
- **`llm.py`**: LLM factory via `get_llm()`. Default provider is Azure OpenAI; supports Claude, Gemini, EXAONE (not implemented). Uses `@lru_cache`
- **`tools.py`**: External search tools built via `build_retrieval_tools()` — arXiv API (direct HTTP + XML parsing), Tavily web search, ScienceON (placeholder). Also contains `bm25_rank()` for BM25Okapi ranking
- **`config.py`**: Loads `.env`, initializes LangSmith tracing, defines `Configuration` dataclass for runtime settings
- **`prompts/system.py`**: Base system prompt + `make_system_prompt()` helper used by all agents
- **`utils/parse_json.py`**: Robust JSON parser with fallback regex extraction from LLM output
- **`utils/tavily.py`**: Custom `TavilySearch` LangChain tool wrapper
- **`utils/logging.py`**: LangSmith tracing setup

### State Flow

The `AgentState` accumulates data across nodes:
- Query phase: `refined_query`, `keywords`, `negative_keywords`, `is_ambiguous`, `iteration`
- Retrieval phase: `papers` (list of Paper dicts)
- Analysis phase: `limitations` (list of LimitationItem dicts), `gaps` (list of GapCandidate dicts)
- Evaluation phase: `critic` scores, `critic_loop_count`
- Messages are accumulated via LangGraph's `add_messages` reducer

## Environment Configuration

Configured via `.env` file (loaded by `python-dotenv`). Key variables:
- `LLM_PROVIDER` / `AZURE_OPENAI_*`: LLM provider settings
- `TAVILY_API_KEY`: Web search
- `LANGSMITH_*`: Tracing (project name: GAPAGO)
- `ARXIV_MAX_RESULTS`, `TOP_K_PAPERS`, `MAX_ITERATIONS`, `TAVILY_MAX_RESULTS`: Pipeline tuning

## Conventions

- Agent nodes return partial state dicts (not full state) — LangGraph merges them
- Each agent node attaches an `AIMessage` with a `name` field (e.g., `name="query_analysis"`) for downstream identification
- JSON output from LLM is parsed with fallback regex in `utils/parse_json.py` and `tools._safe_json_loads()`
- Retrieval agent uses `ThreadPoolExecutor` to call all 8 search tools in parallel (no LLM ReAct agent)
- The `recursion_limit` in config (default 30) prevents infinite loops in the graph
- Critic agent has `MAX_CRITIC_LOOPS = 2` as a hard safety bound

## Git Commit Rules

- **커밋 메시지에 `Co-Authored-By` 줄을 절대 추가하지 않는다**
