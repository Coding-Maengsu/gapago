"""에이전트에 넘길 LangChain 툴 조립."""
from __future__ import annotations

import json
from typing import Optional, List

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from core.config import Configuration
from utils.tavily import TavilySearch

from ._common import _norm, _safe_json_loads, _tokenize
from .arxiv import arxiv_api_call
from .crossref import crossref_search
from .openalex import openalex_search
from .ranking import bm25_rank
from .scienceon import scienceon_search, scienceon_patent_search, scienceon_report_search
from .semantic_scholar import semantic_scholar_search
from .schemas import (
    ArxivApiCallInput, CrossrefSearchInput, WebSearchInput,
    SemanticScholarSearchInput, OpenAlexSearchInput,
    ScienceOnSearchInput, ScienceOnPatentSearchInput, ScienceOnReportSearchInput,
)


def build_retrieval_tools(config: Optional[RunnableConfig] = None) -> List:
    """
    Retrieval Agent가 선택할 수 있는 외부 검색 툴만 노출한다.
    - web_search_tool
    - arxiv_api_call_tool
    - scienceon_search_tool (placeholder)
    """
    cfg = Configuration.from_runnable_config(config)
    tavily_tool = TavilySearch(max_results=cfg.tavily_max_results)

    @tool(args_schema=ArxivApiCallInput)
    def arxiv_api_call_tool(
        search_query: str,
        max_total: int = 100,
        page_size: int = 100,
        max_pages: int = 1,
        year: str = "",
    ) -> str:
        """Call arXiv API directly and return a paper list as JSON string. Use year param for date filtering (e.g. '2022-2026')."""
        try:
            # Embed year filter into arXiv query via submittedDate
            if year and "-" in year:
                parts = year.split("-")
                from_year = parts[0].strip()
                to_year = parts[1].strip() if parts[1].strip() else "2026"
                date_filter = f" AND submittedDate:[{from_year}01010000 TO {to_year}12312359]"
                search_query = search_query + date_filter
            results = arxiv_api_call(
                search_query=search_query,
                max_total=max_total,
                page_size=page_size,
                max_pages=max_pages,
            )
            return json.dumps({
                "source": "arxiv",
                "query": search_query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Arxiv API call failed: {str(e)}</Error>"

    @tool(args_schema=CrossrefSearchInput)
    def crossref_search_tool(query: str, rows: int = 60, year: str = "") -> str:
        """Search Crossref for academic papers. Covers 150M+ scholarly works with DOI, abstracts, and citation metadata. Very reliable with no rate limit issues. Use year param for filtering (e.g. '2022-2026')."""
        try:
            results = crossref_search(query=query, rows=rows, year=year)
            return json.dumps({
                "source": "crossref",
                "query": query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Crossref search failed: {str(e)}</Error>"

    @tool(args_schema=WebSearchInput)
    def web_search_tool(query: str) -> str:
        """Search the web API and return results as JSON string."""
        try:
            results = tavily_tool.search(query)
            return json.dumps({
                "source": "web",
                "query": query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Web search failed: {str(e)}</Error>"

    @tool(args_schema=SemanticScholarSearchInput)
    def semantic_scholar_search_tool(query: str, limit: int = 50, year: str = "") -> str:
        """Search Semantic Scholar for academic papers. Returns papers with metadata. Good for finding highly-cited and cross-domain papers."""
        try:
            results = semantic_scholar_search(query=query, limit=limit, year=year)
            return json.dumps({
                "source": "semantic_scholar",
                "query": query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Semantic Scholar search failed: {str(e)}</Error>"

    @tool(args_schema=OpenAlexSearchInput)
    def openalex_search_tool(query: str, per_page: int = 40, year: str = "") -> str:
        """Search OpenAlex for academic papers. Covers 200M+ works across all disciplines. No API key needed. Use year param for filtering (e.g. '2022-2026')."""
        try:
            results = openalex_search(query=query, per_page=per_page, year=year)
            return json.dumps({
                "source": "openalex",
                "query": query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>OpenAlex search failed: {str(e)}</Error>"

    @tool(args_schema=ScienceOnSearchInput)
    def scienceon_search_tool(query: str, target: str = "ARTI", cur_page: int = 1, row_count: int = 15, year: str = "") -> str:
        """Search ScienceON paper records. Use year param for filtering (e.g. '2022-2026')."""
        if not cfg.scienceon_client_id:
            return "<Error>ScienceON client_id is not configured. Set SCIENCEON_CLIENT_ID.</Error>"
        try:
            result = scienceon_search(
                client_id=cfg.scienceon_client_id,
                query=query,
                target=target or cfg.scienceon_default_target,
                cur_page=cur_page,
                row_count=row_count or cfg.scienceon_default_row_count,
                mac_address=cfg.scienceon_mac_address,
                key=cfg.scienceon_key,
                year=year,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"<Error>ScienceON search failed: {str(e)}</Error>"

    @tool(args_schema=ScienceOnPatentSearchInput)
    def scienceon_patent_search_tool(query: str, cur_page: int = 1, row_count: int = 10, year: str = "") -> str:
        """Search ScienceON for Korean patents. Use year param for filtering (e.g. '2022-2026')."""
        if not cfg.scienceon_client_id:
            return "<Error>ScienceON client_id is not configured. Set SCIENCEON_CLIENT_ID.</Error>"
        try:
            result = scienceon_patent_search(
                client_id=cfg.scienceon_client_id,
                query=query,
                cur_page=cur_page,
                row_count=row_count or cfg.scienceon_default_row_count,
                mac_address=cfg.scienceon_mac_address,
                key=cfg.scienceon_key,
                year=year,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"<Error>ScienceON patent search failed: {str(e)}</Error>"

    @tool(args_schema=ScienceOnReportSearchInput)
    def scienceon_report_search_tool(query: str, cur_page: int = 1, row_count: int = 10, year: str = "") -> str:
        """Search ScienceON for Korean national R&D reports. Use year param for filtering (e.g. '2022-2026')."""
        if not cfg.scienceon_client_id:
            return "<Error>ScienceON client_id is not configured. Set SCIENCEON_CLIENT_ID.</Error>"
        try:
            result = scienceon_report_search(
                client_id=cfg.scienceon_client_id,
                query=query,
                cur_page=cur_page,
                row_count=row_count or cfg.scienceon_default_row_count,
                mac_address=cfg.scienceon_mac_address,
                key=cfg.scienceon_key,
                year=year,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"<Error>ScienceON report search failed: {str(e)}</Error>"

    return [
        web_search_tool,
        crossref_search_tool,
        arxiv_api_call_tool,
        semantic_scholar_search_tool,
        openalex_search_tool,
        scienceon_search_tool,
        scienceon_patent_search_tool,
        scienceon_report_search_tool,
    ]


def build_role_tools(config: Optional[RunnableConfig] = None) -> dict:
    retrieval_tools = build_retrieval_tools(config)
    return {
        "QUERY_TOOLS": [],
        "RETRIEVAL_TOOLS": retrieval_tools,
        "LIMITATION_TOOLS": [],
        "GAP_INFER_TOOLS": [],
        "CRITIC_TOOLS": [],
        "RESPONSE_TOOLS": [],
    }
