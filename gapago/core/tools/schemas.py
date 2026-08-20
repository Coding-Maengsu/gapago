"""LangChain 툴 입력 스키마."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ArxivApiCallInput(BaseModel):
    search_query: str = Field(description="arXiv API search_query")
    max_total: int = Field(default=100, description="총 최대 결과 수")
    page_size: int = Field(default=100, description="페이지당 결과 수")
    max_pages: int = Field(default=1, description="최대 페이지 수")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026'). submittedDate 범위로 변환")


class CrossrefSearchInput(BaseModel):
    query: str = Field(description="Crossref 검색 쿼리")
    rows: int = Field(default=60, description="최대 결과 수 (max 1000)")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026')")


class WebSearchInput(BaseModel):
    query: str = Field(description="웹 검색 쿼리")


class SemanticScholarSearchInput(BaseModel):
    query: str = Field(description="Semantic Scholar 검색 쿼리")
    limit: int = Field(default=50, description="최대 결과 수 (max 100)")
    year: str = Field(default="", description="연도 필터 (e.g. '2020-2025', '2023-')")


class OpenAlexSearchInput(BaseModel):
    query: str = Field(description="OpenAlex 검색 쿼리")
    per_page: int = Field(default=40, description="최대 결과 수 (max 200)")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026')")


class ScienceOnSearchInput(BaseModel):
    query: str = Field(description="ScienceON 검색 쿼리")
    target: str = Field(default="ARTI", description="ScienceON target")
    cur_page: int = Field(default=1, description="현재 페이지 번호")
    row_count: int = Field(default=15, description="가져올 결과 수")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026')")


class ScienceOnPatentSearchInput(BaseModel):
    query: str = Field(description="ScienceON 특허 검색 쿼리")
    cur_page: int = Field(default=1, description="현재 페이지 번호")
    row_count: int = Field(default=10, description="가져올 결과 수")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026')")


class ScienceOnReportSearchInput(BaseModel):
    query: str = Field(description="ScienceON 국가 R&D 보고서 검색 쿼리")
    cur_page: int = Field(default=1, description="현재 페이지 번호")
    row_count: int = Field(default=10, description="가져올 결과 수")
    year: str = Field(default="", description="연도 필터 (e.g. '2022-2026')")
