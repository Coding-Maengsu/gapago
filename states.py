"""
GAPAGO State Definitions
This module defines the states used at each stage of the research process.
"""

from typing import List, Annotated, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages
from typing import Sequence
from typing_extensions import Annotated
import operator


# =====================================================================
# ======================== Data Models(Schema) ========================
# =====================================================================


# =====================================================================
# -1- QUERY AGENT
#
# 설계 원칙:
#   목적: 연구 방향성 → 논문 검색 가능한 쿼리 생성
#   GAP 분석과 무관. 검색 가능성만 판단.
#
# 판정 기준 (SemRank, Zhang et al. EMNLP 2025 기반):
#   general_topic   : 큰 연구 분야 (이것만 있으면 TOO_BROAD)
#   specific_phrases: 실제 검색에 쓸 구체적 키워드
#   → specific_phrases 없음  → TOO_BROAD
#   → specific_phrases 1개+  → SEARCHABLE
#   → 조합이 너무 희귀        → TOO_NARROW
#
# 상호작용 (CoQuest, Liu et al. CHI 2024 기반):
#   TOO_BROAD  → breadth-first: 하위 방향 후보 3개 동시 제시
#   SEARCHABLE → AI Thoughts: 판정 근거 + 키워드 함께 표시
#   TOO_NARROW → 확장 제안
# =====================================================================

class ScopeCandidate(BaseModel):
    """TOO_BROAD일 때 breadth-first로 제시하는 하위 방향 후보"""
    direction: str = Field(
        description="구체적인 하위 연구 방향 (예: '오디오-비주얼 멀티모달 딥페이크 탐지')"
    )
    rationale: str = Field(
        description="이 방향이 검색 가능한 이유 (AI Thoughts)"
    )
    sample_keywords: List[str] = Field(
        default_factory=list,
        description="이 방향으로 검색 시 사용할 키워드 예시"
    )


class ScopeAssessment(BaseModel):
    """검색 가능성 판정 결과 (SemRank 개념 기반)"""
    scope_level: Literal["TOO_BROAD", "SEARCHABLE", "TOO_NARROW"] = Field(
        description="검색 가능성 판정 결과"
    )
    general_topic: str = Field(
        description="연구가 속하는 큰 연구 분야"
    )
    specific_phrases: List[str] = Field(
        default_factory=list,
        description="검색에 직접 쓸 수 있는 구체적 키워드/구문"
    )
    rationale: str = Field(
        description="판정 근거 (AI Thoughts)"
    )
    breadth_candidates: List[ScopeCandidate] = Field(
        default_factory=list,
        description="TOO_BROAD일 때만 채움. 하위 방향 3개 (CoQuest breadth-first)"
    )
    expansion_suggestion: str = Field(
        default="",
        description="TOO_NARROW일 때만 채움. 확장 제안"
    )


class QueryResult(BaseModel):
    """LLM이 생성하는 최종 쿼리 분석 결과"""
    scope_assessment: ScopeAssessment = Field(
        description="검색 가능성 판정 결과"
    )
    refined_query: str = Field(
        default="",
        description="SEARCHABLE인 경우 검색용 자연어 쿼리. 아니면 빈 문자열."
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="SEARCHABLE인 경우 arXiv 검색 키워드 2~5개."
    )
    negative_keywords: List[str] = Field(
        default_factory=list,
        description="검색 제외 키워드. 필요한 경우만 1~3개."
    )


# =====================================================================
# -2- RETRIEVE AGENT
class Paper(BaseModel):
    """Individual paper metadata from arXiv."""

    paper_id: str
    title: str
    abstract: str
    url: str
    year: int
    authors: List[str] = Field(default_factory=list)
    score_bm25: float = 0.0
    full_text_sections: dict = Field(default_factory=dict)


# =====================================================================
# -3- LIMITATION AGENT
class LimitationItem(BaseModel):
    """Extracted limitation from a paper."""

    paper_id: str
    claim: str
    evidence_quote: str
    track: str = "author_stated"
    source_section: str = ""


# =====================================================================
# -4- GAP INFER AGENT
class GapCandidate(BaseModel):
    """Research gap identified from limitations."""

    axis: str
    axis_label: str = ""
    axis_type: str = "fixed"
    gap_statement: str
    elaboration: str = ""
    proposed_topic: str = ""
    repeat_count: int = 0
    supporting_papers: List[str] = Field(default_factory=list)
    supporting_quotes: List[str] = Field(default_factory=list)


# =====================================================================
# -5- CRITIC AGENT
class CriticScores(BaseModel):
    """Quality scores for the analysis."""

    query_specificity: float = Field(0.0, ge=0.0, le=1.0)
    paper_relevance: float = Field(0.0, ge=0.0, le=1.0)
    groundedness: float = Field(0.0, ge=0.0, le=1.0)


class DimensionScore(BaseModel):
    """개별 평가 차원 점수"""

    dimension: str
    label: str
    score: int = Field(0, ge=0, le=10)
    reasoning: str = ""


class EvaluationResult(BaseModel):
    """LLM-as-a-Judge 최종 평가 결과"""

    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    average_score: float = Field(0.0, ge=0.0, le=10.0)
    summary: str = ""


# =====================================================================
# ========================= State Definitions =========================
# =====================================================================

class AgentState(TypedDict):
    # ==================================================================
    # -0- ORCHESTRATION
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sender: Annotated[str, "The sender of the last message"]
    errors: List[str]

    # ==================================================================
    # -1- QUERY AGENT
    iteration: int
    max_iterations: int

    # 검색 가능성 판정 결과
    scope_level: str                    # TOO_BROAD / SEARCHABLE / TOO_NARROW
    scope_rationale: str                # 판정 근거 (AI Thoughts)
    breadth_candidates: List[dict]      # TOO_BROAD일 때 후보 3개
    expansion_suggestion: str           # TOO_NARROW일 때 확장 제안

    # 확정된 검색 정보 (SEARCHABLE 확정 후)
    keywords: List[str]
    negative_keywords: List[str]
    refined_query: str
    user_question: str

    # 진행 제어
    needs_user_input: bool              # True면 human_clarify로 분기

    # ==================================================================
    # -2- RETRIEVE AGENT
    papers: List[dict]
    web_results: List[dict]  # 웹 검색 결과 (논문 풀과 분리, recency_check에서 사용)
    research_domain: str     # 연구 도메인 (auto/ai_cs/biomedical/materials_chemistry/physics/general)
    llm_provider: str        # LLM 프로바이더 (azure/claude/gemini/exaone)
    year_range: str          # 연도 필터 (auto/1y/3y/5y)

    # ==================================================================
    # -3- LIMITATION AGENT
    limitations: List[dict]

    # ==================================================================
    # -3.5- LIMITATION EVAL AGENT
    limitation_eval: dict           # 평가 상세 결과 (call1/call2/decision)
    eval_warnings: List[str]        # 평가 경고 메시지
    eval_retry_count: int           # 평가 RETRY 횟수

    # ==================================================================
    # -4- GAP INFER AGENT
    gaps: List[dict]

    # ==================================================================
    # -5- CRITIC AGENT
    critic: Optional[dict]
    critic_loop_count: int

    trace: dict