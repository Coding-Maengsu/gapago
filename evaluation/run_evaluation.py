"""
Run LitSearch Benchmark Evaluation

Usage:
    python evaluation/run_evaluation.py --queries data/litsearch_queries.json \
                                        --corpus data/litsearch_corpus.json \
                                        --output results/litsearch_eval.json
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.litsearch_benchmark import (
    LitSearchDataLoader,
    QueryReformulationResult,
    RetrievalResult,
    EndToEndEvaluator,
)
from core.states import AgentState
from langchain_core.messages import HumanMessage, AIMessage


def query_agent_wrapper(original_query: str, llm_provider: str = "azure") -> QueryReformulationResult:
    """
    Wrapper for Query Agent that matches LitSearch evaluation interface.

    Args:
        original_query: Original user query
        llm_provider: LLM provider to use

    Returns:
        QueryReformulationResult object
    """
    from agents.query_agent.query_analysis import query_analysis_node

    state = AgentState(
        messages=[HumanMessage(content=original_query)],
        sender="",
        errors=[],
        iteration=0,
        max_iterations=1,
        scope_level="",
        scope_rationale="",
        breadth_candidates=[],
        expansion_suggestion="",
        keywords=[],
        negative_keywords=[],
        refined_query="",
        user_question=original_query,
        needs_user_input=False,
        papers=[],
        total_candidates_count=0,
        web_results=[],
        research_domain="auto",
        llm_provider=llm_provider,
        year_range="auto",
        output_language="en",
        session_id="eval",
        limitations=[],
        limitation_eval={},
        eval_warnings=[],
        eval_retry_count=0,
        gaps=[],
        critic=None,
        critic_loop_count=0,
        trace={},
    )

    try:
        result_state = query_analysis_node(state)

        # Extract specific phrases from scope_assessment if available
        specific_phrases = []
        if result_state.get("messages"):
            for msg in result_state["messages"]:
                if hasattr(msg, "name") and msg.name == "query_analysis":
                    try:
                        data = json.loads(msg.content)
                        sa = data.get("scope_assessment", {})
                        specific_phrases = sa.get("specific_phrases", [])
                    except:
                        pass

        return QueryReformulationResult(
            query_id="",  # Will be filled by runner
            original_query=original_query,
            scope_level=result_state.get("scope_level", "TOO_BROAD"),
            refined_query=result_state.get("refined_query", ""),
            keywords=result_state.get("keywords", []),
            specific_phrases=specific_phrases,
        )

    except Exception as e:
        print(f"❌ Query Agent error: {e}")
        return QueryReformulationResult(
            query_id="",
            original_query=original_query,
            scope_level="ERROR",
            refined_query="",
            keywords=[],
            specific_phrases=[],
        )


def retrieval_agent_wrapper(
    refined_query: str,
    keywords: List[str],
    llm_provider: str = "azure",
    year_range: str = "5y"
) -> RetrievalResult:
    """
    Wrapper for Retrieval Agent that matches LitSearch evaluation interface.

    Args:
        refined_query: Refined query from Query Agent
        keywords: Keywords from Query Agent
        llm_provider: LLM provider to use
        year_range: Year filter (auto/1y/3y/5y)

    Returns:
        RetrievalResult object
    """
    from agents.retrieval_agent import paper_retrieval_node

    # Build meaning_expand style message
    meaning_expand_content = json.dumps({
        "refined_query": refined_query,
        "keywords": keywords,
        "expanded_terms": keywords,
        "arxiv_query_candidates": [refined_query],
    }, ensure_ascii=False)

    state = AgentState(
        messages=[
            AIMessage(content=meaning_expand_content, name="meaning_expand")
        ],
        sender="meaning_expand",
        errors=[],
        iteration=0,
        max_iterations=1,
        scope_level="SEARCHABLE",
        scope_rationale="",
        breadth_candidates=[],
        expansion_suggestion="",
        keywords=keywords,
        negative_keywords=[],
        refined_query=refined_query,
        user_question=refined_query,
        needs_user_input=False,
        papers=[],
        total_candidates_count=0,
        web_results=[],
        research_domain="auto",
        llm_provider=llm_provider,
        year_range=year_range,
        output_language="en",
        session_id="eval",
        limitations=[],
        limitation_eval={},
        eval_warnings=[],
        eval_retry_count=0,
        gaps=[],
        critic=None,
        critic_loop_count=0,
        trace={},
    )

    try:
        result_state = paper_retrieval_node(state)

        papers = result_state.get("papers", [])
        retrieved_ids = []
        scores = []

        for p in papers:
            if hasattr(p, "paper_id"):
                retrieved_ids.append(p.paper_id)
                scores.append(p.score_bm25)
            elif isinstance(p, dict):
                retrieved_ids.append(p.get("paper_id", ""))
                scores.append(p.get("score_bm25", 0.0))

        return RetrievalResult(
            query_id="",  # Will be filled by runner
            retrieved_papers=retrieved_ids,
            scores=scores,
        )

    except Exception as e:
        print(f"❌ Retrieval Agent error: {e}")
        return RetrievalResult(
            query_id="",
            retrieved_papers=[],
            scores=[],
        )


def run_evaluation_pipeline(
    queries_path: str,
    corpus_path: str,
    output_path: str,
    llm_provider: str = "azure",
    year_range: str = "5y",
    max_queries: int = None,
) -> Dict:
    """
    Run full evaluation pipeline on LitSearch benchmark.

    Args:
        queries_path: Path to LitSearch queries JSON
        corpus_path: Path to LitSearch paper corpus JSON
        output_path: Path to save evaluation results
        llm_provider: LLM provider (azure/claude/gemini)
        year_range: Year filter for retrieval
        max_queries: Max number of queries to evaluate (for testing)

    Returns:
        Evaluation results dictionary
    """
    print("=" * 70)
    print("LitSearch Benchmark Evaluation")
    print("=" * 70)

    # Load dataset
    print(f"\n📂 Loading dataset...")
    print(f"  Queries: {queries_path}")
    print(f"  Corpus:  {corpus_path}")

    loader = LitSearchDataLoader()
    queries = loader.load_queries(queries_path)
    corpus = loader.load_paper_corpus(corpus_path)

    if max_queries:
        queries = queries[:max_queries]

    print(f"  ✓ Loaded {len(queries)} queries")
    print(f"  ✓ Loaded {len(corpus)} papers in corpus")

    # Run agents
    print(f"\n🤖 Running agents...")
    query_results = []
    retrieval_results = []

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query.query_id}")
        print(f"  Query: {query.original_query[:80]}...")

        # Query Agent
        print(f"  → Running Query Agent...")
        qr = query_agent_wrapper(query.original_query, llm_provider)
        qr.query_id = query.query_id
        query_results.append(qr)

        print(f"    Scope: {qr.scope_level}")
        print(f"    Refined: {qr.refined_query[:80] if qr.refined_query else 'N/A'}...")
        print(f"    Keywords: {qr.keywords}")

        # Retrieval Agent (only if SEARCHABLE)
        if qr.scope_level == "SEARCHABLE" and qr.refined_query:
            print(f"  → Running Retrieval Agent...")
            rr = retrieval_agent_wrapper(
                qr.refined_query,
                qr.keywords,
                llm_provider,
                year_range
            )
            rr.query_id = query.query_id
            retrieval_results.append(rr)

            print(f"    Retrieved: {len(rr.retrieved_papers)} papers")
        else:
            print(f"  ⏭️  Skipping retrieval (not SEARCHABLE)")

    # Evaluate
    print(f"\n📊 Evaluating results...")
    evaluator = EndToEndEvaluator()
    results = evaluator.evaluate(query_results, retrieval_results, queries)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n📝 Query Agent Metrics:")
    for metric, value in results["query_metrics"].items():
        print(f"  {metric:25s}: {value:.4f}")

    print("\n🔍 Retrieval Agent Metrics:")
    for metric, value in results["retrieval_metrics"].items():
        print(f"  {metric:25s}: {value:.4f}")

    print(f"\n⭐ Overall Score: {results['overall_score']:.4f}")

    # Save results
    print(f"\n💾 Saving results to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    detailed_results = {
        "summary": results,
        "per_query": [
            {
                "query_id": qr.query_id,
                "original_query": qr.original_query,
                "scope_level": qr.scope_level,
                "refined_query": qr.refined_query,
                "keywords": qr.keywords,
                "retrieved_count": len([r for r in retrieval_results if r.query_id == qr.query_id]),
            }
            for qr in query_results
        ],
        "config": {
            "llm_provider": llm_provider,
            "year_range": year_range,
            "num_queries": len(queries),
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print("  ✓ Done!")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run LitSearch benchmark evaluation on GAPAGO agents"
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to LitSearch queries JSON file"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to LitSearch paper corpus JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/litsearch_eval.json",
        help="Path to save evaluation results (default: results/litsearch_eval.json)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="azure",
        choices=["azure", "claude", "gemini", "exaone"],
        help="LLM provider to use (default: azure)"
    )
    parser.add_argument(
        "--year-range",
        type=str,
        default="5y",
        choices=["auto", "1y", "3y", "5y"],
        help="Year filter for retrieval (default: 5y)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Max number of queries to evaluate (for testing, default: all)"
    )

    args = parser.parse_args()

    # Validate input files
    if not Path(args.queries).exists():
        print(f"❌ Error: Queries file not found: {args.queries}")
        sys.exit(1)

    if not Path(args.corpus).exists():
        print(f"❌ Error: Corpus file not found: {args.corpus}")
        sys.exit(1)

    # Run evaluation
    try:
        run_evaluation_pipeline(
            queries_path=args.queries,
            corpus_path=args.corpus,
            output_path=args.output,
            llm_provider=args.provider,
            year_range=args.year_range,
            max_queries=args.max_queries,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
