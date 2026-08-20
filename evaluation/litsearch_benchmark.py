"""
LitSearch Benchmark Evaluation Framework

This module provides evaluation capabilities for both Query Agent and Retrieval Agent
using the LitSearch benchmark dataset.

Reference:
    LitSearch: A Retrieval Benchmark for Scientific Literature Search
    https://arxiv.org/abs/2407.14228
"""

import asyncio
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LitSearchQuery:
    """Single query from LitSearch dataset."""
    query_id: str
    original_query: str
    domain: str  # e.g., "biomedical", "computer_science"
    information_need: str  # Long-form description
    relevant_papers: List[str] = field(default_factory=list)  # Paper IDs
    relevance_scores: Dict[str, float] = field(default_factory=dict)  # paper_id -> score


@dataclass
class QueryReformulationResult:
    """Output from Query Agent."""
    query_id: str
    original_query: str
    scope_level: str  # TOO_BROAD / SEARCHABLE / TOO_NARROW
    refined_query: str
    keywords: List[str]
    specific_phrases: List[str]


@dataclass
class RetrievalResult:
    """Output from Retrieval Agent."""
    query_id: str
    retrieved_papers: List[str]  # Paper IDs in ranked order
    scores: List[float]  # Relevance scores


# =====================================================================
# Evaluation Metrics
# =====================================================================

def compute_ndcg(retrieved: List[str], relevance: Dict[str, float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    Args:
        retrieved: List of retrieved paper IDs in ranked order
        relevance: Dict mapping paper_id to relevance score (0-3 scale typical)
        k: Cutoff rank

    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    if not retrieved or not relevance:
        return 0.0

    # DCG@k
    dcg = 0.0
    for i, paper_id in enumerate(retrieved[:k]):
        rel = relevance.get(paper_id, 0.0)
        dcg += (2**rel - 1) / np.log2(i + 2)  # i+2 because rank starts at 1

    # IDCG@k (ideal DCG)
    ideal_ranking = sorted(relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_ranking):
        idcg += (2**rel - 1) / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Compute Recall@k.

    Args:
        retrieved: List of retrieved paper IDs in ranked order
        relevant: List of relevant paper IDs (ground truth)
        k: Cutoff rank

    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant:
        return 0.0

    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)

    return len(retrieved_set & relevant_set) / len(relevant_set)


def compute_precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Compute Precision@k.

    Args:
        retrieved: List of retrieved paper IDs in ranked order
        relevant: List of relevant paper IDs (ground truth)
        k: Cutoff rank

    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if not retrieved:
        return 0.0

    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)

    return len(retrieved_set & relevant_set) / min(k, len(retrieved))


def compute_mrr(retrieved: List[str], relevant: List[str]) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        retrieved: List of retrieved paper IDs in ranked order
        relevant: List of relevant paper IDs (ground truth)

    Returns:
        Reciprocal rank of first relevant document (0.0 to 1.0)
    """
    relevant_set = set(relevant)

    for i, paper_id in enumerate(retrieved):
        if paper_id in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


# =====================================================================
# Query Reformulation Evaluation
# =====================================================================

class QueryReformulationEvaluator:
    """Evaluates Query Agent's query reformulation quality."""

    def evaluate_specificity(self, original: str, refined: str) -> float:
        """
        Measure query specificity improvement.

        Heuristic: refined query should be longer and contain more specific terms.
        """
        if not refined:
            return 0.0

        # Token count increase
        orig_tokens = set(original.lower().split())
        refined_tokens = set(refined.lower().split())

        # New specific terms added
        new_terms = refined_tokens - orig_tokens

        # Normalize by original length
        if len(orig_tokens) == 0:
            return 0.0

        specificity_score = len(new_terms) / len(orig_tokens)
        return min(specificity_score, 1.0)

    def evaluate_keyword_quality(
        self,
        keywords: List[str],
        ground_truth_papers: List[Dict[str, str]]
    ) -> float:
        """
        Evaluate extracted keywords against ground truth papers.

        Uses BM25 to measure how well keywords match relevant papers.

        Args:
            keywords: Extracted keywords from Query Agent
            ground_truth_papers: List of relevant papers (with title, abstract)

        Returns:
            Average BM25 score across relevant papers (normalized)
        """
        if not keywords or not ground_truth_papers:
            return 0.0

        from rank_bm25 import BM25Okapi

        # Tokenize ground truth corpus
        corpus = [
            f"{p.get('title', '')} {p.get('abstract', '')}".lower().split()
            for p in ground_truth_papers
        ]

        bm25 = BM25Okapi(corpus)
        query_tokens = " ".join(keywords).lower().split()
        scores = bm25.get_scores(query_tokens)

        # Normalize by max possible score
        max_score = max(scores) if len(scores) > 0 else 1.0
        if max_score == 0:
            return 0.0

        avg_score = np.mean(scores) / max_score
        return float(avg_score)

    def evaluate_scope_assessment(
        self,
        scope_level: str,
        query_complexity: str
    ) -> float:
        """
        Evaluate whether scope assessment matches query complexity.

        Args:
            scope_level: TOO_BROAD / SEARCHABLE / TOO_NARROW
            query_complexity: Expected complexity (vague / moderate / specific)

        Returns:
            1.0 if correct, 0.0 if incorrect
        """
        mapping = {
            "vague": "TOO_BROAD",
            "moderate": "SEARCHABLE",
            "specific": "SEARCHABLE"
        }

        expected = mapping.get(query_complexity, "SEARCHABLE")
        return 1.0 if scope_level == expected else 0.0


# =====================================================================
# Retrieval Evaluation
# =====================================================================

class RetrievalEvaluator:
    """Evaluates Retrieval Agent's performance against LitSearch ground truth."""

    def __init__(self, k_values: List[int] = [5, 10, 20]):
        self.k_values = k_values

    def evaluate_single_query(
        self,
        retrieved: List[str],
        relevance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval for a single query.

        Args:
            retrieved: List of retrieved paper IDs in ranked order
            relevance: Dict mapping paper_id to relevance score

        Returns:
            Dict with metrics: ndcg@k, recall@k, precision@k, mrr
        """
        relevant_papers = [pid for pid, score in relevance.items() if score > 0]

        metrics = {}

        # Compute metrics at different k values
        for k in self.k_values:
            metrics[f"ndcg@{k}"] = compute_ndcg(retrieved, relevance, k)
            metrics[f"recall@{k}"] = compute_recall_at_k(retrieved, relevant_papers, k)
            metrics[f"precision@{k}"] = compute_precision_at_k(retrieved, relevant_papers, k)

        # MRR
        metrics["mrr"] = compute_mrr(retrieved, relevant_papers)

        return metrics

    def evaluate_batch(
        self,
        results: List[RetrievalResult],
        ground_truth: List[LitSearchQuery]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval across multiple queries.

        Args:
            results: List of RetrievalResult objects
            ground_truth: List of LitSearchQuery objects

        Returns:
            Dict with averaged metrics
        """
        # Build lookup for ground truth
        gt_lookup = {q.query_id: q for q in ground_truth}

        all_metrics = defaultdict(list)

        for result in results:
            if result.query_id not in gt_lookup:
                continue

            gt = gt_lookup[result.query_id]
            metrics = self.evaluate_single_query(result.retrieved_papers, gt.relevance_scores)

            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)

        # Average across queries
        avg_metrics = {
            metric: np.mean(values) if values else 0.0
            for metric, values in all_metrics.items()
        }

        return avg_metrics


# =====================================================================
# End-to-End Evaluation
# =====================================================================

class EndToEndEvaluator:
    """Evaluates full pipeline: Query Agent → Retrieval Agent."""

    def __init__(self):
        self.query_evaluator = QueryReformulationEvaluator()
        self.retrieval_evaluator = RetrievalEvaluator()

    def evaluate(
        self,
        query_results: List[QueryReformulationResult],
        retrieval_results: List[RetrievalResult],
        ground_truth: List[LitSearchQuery]
    ) -> Dict[str, any]:
        """
        Evaluate full pipeline.

        Returns:
            Dict with sections: query_metrics, retrieval_metrics, overall
        """
        # Query Agent metrics
        query_metrics = {
            "avg_specificity": 0.0,
            "avg_keyword_quality": 0.0,
            "searchable_rate": 0.0,
        }

        specificity_scores = []
        searchable_count = 0

        for qr in query_results:
            spec_score = self.query_evaluator.evaluate_specificity(
                qr.original_query, qr.refined_query
            )
            specificity_scores.append(spec_score)

            if qr.scope_level == "SEARCHABLE":
                searchable_count += 1

        query_metrics["avg_specificity"] = np.mean(specificity_scores) if specificity_scores else 0.0
        query_metrics["searchable_rate"] = searchable_count / len(query_results) if query_results else 0.0

        # Retrieval Agent metrics
        retrieval_metrics = self.retrieval_evaluator.evaluate_batch(
            retrieval_results, ground_truth
        )

        # Overall score (weighted combination)
        overall_score = (
            0.3 * query_metrics["avg_specificity"] +
            0.3 * query_metrics["searchable_rate"] +
            0.4 * retrieval_metrics.get("ndcg@10", 0.0)
        )

        return {
            "query_metrics": query_metrics,
            "retrieval_metrics": retrieval_metrics,
            "overall_score": overall_score,
        }


# =====================================================================
# LitSearch Dataset Loader
# =====================================================================

class LitSearchDataLoader:
    """Load and parse LitSearch benchmark dataset."""

    @staticmethod
    def load_queries(file_path: str) -> List[LitSearchQuery]:
        """
        Load queries from LitSearch JSON file.

        Expected format:
        {
            "queries": [
                {
                    "query_id": "...",
                    "query": "...",
                    "domain": "...",
                    "information_need": "...",
                    "relevant_papers": ["paper_id1", ...],
                    "relevance_scores": {"paper_id1": 2.0, ...}
                }
            ]
        }
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        queries = []
        for q in data.get("queries", []):
            queries.append(LitSearchQuery(
                query_id=q["query_id"],
                original_query=q["query"],
                domain=q.get("domain", "general"),
                information_need=q.get("information_need", ""),
                relevant_papers=q.get("relevant_papers", []),
                relevance_scores=q.get("relevance_scores", {})
            ))

        return queries

    @staticmethod
    def load_paper_corpus(file_path: str) -> Dict[str, Dict]:
        """
        Load paper corpus (titles, abstracts) for LitSearch.

        Expected format:
        {
            "papers": {
                "paper_id1": {
                    "title": "...",
                    "abstract": "...",
                    ...
                }
            }
        }
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get("papers", {})


# =====================================================================
# Evaluation Runner
# =====================================================================

def run_evaluation(
    queries_path: str,
    corpus_path: str,
    query_agent_fn,
    retrieval_agent_fn,
    output_path: Optional[str] = None
) -> Dict:
    """
    Run full evaluation pipeline.

    Args:
        queries_path: Path to LitSearch queries JSON
        corpus_path: Path to LitSearch paper corpus JSON
        query_agent_fn: Function that takes original_query, returns QueryReformulationResult
        retrieval_agent_fn: Function that takes refined_query, returns RetrievalResult
        output_path: Optional path to save results JSON

    Returns:
        Evaluation results dict
    """
    # Load data
    loader = LitSearchDataLoader()
    queries = loader.load_queries(queries_path)
    corpus = loader.load_paper_corpus(corpus_path)

    # Run agents
    query_results = []
    retrieval_results = []

    for query in queries:
        # Query Agent
        qr = query_agent_fn(query.original_query)
        query_results.append(qr)

        # Retrieval Agent (only if SEARCHABLE)
        if qr.scope_level == "SEARCHABLE":
            rr = retrieval_agent_fn(qr.refined_query)
            retrieval_results.append(rr)

    # Evaluate
    evaluator = EndToEndEvaluator()
    results = evaluator.evaluate(query_results, retrieval_results, queries)

    # Save if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# =====================================================================
# Example Usage
# =====================================================================

if __name__ == "__main__":
    """
    Example: How to use this evaluation framework.

    1. Download LitSearch dataset
    2. Implement adapter functions for your agents
    3. Run evaluation
    """

    # Example adapter for Query Agent
    def query_agent_adapter(original_query: str) -> QueryReformulationResult:
        from gapago.agents.query_agent.query_analysis import query_analysis_node
        from gapago.core.states import AgentState
        from langchain_core.messages import HumanMessage

        state = {
            "messages": [HumanMessage(content=original_query)],
            "iteration": 0,
            "max_iterations": 1,
            "llm_provider": "azure",
            "output_language": "en",
        }

        result_state = query_analysis_node(state)

        return QueryReformulationResult(
            query_id="test",
            original_query=original_query,
            scope_level=result_state["scope_level"],
            refined_query=result_state["refined_query"],
            keywords=result_state["keywords"],
            specific_phrases=[]  # Extract from scope_assessment if needed
        )

    # Example adapter for Retrieval Agent
    def retrieval_agent_adapter(refined_query: str) -> RetrievalResult:
        from gapago.agents.retrieval_agent import paper_retrieval_node
        from gapago.core.states import AgentState
        from langchain_core.messages import AIMessage
        import json

        state = {
            "messages": [AIMessage(
                content=json.dumps({"refined_query": refined_query}),
                name="query_analysis"
            )],
            "refined_query": refined_query,
            "keywords": [],
            "llm_provider": "azure",
            "year_range": "auto",
            "session_id": "eval",
        }

        result_state = asyncio.run(paper_retrieval_node(state))  # async 노드

        retrieved_ids = [p.paper_id for p in result_state["papers"]]
        scores = [p.score_bm25 for p in result_state["papers"]]

        return RetrievalResult(
            query_id="test",
            retrieved_papers=retrieved_ids,
            scores=scores
        )

    # Run evaluation
    # results = run_evaluation(
    #     queries_path="data/litsearch_queries.json",
    #     corpus_path="data/litsearch_corpus.json",
    #     query_agent_fn=query_agent_adapter,
    #     retrieval_agent_fn=retrieval_agent_adapter,
    #     output_path="results/evaluation_results.json"
    # )

    print("Evaluation framework ready. Uncomment example code to run.")
