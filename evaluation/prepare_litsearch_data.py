"""
Prepare LitSearch dataset for GAPAGO evaluation.

Converts LitSearch JSONL format to GAPAGO evaluation format.

Usage:
    python evaluation/prepare_litsearch_data.py \
        --input LitSearch/data/queries.jsonl \
        --corpus LitSearch/data/corpus.jsonl \
        --output-queries data/litsearch_queries.json \
        --output-corpus data/litsearch_corpus.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def convert_queries(queries_data: List[Dict]) -> Dict:
    """
    Convert LitSearch queries to GAPAGO evaluation format.

    LitSearch format:
    {
        "query_id": "...",
        "query": "...",
        "domain": "...",
        "information_need": "...",
        "qrels": {"paper_id": relevance_score, ...}
    }

    GAPAGO format:
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
    converted = []

    for q in queries_data:
        query_id = q.get("query_id", q.get("_id", ""))
        query_text = q.get("query", q.get("text", ""))
        domain = q.get("domain", "general")
        info_need = q.get("information_need", q.get("description", ""))

        # Extract relevance judgments
        qrels = q.get("qrels", {})
        if isinstance(qrels, dict):
            relevance_scores = qrels
        elif isinstance(qrels, list):
            # Convert list format: [{"paper_id": "...", "score": 2}, ...]
            relevance_scores = {item["paper_id"]: item["score"] for item in qrels}
        else:
            relevance_scores = {}

        # Extract relevant papers (score > 0)
        relevant_papers = [
            pid for pid, score in relevance_scores.items()
            if float(score) > 0
        ]

        converted.append({
            "query_id": query_id,
            "query": query_text,
            "domain": domain,
            "information_need": info_need,
            "relevant_papers": relevant_papers,
            "relevance_scores": relevance_scores,
        })

    return {"queries": converted}


def convert_corpus(corpus_data: List[Dict]) -> Dict:
    """
    Convert LitSearch corpus to GAPAGO evaluation format.

    LitSearch format:
    {
        "paper_id": "...",
        "title": "...",
        "abstract": "...",
        "authors": [...],
        "year": 2024,
        "venue": "...",
        ...
    }

    GAPAGO format:
    {
        "papers": {
            "paper_id": {
                "title": "...",
                "abstract": "...",
                "authors": [...],
                "year": 2024,
                "venue": "...",
                ...
            }
        }
    }
    """
    papers = {}

    for doc in corpus_data:
        paper_id = doc.get("paper_id", doc.get("_id", ""))
        if not paper_id:
            continue

        papers[paper_id] = {
            "title": doc.get("title", ""),
            "abstract": doc.get("abstract", ""),
            "authors": doc.get("authors", []),
            "year": doc.get("year", 0),
            "venue": doc.get("venue", ""),
            "doi": doc.get("doi", ""),
            "url": doc.get("url", ""),
        }

    return {"papers": papers}


def generate_training_data(
    queries_path: str,
    output_path: str,
    split: str = "train"
) -> None:
    """
    Generate training data for query reformulation fine-tuning.

    Creates training examples in the format:
    {
        "input": "original vague query",
        "output": {
            "scope_level": "TOO_BROAD",
            "refined_query": "more specific query",
            "keywords": ["keyword1", "keyword2"],
            "breadth_candidates": [...]
        }
    }

    This can be used to fine-tune the Query Agent.
    """
    with open(queries_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = data["queries"]
    training_examples = []

    for q in queries:
        # Heuristic: queries with few relevant papers are likely too narrow
        # queries with many relevant papers might be too broad
        num_relevant = len(q["relevant_papers"])

        if num_relevant > 50:
            # Likely too broad - needs refinement
            scope_level = "TOO_BROAD"
            # Extract domain from information_need as a hint
            info_need = q.get("information_need", "")
            keywords = extract_keywords_from_text(info_need)

            example = {
                "input": q["query"],
                "output": {
                    "scope_level": scope_level,
                    "rationale": "Query is too broad and needs more specificity",
                    "keywords": keywords[:5],
                }
            }
        elif num_relevant < 5:
            # Likely too narrow or well-scoped
            scope_level = "TOO_NARROW" if num_relevant < 2 else "SEARCHABLE"
            info_need = q.get("information_need", "")
            keywords = extract_keywords_from_text(info_need)

            example = {
                "input": q["query"],
                "output": {
                    "scope_level": scope_level,
                    "refined_query": info_need if info_need else q["query"],
                    "keywords": keywords[:5],
                }
            }
        else:
            # Well-scoped
            scope_level = "SEARCHABLE"
            info_need = q.get("information_need", "")
            keywords = extract_keywords_from_text(info_need)

            example = {
                "input": q["query"],
                "output": {
                    "scope_level": scope_level,
                    "refined_query": info_need if info_need else q["query"],
                    "keywords": keywords[:5],
                }
            }

        training_examples.append(example)

    # Save as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in training_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"Generated {len(training_examples)} training examples → {output_path}")


def extract_keywords_from_text(text: str) -> List[str]:
    """
    Simple keyword extraction from text.

    Uses basic heuristics:
    - Extract noun phrases
    - Filter common words
    - Keep technical terms
    """
    import re

    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())

    # Split into words
    words = text.split()

    # Filter stop words and short words
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'this', 'these', 'those', 'such',
        'can', 'may', 'have', 'been', 'do', 'does', 'we', 'our', 'how',
        'what', 'when', 'where', 'which', 'who', 'why'
    }

    keywords = []
    for word in words:
        if len(word) > 3 and word not in stop_words:
            keywords.append(word)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            deduped.append(kw)

    return deduped


def analyze_dataset(queries_path: str, corpus_path: str) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 70)
    print("LitSearch Dataset Statistics")
    print("=" * 70)

    # Load data
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    queries = queries_data["queries"]
    papers = corpus_data["papers"]

    print(f"\n📊 Queries: {len(queries)}")

    # Domain distribution
    domain_counts = defaultdict(int)
    for q in queries:
        domain_counts[q["domain"]] += 1

    print("\n  Domain Distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {domain:20s}: {count:4d} ({count/len(queries)*100:.1f}%)")

    # Relevant papers distribution
    rel_counts = [len(q["relevant_papers"]) for q in queries]
    print(f"\n  Relevant Papers per Query:")
    print(f"    Mean:   {sum(rel_counts)/len(rel_counts):.2f}")
    print(f"    Median: {sorted(rel_counts)[len(rel_counts)//2]}")
    print(f"    Min:    {min(rel_counts)}")
    print(f"    Max:    {max(rel_counts)}")

    print(f"\n📚 Corpus: {len(papers)} papers")

    # Year distribution
    years = [p.get("year", 0) for p in papers.values() if p.get("year")]
    if years:
        print(f"\n  Year Range:")
        print(f"    Min:    {min(years)}")
        print(f"    Max:    {max(years)}")
        print(f"    Median: {sorted(years)[len(years)//2]}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LitSearch dataset for GAPAGO evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to LitSearch queries JSONL file"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to LitSearch corpus JSONL file"
    )
    parser.add_argument(
        "--output-queries",
        type=str,
        default="data/litsearch_queries.json",
        help="Output path for converted queries (default: data/litsearch_queries.json)"
    )
    parser.add_argument(
        "--output-corpus",
        type=str,
        default="data/litsearch_corpus.json",
        help="Output path for converted corpus (default: data/litsearch_corpus.json)"
    )
    parser.add_argument(
        "--generate-train",
        action="store_true",
        help="Also generate training data for fine-tuning"
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/query_reformulation_train.jsonl",
        help="Output path for training data (default: data/query_reformulation_train.jsonl)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print dataset statistics after conversion"
    )

    args = parser.parse_args()

    # Validate input files
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return

    if not Path(args.corpus).exists():
        print(f"❌ Error: Corpus file not found: {args.corpus}")
        return

    # Create output directories
    Path(args.output_queries).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_corpus).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LitSearch Data Preparation")
    print("=" * 70)

    # Load original data
    print(f"\n📂 Loading LitSearch data...")
    print(f"  Queries: {args.input}")
    print(f"  Corpus:  {args.corpus}")

    queries_raw = load_jsonl(args.input)
    corpus_raw = load_jsonl(args.corpus)

    print(f"  ✓ Loaded {len(queries_raw)} queries")
    print(f"  ✓ Loaded {len(corpus_raw)} papers")

    # Convert queries
    print(f"\n🔄 Converting queries...")
    queries_converted = convert_queries(queries_raw)
    with open(args.output_queries, 'w', encoding='utf-8') as f:
        json.dump(queries_converted, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved to {args.output_queries}")

    # Convert corpus
    print(f"\n🔄 Converting corpus...")
    corpus_converted = convert_corpus(corpus_raw)
    with open(args.output_corpus, 'w', encoding='utf-8') as f:
        json.dump(corpus_converted, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved to {args.output_corpus}")

    # Generate training data if requested
    if args.generate_train:
        print(f"\n🎯 Generating training data...")
        generate_training_data(args.output_queries, args.train_output)

    # Analyze if requested
    if args.analyze:
        analyze_dataset(args.output_queries, args.output_corpus)

    print("\n✅ Data preparation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
