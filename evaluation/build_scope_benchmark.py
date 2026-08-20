"""
Build Scope Assessment Benchmark (LitSearch-inspired)

Constructs a dataset for evaluating Query Agent's scope classification:
- TOO_BROAD: Only [D] present
- SEARCHABLE: [D]+[T] or specific combinations
- TOO_NARROW: [D]+[T]+[M]+[P] all present (overly specific)

Workflow:
1. Sample arXiv papers across domains
2. Extract citation contexts from papers
3. Generate 3 query variants per context using LLM:
   - Type A: [D] only → GT: TOO_BROAD
   - Type B: [D]+[T] → GT: SEARCHABLE
   - Type C: [D]+[T]+[M]+[P] → GT: TOO_NARROW or SEARCHABLE
4. Human review samples
5. Export benchmark dataset

Usage:
    python evaluation/build_scope_benchmark.py \
        --papers-json data/arxiv_papers_sample.json \
        --output data/scope_benchmark.json \
        --num-samples 200 \
        --domains cs biomedical physics
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class PaperSample:
    """arXiv paper sample for query generation."""
    paper_id: str
    title: str
    abstract: str
    domain: str
    citation_contexts: List[str]  # Citation sentences from the paper


@dataclass
class QueryVariant:
    """Generated query variant with ground truth label."""
    query_id: str
    query_text: str
    query_type: str  # "broad", "searchable", "narrow"
    ground_truth_label: str  # "TOO_BROAD", "SEARCHABLE", "TOO_NARROW"

    # Component analysis
    domain: str  # [D]
    task: str    # [T]
    modality: str  # [M]
    problem: str   # [P]

    # Metadata
    source_paper_id: str
    source_context: str
    generation_prompt: str
    human_verified: bool = False
    human_label: Optional[str] = None
    notes: str = ""


def extract_citation_contexts(abstract: str, title: str) -> List[str]:
    """
    Extract citation-like contexts from paper abstract.

    These are sentences that could inspire research questions:
    - Sentences mentioning limitations, challenges, or future work
    - Sentences describing methodology or problem setup
    - Sentences with contrastive structures ("however", "despite")
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', abstract)

    citation_contexts = []

    # Keywords indicating good citation contexts
    good_indicators = [
        'however', 'despite', 'although', 'but', 'yet',
        'challenge', 'difficult', 'limitation', 'issue', 'problem',
        'future work', 'remains', 'lacking', 'insufficient',
        'propose', 'introduce', 'present', 'develop',
        'unlike', 'different from', 'improve', 'outperform'
    ]

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30:  # Too short
            continue

        sent_lower = sent.lower()

        # Check for good indicators
        if any(indicator in sent_lower for indicator in good_indicators):
            citation_contexts.append(sent)

        # Also include first sentence (problem setup)
        elif sent == sentences[0].strip():
            citation_contexts.append(sent)

    # Fallback: include first 2-3 sentences if nothing found
    if not citation_contexts:
        citation_contexts = [s.strip() for s in sentences[:3] if len(s.strip()) > 30]

    return citation_contexts[:5]  # Max 5 contexts per paper


QUERY_GENERATION_PROMPT_TEMPLATE = """You are a research query generator. Given a citation context from a scientific paper, generate THREE query variants that a researcher might use to find related work.

Citation Context:
"{context}"

Domain: {domain}

Generate exactly 3 query variants:

1. BROAD Query (TOO_BROAD)
   - Include ONLY the application domain [D]
   - Generic field mention without specific task
   - Example: "natural language processing", "computer vision", "robotics"

2. SEARCHABLE Query (SEARCHABLE)
   - Include domain [D] + specific task [T]
   - OR domain [D] + data modality [M]
   - OR task [T] + modality [M]
   - Example: "sentiment analysis in social media", "CNN for image classification"

3. NARROW Query (TOO_NARROW or SEARCHABLE)
   - Include all components: domain [D] + task [T] + modality [M] + problem [P]
   - Highly specific, possibly too specific
   - Example: "transfer learning for low-resource named entity recognition in clinical notes with class imbalance"

For each query, also identify:
- [D] Domain
- [T] Task
- [M] Modality (if applicable)
- [P] Problem (if applicable)

Return JSON:
{{
  "broad_query": {{
    "query": "...",
    "domain": "...",
    "task": "",
    "modality": "",
    "problem": ""
  }},
  "searchable_query": {{
    "query": "...",
    "domain": "...",
    "task": "...",
    "modality": "",
    "problem": ""
  }},
  "narrow_query": {{
    "query": "...",
    "domain": "...",
    "task": "...",
    "modality": "...",
    "problem": "..."
  }}
}}"""


def generate_query_variants(
    paper: PaperSample,
    llm_provider: str = "azure"
) -> List[QueryVariant]:
    """
    Generate 3 query variants (broad, searchable, narrow) per citation context.

    Args:
        paper: Paper sample with citation contexts
        llm_provider: LLM provider for generation

    Returns:
        List of QueryVariant objects
    """
    from gapago.core.llm import get_llm
    from langchain_core.messages import HumanMessage

    llm = get_llm(provider=llm_provider)
    variants = []

    for ctx_idx, context in enumerate(paper.citation_contexts):
        prompt = QUERY_GENERATION_PROMPT_TEMPLATE.format(
            context=context,
            domain=paper.domain
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)

            # Parse JSON response
            data = json.loads(content)

            # Create variants
            for query_type in ["broad_query", "searchable_query", "narrow_query"]:
                q_data = data.get(query_type, {})

                if not q_data or not q_data.get("query"):
                    continue

                # Determine ground truth label
                if query_type == "broad_query":
                    gt_label = "TOO_BROAD"
                elif query_type == "searchable_query":
                    gt_label = "SEARCHABLE"
                else:  # narrow_query
                    # Heuristic: if all 4 components present, likely TOO_NARROW
                    num_components = sum([
                        bool(q_data.get("domain")),
                        bool(q_data.get("task")),
                        bool(q_data.get("modality")),
                        bool(q_data.get("problem")),
                    ])
                    gt_label = "TOO_NARROW" if num_components >= 4 else "SEARCHABLE"

                variant = QueryVariant(
                    query_id=f"{paper.paper_id}_ctx{ctx_idx}_{query_type}",
                    query_text=q_data.get("query", ""),
                    query_type=query_type.replace("_query", ""),
                    ground_truth_label=gt_label,
                    domain=q_data.get("domain", ""),
                    task=q_data.get("task", ""),
                    modality=q_data.get("modality", ""),
                    problem=q_data.get("problem", ""),
                    source_paper_id=paper.paper_id,
                    source_context=context,
                    generation_prompt=prompt,
                )

                variants.append(variant)

        except Exception as e:
            print(f"  ⚠️ Failed to generate variants for {paper.paper_id} ctx {ctx_idx}: {e}")
            continue

    return variants


def load_arxiv_papers_sample(file_path: str) -> List[PaperSample]:
    """
    Load arXiv papers sample from JSON.

    Expected format:
    {
      "papers": [
        {
          "paper_id": "arxiv:2301.12345",
          "title": "...",
          "abstract": "...",
          "domain": "cs"
        }
      ]
    }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    papers = []
    for p in data.get("papers", []):
        citation_contexts = extract_citation_contexts(
            p.get("abstract", ""),
            p.get("title", "")
        )

        if not citation_contexts:
            continue

        papers.append(PaperSample(
            paper_id=p.get("paper_id", ""),
            title=p.get("title", ""),
            abstract=p.get("abstract", ""),
            domain=p.get("domain", "general"),
            citation_contexts=citation_contexts,
        ))

    return papers


def sample_papers_by_domain(
    papers: List[PaperSample],
    domains: List[str],
    samples_per_domain: int
) -> List[PaperSample]:
    """Sample papers evenly across domains."""
    sampled = []

    for domain in domains:
        domain_papers = [p for p in papers if p.domain == domain]

        if len(domain_papers) <= samples_per_domain:
            sampled.extend(domain_papers)
        else:
            sampled.extend(random.sample(domain_papers, samples_per_domain))

    return sampled


def build_benchmark_dataset(
    papers_json: str,
    output_path: str,
    num_samples: int = 200,
    domains: List[str] = None,
    llm_provider: str = "azure",
) -> Dict:
    """
    Build complete scope assessment benchmark dataset.

    Args:
        papers_json: Path to arXiv papers JSON
        output_path: Output path for benchmark JSON
        num_samples: Number of papers to sample
        domains: List of domains to include
        llm_provider: LLM provider for query generation

    Returns:
        Benchmark dataset dict
    """
    print("=" * 70)
    print("Building Scope Assessment Benchmark")
    print("=" * 70)

    # Load papers
    print(f"\n📂 Loading papers from {papers_json}...")
    papers = load_arxiv_papers_sample(papers_json)
    print(f"  ✓ Loaded {len(papers)} papers")

    # Domain distribution
    domain_counts = defaultdict(int)
    for p in papers:
        domain_counts[p.domain] += 1

    print(f"\n  Domain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"    {domain:20s}: {count:4d}")

    # Sample papers
    if domains:
        print(f"\n🎯 Sampling {num_samples} papers from domains: {', '.join(domains)}...")
        samples_per_domain = num_samples // len(domains)
        sampled_papers = sample_papers_by_domain(papers, domains, samples_per_domain)
    else:
        print(f"\n🎯 Sampling {num_samples} papers...")
        sampled_papers = random.sample(papers, min(num_samples, len(papers)))

    print(f"  ✓ Sampled {len(sampled_papers)} papers")

    # Generate query variants
    print(f"\n🤖 Generating query variants (3 per citation context)...")
    all_variants = []

    for i, paper in enumerate(sampled_papers, 1):
        print(f"  [{i}/{len(sampled_papers)}] {paper.paper_id} ({len(paper.citation_contexts)} contexts)")

        variants = generate_query_variants(paper, llm_provider)
        all_variants.extend(variants)

        print(f"    → Generated {len(variants)} variants")

    print(f"\n  ✓ Total variants: {len(all_variants)}")

    # Label distribution
    label_counts = defaultdict(int)
    for v in all_variants:
        label_counts[v.ground_truth_label] += 1

    print(f"\n  Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:20s}: {count:4d} ({count/len(all_variants)*100:.1f}%)")

    # Build dataset
    dataset = {
        "metadata": {
            "num_papers": len(sampled_papers),
            "num_variants": len(all_variants),
            "domains": list(set(p.domain for p in sampled_papers)),
            "label_distribution": dict(label_counts),
            "llm_provider": llm_provider,
        },
        "variants": [asdict(v) for v in all_variants],
    }

    # Save
    print(f"\n💾 Saving benchmark to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(all_variants)} query variants")

    # Export human review format
    review_path = output_path.replace(".json", "_review.jsonl")
    print(f"\n📋 Exporting human review format to {review_path}...")

    with open(review_path, 'w', encoding='utf-8') as f:
        for v in all_variants:
            review_item = {
                "query_id": v.query_id,
                "query": v.query_text,
                "suggested_label": v.ground_truth_label,
                "components": {
                    "domain": v.domain,
                    "task": v.task,
                    "modality": v.modality,
                    "problem": v.problem,
                },
                "source_context": v.source_context,
                "human_label": "",  # To be filled
                "notes": "",  # To be filled
            }
            f.write(json.dumps(review_item, ensure_ascii=False) + '\n')

    print(f"  ✓ Exported {len(all_variants)} items for human review")

    print("\n" + "=" * 70)
    print("✅ Benchmark dataset built successfully!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review queries in: {review_path}")
    print(f"2. Update human_label field for each item")
    print(f"3. Run evaluation: python evaluation/evaluate_scope_classification.py")

    return dataset


def sample_arxiv_papers_via_api(
    domains: List[str],
    samples_per_domain: int = 50,
    output_path: str = "data/arxiv_papers_sample.json"
) -> None:
    """
    Sample arXiv papers via API for benchmark construction.

    Args:
        domains: List of arXiv domain codes (cs, physics, math, etc.)
        samples_per_domain: Number of papers to sample per domain
        output_path: Output path for sampled papers JSON
    """
    import arxiv

    print("=" * 70)
    print("Sampling arXiv Papers")
    print("=" * 70)

    all_papers = []

    for domain in domains:
        print(f"\n📚 Sampling {samples_per_domain} papers from {domain}...")

        # Query arXiv
        search = arxiv.Search(
            query=f"cat:{domain}.*",
            max_results=samples_per_domain * 2,  # Get more to filter
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        count = 0
        for result in search.results():
            if count >= samples_per_domain:
                break

            # Filter: abstract must be substantial
            if not result.summary or len(result.summary) < 200:
                continue

            paper = {
                "paper_id": f"arxiv:{result.entry_id.split('/')[-1]}",
                "title": result.title,
                "abstract": result.summary,
                "domain": domain,
                "authors": [str(a) for a in result.authors],
                "published": result.published.isoformat() if result.published else "",
                "url": result.entry_id,
            }

            all_papers.append(paper)
            count += 1

        print(f"  ✓ Sampled {count} papers")

    # Save
    print(f"\n💾 Saving {len(all_papers)} papers to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    dataset = {"papers": all_papers}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Build scope assessment benchmark dataset"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Sample papers command
    sample_parser = subparsers.add_parser("sample", help="Sample arXiv papers")
    sample_parser.add_argument(
        "--domains",
        nargs="+",
        default=["cs", "physics", "q-bio", "eess", "math"],
        help="arXiv domain codes (default: cs physics q-bio eess math)"
    )
    sample_parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=50,
        help="Number of papers per domain (default: 50)"
    )
    sample_parser.add_argument(
        "--output",
        type=str,
        default="data/arxiv_papers_sample.json",
        help="Output path (default: data/arxiv_papers_sample.json)"
    )

    # Build benchmark command
    build_parser = subparsers.add_parser("build", help="Build benchmark from sampled papers")
    build_parser.add_argument(
        "--papers-json",
        type=str,
        required=True,
        help="Path to arXiv papers JSON"
    )
    build_parser.add_argument(
        "--output",
        type=str,
        default="data/scope_benchmark.json",
        help="Output path (default: data/scope_benchmark.json)"
    )
    build_parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of papers to sample (default: 200)"
    )
    build_parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Filter to specific domains (default: all)"
    )
    build_parser.add_argument(
        "--provider",
        type=str,
        default="azure",
        choices=["azure", "claude", "gemini"],
        help="LLM provider for query generation (default: azure)"
    )

    args = parser.parse_args()

    if args.command == "sample":
        sample_arxiv_papers_via_api(
            domains=args.domains,
            samples_per_domain=args.samples_per_domain,
            output_path=args.output,
        )

    elif args.command == "build":
        build_benchmark_dataset(
            papers_json=args.papers_json,
            output_path=args.output,
            num_samples=args.num_samples,
            domains=args.domains,
            llm_provider=args.provider,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
