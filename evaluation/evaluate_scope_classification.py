"""
Evaluate Query Agent's Scope Classification Performance

Measures how accurately the Query Agent classifies queries into:
- TOO_BROAD
- SEARCHABLE
- TOO_NARROW

Metrics:
- Accuracy (overall)
- Precision, Recall, F1 (per class)
- Confusion Matrix
- Misclassification analysis

Usage:
    python evaluation/evaluate_scope_classification.py \
        --benchmark data/scope_benchmark.json \
        --output results/scope_classification_eval.json \
        --provider azure
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.states import AgentState
from langchain_core.messages import HumanMessage


def load_benchmark(file_path: str) -> Dict:
    """Load scope benchmark dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_human_reviewed_labels(review_file_path: str) -> Dict[str, str]:
    """
    Load human-reviewed labels from JSONL file.

    Format:
    {"query_id": "...", "human_label": "SEARCHABLE", ...}

    Returns:
        Dict mapping query_id to human_label
    """
    if not Path(review_file_path).exists():
        return {}

    human_labels = {}

    with open(review_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get("human_label"):
                human_labels[item["query_id"]] = item["human_label"]

    return human_labels


def query_agent_predict(query_text: str, llm_provider: str = "azure") -> Tuple[str, Dict]:
    """
    Run Query Agent on a single query and get scope classification.

    Args:
        query_text: Query to classify
        llm_provider: LLM provider

    Returns:
        Tuple of (predicted_label, full_result_dict)
    """
    from agents.query_agent.query_analysis import query_analysis_node

    state = AgentState(
        messages=[HumanMessage(content=query_text)],
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
        user_question=query_text,
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

        predicted_label = result_state.get("scope_level", "UNKNOWN")
        rationale = result_state.get("scope_rationale", "")
        refined_query = result_state.get("refined_query", "")
        keywords = result_state.get("keywords", [])

        return predicted_label, {
            "scope_level": predicted_label,
            "rationale": rationale,
            "refined_query": refined_query,
            "keywords": keywords,
        }

    except Exception as e:
        print(f"    ⚠️ Prediction failed: {e}")
        return "ERROR", {"error": str(e)}


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = ["TOO_BROAD", "SEARCHABLE", "TOO_NARROW"]
) -> Dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of possible labels

    Returns:
        Dict with metrics: accuracy, precision, recall, f1, confusion_matrix
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Build results
    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
    }

    for i, label in enumerate(labels):
        metrics["per_class"][label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return metrics


def analyze_misclassifications(
    variants: List[Dict],
    predictions: List[Dict],
    human_labels: Dict[str, str]
) -> Dict:
    """
    Analyze misclassification patterns.

    Returns:
        Dict with misclassification analysis
    """
    misclassified = []

    for variant, pred in zip(variants, predictions):
        query_id = variant["query_id"]

        # Use human label if available, else ground truth
        true_label = human_labels.get(query_id, variant["ground_truth_label"])
        pred_label = pred["predicted_label"]

        if true_label != pred_label:
            misclassified.append({
                "query_id": query_id,
                "query": variant["query_text"],
                "true_label": true_label,
                "predicted_label": pred_label,
                "components": {
                    "domain": variant.get("domain", ""),
                    "task": variant.get("task", ""),
                    "modality": variant.get("modality", ""),
                    "problem": variant.get("problem", ""),
                },
                "rationale": pred["result"].get("rationale", ""),
            })

    # Analyze patterns
    error_types = defaultdict(int)
    for m in misclassified:
        error_type = f"{m['true_label']} → {m['predicted_label']}"
        error_types[error_type] += 1

    # Component analysis
    component_patterns = defaultdict(lambda: {"correct": 0, "incorrect": 0})

    for variant, pred in zip(variants, predictions):
        query_id = variant["query_id"]
        true_label = human_labels.get(query_id, variant["ground_truth_label"])
        pred_label = pred["predicted_label"]

        is_correct = (true_label == pred_label)

        # Count components
        num_components = sum([
            bool(variant.get("domain")),
            bool(variant.get("task")),
            bool(variant.get("modality")),
            bool(variant.get("problem")),
        ])

        key = f"{num_components}_components"
        if is_correct:
            component_patterns[key]["correct"] += 1
        else:
            component_patterns[key]["incorrect"] += 1

    return {
        "misclassified_count": len(misclassified),
        "misclassified_examples": misclassified[:20],  # Top 20
        "error_type_distribution": dict(error_types),
        "component_accuracy": {
            k: {
                "correct": v["correct"],
                "incorrect": v["incorrect"],
                "accuracy": v["correct"] / (v["correct"] + v["incorrect"]) if (v["correct"] + v["incorrect"]) > 0 else 0,
            }
            for k, v in component_patterns.items()
        },
    }


def run_evaluation(
    benchmark_path: str,
    output_path: str,
    llm_provider: str = "azure",
    human_review_path: str = None,
    max_samples: int = None,
) -> Dict:
    """
    Run full scope classification evaluation.

    Args:
        benchmark_path: Path to scope benchmark JSON
        output_path: Output path for results
        llm_provider: LLM provider
        human_review_path: Optional path to human-reviewed labels
        max_samples: Max number of samples to evaluate (for testing)

    Returns:
        Evaluation results dict
    """
    print("=" * 70)
    print("Scope Classification Evaluation")
    print("=" * 70)

    # Load benchmark
    print(f"\n📂 Loading benchmark from {benchmark_path}...")
    benchmark = load_benchmark(benchmark_path)
    variants = benchmark["variants"]

    if max_samples:
        variants = variants[:max_samples]

    print(f"  ✓ Loaded {len(variants)} query variants")

    # Load human labels if available
    human_labels = {}
    if human_review_path:
        print(f"\n👤 Loading human-reviewed labels from {human_review_path}...")
        human_labels = load_human_reviewed_labels(human_review_path)
        print(f"  ✓ Loaded {len(human_labels)} human labels")

    # Run predictions
    print(f"\n🤖 Running Query Agent predictions...")
    predictions = []

    for i, variant in enumerate(variants, 1):
        query_id = variant["query_id"]
        query_text = variant["query_text"]

        print(f"  [{i}/{len(variants)}] {query_id}")
        print(f"    Query: {query_text[:60]}...")

        pred_label, result = query_agent_predict(query_text, llm_provider)

        print(f"    GT: {variant['ground_truth_label']} | Pred: {pred_label}")

        predictions.append({
            "query_id": query_id,
            "query": query_text,
            "ground_truth_label": variant["ground_truth_label"],
            "predicted_label": pred_label,
            "result": result,
        })

    # Prepare labels for metrics
    y_true = []
    y_pred = []

    for variant, pred in zip(variants, predictions):
        query_id = variant["query_id"]

        # Use human label if available, else ground truth
        true_label = human_labels.get(query_id, variant["ground_truth_label"])

        # Filter out ERROR predictions
        if pred["predicted_label"] != "ERROR":
            y_true.append(true_label)
            y_pred.append(pred["predicted_label"])

    print(f"\n📊 Computing metrics...")
    print(f"  Valid predictions: {len(y_pred)}/{len(predictions)}")

    # Compute metrics
    labels = ["TOO_BROAD", "SEARCHABLE", "TOO_NARROW"]
    metrics = compute_classification_metrics(y_true, y_pred, labels)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n📈 Overall Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")

    print(f"\n📋 Per-Class Metrics:")
    for label in labels:
        if label in metrics["per_class"]:
            m = metrics["per_class"][label]
            print(f"\n  {label}:")
            print(f"    Precision: {m['precision']:.4f}")
            print(f"    Recall:    {m['recall']:.4f}")
            print(f"    F1:        {m['f1']:.4f}")
            print(f"    Support:   {m['support']}")

    print(f"\n🔀 Confusion Matrix:")
    print(f"  Rows: True labels | Cols: Predicted labels")
    print(f"  Labels: {labels}")

    cm = metrics["confusion_matrix"]
    for i, true_label in enumerate(labels):
        row = cm[i]
        print(f"  {true_label:15s}: {row}")

    # Misclassification analysis
    print(f"\n🔍 Analyzing misclassifications...")
    misclass_analysis = analyze_misclassifications(variants, predictions, human_labels)

    print(f"\n  Total misclassified: {misclass_analysis['misclassified_count']}/{len(y_true)}")
    print(f"\n  Error type distribution:")
    for error_type, count in sorted(
        misclass_analysis["error_type_distribution"].items(),
        key=lambda x: -x[1]
    ):
        print(f"    {error_type:30s}: {count:3d}")

    print(f"\n  Accuracy by component count:")
    for comp_key, stats in sorted(misclass_analysis["component_accuracy"].items()):
        print(f"    {comp_key:20s}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['correct']+stats['incorrect']})")

    # Save results
    print(f"\n💾 Saving results to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metrics": metrics,
        "predictions": predictions,
        "misclassification_analysis": misclass_analysis,
        "config": {
            "benchmark_path": benchmark_path,
            "llm_provider": llm_provider,
            "num_samples": len(variants),
            "num_valid_predictions": len(y_pred),
            "human_labels_used": len(human_labels),
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved!")

    print("\n" + "=" * 70)
    print("✅ Evaluation complete!")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Query Agent's scope classification"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to scope benchmark JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/scope_classification_eval.json",
        help="Output path (default: results/scope_classification_eval.json)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="azure",
        choices=["azure", "claude", "gemini", "exaone"],
        help="LLM provider (default: azure)"
    )
    parser.add_argument(
        "--human-review",
        type=str,
        default=None,
        help="Path to human-reviewed labels JSONL (optional)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate (for testing, default: all)"
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.benchmark).exists():
        print(f"❌ Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    # Auto-detect human review file if not specified
    if not args.human_review:
        auto_review_path = args.benchmark.replace(".json", "_review.jsonl")
        if Path(auto_review_path).exists():
            args.human_review = auto_review_path
            print(f"ℹ️  Auto-detected human review file: {auto_review_path}")

    # Run evaluation
    try:
        run_evaluation(
            benchmark_path=args.benchmark,
            output_path=args.output,
            llm_provider=args.provider,
            human_review_path=args.human_review,
            max_samples=args.max_samples,
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
