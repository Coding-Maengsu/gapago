"""
Analyze and compare LitSearch evaluation results.

Usage:
    python evaluation/analyze_results.py \
        --results results/eval_azure.json results/eval_claude.json \
        --output results/comparison_report.html
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def _normalize(data: Dict) -> Dict:
    """
    결과 파일 스키마를 summary 형태로 통일한다.

    두 종류가 섞여 들어온다.
      - run_evaluation.py         → {"summary": {"query_metrics": ..., "retrieval_metrics": ...}}
      - evaluate_scope_classification.py → {"metrics": {...}}   (최상위)

    후자를 그대로 두면 아래 모든 summary 조회가 0.0 을 반환해
    scope 결과가 전부 0 으로 보인다.
    """
    if "summary" in data:
        return data

    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        data = dict(data)
        data["summary"] = {
            "query_metrics": metrics,
            "retrieval_metrics": {},
            "overall_score": metrics.get("accuracy", 0.0),
        }
    return data


def load_results(file_paths: List[str]) -> Dict[str, Dict]:
    """Load evaluation results from JSON files."""
    results = {}

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract name from file path
        name = Path(file_path).stem
        results[name] = _normalize(data)

    return results


def print_comparison_table(results: Dict[str, Dict]) -> None:
    """Print comparison table to console."""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 100)

    # Extract all metric names
    all_metrics = set()
    for result in results.values():
        summary = result.get("summary", {})
        all_metrics.update(summary.get("query_metrics", {}).keys())
        all_metrics.update(summary.get("retrieval_metrics", {}).keys())

    # Print header
    print(f"\n{'Metric':<30s}", end="")
    for name in results.keys():
        print(f"{name:>15s}", end="")
    print()
    print("-" * 100)

    # Print Query Agent metrics
    print(f"\n{'QUERY AGENT METRICS':>30s}")
    print("-" * 100)

    query_metrics = ["avg_specificity", "avg_keyword_quality", "searchable_rate"]
    for metric in query_metrics:
        if metric in all_metrics:
            print(f"{metric:<30s}", end="")
            for name, result in results.items():
                value = result.get("summary", {}).get("query_metrics", {}).get(metric, 0.0)
                print(f"{value:>15.4f}", end="")
            print()

    # Print Retrieval Agent metrics
    print(f"\n{'RETRIEVAL AGENT METRICS':>30s}")
    print("-" * 100)

    retrieval_metrics = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20", "mrr"]
    for metric in retrieval_metrics:
        if metric in all_metrics:
            print(f"{metric:<30s}", end="")
            for name, result in results.items():
                value = result.get("summary", {}).get("retrieval_metrics", {}).get(metric, 0.0)
                print(f"{value:>15.4f}", end="")
            print()

    # Print Overall Score
    print(f"\n{'OVERALL':>30s}")
    print("-" * 100)
    print(f"{'overall_score':<30s}", end="")
    for name, result in results.items():
        value = result.get("summary", {}).get("overall_score", 0.0)
        print(f"{value:>15.4f}", end="")
    print()

    # Print Config
    print(f"\n{'CONFIG':>30s}")
    print("-" * 100)
    for config_key in ["llm_provider", "year_range", "num_queries"]:
        print(f"{config_key:<30s}", end="")
        for name, result in results.items():
            value = result.get("config", {}).get(config_key, "N/A")
            print(f"{str(value):>15s}", end="")
        print()

    print("\n" + "=" * 100)


def generate_html_report(results: Dict[str, Dict], output_path: str) -> None:
    """Generate HTML comparison report."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LitSearch Evaluation Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        h1, h2 {
            color: #333;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .metric-name {
            font-weight: 500;
        }
        .best-score {
            background-color: #d4edda;
            font-weight: bold;
        }
        .section-header {
            background-color: #6c757d;
            color: white;
            font-weight: bold;
        }
        .chart {
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 LitSearch Evaluation Results</h1>
        <p><strong>Generated:</strong> {timestamp}</p>

        <h2>📊 Overall Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    {header_cols}
                </tr>
            </thead>
            <tbody>
                <tr class="section-header">
                    <td colspan="{total_cols}">QUERY AGENT METRICS</td>
                </tr>
                {query_rows}

                <tr class="section-header">
                    <td colspan="{total_cols}">RETRIEVAL AGENT METRICS</td>
                </tr>
                {retrieval_rows}

                <tr class="section-header">
                    <td colspan="{total_cols}">OVERALL</td>
                </tr>
                {overall_row}
            </tbody>
        </table>

        <h2>⚙️ Configuration</h2>
        <table>
            <thead>
                <tr>
                    <th>Setting</th>
                    {header_cols}
                </tr>
            </thead>
            <tbody>
                {config_rows}
            </tbody>
        </table>

        <h2>📈 Per-Query Analysis</h2>
        {per_query_section}

        <h2>💡 Insights</h2>
        {insights_section}
    </div>
</body>
</html>
"""

    from datetime import datetime

    # Build header columns
    header_cols = "".join([f"<th>{name}</th>" for name in results.keys()])
    total_cols = len(results) + 1

    # Build query metrics rows
    query_metrics = ["avg_specificity", "avg_keyword_quality", "searchable_rate"]
    query_rows = []
    for metric in query_metrics:
        values = []
        for name, result in results.items():
            value = result.get("summary", {}).get("query_metrics", {}).get(metric, 0.0)
            values.append(value)

        max_value = max(values) if values else 0
        row = f"<tr><td class='metric-name'>{metric}</td>"
        for value in values:
            cell_class = "best-score" if value == max_value and value > 0 else ""
            row += f"<td class='{cell_class}'>{value:.4f}</td>"
        row += "</tr>"
        query_rows.append(row)

    # Build retrieval metrics rows
    retrieval_metrics = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20", "mrr"]
    retrieval_rows = []
    for metric in retrieval_metrics:
        values = []
        for name, result in results.items():
            value = result.get("summary", {}).get("retrieval_metrics", {}).get(metric, 0.0)
            values.append(value)

        max_value = max(values) if values else 0
        row = f"<tr><td class='metric-name'>{metric}</td>"
        for value in values:
            cell_class = "best-score" if value == max_value and value > 0 else ""
            row += f"<td class='{cell_class}'>{value:.4f}</td>"
        row += "</tr>"
        retrieval_rows.append(row)

    # Build overall score row
    overall_values = []
    for name, result in results.items():
        value = result.get("summary", {}).get("overall_score", 0.0)
        overall_values.append(value)

    max_overall = max(overall_values) if overall_values else 0
    overall_row = "<tr><td class='metric-name'><strong>Overall Score</strong></td>"
    for value in overall_values:
        cell_class = "best-score" if value == max_overall and value > 0 else ""
        overall_row += f"<td class='{cell_class}'><strong>{value:.4f}</strong></td>"
    overall_row += "</tr>"

    # Build config rows
    config_keys = ["llm_provider", "year_range", "num_queries"]
    config_rows = []
    for key in config_keys:
        row = f"<tr><td class='metric-name'>{key}</td>"
        for name, result in results.items():
            value = result.get("config", {}).get(key, "N/A")
            row += f"<td>{value}</td>"
        row += "</tr>"
        config_rows.append(row)

    # Build per-query section
    per_query_section = "<p>Per-query details available in JSON files.</p>"

    # Build insights section
    insights = []
    best_config = max(results.items(), key=lambda x: x[1].get("summary", {}).get("overall_score", 0))
    insights.append(f"<li><strong>Best Configuration:</strong> {best_config[0]} (score: {best_config[1].get('summary', {}).get('overall_score', 0):.4f})</li>")

    # Query Agent insights
    best_query_spec = max(
        results.items(),
        key=lambda x: x[1].get("summary", {}).get("query_metrics", {}).get("avg_specificity", 0)
    )
    insights.append(f"<li><strong>Best Query Specificity:</strong> {best_query_spec[0]} ({best_query_spec[1].get('summary', {}).get('query_metrics', {}).get('avg_specificity', 0):.4f})</li>")

    # Retrieval insights
    best_ndcg = max(
        results.items(),
        key=lambda x: x[1].get("summary", {}).get("retrieval_metrics", {}).get("ndcg@10", 0)
    )
    insights.append(f"<li><strong>Best nDCG@10:</strong> {best_ndcg[0]} ({best_ndcg[1].get('summary', {}).get('retrieval_metrics', {}).get('ndcg@10', 0):.4f})</li>")

    insights_section = "<ul>" + "".join(insights) + "</ul>"

    # Fill template
    html = html.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        header_cols=header_cols,
        total_cols=total_cols,
        query_rows="\n".join(query_rows),
        retrieval_rows="\n".join(retrieval_rows),
        overall_row=overall_row,
        config_rows="\n".join(config_rows),
        per_query_section=per_query_section,
        insights_section=insights_section,
    )

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n📄 HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare LitSearch evaluation results"
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Paths to evaluation result JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison_report.html",
        help="Output path for HTML report (default: results/comparison_report.html)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["console", "html", "both"],
        help="Output format (default: both)"
    )

    args = parser.parse_args()

    # Validate input files
    for file_path in args.results:
        if not Path(file_path).exists():
            print(f"❌ Error: Result file not found: {file_path}")
            sys.exit(1)

    # Load results
    print(f"📂 Loading {len(args.results)} result files...")
    results = load_results(args.results)
    print(f"  ✓ Loaded results for: {', '.join(results.keys())}")

    # Generate output
    if args.format in ["console", "both"]:
        print_comparison_table(results)

    if args.format in ["html", "both"]:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        generate_html_report(results, args.output)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
