# Complete Evaluation Workflow

This document provides a step-by-step guide for the complete evaluation workflow, combining both LitSearch and Scope Classification benchmarks.

## 🎯 Evaluation Strategy Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION STRATEGY                        │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌─────────────────────┐    ┌─────────────────────┐
    │ Scope Classification │    │  LitSearch Benchmark│
    │    Benchmark         │    │   (End-to-End)      │
    └─────────────────────┘    └─────────────────────┘
                │                           │
                ▼                           ▼
    ┌─────────────────────┐    ┌─────────────────────┐
    │ TOO_BROAD/          │    │  nDCG, Recall, MRR  │
    │ SEARCHABLE/         │    │  Retrieval Quality  │
    │ TOO_NARROW          │    │                     │
    │ Classification      │    │                     │
    └─────────────────────┘    └─────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    ┌───────────────────┐
                    │ Overall Assessment│
                    │ + Improvements    │
                    └───────────────────┘
```

## 📋 Workflow

### Phase 1: Setup (One-time)

#### 1.1 Install Dependencies

```bash
pip install -r requirements.txt
pip install arxiv sklearn  # Additional for evaluation
```

#### 1.2 Download LitSearch Dataset (Optional)

```bash
git clone https://github.com/yale-nlp/LitSearch.git
python evaluation/prepare_litsearch_data.py \
    --input LitSearch/data/queries.jsonl \
    --corpus LitSearch/data/corpus.jsonl \
    --output-queries data/litsearch_queries.json \
    --output-corpus data/litsearch_corpus.json
```

### Phase 2: Baseline Evaluation

#### 2.1 Build Scope Classification Benchmark

```bash
# Step 1: Sample papers (5 major science & technology domains)
python evaluation/build_scope_benchmark.py sample \
    --domains cs physics q-bio eess math \
    --samples-per-domain 50 \
    --output data/arxiv_papers_sample.json

# Step 2: Generate query variants
python evaluation/build_scope_benchmark.py build \
    --papers-json data/arxiv_papers_sample.json \
    --output data/scope_benchmark.json \
    --num-samples 250 \
    --provider azure
```

**Output**:
- `data/scope_benchmark.json` - 750 query variants (3 per paper × 250 papers)
- `data/scope_benchmark_review.jsonl` - Human review format

#### 2.2 Human Review (Recommended)

```bash
# Review at least 120 samples (16%)
python evaluation/review_scope_labels.py review \
    --input data/scope_benchmark_review.jsonl

# Check progress
python evaluation/review_scope_labels.py summary \
    --input data/scope_benchmark_review.jsonl
```

**Time estimate**: ~30 minutes for 100 samples

#### 2.3 Run Scope Classification Evaluation

```bash
python evaluation/evaluate_scope_classification.py \
    --benchmark data/scope_benchmark.json \
    --human-review data/scope_benchmark_review.jsonl \
    --output results/scope_baseline.json \
    --provider azure
```

**Expected time**: ~10 minutes for 600 queries

#### 2.4 Run LitSearch Evaluation (Optional)

```bash
# Full evaluation (slow, ~2-3 hours)
python evaluation/run_evaluation.py \
    --queries data/litsearch_queries.json \
    --corpus data/litsearch_corpus.json \
    --output results/litsearch_baseline.json \
    --provider azure \
    --year-range 5y

# Quick test (10 queries, ~10 minutes)
python evaluation/run_evaluation.py \
    --queries data/litsearch_queries.json \
    --corpus data/litsearch_corpus.json \
    --output results/litsearch_test.json \
    --max-queries 10
```

### Phase 3: Iterative Improvement

```
┌──────────────────────────────────────────────────────┐
│ Improvement Loop                                     │
│                                                      │
│  1. Analyze Results                                  │
│  2. Identify Weaknesses                              │
│  3. Modify SYSTEM_PROMPT or Code                     │
│  4. Re-evaluate (Scope Benchmark - fast)             │
│  5. Validate with LitSearch (slow, periodic)         │
│  6. Repeat                                           │
└──────────────────────────────────────────────────────┘
```

#### 3.1 Analyze Baseline Results

```bash
# Scope classification
cat results/scope_baseline.json | jq '.metrics'

# Check misclassifications
cat results/scope_baseline.json | jq '.misclassification_analysis'
```

**Key questions**:
- Is the system too strict (high TOO_BROAD false positives)?
- Is the system too lenient (low TOO_BROAD recall)?
- Which component combinations cause confusion?

#### 3.2 Modify SYSTEM_PROMPT

Edit `agents/query_agent/query_analysis.py:45-170`

**Example changes**:
1. **Stricter TOO_BROAD**: Require [D]+[T] minimum for SEARCHABLE
2. **Looser TOO_BROAD**: Allow [D]+[M] as SEARCHABLE
3. **Better Type C detection**: Improve methodology vs. application domain distinction

#### 3.3 Quick Validation (Scope Benchmark)

```bash
python evaluation/evaluate_scope_classification.py \
    --benchmark data/scope_benchmark.json \
    --output results/scope_iteration_1.json \
    --provider azure

# Compare with baseline
diff results/scope_baseline.json results/scope_iteration_1.json
```

**Iteration time**: ~10 minutes

#### 3.4 Full Validation (LitSearch)

Once you're satisfied with scope classification:

```bash
python evaluation/run_evaluation.py \
    --queries data/litsearch_queries.json \
    --corpus data/litsearch_corpus.json \
    --output results/litsearch_iteration_1.json
```

**Validation time**: ~2-3 hours

### Phase 4: Multi-Configuration Testing

#### 4.1 Test Different LLM Providers

```bash
for provider in azure claude gemini; do
    # Scope benchmark
    python evaluation/evaluate_scope_classification.py \
        --benchmark data/scope_benchmark.json \
        --output results/scope_${provider}.json \
        --provider $provider

    # LitSearch benchmark (optional)
    python evaluation/run_evaluation.py \
        --queries data/litsearch_queries.json \
        --corpus data/litsearch_corpus.json \
        --output results/litsearch_${provider}.json \
        --provider $provider \
        --max-queries 50  # Subset for speed
done
```

#### 4.2 Test Different Year Filters

```bash
for year in auto 1y 3y 5y; do
    python evaluation/run_evaluation.py \
        --queries data/litsearch_queries.json \
        --corpus data/litsearch_corpus.json \
        --output results/litsearch_year_${year}.json \
        --year-range $year \
        --max-queries 50
done
```

#### 4.3 Compare Results

```bash
# Generate HTML comparison report
python evaluation/analyze_results.py \
    --results results/scope_*.json results/litsearch_*.json \
    --output results/comparison_report.html \
    --format both
```

### Phase 5: Reporting

#### 5.1 Generate Final Report

Create a summary of your findings:

```markdown
# Evaluation Report

## Scope Classification Results

- **Accuracy**: 0.853
- **Macro F1**: 0.837

### Per-Class Performance
- TOO_BROAD: P=0.92, R=0.88, F1=0.90
- SEARCHABLE: P=0.85, R=0.89, F1=0.87
- TOO_NARROW: P=0.76, R=0.73, F1=0.74

### Key Findings
1. System performs well on TOO_BROAD detection
2. Confusion between SEARCHABLE and TOO_NARROW (~15% misclassification)
3. Methodology-only queries (Type C) still challenging

## LitSearch Results

- **nDCG@10**: 0.42
- **Recall@10**: 0.37
- **MRR**: 0.33

### Comparison with Baselines
- Outperforms BM25 baseline (+0.07 nDCG)
- Competitive with dense retrieval systems
- Multi-source strategy adds +12% recall

## Recommendations

1. Improve TOO_NARROW boundary detection
2. Add more examples for methodology-only queries
3. Consider domain-specific prompts for biomedical vs CS
```

## 🎯 Success Criteria

### Scope Classification

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Overall Accuracy | 0.75 | 0.82 | 0.90 |
| TOO_BROAD F1 | 0.80 | 0.87 | 0.93 |
| SEARCHABLE F1 | 0.75 | 0.86 | 0.92 |
| TOO_NARROW F1 | 0.60 | 0.70 | 0.80 |

### LitSearch (End-to-End)

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| nDCG@10 | 0.35 | 0.40 | 0.50 |
| Recall@10 | 0.30 | 0.35 | 0.45 |
| MRR | 0.25 | 0.30 | 0.40 |

## 📊 Tracking Progress

Create a tracking sheet:

| Date | Iteration | Scope F1 | nDCG@10 | Changes Made |
|------|-----------|----------|---------|--------------|
| 2026-03-31 | baseline | 0.837 | 0.42 | Initial evaluation |
| 2026-04-02 | iter_1 | 0.851 | 0.44 | Improved Type C detection |
| 2026-04-05 | iter_2 | 0.863 | 0.45 | Added domain-specific rules |

## 🔄 Continuous Evaluation

### Weekly Check

```bash
# Quick scope check (10 minutes)
python evaluation/evaluate_scope_classification.py \
    --benchmark data/scope_benchmark.json \
    --output results/weekly_$(date +%Y%m%d).json \
    --max-samples 100
```

### Monthly Deep Dive

```bash
# Full evaluation suite (~3 hours)
python evaluation/evaluate_scope_classification.py \
  --benchmark data/scope_benchmark.json --output results/scope_full.json
python evaluation/run_evaluation.py \
  --queries data/litsearch_queries.json --corpus data/litsearch_corpus.json \
  --output results/litsearch_full.json
python evaluation/analyze_results.py --results results/*.json \
  --output results/comparison_report.html --format both
```

## 📚 Additional Resources

- [LitSearch Paper](https://arxiv.org/abs/2407.14228)
- [SemRank Paper](https://arxiv.org/abs/2505.21815) - Multi-granular concepts
- [CoQuest Paper](https://arxiv.org/abs/2310.06155) - Human-AI co-creation
- [sklearn.metrics docs](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Last updated**: 2026-03-31
