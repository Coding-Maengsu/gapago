# Evaluation Framework for GAPAGO

> ### ⚠️ 실행 전 알아두세요
>
> - 이 벤치마크는 **외부 데이터셋(LitSearch)** 이 필요하며 저장소에 포함돼 있지 않습니다
>   (`.gitignore` 가 `data/*` 를 제외). 아래 Quick Start 의 준비 단계를 먼저 수행하세요.
> - CI 에서 실행되지 않습니다. 수동 실행 도구입니다.
> - **2026-08 이전에 생성된 retrieval 지표(nDCG/Recall/MRR)는 신뢰할 수 없습니다.**
>   `paper_retrieval_node` 를 `await` 없이 호출하던 버그로 값이 항상 0 이었습니다.
>   scope 분류 지표는 영향받지 않았습니다.

This directory contains two complementary evaluation approaches:

1. **LitSearch Benchmark**: End-to-end evaluation of Query + Retrieval Agents
2. **Scope Classification Benchmark**: Targeted evaluation of Query Agent's scope assessment (TOO_BROAD/SEARCHABLE/TOO_NARROW)

---

## 📚 Part 1: LitSearch Benchmark (End-to-End)

**LitSearch: A Retrieval Benchmark for Scientific Literature Search**
- Paper: https://arxiv.org/abs/2407.14228
- Dataset: https://github.com/yale-nlp/LitSearch

LitSearch is a benchmark designed to evaluate retrieval systems for scientific literature. It contains:
- Real information needs from scientists
- Queries across multiple domains (CS, biomedical, physics, etc.)
- Relevance judgments for retrieved papers
- Multi-level relevance scores (0-3 scale)

### 🚀 Quick Start (LitSearch)

### 1. Download LitSearch Dataset

```bash
# Clone LitSearch repository
git clone https://github.com/yale-nlp/LitSearch.git

# Or download directly from their releases
wget https://github.com/yale-nlp/LitSearch/releases/download/v1.0/litsearch_data.zip
unzip litsearch_data.zip
```

### 2. Convert to GAPAGO Format

```bash
# Convert LitSearch format to GAPAGO evaluation format
python evaluation/prepare_litsearch_data.py \
    --input LitSearch/data/queries.jsonl \
    --corpus LitSearch/data/corpus.jsonl \
    --output-queries data/litsearch_queries.json \
    --output-corpus data/litsearch_corpus.json
```

### 3. Run Evaluation

```bash
# Run full evaluation
python evaluation/run_evaluation.py \
    --queries data/litsearch_queries.json \
    --corpus data/litsearch_corpus.json \
    --output results/litsearch_eval.json \
    --provider azure \
    --year-range 5y

# Test with subset (first 10 queries)
python evaluation/run_evaluation.py \
    --queries data/litsearch_queries.json \
    --corpus data/litsearch_corpus.json \
    --output results/litsearch_eval_test.json \
    --max-queries 10
```

## 📊 Evaluation Metrics

### Query Agent Metrics

- **Specificity Score**: Measures how much the refined query improves over the original
- **Keyword Quality**: BM25-based evaluation of extracted keywords against ground truth
- **Scope Assessment Accuracy**: Whether TOO_BROAD/SEARCHABLE/TOO_NARROW is correctly identified
- **Searchable Rate**: Percentage of queries successfully refined to SEARCHABLE level

### Retrieval Agent Metrics

- **nDCG@k** (k=5, 10, 20): Normalized Discounted Cumulative Gain
- **Recall@k**: Percentage of relevant papers found in top-k results
- **Precision@k**: Percentage of top-k results that are relevant
- **MRR**: Mean Reciprocal Rank of first relevant paper

### Overall Score

Weighted combination:
- 30% Query specificity
- 30% Searchable rate
- 40% Retrieval nDCG@10

## 📁 File Structure

```
evaluation/
├── README.md                           # This file
│
├── LitSearch Benchmark (End-to-End)
│   ├── litsearch_benchmark.py          # Core evaluation framework
│   ├── run_evaluation.py               # Evaluation runner script
│   ├── prepare_litsearch_data.py       # Data format converter
│   └── analyze_results.py              # Results analysis and visualization
│
└── Scope Classification Benchmark
    ├── build_scope_benchmark.py        # Generate scope classification dataset
    ├── evaluate_scope_classification.py # Run classification evaluation
    └── review_scope_labels.py          # Human review interface
```

## 🔧 Advanced Usage

### Custom Evaluation

```python
from evaluation.litsearch_benchmark import (
    LitSearchDataLoader,
    QueryReformulationEvaluator,
    RetrievalEvaluator,
    EndToEndEvaluator
)

# Load data
loader = LitSearchDataLoader()
queries = loader.load_queries("data/litsearch_queries.json")
corpus = loader.load_paper_corpus("data/litsearch_corpus.json")

# Evaluate query reformulation only
query_eval = QueryReformulationEvaluator()
spec_score = query_eval.evaluate_specificity(
    original="deep learning",
    refined="deep learning for medical image segmentation"
)

# Evaluate retrieval only
retrieval_eval = RetrievalEvaluator(k_values=[5, 10, 20])
metrics = retrieval_eval.evaluate_single_query(
    retrieved=["paper1", "paper2", "paper3"],
    relevance={"paper1": 3.0, "paper2": 1.0, "paper4": 2.0}
)
```

### Ablation Studies

Test different configurations:

```bash
# Test different year filters
for year in auto 1y 3y 5y; do
    python evaluation/run_evaluation.py \
        --queries data/litsearch_queries.json \
        --corpus data/litsearch_corpus.json \
        --output results/eval_${year}.json \
        --year-range $year
done

# Test different LLM providers
for provider in azure claude gemini; do
    python evaluation/run_evaluation.py \
        --queries data/litsearch_queries.json \
        --corpus data/litsearch_corpus.json \
        --output results/eval_${provider}.json \
        --provider $provider
done
```

### Analyze Results

```bash
# Generate comparison report
python evaluation/analyze_results.py \
    --results results/eval_*.json \
    --output results/comparison_report.html
```

## 🎯 Fine-tuning with LitSearch

### 1. Generate Training Data

Use LitSearch to create training examples for query reformulation:

```python
from evaluation.prepare_litsearch_data import generate_training_data

# Create query reformulation training pairs
train_data = generate_training_data(
    queries_path="data/litsearch_queries.json",
    output_path="data/query_reformulation_train.jsonl"
)

# Format:
# {
#   "input": "vague query",
#   "output": {
#     "scope_level": "TOO_BROAD",
#     "breadth_candidates": [...]
#   }
# }
```

### 2. Fine-tune Query Analysis

```bash
# Fine-tune on query reformulation task
python training/finetune_query_agent.py \
    --train-data data/query_reformulation_train.jsonl \
    --model gpt-4 \
    --output models/query_agent_finetuned
```

### 3. Evaluate Fine-tuned Model

```bash
# Compare original vs fine-tuned
python evaluation/run_evaluation.py \
    --queries data/litsearch_queries.json \
    --corpus data/litsearch_corpus.json \
    --output results/eval_finetuned.json \
    --model models/query_agent_finetuned
```

## 📈 Expected Performance

Based on LitSearch baseline systems:

| Metric | Random | BM25 | Dense Retrieval | Expected GAPAGO |
|--------|--------|------|----------------|-----------------|
| nDCG@10 | 0.15 | 0.35 | 0.45 | **0.40+** |
| Recall@10 | 0.10 | 0.30 | 0.40 | **0.35+** |
| MRR | 0.08 | 0.28 | 0.38 | **0.30+** |

GAPAGO's advantage:
- ✅ Multi-source retrieval (arXiv + Semantic Scholar + OpenAlex + ScienceON)
- ✅ LLM-based query refinement
- ✅ Adaptive BM25 + LLM reranking
- ✅ Full-text filtering

## 🐛 Troubleshooting

### Issue: "Module not found"

```bash
# Make sure you're in the project root
cd AI-Co-Scientist-Challenge-New
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or use relative imports
python -m evaluation.run_evaluation ...
```

### Issue: "API rate limit exceeded"

```bash
# Reduce evaluation batch size
python evaluation/run_evaluation.py \
    --max-queries 50 \
    --delay 2  # Add 2 second delay between queries
```

### Issue: "Paper not found in corpus"

LitSearch uses Semantic Scholar paper IDs. Make sure your retrieval agent returns papers in the format `s2:PAPER_ID` or maps arXiv IDs to Semantic Scholar IDs.

## 📖 References

```bibtex
@article{litsearch2024,
  title={LitSearch: A Retrieval Benchmark for Scientific Literature Search},
  author={...},
  journal={arXiv preprint arXiv:2407.14228},
  year={2024}
}

@inproceedings{semrank2025,
  title={SemRank: Semantic-aware Re-ranking with Multi-granular Scientific Concepts},
  author={Zhang et al.},
  booktitle={EMNLP},
  year={2025}
}

@inproceedings{coquest2024,
  title={CoQuest: Co-creating Research Questions with AI},
  author={Liu et al.},
  booktitle={CHI},
  year={2024}
}
```

---

## 🎯 Part 2: Scope Classification Benchmark

**LitSearch-inspired approach for targeted Query Agent evaluation**

This benchmark specifically tests the Query Agent's ability to correctly classify queries into:
- **TOO_BROAD**: Only domain [D], no specific task or problem
- **SEARCHABLE**: Domain + task [D+T], or other valid combinations
- **TOO_NARROW**: All components [D+T+M+P], overly specific

### Why This Matters

The scope classification is the **first and most critical decision** in the pipeline:
- ❌ **TOO_BROAD** → System requests refinement (poor UX if wrong)
- ✅ **SEARCHABLE** → Proceeds to retrieval (must be accurate)
- ⚠️ **TOO_NARROW** → Suggests broadening (helpful if correct)

**Key advantage**: Direct evaluation with ground truth labels, unlike LitSearch which only measures end-to-end performance.

### 🚀 Quick Start (Scope Benchmark)

#### Step 1: Sample arXiv Papers

```bash
# Sample papers across 7 major scientific domains
python evaluation/build_scope_benchmark.py sample \
    --domains cs physics q-bio eess math \
    --samples-per-domain 50 \
    --output data/arxiv_papers_sample.json
```

#### Step 2: Generate Query Variants

For each paper, LLM generates 3 query variants:
- **Broad**: [D] only → Ground truth: TOO_BROAD
- **Searchable**: [D+T] → Ground truth: SEARCHABLE
- **Narrow**: [D+T+M+P] → Ground truth: TOO_NARROW or SEARCHABLE

```bash
python evaluation/build_scope_benchmark.py build \
    --papers-json data/arxiv_papers_sample.json \
    --output data/scope_benchmark.json \
    --num-samples 200 \
    --provider azure
```

This creates:
- `data/scope_benchmark.json` - Full benchmark dataset
- `data/scope_benchmark_review.jsonl` - Human review format

#### Step 3: Human Review (Recommended)

Verify/correct a sample of generated labels:

```bash
# Interactive review session
python evaluation/review_scope_labels.py review \
    --input data/scope_benchmark_review.jsonl \
    --output data/scope_benchmark_review_updated.jsonl

# Check progress
python evaluation/review_scope_labels.py summary \
    --input data/scope_benchmark_review_updated.jsonl
```

**Review interface**:
```
================================================================
Item 45/600
================================================================

Query ID: arxiv:2301.12345_ctx0_broad_query

Query:
  computer vision

Suggested Label: TOO_BROAD

Components:
  [D] domain    : computer vision
  [ ] task      : (none)
  [ ] modality  : (none)
  [ ] problem   : (none)

Source Context:
  Despite recent advances in deep learning for image classification...

Options:
  [1] TOO_BROAD    [2] SEARCHABLE    [3] TOO_NARROW
  [a] ACCEPT       [s] SKIP          [n] ADD NOTE    [q] QUIT
```

#### Step 4: Run Evaluation

```bash
# Evaluate with human-reviewed labels
python evaluation/evaluate_scope_classification.py \
    --benchmark data/scope_benchmark.json \
    --human-review data/scope_benchmark_review_updated.jsonl \
    --output results/scope_classification_eval.json \
    --provider azure
```

**Output metrics**:
```
Overall Metrics:
  Accuracy:        0.8533
  Macro Precision: 0.8421
  Macro Recall:    0.8312
  Macro F1:        0.8366

Per-Class Metrics:
  TOO_BROAD:
    Precision: 0.92
    Recall:    0.88
    F1:        0.90
    Support:   200

  SEARCHABLE:
    Precision: 0.85
    Recall:    0.89
    F1:        0.87
    Support:   250

  TOO_NARROW:
    Precision: 0.76
    Recall:    0.73
    F1:        0.74
    Support:   150
```

### Evaluation Metrics

**Classification Metrics**:
- **Accuracy**: Overall correct classification rate
- **Precision/Recall/F1** per class (TOO_BROAD, SEARCHABLE, TOO_NARROW)
- **Confusion Matrix**: Where does the model get confused?

**Error Analysis**:
- **Misclassification patterns**: Which transitions are most common?
  - Example: TOO_BROAD → SEARCHABLE (false positive, too lenient)
  - Example: SEARCHABLE → TOO_BROAD (false negative, too strict)
- **Component accuracy**: Performance by # of components (1, 2, 3, 4)

### Example Use Cases

#### 1. Test SYSTEM_PROMPT changes

```bash
# Baseline
python evaluation/evaluate_scope_classification.py \
    --benchmark data/scope_benchmark.json \
    --output results/baseline.json

# After modifying SYSTEM_PROMPT in query_analysis.py
python evaluation/evaluate_scope_classification.py \
    --benchmark data/scope_benchmark.json \
    --output results/modified_prompt.json

# Compare
diff results/baseline.json results/modified_prompt.json
```

#### 2. Test different LLM providers

```bash
for provider in azure claude gemini; do
    python evaluation/evaluate_scope_classification.py \
        --benchmark data/scope_benchmark.json \
        --output results/scope_${provider}.json \
        --provider $provider
done
```

#### 3. Ablation: Component thresholds

Test different thresholds for TOO_BROAD detection:
- Current: "lacks at least one of [D], [T], [M]"
- Stricter: "lacks any two of [D], [T], [M], [P]"
- Looser: "lacks all of [T], [M], [P]"

Modify SYSTEM_PROMPT and re-evaluate.

### Expected Performance

Target metrics based on SemRank paper and our SYSTEM_PROMPT design:

| Class | Target Precision | Target Recall | Target F1 |
|-------|-----------------|---------------|-----------|
| TOO_BROAD | **0.90+** | **0.85+** | **0.87+** |
| SEARCHABLE | **0.85+** | **0.88+** | **0.86+** |
| TOO_NARROW | **0.70+** | **0.70+** | **0.70+** |
| **Overall** | **0.82+** | **0.81+** | **0.82+** |

**Why lower for TOO_NARROW?**
- Boundary between SEARCHABLE and TOO_NARROW is fuzzy
- Real-world papers may exist even for "narrow" queries
- Conservative approach preferred (better to search than reject)

### Comparison: LitSearch vs Scope Benchmark

| Aspect | LitSearch | Scope Benchmark |
|--------|-----------|-----------------|
| **What it measures** | End-to-end retrieval | Query classification only |
| **Ground truth** | Relevance judgments | Scope labels (TOO_BROAD/etc) |
| **Metrics** | nDCG, Recall, MRR | Accuracy, Precision, F1 |
| **Setup cost** | High (need paper corpus) | Medium (need LLM generation) |
| **Iteration speed** | Slow (full retrieval) | Fast (classification only) |
| **Best for** | System validation | Prompt engineering |

**Recommendation**: Use **both**
1. **Scope Benchmark** for rapid iteration on SYSTEM_PROMPT
2. **LitSearch** for final end-to-end validation

---

## 💡 Tips

1. **Start Small**: Test with `--max-queries 10` (LitSearch) or `--max-samples 50` (Scope) first
2. **Monitor Costs**: LLM calls can add up; consider using cheaper models for initial testing
3. **Cache Results**: Retrieval results can be cached to avoid redundant API calls
4. **Domain-Specific**: Evaluate per-domain (CS vs biomedical) for insights
5. **Iterate**: Use results to identify weaknesses and improve prompts
6. **Human Review**: Review at least 10-20% of scope labels to ensure quality

## 🤝 Contributing

Found a bug or have an improvement? Please open an issue or PR!

---

**Last updated**: 2026-03-31
