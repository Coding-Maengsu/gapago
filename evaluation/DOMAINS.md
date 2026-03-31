# Supported Scientific Domains

This document describes the scientific domains supported by the scope classification benchmark.

## 📚 arXiv Domain Codes

The benchmark uses arXiv's domain taxonomy to ensure broad coverage of **science and technology** fields.

### 1. Computer Science (cs)

**arXiv code**: `cs`

**Subcategories include**:
- Artificial Intelligence (cs.AI)
- Computer Vision (cs.CV)
- Machine Learning (cs.LG)
- Natural Language Processing (cs.CL)
- Robotics (cs.RO)
- Human-Computer Interaction (cs.HC)
- Software Engineering (cs.SE)
- Databases (cs.DB)
- Cryptography (cs.CR)

**Example queries**:
- TOO_BROAD: "computer vision"
- SEARCHABLE: "object detection in autonomous driving"
- TOO_NARROW: "few-shot learning for 3D point cloud segmentation in autonomous vehicles with limited labeled data"

### 2. Physics (physics)

**arXiv code**: `physics`

**Subcategories include**:
- Computational Physics (physics.comp-ph)
- Data Analysis (physics.data-an)
- Optics (physics.optics)
- Plasma Physics (physics.plasm-ph)
- Quantum Physics (quant-ph)
- Condensed Matter (cond-mat)

**Example queries**:
- TOO_BROAD: "quantum computing"
- SEARCHABLE: "quantum error correction for superconducting qubits"
- TOO_NARROW: "surface code implementation on IBM quantum processors with T1/T2 coherence times under 100μs"

### 3. Quantitative Biology (q-bio)

**arXiv code**: `q-bio`

**Subcategories include**:
- Biomolecules (q-bio.BM)
- Genomics (q-bio.GN)
- Neurons and Cognition (q-bio.NC)
- Quantitative Methods (q-bio.QM)
- Populations and Evolution (q-bio.PE)

**Example queries**:
- TOO_BROAD: "genomics"
- SEARCHABLE: "gene expression analysis in cancer cells"
- TOO_NARROW: "single-cell RNA-seq analysis for identifying rare cell populations in triple-negative breast cancer with batch effect correction"

### 4. Electrical Engineering and Systems Science (eess)

**arXiv code**: `eess`

**Subcategories include**:
- Audio and Speech Processing (eess.AS)
- Image and Video Processing (eess.IV)
- Signal Processing (eess.SP)
- Systems and Control (eess.SY)

**Example queries**:
- TOO_BROAD: "signal processing"
- SEARCHABLE: "speech enhancement in noisy environments"
- TOO_NARROW: "real-time multi-channel speech enhancement using transformer-based deep neural networks for cocktail party scenarios with 5+ speakers"

### 5. Mathematics (math)

**arXiv code**: `math`

**Subcategories include**:
- Optimization and Control (math.OC)
- Numerical Analysis (math.NA)
- Probability (math.PR)
- Statistics Theory (math.ST)
- Machine Learning (stat.ML)

**Example queries**:
- TOO_BROAD: "optimization"
- SEARCHABLE: "convex optimization for large-scale machine learning"
- TOO_NARROW: "stochastic gradient descent with momentum and adaptive learning rates for non-convex deep neural networks with batch normalization"

## 📊 Domain Distribution

For balanced benchmarks, we recommend:

### Small Benchmark (100 papers total)
- Each domain: 20 papers
- Total variants: ~300 (100 × 3)

### Medium Benchmark (250 papers total)
- Each domain: 50 papers
- Total variants: ~750 (250 × 3)

### Large Benchmark (500 papers total)
- Each domain: 100 papers
- Total variants: ~1,500 (500 × 3)

## 🎯 Domain-Specific Considerations

### High Specificity Domains (Narrow by Nature)

**q-bio, eess**: Research in these fields often naturally contains many components [D+T+M+P], making TOO_NARROW classification more common.

**Implication**: May need domain-specific thresholds or more lenient TOO_NARROW criteria.

### Broad Methodology Domains

**cs, math**: These fields often have methodology-focused papers where techniques are discussed without specific applications.

**Implication**: Type C (methodology-only) queries are more common and require careful detection.

### Multi-Domain Research

**physics**: Often intersects with other domains (e.g., computational physics, biophysics).

**Implication**: Queries may legitimately span multiple domains, requiring flexible classification.

## 🔧 Usage Examples

### Balanced Sampling (Recommended)

```bash
# Equal representation across all 5 domains
python evaluation/build_scope_benchmark.py sample \
    --domains cs physics q-bio eess math \
    --samples-per-domain 50 \
    --output data/arxiv_papers_sample.json
```

### Domain-Specific Benchmarks

```bash
# CS + Math only (methodology-heavy)
python evaluation/build_scope_benchmark.py sample \
    --domains cs math \
    --samples-per-domain 100 \
    --output data/arxiv_papers_cs_math.json

# Applied sciences (q-bio, eess)
python evaluation/build_scope_benchmark.py sample \
    --domains q-bio eess \
    --samples-per-domain 100 \
    --output data/arxiv_papers_applied.json
```

### Time Period Sampling

```bash
# Recent papers (2024-2026)
# Modify build_scope_benchmark.py:
search = arxiv.Search(
    query=f"cat:{domain}.* AND submittedDate:[20240101 TO 20261231]",
    max_results=samples_per_domain * 2,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)
```

## 📖 Domain-Specific Examples in Benchmark

Each domain should have representative examples covering:

### Domain Taxonomy
- Pure domain queries (TOO_BROAD)
- Domain + task (SEARCHABLE)
- Domain + task + constraints (TOO_NARROW or SEARCHABLE)

### Methodologies
- General methodology (TOO_BROAD)
- Methodology + application (SEARCHABLE)
- Methodology + application + modality + problem (TOO_NARROW)

### Applications
- General application area (TOO_BROAD)
- Application + specific task (SEARCHABLE)
- Application + task + data type + constraints (TOO_NARROW)

## 🎓 Academic Coverage

This domain selection covers:
- **~95%** of computational/quantitative science & technology research on arXiv
- **Major** AI/ML conferences (NeurIPS, ICML, ICLR, CVPR, ACL)
- **Core** natural sciences (physics, biology)
- **Engineering** applications (signal processing, control systems)
- **Mathematical** foundations (optimization, statistics)

**What's included**:
- Computer Science & Engineering
- Physics & Applied Physics
- Computational & Quantitative Biology
- Electrical Engineering & Systems
- Mathematics & Statistics

**What's excluded** (not science/technology):
- Economics (social science)
- Finance (business/social science)
- Humanities
- Pure social sciences

## 💡 Tips for Domain Selection

1. **Start Broad**: Use all 5 domains for initial benchmark
2. **Specialize Later**: Create domain-specific benchmarks as needed
3. **Balance by Complexity**: Mix methodology-heavy (cs, math) with application-heavy (q-bio, eess)
4. **Consider User Base**: If your users are primarily CS, weight cs more heavily
5. **Update Periodically**: Refresh papers yearly to capture new research trends

---

**Last updated**: 2026-03-31
