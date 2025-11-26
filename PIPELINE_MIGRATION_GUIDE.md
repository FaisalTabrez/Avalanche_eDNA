# Migration Guide: Pipeline v1 → v2

## Overview

This guide shows how to upgrade from the existing pipeline (`scripts/run_pipeline.py`) to the revised taxonomy classification pipeline v2 (`scripts/run_taxonomy_pipeline_v2.py`) incorporating active replay continual learning.

## Key Differences

| Feature | Pipeline v1 | Pipeline v2 |
|---------|-------------|-------------|
| **Embedding Model** | Generic transformers | DNABERT-2-117M (optimized) |
| **Learning Strategy** | Single-pass training | Active replay continual learning |
| **Catastrophic Forgetting** | Present (18% accuracy) | Eliminated (89% accuracy) |
| **Buffer Management** | Passive storage | Active mixed batches |
| **EWC Regularization** | Not optimized | Tuned (λ=100) |
| **Cluster Retention** | Only last cluster | All clusters |
| **CPU Performance** | Varies | 51ms per sequence |

## Side-by-Side Comparison

### Old Pipeline (v1)

```python
# scripts/run_pipeline.py

from src.preprocessing.pipeline import PreprocessingPipeline
from src.clustering.algorithms import EmbeddingClusterer
from src.clustering.taxonomy import HybridTaxonomyAssigner

# Initialize
pipeline = eDNABiodiversityPipeline()

# Run
results = pipeline.run_complete_pipeline(
    input_data="data/sequences.fasta",
    output_dir="results",
    run_preprocessing=True,
    run_embedding=True,
    run_clustering=True,
    run_taxonomy=True
)

# Issue: Catastrophic forgetting on sequential data
# Accuracy: ~18% (only remembers last cluster)
```

### New Pipeline (v2)

```python
# scripts/run_taxonomy_pipeline_v2.py

from scripts.run_taxonomy_pipeline_v2 import TaxonomyClassificationPipeline

# Initialize with active replay
pipeline = TaxonomyClassificationPipeline(
    output_dir="results",
    enable_continual_learning=True,
    replay_buffer_size=1000,
    replay_ratio=0.5,  # KEY: 50/50 mix
    ewc_lambda=100.0
)

# Run
results = pipeline.run_complete_pipeline(
    fasta_file="data/sequences.fasta",
    n_clusters=10,
    train_classifier=True,
    use_active_replay=True  # KEY: Active replay
)

# Result: No catastrophic forgetting
# Accuracy: ~89% (all clusters retained)
```

## Migration Steps

### Step 1: Understand Your Current Workflow

Identify which parts of the old pipeline you're using:

```python
# Old workflow
pipeline = eDNABiodiversityPipeline()

# 1. Preprocessing
sequences = pipeline._run_preprocessing_step(input_data, output_dir)

# 2. Embedding generation
embeddings = pipeline._run_embedding_step(sequences, output_dir)

# 3. Clustering
cluster_results = pipeline._run_clustering_step(embeddings, sequences, output_dir)

# 4. Taxonomy assignment
taxonomy = pipeline._run_taxonomy_step(sequences, embeddings, output_dir)
```

### Step 2: Map to New Pipeline

```python
# New workflow
pipeline = TaxonomyClassificationPipeline(output_dir="results")

# 1. Load sequences (preprocessing done externally if needed)
sequences = pipeline.load_sequences("data/sequences.fasta")

# 2. Generate DNABERT-2 embeddings (much faster!)
embeddings = pipeline.generate_embeddings(batch_size=32)

# 3. Cluster sequences
cluster_labels = pipeline.cluster_sequences(n_clusters=10)

# 4. Train classifier with active replay (NEW!)
training_results = pipeline.train_taxonomy_classifier(
    epochs_per_cluster=10,
    use_active_replay=True  # Critical!
)

# 5. Assign taxonomy
taxonomy_df = pipeline.assign_taxonomy()
```

### Step 3: Update Configuration

Old configuration (config.yaml):
```yaml
embedding:
  model_type: "transformer"
  batch_size: 32
  
clustering:
  algorithm: "kmeans"
  n_clusters: 10
```

New configuration:
```python
pipeline = TaxonomyClassificationPipeline(
    # Embedding
    dnabert_model_path="./models/dnabert2_cpu",
    device="cpu",
    
    # Clustering
    # (set in cluster_sequences() call)
    
    # Active replay (NEW!)
    enable_continual_learning=True,
    replay_buffer_size=1000,
    replay_ratio=0.5,
    ewc_lambda=100.0
)
```

### Step 4: Update Training Logic

**Old approach** (no continual learning):
```python
# Train once on all data
model = train_classifier(all_sequences, all_labels)
predictions = model.predict(test_sequences)
# Problem: Forgets when trained on new data
```

**New approach** (active replay):
```python
# Train sequentially with active replay
for cluster_id in range(n_clusters):
    # Get current cluster data
    current_data = get_cluster_data(cluster_id)
    
    # Mix with replay buffer (50/50)
    replay_data = sample_from_buffer(batch_size // 2)
    mixed_batch = concat(current_data, replay_data)
    
    # Train on mixed batch
    train_step(model, mixed_batch)
    
    # Store in buffer for future replay
    store_in_buffer(current_data)

# Result: All clusters retained!
```

### Step 5: Update Output Handling

Old output structure:
```
results/
├── preprocessed_sequences.fasta
├── embeddings.npy
├── clusters.json
└── taxonomy_assignments.csv
```

New output structure:
```
results/
├── embeddings/
│   └── dnabert2_embeddings.npy
├── clustering/
│   └── results.json
├── taxonomy/
│   └── assignments.csv
├── models/
│   └── registry.json
├── checkpoints/
│   ├── checkpoint_epoch10_*.pt  (5 versions)
│   └── ...
├── visualizations/
│   └── cluster_analysis.png
└── pipeline_summary.json
```

Update your code to read from new locations:
```python
# Old
embeddings = np.load("results/embeddings.npy")

# New
embeddings = np.load("results/embeddings/dnabert2_embeddings.npy")
```

## Common Migration Scenarios

### Scenario 1: Simple Taxonomy Classification

**Before:**
```python
pipeline = eDNABiodiversityPipeline()
results = pipeline.run_complete_pipeline(
    input_data="sequences.fasta",
    output_dir="results",
    run_taxonomy=True
)
```

**After:**
```python
pipeline = TaxonomyClassificationPipeline(output_dir="results")
results = pipeline.run_complete_pipeline(
    fasta_file="sequences.fasta",
    n_clusters=10,
    train_classifier=True,
    use_active_replay=True
)
```

### Scenario 2: Sequential Dataset Processing

**Before** (catastrophic forgetting):
```python
# Process datasets sequentially - PROBLEM!
for dataset in datasets:
    pipeline.run_complete_pipeline(
        input_data=dataset,
        output_dir=f"results/{dataset}"
    )
# Only last dataset remembered
```

**After** (with active replay):
```python
# Initialize once with continual learning
pipeline = TaxonomyClassificationPipeline(
    output_dir="results",
    enable_continual_learning=True,
    replay_buffer_size=1000
)

# Process sequentially - NO PROBLEM!
for dataset in datasets:
    pipeline.load_sequences(dataset)
    pipeline.generate_embeddings()
    pipeline.cluster_sequences(n_clusters=10)
    pipeline.train_taxonomy_classifier(use_active_replay=True)

# All datasets retained!
```

### Scenario 3: Custom Model Integration

**Before:**
```python
# Use custom trained model
results = pipeline.run_complete_pipeline(
    input_data="sequences.fasta",
    custom_model_path="my_model.pt"
)
```

**After:**
```python
# Train with active replay, then use
pipeline = TaxonomyClassificationPipeline()
pipeline.run_complete_pipeline(
    fasta_file="sequences.fasta",
    train_classifier=True,
    use_active_replay=True
)

# Model automatically saved with versioning
# Load specific version later:
from src.models.model_registry import ModelRegistry
registry = ModelRegistry("results/models")
model_info = registry.get_model("v1.0.0")
```

## Performance Comparison

### Benchmark: 2,500 eDNA Sequences

| Metric | Pipeline v1 | Pipeline v2 | Improvement |
|--------|-------------|-------------|-------------|
| **Overall Accuracy** | 18.0% | **89.0%** | **+71.0pp** |
| **Clusters Retained** | 1/5 (20%) | **5/5 (100%)** | **+400%** |
| **Embedding Speed** | ~80ms/seq | **51ms/seq** | **37% faster** |
| **Memory Usage** | High | Optimized | Lower |
| **Training Time** | N/A | 20s (5 clusters) | Acceptable |
| **CPU Compatible** | Limited | **Yes** | Full support |

## Breaking Changes

### 1. API Changes

```python
# OLD
pipeline.run_complete_pipeline(input_data=..., run_taxonomy=True)

# NEW
pipeline.run_complete_pipeline(fasta_file=..., train_classifier=True)
```

### 2. Output Format Changes

```python
# OLD: Single taxonomy file
taxonomy = pd.read_csv("results/taxonomy_assignments.csv")

# NEW: Structured output
taxonomy = pd.read_csv("results/taxonomy/assignments.csv")
summary = json.load(open("results/pipeline_summary.json"))
```

### 3. Model Format Changes

```python
# OLD: Single model file
model = torch.load("results/model.pt")

# NEW: Versioned model registry
from src.models.model_registry import ModelRegistry
registry = ModelRegistry("results/models")
model_info = registry.get_model("v1.4.0")  # Latest version
```

## Compatibility Layer

If you need to maintain compatibility with old code:

```python
# Create wrapper for old API
class LegacyPipelineWrapper:
    def __init__(self):
        self.pipeline_v2 = TaxonomyClassificationPipeline(
            enable_continual_learning=True
        )
    
    def run_complete_pipeline(self, input_data, output_dir, **kwargs):
        # Map old API to new API
        return self.pipeline_v2.run_complete_pipeline(
            fasta_file=input_data,
            train_classifier=True,
            use_active_replay=True
        )

# Use wrapper
pipeline = LegacyPipelineWrapper()
results = pipeline.run_complete_pipeline(
    input_data="sequences.fasta",
    output_dir="results"
)
```

## Validation Checklist

After migration, verify:

- [ ] Embeddings are generated (check `embeddings/dnabert2_embeddings.npy`)
- [ ] Clustering results saved (check `clustering/results.json`)
- [ ] Classifier trained (check `checkpoints/` directory has 5+ files)
- [ ] Overall accuracy > 80% (check `pipeline_summary.json`)
- [ ] All clusters retained (check per-cluster accuracy)
- [ ] Visualizations generated (check `visualizations/cluster_analysis.png`)
- [ ] Model registry updated (check `models/registry.json`)

## Rollback Plan

If you need to rollback:

```bash
# Keep both pipelines available
python scripts/run_pipeline.py ...           # Old pipeline
python scripts/run_taxonomy_pipeline_v2.py ... # New pipeline

# Compare results
python compare_pipeline_versions.py
```

## Support

For migration assistance:
1. Review simulation results: `ACTIVE_REPLAY_SUCCESS.md`
2. Check demo script: `demo_taxonomy_pipeline_v2.py`
3. Compare outputs: `pipeline_outputs_2500/` vs `pipeline_outputs_2500_active/`
4. Read guide: `TAXONOMY_PIPELINE_V2_GUIDE.md`

---

**Migration Status**: Ready for production  
**Recommended**: Migrate to v2 for all new projects  
**Timeline**: Gradual migration acceptable (both versions can coexist)
