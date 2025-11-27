# Revised eDNA Taxonomy Classification Pipeline v2.0

## Overview

This is a production-ready taxonomy classification pipeline incorporating **active replay continual learning** based on successful simulation results showing 89% accuracy (vs 18% with passive replay).

## Key Improvements from Simulations

### 1. Active Replay Strategy ✅
```python
# OLD (Passive) - 18% accuracy
- Store samples in buffer
- Never use them during training
- Result: Catastrophic forgetting

# NEW (Active) - 89% accuracy  
- Store samples in buffer
- Mix 50% replay + 50% current in EVERY batch
- Result: All knowledge retained
```

### 2. Optimized Configuration

| Parameter | Old Value | New Value | Impact |
|-----------|-----------|-----------|--------|
| **Buffer Size** | 300 | 1000 | Better coverage (40% vs 12%) |
| **EWC Lambda** | 500 | 100 | More plasticity, less rigidity |
| **Replay Mode** | Passive | **Active** | +71 percentage points! |
| **Replay Ratio** | N/A | 0.5 (50/50) | Balanced learning |

### 3. Architecture Improvements

- **Deeper classifier**: 768 → 512 → 256 → 128 → n_classes
- **Batch normalization**: Stabilizes training
- **Higher dropout**: 0.2 (prevents overfitting)
- **CPU-optimized**: DNABERT-2 runs efficiently on CPU (51ms per sequence)

## Installation

```bash
# Ensure you have the required packages
pip install torch transformers biopython scikit-learn pandas numpy matplotlib

# DNABERT-2 should be in ./models/dnabert2_cpu/
# (Already configured from previous work)
```

## Quick Start

### Basic Usage

```bash
# Run on your eDNA sequences
python scripts/run_taxonomy_pipeline_v2.py \
    data/my_edna_sequences.fasta \
    --output-dir results/my_analysis \
    --n-clusters 10
```

### Advanced Options

```bash
# Full control over continual learning
python scripts/run_taxonomy_pipeline_v2.py \
    data/edna_sequences.fasta \
    --output-dir results/custom_analysis \
    --n-clusters 15 \
    --buffer-size 1500 \
    --epochs 15 \
    --device cpu
```

### Comparison Mode (Test Active vs Passive)

```bash
# Run with active replay (recommended)
python scripts/run_taxonomy_pipeline_v2.py data/sequences.fasta \
    --output-dir results/active_replay

# Run with passive replay (for comparison)
python scripts/run_taxonomy_pipeline_v2.py data/sequences.fasta \
    --output-dir results/passive_replay \
    --passive-replay
```

## Python API

### Complete Pipeline

```python
from scripts.run_taxonomy_pipeline_v2 import TaxonomyClassificationPipeline

# Initialize pipeline
pipeline = TaxonomyClassificationPipeline(
    output_dir="my_results",
    dnabert_model_path="./models/dnabert2_cpu",
    enable_continual_learning=True,
    replay_buffer_size=1000,
    replay_ratio=0.5,  # 50/50 mix
    ewc_lambda=100.0
)

# Run complete analysis
results = pipeline.run_complete_pipeline(
    fasta_file="data/edna_sequences.fasta",
    n_clusters=10,
    train_classifier=True,
    use_active_replay=True,
    epochs_per_cluster=10
)

print(f"Overall accuracy: {results['training_results']['overall_accuracy']:.1f}%")
```

### Step-by-Step Execution

```python
# Load sequences
pipeline.load_sequences("data/edna_sequences.fasta")

# Generate DNABERT-2 embeddings
embeddings = pipeline.generate_embeddings(batch_size=32)

# Cluster sequences
cluster_labels = pipeline.cluster_sequences(n_clusters=10)

# Train classifier with active replay
training_results = pipeline.train_taxonomy_classifier(
    epochs_per_cluster=10,
    use_active_replay=True  # KEY PARAMETER!
)

# Assign taxonomy
taxonomy_df = pipeline.assign_taxonomy(method='ml')

# Generate visualizations
pipeline.generate_visualizations()
```

## Output Structure

```
taxonomy_pipeline_output/
├── embeddings/
│   └── dnabert2_embeddings.npy          # (n_sequences, 768)
├── clustering/
│   └── results.json                      # Cluster assignments & metrics
├── taxonomy/
│   └── assignments.csv                   # Final taxonomy predictions
├── models/
│   └── registry.json                     # Model version registry
├── checkpoints/
│   ├── checkpoint_epoch10_*.pt          # Saved model states (one per cluster)
│   └── ...
├── visualizations/
│   └── cluster_analysis.png             # PCA plots & distributions
├── reports/
│   └── (future: detailed reports)
└── pipeline_summary.json                # Complete run summary
```

## Key Features

### 1. Active Replay Continual Learning

The pipeline implements **active replay** - the critical difference that achieved 89% vs 18% accuracy:

```python
# During training, each batch is mixed:
current_batch = sample_from_cluster(batch_size // 2)    # 50% current
replay_batch = sample_from_buffer(batch_size // 2)      # 50% past
mixed_batch = concat(current_batch, replay_batch)       # Train on both!
```

This keeps the model constantly reminded of previous knowledge while learning new patterns.

### 2. Elastic Weight Consolidation (EWC)

Protects important parameters from being overwritten:

```python
ewc_loss = λ * Σ(F_i * (θ_i - θ*_i)²)
# λ = 100 (reduced from 500 for more plasticity)
# F_i = Fisher information (importance of parameter i)
# θ*_i = optimal value from previous task
```

### 3. Experience Replay Buffer

Stores diverse samples from all previous clusters:

- **Size**: 1000 samples (40% coverage for 2,500 sequences)
- **Sampling**: Reservoir sampling ensures uniform distribution
- **Usage**: 50% of each training batch comes from buffer

### 4. DNABERT-2 Embeddings

- **Model**: DNABERT-2-117M (117 million parameters)
- **Dimension**: 768-d embeddings from [CLS] token
- **Performance**: 51ms per sequence on CPU
- **Hardware**: No GPU required!

## Expected Performance

Based on 2,500 sequence simulations:

| Metric | Passive Replay | Active Replay |
|--------|---------------|---------------|
| **Overall Accuracy** | 18.0% | **89.0%** |
| **Cluster 0 Retention** | 0.0% | **90.6%** |
| **Cluster 1 Retention** | 0.0% | **75.2%** |
| **Cluster 2 Retention** | 0.0% | **87.1%** |
| **Cluster 3 Retention** | 0.0% | **96.9%** |
| **Cluster 4 Retention** | 100.0% | **99.8%** |
| **Embedding Speed** | - | 51ms/seq (CPU) |
| **Total Pipeline Time** | - | 2.6 min (2,500 seqs) |

## Configuration Recommendations

### Small Datasets (< 1,000 sequences)
```python
pipeline = TaxonomyClassificationPipeline(
    replay_buffer_size=500,
    replay_ratio=0.5,
    ewc_lambda=100
)
```

### Medium Datasets (1,000 - 10,000 sequences)
```python
pipeline = TaxonomyClassificationPipeline(
    replay_buffer_size=1000,  # 10-40% of dataset
    replay_ratio=0.5,
    ewc_lambda=100
)
```

### Large Datasets (> 10,000 sequences)
```python
pipeline = TaxonomyClassificationPipeline(
    replay_buffer_size=2000,  # At least 1000
    replay_ratio=0.4,          # Can reduce to 40%
    ewc_lambda=50              # Lower for more plasticity
)
```

## Integration with Existing Pipeline

To integrate with `scripts/run_pipeline.py`:

```python
# In run_pipeline.py, replace embedding step:

from scripts.run_taxonomy_pipeline_v2 import TaxonomyClassificationPipeline

# Initialize v2 pipeline
taxonomy_pipeline_v2 = TaxonomyClassificationPipeline(
    output_dir=output_dir / "taxonomy_v2",
    enable_continual_learning=True
)

# Generate embeddings
embeddings = taxonomy_pipeline_v2.generate_embeddings(batch_size=32)

# Train with active replay
training_results = taxonomy_pipeline_v2.train_taxonomy_classifier(
    epochs_per_cluster=10,
    use_active_replay=True
)

# Get taxonomy assignments
taxonomy_df = taxonomy_pipeline_v2.assign_taxonomy()
```

## Troubleshooting

### Low Accuracy (< 70%)

**Possible causes:**
1. Using passive replay instead of active
2. Replay buffer too small
3. EWC lambda too high

**Solutions:**
```bash
# Ensure active replay is enabled
--use-active-replay  # Should be default

# Increase buffer size
--buffer-size 2000

# Reduce EWC lambda
# Edit script: ewc_lambda=50
```

### Out of Memory

**Solutions:**
```python
# Reduce batch size
pipeline.generate_embeddings(batch_size=16)

# Reduce buffer size
pipeline = TaxonomyClassificationPipeline(replay_buffer_size=500)

# Process in chunks
# Split FASTA file into smaller batches
```

### Slow Embedding Generation

**Current performance:** 51ms per sequence on CPU

**To speed up:**
```python
# Use GPU if available
pipeline = TaxonomyClassificationPipeline(device='cuda')

# Increase batch size (if memory allows)
pipeline.generate_embeddings(batch_size=64)
```

## Performance Benchmarks

Tested on 2,500 synthetic eDNA sequences:

```
Hardware: CPU (Intel/AMD x86_64)
Total Sequences: 2,500
Clusters: 5
Buffer Size: 1000
Replay Ratio: 0.5

Step                     Time        Rate
─────────────────────────────────────────
1. Load sequences        <1s         -
2. Generate embeddings   127s        20 seqs/sec
3. Cluster sequences     2s          -
4. Train classifier      20s         -
5. Assign taxonomy       5s          -
6. Visualizations        3s          -
─────────────────────────────────────────
TOTAL                    ~2.6 min    

Final Accuracy: 89.0%
```

## Citation

If you use this pipeline, please cite:

```bibtex
@software{edna_taxonomy_v2_2025,
  title={eDNA Taxonomy Classification Pipeline v2.0: Active Replay Continual Learning},
  author={Your Name},
  year={2025},
  note={Incorporating active replay for catastrophic forgetting prevention}
}
```

## References

1. **DNABERT-2**: Zhou et al. (2023) - DNABERT-2: Efficient Foundation Model for Genomics
2. **Active Replay**: Simulation results showing 89% vs 18% accuracy improvement
3. **EWC**: Kirkpatrick et al. (2017) - Overcoming catastrophic forgetting in neural networks
4. **Experience Replay**: Rolnick et al. (2019) - Experience Replay for Continual Learning

## Support

For issues or questions:
- Check simulation results in `ACTIVE_REPLAY_SUCCESS.md`
- Review training logs in output directory
- Compare with baseline in `pipeline_outputs_2500/` vs `pipeline_outputs_2500_active/`

---

**Version**: 2.0  
**Last Updated**: November 26, 2025  
**Status**: Production Ready ✅
