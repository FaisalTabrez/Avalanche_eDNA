# Configuration Guide

## Main Configuration File: `config/config.yaml`

The system uses a centralized YAML configuration file to manage all analysis parameters, file paths, and model settings. This allows for easy customization without modifying source code.

### Data Paths
```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  reference_dir: "data/reference"
  output_dir: "data/output"
```

### Preprocessing Parameters
```yaml
preprocessing:
  quality_threshold: 20        # Minimum average quality score
  min_length: 50              # Minimum sequence length
  max_length: 500             # Maximum sequence length
  adapter_sequences:          # Adapter sequences to remove
    - "AGATCGGAAGAGC"
    - "CTGTCTCTTATA"
```

### Model Configuration
```yaml
embedding:
  model_type: "transformer"   # or "autoencoder"
  kmer_size: 6               # K-mer size for tokenization
  max_sequence_length: 512   # Maximum sequence length for model
  embedding_dim: 256         # Embedding dimension

  transformer:
    num_layers: 6
    num_heads: 8
    dropout: 0.1
```

### Clustering Settings
```yaml
clustering:
  method: "hdbscan"          # "hdbscan", "kmeans", "dbscan"
  min_cluster_size: 10
  min_samples: 5
  metric: "euclidean"
```

### Taxonomy Assignment
```yaml
taxonomy:
  blast:
    database: "data/reference/nt"
    evalue: 1e-5
    max_targets: 10
    identity_threshold: 97.0
```

### Novelty Detection
```yaml
novelty:
  similarity_threshold: 0.85   # Similarity threshold for known taxa
  abundance_threshold: 0.001   # Minimum abundance for consideration
  cluster_coherence: 0.7       # Minimum cluster coherence
```

## Programmatic Configuration

You can also modify configuration values programmatically using the Config class:

```python
from utils.config import config

# Get configuration values
batch_size = config.get('embedding.training.batch_size', 32)
output_dir = config.get('data.output_dir')

# Set configuration values
config.set('clustering.method', 'hdbscan')
config.save()
```

**Methods:**
- `get(key: str, default: Any = None) -> Any`: Get configuration value
- `set(key: str, value: Any) -> None`: Set configuration value
- `save(path: Optional[str] = None) -> None`: Save configuration to file

## Performance Configuration

For large datasets or GPU acceleration:

```yaml
performance:
  use_gpu: true
  gpu_memory_fraction: 0.8
  n_jobs: -1  # Use all CPU cores
  chunk_size: 1000
```

## Custom Configuration Files

You can create custom configuration files for different analysis scenarios:

```bash
# Use custom config
python scripts/run_pipeline.py --config config/custom.yaml
```