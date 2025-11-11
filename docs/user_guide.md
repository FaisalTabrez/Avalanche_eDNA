# Deep-Sea eDNA Biodiversity Assessment System - User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Running the Pipeline](#running-the-pipeline)
4. [Using the Dashboard](#using-the-dashboard)
5. [API Reference](#api-reference)

> **Note:** For installation instructions, see [Installation Guide](installation.md). For configuration options, see [Configuration Guide](configuration.md). For troubleshooting, see [Troubleshooting Guide](troubleshooting.md).

## Quick Start

### 1. Create Sample Data
```bash
python scripts/run_pipeline.py --create-sample --input data/sample --output results/sample_analysis
```

### 2. Run Complete Analysis
```bash
python scripts/run_pipeline.py --input data/sample/sample_edna_sequences.fasta --output results/sample_analysis
```

### 3. Launch Dashboard
```bash
streamlit run src/visualization/dashboard.py
```

## Data Preparation

### Supported File Formats
- **FASTQ**: Raw sequencing data with quality scores
- **FASTA**: Processed sequences without quality scores

### Data Requirements
- Sequences should be 50-500 bp in length
- Quality scores ≥20 recommended for FASTQ files
- Remove adapters and primers before analysis

### Example Data Structure
```
data/
├── raw/
│   ├── sample1.fastq
│   ├── sample2.fastq
│   └── sample3.fastq
├── processed/
└── reference/
    ├── nt/                    # BLAST nucleotide database
    └── silva_138.1.fasta      # SILVA reference
```

## Running the Pipeline

### Command Line Interface

The main pipeline script provides a complete end-to-end analysis:

```bash
python scripts/run_pipeline.py [OPTIONS]
```

#### Basic Usage
```bash
# Analyze single file
python scripts/run_pipeline.py --input data/sequences.fasta --output results/analysis1

# Analyze directory of files
python scripts/run_pipeline.py --input data/raw/ --output results/batch_analysis

# Skip specific steps
python scripts/run_pipeline.py --input data/sequences.fasta --output results/analysis2 --skip-preprocessing --skip-taxonomy
```

#### Pipeline Steps

1. **Preprocessing**: Quality filtering, adapter trimming, chimera removal
2. **Embedding**: Generate sequence embeddings using deep learning
3. **Clustering**: Group similar sequences into taxonomic units
4. **Taxonomy**: Assign taxonomic labels using BLAST and ML
5. **Novelty**: Detect potentially novel taxa
6. **Visualization**: Generate plots and interactive dashboard

### Configuration

Modify `config/config.yaml` to customize analysis parameters:

```yaml
# Example configuration
preprocessing:
  quality_threshold: 20
  min_length: 50
  max_length: 500

embedding:
  model_type: "transformer"
  embedding_dim: 256
  kmer_size: 6

clustering:
  method: "hdbscan"
  min_cluster_size: 10

novelty:
  similarity_threshold: 0.85
  abundance_threshold: 0.001
```

## Using the Dashboard

### Launching the Dashboard
```bash
streamlit run src/visualization/dashboard.py
```

The dashboard will be available at `http://localhost:8501`

### Dashboard Features

#### 1. Data Upload
- Upload FASTQ/FASTA files
- Load from directory
- Use sample datasets

#### 2. Preprocessing
- Configure quality filters
- Set sequence length limits
- Enable chimera detection

#### 3. Analysis
- Choose clustering algorithms
- Set novelty detection thresholds
- Configure taxonomy assignment

#### 4. Results Visualization
- Interactive cluster plots
- Taxonomic composition charts
- Novelty detection summaries
- Biodiversity metrics

#### 5. Export Options
- Download results as CSV/JSON
- Generate analysis reports
- Export visualizations





## API Reference

### Core Classes

#### DNATokenizer
```python
from models.tokenizer import DNATokenizer

tokenizer = DNATokenizer(encoding_type="kmer", kmer_size=6)
encoded = tokenizer.encode_sequence("ATCGATCGATCG")
```

#### EmbeddingClusterer
```python
from clustering.algorithms import EmbeddingClusterer

clusterer = EmbeddingClusterer(method="hdbscan")
labels = clusterer.fit(embeddings)
```

#### NoveltyAnalyzer
```python
from novelty.detection import NoveltyAnalyzer

analyzer = NoveltyAnalyzer()
results = analyzer.analyze_novelty(query_embeddings, reference_embeddings, sequences)
```

### Pipeline Integration

#### Custom Pipeline
```python
from scripts.run_pipeline import eDNABiodiversityPipeline

pipeline = eDNABiodiversityPipeline()
results = pipeline.run_complete_pipeline(
    input_data="data/sequences.fasta",
    output_dir="results/custom_analysis"
)
```

### Example Workflows

#### 1. Basic Analysis
```python
import numpy as np
from models.tokenizer import DNATokenizer
from clustering.algorithms import EmbeddingClusterer

# Prepare sequences
sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA", "TTAATTAATTAA"]

# Tokenize
tokenizer = DNATokenizer()
encoded = tokenizer.encode_sequences(sequences)

# Generate mock embeddings (replace with actual model)
embeddings = np.random.randn(len(sequences), 256)

# Cluster
clusterer = EmbeddingClusterer()
labels = clusterer.fit(embeddings)
```

#### 2. Novelty Detection
```python
from novelty.detection import NoveltyDetector

# Prepare reference and query data
reference_embeddings = np.random.randn(100, 256)
query_embeddings = np.random.randn(20, 256)

# Detect novelty
detector = NoveltyDetector(method="isolation_forest")
detector.fit(reference_embeddings)
predictions = detector.predict(query_embeddings)
```

#### 3. Visualization
```python
from visualization.plots import BiodiversityPlotter

plotter = BiodiversityPlotter()

# Plot taxonomic composition
taxonomy_counts = {"Bacteria": 500, "Archaea": 200, "Eukaryota": 150}
fig = plotter.plot_taxonomic_composition(taxonomy_counts)
fig.show()
```

## Advanced Usage

### Custom Models

#### Training Custom Embedding Model
```python
from models.trainer import EmbeddingTrainer
from models.embeddings import DNATransformerEmbedder

# Create model
model = DNATransformerEmbedder(vocab_size=tokenizer.vocab_size)

# Train
trainer = EmbeddingTrainer(model, tokenizer)
train_loader, val_loader = trainer.prepare_data(sequences)
history = trainer.train_contrastive(train_loader, val_loader, epochs=50)
```

#### Custom Clustering
```python
from clustering.algorithms import EmbeddingClusterer

# Custom HDBSCAN parameters
clusterer = EmbeddingClusterer(
    method="hdbscan",
    min_cluster_size=20,
    min_samples=10,
    cluster_selection_epsilon=0.5
)
```

### Batch Processing

#### Process Multiple Datasets
```python
import glob
from pathlib import Path

pipeline = eDNABiodiversityPipeline()

for fasta_file in glob.glob("data/samples/*.fasta"):
    output_dir = f"results/{Path(fasta_file).stem}"
    results = pipeline.run_complete_pipeline(fasta_file, output_dir)
    print(f"Processed {fasta_file}: {results['summary']}")
```

### Cloud Deployment

#### Docker Setup
```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["streamlit", "run", "src/visualization/dashboard.py"]
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edna-dashboard
spec:
  replicas: 1
  selector:
    matchLabels:
      app: edna-dashboard
  template:
    metadata:
      labels:
        app: edna-dashboard
    spec:
      containers:
      - name: dashboard
        image: edna-biodiversity:latest
        ports:
        - containerPort: 8501
```

## Support

### Getting Help
- Check the troubleshooting section
- Review example notebooks in `notebooks/`
- Run test suite: `python tests/test_system.py`
- Check logs in `logs/` directory

### Contributing
- Fork the repository
- Create feature branch
- Add tests for new functionality
- Submit pull request

### Citation
If you use this system in your research, please cite:
```
eDNA Biodiversity Assessment System. (2024). 
Deep-Sea Environmental DNA Analysis Platform.
```