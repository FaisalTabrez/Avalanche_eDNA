# Deep-Sea eDNA Biodiversity Assessment System

An end-to-end system for identifying taxonomic diversity and assessing biological richness in deep-sea environmental DNA (eDNA) datasets using advanced machine learning and bioinformatics techniques.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🌊 Overview

This system addresses the challenges of deep-sea eDNA analysis by:
- Processing massive, complex eDNA datasets efficiently
- Discovering novel taxa without relying solely on existing reference databases
- Providing scalable, accurate taxonomic classification
- Offering intuitive visualization and analysis tools

## 🔧 Features

- **Data Preprocessing Pipeline**: Quality filtering, adapter trimming, chimera removal
- **Transformer Embeddings**: Nucleotide Transformer (HF) with chunked mean-pooling, optional PCA to 256 dims, and L2 normalization
- **Advanced Clustering**: Unsupervised taxonomic grouping with novelty detection
- **Interactive Dashboard**: Web-based visualization and analysis interface
- **Scalable Architecture**: GPU acceleration and cloud deployment ready
- **NCBI SRA Integration**: Direct access to 1000+ eDNA studies from NCBI Sequence Read Archive
- **Multi-Format Support**: Universal support for FASTA, FASTQ, Swiss-Prot, GenBank, EMBL, and SRA formats
- **Real-time Processing**: Live progress tracking with interactive visualizations

## 📁 Project Structure

```
├── data/                        # Sample datasets and test data
├── src/                         # Source code
│   ├── analysis/                # Dataset analysis utilities
│   ├── api/                     # Report management API
│   ├── clustering/              # Clustering algorithms and taxonomy helpers
│   ├── dashboards/              # Streamlit dashboard modules
│   ├── database/                # Database models and manager
│   ├── novelty/                 # Novelty detection logic
│   ├── organism_profiling/      # Organism profiling modules
│   ├── preprocessing/           # Data cleaning and preparation
│   ├── report_management/       # Report/catalogue management
│   ├── similarity/              # Cross-analysis engine
│   ├── utils/                   # Shared utilities and config
│   └── visualization/           # Plotting and dashboard utilities
├── notebooks/                   # Jupyter notebooks for analysis
├── tests/                       # Unit and integration tests
├── docs/                        # Documentation
├── scripts/                     # Pipeline and automation scripts
├── streamlit_app.py             # Streamlit UI entrypoint
└── requirements*.txt            # Python dependencies
```

> Note: The current pipeline uses placeholder embeddings and a demo ML taxonomy classifier trained on synthetic data. Replace the embedding step with real models and training when src/models is introduced.

## Installation

See [Installation Guide](docs/installation.md) for detailed setup instructions including prerequisites, environment setup, and optional dependencies.

## 🚀 Quick Start

Note: The default embedding backend uses a pretrained Nucleotide Transformer from Hugging Face. The first run will download the model weights to your local cache. Embedding post-processing (optional PCA to 256 and L2 normalization) is configurable in config/config.yaml under embedding.postprocess.

1. **Create Sample Data and Run Analysis**
    ```bash
    # Create sample eDNA dataset
    python scripts/run_pipeline.py --create-sample --input data/sample --output results/demo

    # Run complete analysis pipeline
    python scripts/run_pipeline.py --input data/sample/sample_edna_sequences.fasta --output results/demo
    ```

2. **Launch Interactive Dashboard**
    ```bash
    python scripts/launch_dashboard.py
    ```
    Then open http://localhost:8504 in your browser

3. **View Results**
    ```bash
    # Results are saved in results/demo/
    # - pipeline_results.json: Complete analysis results
    # - visualizations/: Interactive plots
    # - clustering/: Clustering analysis
    # - taxonomy/: Taxonomic assignments
    # - novelty/: Novel taxa detection
    ```

## Documentation

- [User Guide](docs/user_guide.md) - Complete usage instructions and tutorials
- [API Reference](docs/api_reference.md) - Detailed API documentation
- [Configuration](docs/configuration.md) - Configuration options and examples
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## 📊 Usage

### Command Line Interface

```bash
# Complete end-to-end analysis
python scripts/run_pipeline.py --input sequences.fasta --output results/

# Skip specific steps
python scripts/run_pipeline.py --input sequences.fasta --output results/ --skip-preprocessing

# Create sample data for testing
python scripts/run_pipeline.py --create-sample --input data/sample --output results/demo
```

### Interactive Dashboard

```bash
# Launch web interface
python scripts/launch_dashboard.py
```

### Python API

```python
from scripts.run_pipeline import eDNABiodiversityPipeline

# Initialize pipeline
pipeline = eDNABiodiversityPipeline()

# Run analysis
results = pipeline.run_complete_pipeline(
    input_data="sequences.fasta",
    output_dir="results/analysis"
)

print(f"Found {results['summary']['novel_taxa_candidates']} novel taxa candidates")
```

### Example Analysis Workflow

1. **Data Upload**: Load FASTQ/FASTA files
2. **Preprocessing**: Quality filtering, adapter removal, chimera detection
3. **Embedding Generation**: Deep learning sequence representations
4. **Clustering**: Group sequences into taxonomic units
5. **Taxonomy Assignment**: BLAST + ML classification
6. **Novelty Detection**: Identify potential new species
7. **Visualization**: Interactive plots and reports

## 🧬 NCBI SRA Integration

The system includes comprehensive NCBI SRA integration for accessing real-world eDNA datasets. See [User Guide](docs/user_guide.md) for SRA usage examples and configuration.

## 🧪 Testing

```bash
pytest tests/
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
