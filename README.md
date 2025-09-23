# Deep-Sea eDNA Biodiversity Assessment System

An end-to-end system for identifying taxonomic diversity and assessing biological richness in deep-sea environmental DNA (eDNA) datasets using advanced machine learning and bioinformatics techniques.

## 🌊 Overview

This system addresses the challenges of deep-sea eDNA analysis by:
- Processing massive, complex eDNA datasets efficiently
- Discovering novel taxa without relying solely on existing reference databases
- Providing scalable, accurate taxonomic classification
- Offering intuitive visualization and analysis tools

## 🔧 Features

- **Data Preprocessing Pipeline**: Quality filtering, adapter trimming, chimera removal
- **Deep Learning Embeddings**: Transformer-based sequence representation learning
- **Advanced Clustering**: Unsupervised taxonomic grouping with novelty detection
- **Interactive Dashboard**: Web-based visualization and analysis interface
- **Scalable Architecture**: GPU acceleration and cloud deployment ready

## 📁 Project Structure

```
├── data/                   # Sample datasets and test data
├── src/                   # Source code
│   ├── preprocessing/     # Data cleaning and preparation
│   ├── models/           # Deep learning models
│   ├── clustering/       # Taxonomic clustering algorithms
│   ├── visualization/    # Dashboard and plotting utilities
│   └── utils/           # Shared utilities and helpers
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Unit and integration tests
├── docs/                # Documentation
├── scripts/             # Pipeline and automation scripts
└── requirements.txt     # Python dependencies
```

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   conda create -n edna-biodiversity python=3.9
   conda activate edna-biodiversity
   pip install -r requirements.txt
   ```

2. **Create Sample Data and Run Analysis**
   ```bash
   # Create sample eDNA dataset
   python scripts/run_pipeline.py --create-sample --input data/sample --output results/demo
   
   # Run complete analysis pipeline
   python scripts/run_pipeline.py --input data/sample/sample_edna_sequences.fasta --output results/demo
   ```

3. **Launch Interactive Dashboard**
   ```bash
   python scripts/launch_dashboard.py
   ```
   Then open http://localhost:8501 in your browser

4. **View Results**
   ```bash
   # Results are saved in results/demo/
   # - pipeline_results.json: Complete analysis results
   # - visualizations/: Interactive plots
   # - clustering/: Clustering analysis
   # - taxonomy/: Taxonomic assignments
   # - novelty/: Novel taxa detection
   ```

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

See the [documentation](docs/) for detailed usage instructions and tutorials.

## 🧪 Testing

```bash
pytest tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.