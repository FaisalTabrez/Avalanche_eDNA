# Avalanche eDNA - Deep-Sea Biodiversity Assessment System

An advanced end-to-end system for identifying taxonomic diversity and assessing biological richness in deep-sea environmental DNA (eDNA) datasets using state-of-the-art machine learning, continual learning, and bioinformatics techniques.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [What's New](#whats-new)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Continual Learning](#continual-learning)
- [Documentation](#documentation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🌊 Overview

Avalanche eDNA is a comprehensive platform for deep-sea eDNA analysis that combines cutting-edge deep learning with robust bioinformatics workflows. The system features:

- **DNABERT-2 Embeddings**: Utilizing state-of-the-art 117M parameter transformer models for DNA sequence understanding
- **Continual Learning**: Models that grow and improve with each new dataset without forgetting previous knowledge
- **Real-time Monitoring**: Live pipeline progress tracking and model training visualization
- **Scalable Processing**: GPU-accelerated workflows for handling massive eDNA datasets
- **Novel Taxa Discovery**: Advanced unsupervised methods for identifying previously unknown organisms
- **Comprehensive Taxonomy**: Multi-source taxonomic assignment combining BLAST, k-NN, and ML classifiers

## 🚀 Key Features

### Core Capabilities
- **Advanced DNA Embeddings**: DNABERT-2 (117M parameters) with chunked mean-pooling, optional PCA, and L2 normalization
- **Data Preprocessing Pipeline**: Quality filtering, adapter trimming, chimera removal, and format conversion
- **Multi-Format Support**: Universal support for FASTA, FASTQ, Swiss-Prot, GenBank, EMBL, and SRA formats
- **Advanced Clustering**: Unsupervised taxonomic grouping with DBSCAN, HDBSCAN, and hierarchical methods
- **Hybrid Taxonomy Assignment**: 
  - BLAST-based reference matching
  - k-NN with FAISS indexing
  - ML classification (Random Forest, XGBoost)
  - Confidence-weighted consensus scoring
- **Novelty Detection**: Identify novel taxa candidates using isolation forests and similarity metrics

### Continual Learning System 🆕 **[REVISED - Nov 2025]**
- **Active Replay Strategy**: Achieves **89% accuracy** vs 18% with passive replay
  - Mixed batch training (50% current + 50% replay)
  - Large replay buffer (1000+ samples)
  - Optimized EWC (λ=100 for balanced plasticity)
- **Catastrophic Forgetting SOLVED**: All clusters retained (100% vs 20%)
- **Production-Ready Pipeline**: `run_taxonomy_pipeline_v2.py`
  - DNABERT-2 embeddings (51ms per sequence on CPU)
  - Automated clustering and taxonomy assignment
  - Versioned model checkpoints
  - Comprehensive visualizations
- **Validated Performance**: Tested on 2,500 synthetic eDNA sequences
  - See `ACTIVE_REPLAY_SUCCESS.md` for simulation results
  - See `TAXONOMY_PIPELINE_V2_GUIDE.md` for usage guide

### Interactive Dashboards
- **Main Dashboard**: Multi-page Streamlit interface for analysis and visualization
- **Pipeline Progress Monitor**: Real-time batch processing and model loading status
- **Training Dashboard**: Visualize model evolution, compare versions, track lineage
- **SRA Browser**: Search and download from 1000+ NCBI eDNA datasets
- **Biodiversity Results**: Interactive plots for species distribution and diversity metrics
- **Taxonomy Viewer**: Explore taxonomic assignments with confidence scores

### Integration & Scalability
- **NCBI SRA Integration**: Direct dataset download with SRA Toolkit and batch processing
- **GPU Acceleration**: CUDA support for embeddings and training
- **Cloud-Ready**: Containerized deployment with scalable architecture
- **Report Management**: Full-featured API for dataset cataloging and cross-analysis
- **Real-time Monitoring**: Live progress tracking with metrics and visualizations

## 🆕 What's New

### Version 2.0 - Continual Learning Release

**Major Features:**
1. **Continual Learning Infrastructure**
   - Complete checkpoint/resume functionality
   - DNABERT-2 fine-tuning with layer freezing strategies
   - Anti-forgetting mechanisms (EWC, replay, distillation)
   - Model versioning and lineage tracking

2. **Enhanced Pipeline**
   - New flags: `--resume`, `--fine-tune`, `--checkpoint-every`
   - Dataset-specific model training
   - Automatic model registration
   - Performance trend visualization

3. **Training Dashboard**
   - Model overview with status and metrics
   - Performance trends across versions
   - Model lineage tree visualization
   - Side-by-side model comparison
   - Checkpoint management interface

4. **DNABERT-2 Optimization**
   - Replaced Nucleotide Transformer with DNABERT-2
   - Optimized chunking for long sequences
   - Configurable fine-tuning parameters
   - Experience replay buffer for old datasets

**Technical Improvements:**
- Comprehensive configuration in `config.yaml`
- Modular architecture for easy extension
- JSON/SQLite backends for model registry
- Real-time progress monitoring
- Enhanced error handling and logging

## 📁 Project Structure

```
├── data/                        # Sample datasets and test data
│   └── synthetic_edna/          # 🆕 Test datasets (2.5K, 5K sequences)
├── src/                         # Source code
│   ├── analysis/                # Dataset analysis utilities
│   ├── api/                     # Report management API
│   ├── clustering/              # Clustering algorithms and taxonomy
│   ├── dashboards/              # Streamlit dashboard modules
│   ├── database/                # Database models and manager
│   ├── models/                  # 🆕 Model implementations
│   │   ├── checkpoint_manager.py   # Training checkpoint management
│   │   ├── continual_learning.py   # 🔥 Active replay + EWC (89% accuracy)
│   │   ├── dnabert.py              # DNABERT/DNABERT-2 wrapper
│   │   ├── embeddings.py           # Embedding generation
│   │   ├── finetuner.py            # Fine-tuning functionality
│   │   ├── model_registry.py       # Version tracking & lineage
│   │   ├── tokenizer.py            # DNA tokenization
│   │   └── trainer.py              # Model training loops
│   ├── novelty/                 # Novelty detection logic
│   ├── organism_profiling/      # Organism profiling modules
│   ├── preprocessing/           # Data cleaning and preparation
│   ├── report_management/       # Report/catalogue management
│   ├── similarity/              # Cross-analysis engine
│   ├── ui/                      # 🆕 Streamlit UI pages
│   │   └── pages/
│   │       ├── progress_updates.py         # Pipeline monitoring
│   │       ├── model_training_dashboard.py # Training visualization
│   │       └── ...
│   ├── utils/                   # Shared utilities and config
│   └── visualization/           # Plotting and dashboard utilities
├── models/                      # 🆕 Pre-trained models
│   └── dnabert2_cpu/            # DNABERT-2-117M (CPU-optimized)
├── notebooks/                   # Jupyter notebooks for analysis
├── tests/                       # Unit and integration tests
├── docs/                        # Documentation
│   ├── ACTIVE_REPLAY_SUCCESS.md      # 🆕 Simulation results
│   ├── TAXONOMY_PIPELINE_V2_GUIDE.md # 🆕 Usage guide
│   └── PIPELINE_MIGRATION_GUIDE.md   # 🆕 v1 → v2 migration
├── scripts/                     # Pipeline and automation scripts
│   ├── run_pipeline.py               # Original pipeline
│   ├── run_taxonomy_pipeline_v2.py   # 🆕 Revised pipeline (active replay)
│   ├── launch_dashboard.py           # Streamlit dashboard launcher
│   └── ...
├── config/                      # Configuration files
│   └── config.yaml              # Main config (with continual_learning section)
├── streamlit_app.py             # Streamlit UI entrypoint
├── demo_taxonomy_pipeline_v2.py # 🆕 Quick demo script
└── requirements*.txt            # Python dependencies
```

## Installation

See [Installation Guide](docs/installation.md) for detailed setup instructions including prerequisites, environment setup, and optional dependencies.

## 🚀 Quick Start

The system uses DNABERT-2 as the default embedding model. The first run will automatically download model weights (~894MB) to your local HuggingFace cache.

### 1. Basic Analysis

```bash
# Create sample eDNA dataset
python scripts/run_pipeline.py --create-sample --input data/sample --output results/demo

# Run complete analysis pipeline
python scripts/run_pipeline.py \
    --input data/sample/sample_edna_sequences.fasta \
    --output results/demo
```

### 2. Launch Interactive Dashboard

```bash
# Start the main dashboard
python scripts/launch_dashboard.py

# Or directly with streamlit
streamlit run streamlit_app.py --server.port=8504
```

Then open http://localhost:8504 in your browser to access:
- Dataset Analysis
- Pipeline Progress Monitor
- Model Training Dashboard
- SRA Browser
- Biodiversity Results
- Taxonomy Viewer

### 3. View Results

Results are automatically saved in your output directory:
```
results/demo/
├── pipeline_results.json        # Complete analysis summary
├── preprocessed_sequences.fasta # Cleaned sequences
├── sequence_embeddings.npy      # DNABERT-2 embeddings
├── clustering/                  # Cluster assignments
├── taxonomy/                    # Taxonomic predictions
├── novelty/                     # Novel taxa candidates
└── visualizations/              # Interactive plots
```

## 🧬 Continual Learning

Train models that grow with your data - accumulating knowledge from multiple eDNA datasets without forgetting.

### Fine-tune on a New Dataset

```bash
# Fine-tune DNABERT-2 on your dataset
python scripts/run_pipeline.py \
    --input data/marine_dataset \
    --output results/marine_run \
    --fine-tune \
    --checkpoint-every 5 \
    --dataset-name "Marine_eDNA_2024" \
    --model-version "v1.0"
```

### Continue Training on Additional Datasets

```bash
# Train on a second dataset (model accumulates knowledge)
python scripts/run_pipeline.py \
    --input data/freshwater_dataset \
    --output results/freshwater_run \
    --fine-tune \
    --dataset-name "Freshwater_eDNA_2024" \
    --model-version "v1.1"
```

### Resume from Checkpoint

```bash
# Resume interrupted training
python scripts/run_pipeline.py \
    --resume results/marine_run/checkpoints/checkpoint_epoch_10.pt \
    --input data/continue_training \
    --output results/continued
```

### Monitor Training Progress

Access the **Training Dashboard** in the Streamlit UI to:
- View model evolution across datasets
- Compare performance metrics between versions
- Visualize model lineage tree
- Manage checkpoints and resume training
- Track catastrophic forgetting prevention

### Continual Learning Strategies

Configure in `config/config.yaml`:

```yaml
continual_learning:
  strategy:
    method: "combined"  # Use EWC + Replay + LwF together
    
    experience_replay:
      buffer_size: 1000
      replay_ratio: 0.2
    
    ewc:
      lambda: 0.4
      online_ewc: true
    
    lwf:
      temperature: 2.0
      alpha: 0.5
```

**Available Strategies:**
- **EWC (Elastic Weight Consolidation)**: Protects important weights from previous tasks
- **Experience Replay**: Replays samples from old datasets during training
- **LwF (Learning Without Forgetting)**: Distills knowledge from previous model version
- **Combined**: Uses all three strategies for maximum stability

## 📚 Documentation

- [User Guide](docs/user_guide.md) - Complete usage instructions and tutorials
- [API Reference](docs/api_reference.md) - Detailed API documentation
- [Configuration Guide](docs/configuration.md) - Configuration options and examples
- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [SRA Integration Guide](docs/SRA_INTEGRATION_GUIDE.md) - NCBI SRA integration and usage
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## 📊 Usage

### Command Line Interface

#### Standard Analysis
```bash
# Complete end-to-end analysis
python scripts/run_pipeline.py \
    --input sequences.fasta \
    --output results/

# Skip specific steps
python scripts/run_pipeline.py \
    --input sequences.fasta \
    --output results/ \
    --skip-preprocessing \
    --skip-clustering
```

#### Continual Learning Features
```bash
# Fine-tune model on dataset
python scripts/run_pipeline.py \
    --input sequences.fasta \
    --output results/ \
    --fine-tune \
    --checkpoint-every 5 \
    --dataset-name "MyDataset" \
    --model-version "v1.0"

# Resume from checkpoint
python scripts/run_pipeline.py \
    --resume results/checkpoints/checkpoint_epoch_10.pt \
    --input new_sequences.fasta \
    --output results/continued/
```

#### Available Flags
- `--input`: Input file (FASTA/FASTQ) or directory
- `--output`: Output directory for results
- `--fine-tune`: Enable DNABERT-2 fine-tuning
- `--resume`: Resume from checkpoint path
- `--checkpoint-every`: Save checkpoint every N epochs
- `--dataset-name`: Dataset name for registry
- `--model-version`: Model version identifier
- `--skip-preprocessing`: Skip preprocessing step
- `--skip-embedding`: Skip embedding generation
- `--skip-clustering`: Skip clustering step
- `--skip-taxonomy`: Skip taxonomy assignment
- `--skip-novelty`: Skip novelty detection
- `--skip-visualization`: Skip visualization generation
- `--create-sample`: Create sample dataset

### Interactive Dashboard

Launch the multi-page Streamlit dashboard:

```bash
# Method 1: Using launcher script
python scripts/launch_dashboard.py

# Method 2: Direct streamlit
streamlit run streamlit_app.py --server.port=8504
```

**Available Pages:**
- **Home**: System overview and quick start
- **Dataset Analysis**: Upload and analyze eDNA datasets
- **Pipeline Progress**: Monitor real-time processing status
- **Model Training**: 🆕 Track continual learning progress
- **SRA Browser**: Search and download NCBI datasets
- **Biodiversity Results**: Visualize species diversity
- **Taxonomy Viewer**: Explore taxonomic assignments
- **About**: System information and documentation

### Python API

```python
from scripts.run_pipeline import eDNABiodiversityPipeline

# Initialize pipeline
pipeline = eDNABiodiversityPipeline()

# Standard analysis
results = pipeline.run_complete_pipeline(
    input_data="sequences.fasta",
    output_dir="results/analysis"
)

# With continual learning
results = pipeline.run_complete_pipeline(
    input_data="sequences.fasta",
    output_dir="results/analysis",
    enable_fine_tuning=True,
    checkpoint_every=5,
    dataset_name="MyDataset",
    model_version="v1.0"
)

# Access results
print(f"Sequences processed: {results['summary']['total_sequences_processed']}")
print(f"Novel taxa found: {results['summary']['novel_taxa_candidates']}")
print(f"Clusters detected: {results['summary']['clusters_detected']}")
```

### Model Registry API

```python
from src.models.model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry(registry_dir="results/model_registry")

# List all models
models = registry.list_models(status='active')

# Get model lineage
lineage = registry.get_lineage("v1.2")

# Compare two models
comparison = registry.compare_models("v1.0", "v1.2")

# Get best model by metric
best = registry.get_best_model(metric='val_loss', minimize=True)
```

### Example Analysis Workflow

1. **Data Upload**: Load FASTQ/FASTA files or download from SRA
2. **Preprocessing**: Quality filtering, adapter removal, chimera detection
3. **Embedding Generation**: DNABERT-2 sequence representations with optional fine-tuning
4. **Clustering**: Group sequences into taxonomic units (DBSCAN/HDBSCAN)
5. **Taxonomy Assignment**: Hybrid approach (BLAST + k-NN + ML)
6. **Novelty Detection**: Identify potential new species using isolation forests
7. **Visualization**: Interactive plots and comprehensive reports
8. **Model Registry**: Track performance and model evolution

### Configuration Examples

#### Basic Embedding Configuration
```yaml
# config/config.yaml
embedding:
  model_id: "zhihan1996/DNABERT-2-117M"
  max_sequence_length: 512
  transformer:
    stride: 256
    batch_size: 8
  postprocess:
    pca_dims: 256
    normalize: true
```

#### Continual Learning Configuration
```yaml
continual_learning:
  checkpoint:
    enabled: true
    save_frequency: 5
    max_checkpoints: 10
    
  fine_tuning:
    learning_rate: 2e-5
    warmup_ratio: 0.1
    freeze_strategy: "gradual"
    
  strategy:
    method: "combined"
    experience_replay:
      buffer_size: 1000
    ewc:
      lambda: 0.4
    lwf:
      alpha: 0.5
```

## 🖥️ Running the Report System

The system includes a comprehensive report management API for cataloging datasets and cross-analysis.

**Option 1 — Windows helper (convenience)**

```bat
start_report_system.bat
```

This launches two separate command windows:
- FastAPI server on `http://127.0.0.1:8000`
- Streamlit dashboard on `http://localhost:8504`

**Option 2 — Manual (cross-platform)**

Run the FastAPI server:
```bash
python -m uvicorn src.api.report_management_api:app --host 127.0.0.1 --port 8000
```

Run the Streamlit dashboard:
```bash
streamlit run streamlit_app.py --server.port=8504 --server.address=localhost
```

## 🧬 NCBI SRA Integration

The system includes comprehensive NCBI SRA (Sequence Read Archive) integration for accessing thousands of publicly available eDNA datasets.

### Features
- **Search & Browse**: Search NCBI SRA with custom keywords and filters
- **Direct Download**: Download datasets using integrated SRA Toolkit
- **Batch Processing**: Queue and download multiple datasets efficiently
- **Auto-conversion**: Automatic SRA → FASTQ conversion
- **Web Interface**: Full integration in Streamlit SRA Browser page
- **API Access**: Programmatic access via Python API

### Quick Start with SRA

**1. Using the Web Interface**
```bash
# Launch dashboard
streamlit run streamlit_app.py

# Navigate to "SRA Browser" page
# Search for datasets, download, and analyze directly
```

**2. Command Line**
```bash
# Download a specific dataset
python scripts/download_sra_data.py \
    --accession SRR12345678 \
    --output data/sra

# Search and download multiple datasets
python scripts/download_sra_data.py \
    --search "marine eDNA" \
    --max-results 10 \
    --output data/sra_datasets
```

**3. Python API**
```python
from src.utils.sra_integration import SRAIntegrationUI

# Initialize
sra = SRAIntegrationUI()

# Search datasets
results = sra.search_sra_datasets(
    keywords=["eDNA", "18S rRNA"],
    max_results=50
)

# Download specific dataset
success, file_path = sra.download_sra_dataset(
    "SRR12345678",
    output_dir="data/sra"
)

# Process downloaded data
pipeline.run_complete_pipeline(
    input_data=file_path,
    output_dir="results/sra_analysis"
)
```

For complete SRA integration documentation, see [SRA Integration Guide](docs/SRA_INTEGRATION_GUIDE.md).

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_system.py

# Run with coverage
pytest --cov=src tests/

# Run specific test category
pytest tests/ -k "embedding"
```

## 🔧 Advanced Features

### Embedding Management
Scripts for multi-dataset workflows are available in `scripts/`:

```bash
# Build consolidated reference from all run embeddings
python scripts/consolidate_embeddings.py \
    --runs-dir analysis_outputs/runs \
    --output consolidated_data/

# Compress embeddings to save storage (~50% reduction)
python scripts/compress_embeddings.py \
    --input results/embeddings.npy \
    --output results/embeddings_compressed.npz

# Search across all embeddings for similar sequences
python scripts/search_reference.py \
    --query sequence.fasta \
    --reference consolidated_data/

# Track model versions and tag runs
python scripts/manage_embedding_versions.py \
    --registry results/model_registry \
    --action list
```

See [EMBEDDING_MANAGEMENT.md](docs/EMBEDDING_MANAGEMENT.md) for detailed workflows.

### Custom Extensions

The modular architecture allows easy extension:

```python
# Add custom clustering algorithm
from src.clustering.algorithms import EmbeddingClusterer

class MyClusterer(EmbeddingClusterer):
    def cluster(self, embeddings):
        # Your implementation
        return labels

# Add custom taxonomy classifier
from src.clustering.taxonomy import TaxonomyClassifier

class MyClassifier(TaxonomyClassifier):
    def predict(self, embeddings):
        # Your implementation
        return predictions
```


## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for Contribution:**
- New clustering algorithms
- Additional taxonomy databases
- Model architectures and training strategies
- Visualization improvements
- Documentation and tutorials
- Bug reports and feature requests

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This project uses the following key technologies:
- **DNABERT-2** - Pre-trained DNA language model by Zhihan Zhou et al.
- **Transformers** - Hugging Face transformers library
- **Streamlit** - Interactive web applications
- **PyTorch** - Deep learning framework
- **NCBI SRA** - Sequence Read Archive integration
- **BLAST+** - Sequence alignment tools

## 📧 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Check the [documentation](docs/)
- Review [troubleshooting guide](docs/troubleshooting.md)

---

**Built with ❤️ for deep-sea biodiversity research**
