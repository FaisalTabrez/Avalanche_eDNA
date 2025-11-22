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
- **Custom Model Training**: Train your own DNA embedding models using contrastive learning, transformers, or autoencoders
- **Advanced Clustering**: Unsupervised taxonomic grouping with novelty detection
- **Interactive Dashboard**: Web-based visualization and analysis interface
- **Scalable Architecture**: GPU acceleration and cloud deployment ready
- **NCBI SRA Integration**: Search, browse, and download from 1000+ eDNA studies in NCBI Sequence Read Archive
  - Direct dataset download with SRA Toolkit
  - Batch processing for multiple datasets
  - Integrated search with custom filters
  - Automatic format conversion (SRA → FASTQ)
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

### Option 1: Docker (Recommended for Production)

**Prerequisites:**
- Docker 20.10+
- Docker Compose 2.0+

**Quick Start:**
```bash
# Clone the repository
git clone https://github.com/FaisalTabrez/Avalanche_eDNA.git
cd Avalanche_eDNA

# Copy environment template and configure
cp .env.example .env
# Edit .env with your settings (database passwords, etc.)

# Start all services (Streamlit, PostgreSQL, Redis)
docker-compose up -d

# View logs
docker-compose logs -f streamlit

# Access the application
# Navigate to http://localhost:8501
```

**Production Deployment:**
```bash
# Copy production environment template
cp .env.example .env
# Edit .env with production credentials and settings

# Build and start production services
docker-compose -f docker-compose.prod.yml up -d

# Scale application instances
docker-compose -f docker-compose.prod.yml up -d --scale streamlit=3

# View status
docker-compose -f docker-compose.prod.yml ps

# Access the application
# Navigate to http://localhost:8501 (or your configured domain)
```

**Useful Docker Commands:**
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (deletes data!)
docker-compose down -v

# Rebuild after code changes
docker-compose build
docker-compose up -d

# View application logs
docker-compose logs -f streamlit

# Access shell in running container
docker-compose exec streamlit bash

# Run database migrations
docker-compose exec streamlit python scripts/migrate_database.py

# Backup database
docker-compose exec postgres pg_dump -U avalanche avalanche_edna > backup.sql

# Restore database
docker-compose exec -T postgres psql -U avalanche avalanche_edna < backup.sql
```

### Option 2: Local Installation

See [Installation Guide](docs/installation.md) for detailed setup instructions including prerequisites, environment setup, and optional dependencies.

## 🚀 Quick Start

Note: The default embedding backend uses a pretrained Nucleotide Transformer from Hugging Face. The first run will download the model weights to your local cache. Embedding post-processing (optional PCA to 256 and L2 normalization) is configurable in config/config.yaml under embedding.postprocess.

### Basic Analysis

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

### Custom Model Training

Train your own DNA embedding models for improved performance on specific datasets:

1. **Train a Contrastive Learning Model**
    ```bash
    # Train on your own sequences
    python scripts/train_model.py \
        --input data/training_sequences.fasta \
        --output models/my_custom_model \
        --model-type contrastive \
        --epochs 100 \
        --batch-size 32
    
    # With labeled data for supervised training
    python scripts/train_model.py \
        --input data/sequences.fasta \
        --labels data/taxonomy_labels.csv \
        --output models/supervised_model \
        --model-type contrastive \
        --epochs 50
    ```

2. **Use Custom Model in Pipeline**
    ```bash
    # Use pre-trained custom model
    python scripts/run_pipeline.py \
        --input data/sample/sample_edna_sequences.fasta \
        --output results/custom_model_run \
        --model-path models/my_custom_model/model
    
    # Train model and run analysis in one go
    python scripts/run_pipeline.py \
        --input data/sample/sample_edna_sequences.fasta \
        --output results/trained_run \
        --train-model
    ```

3. **Training Configuration**
    
    Edit `config/config.yaml` to customize training parameters:
    ```yaml
    embedding:
      training:
        model_type: "contrastive"  # or "transformer", "autoencoder"
        projection_dim: 128
        temperature: 0.1
        batch_size: 32
        learning_rate: 0.0001
        epochs: 100
        device: "auto"  # auto, cuda, or cpu
    ```

## Documentation
## 🖥️ Running the Report System

You can start the API and dashboard using the included Windows helper or start the services manually.

Option 1 — Windows helper (convenience)

```bat
start_report_system.bat
```

This launches two separate command windows:
- FastAPI server on `http://127.0.0.1:8000`
- Streamlit dashboard on `http://localhost:8504`

Option 2 — Manual (cross-platform)

Run the FastAPI server:

```bash
python -m uvicorn src.api.report_management_api:app --host 127.0.0.1 --port 8000
```

Run the Streamlit dashboard:

```bash
streamlit run streamlit_app.py --server.port=8504 --server.address=localhost
```

Notes:
- The batch helper is a convenience for Windows users; prefer the manual commands for cross-platform workflows.
- If you changed the `.streamlit/config.toml` server address or port, update the commands or the batch file accordingly.


- [User Guide](docs/user_guide.md) - Complete usage instructions and tutorials
- [API Reference](docs/api_reference.md) - Detailed API documentation
- [Configuration](docs/configuration.md) - Configuration options and examples
- [SRA Integration Guide](docs/SRA_INTEGRATION_GUIDE.md) - NCBI SRA integration and usage
- [Monitoring Guide](docs/MONITORING.md) - Prometheus, Grafana, and observability setup
- [Testing Guide](docs/TESTING.md) - Testing infrastructure and best practices
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## 📈 Production Features

### Phase 2.3: Monitoring & Observability
- **Prometheus**: 40+ custom metrics across 7 categories
- **Grafana**: Auto-provisioned dashboards for system, application, and Celery monitoring
- **Alertmanager**: 30+ alert rules with multi-channel notifications
- **Exporters**: PostgreSQL, Redis, and Node exporters for infrastructure metrics
- See [PHASE_2.3_SUMMARY.md](PHASE_2.3_SUMMARY.md) for details

### Phase 2.4: Testing Infrastructure
- **pytest**: 110+ comprehensive tests (unit, integration, e2e)
- **Coverage**: 80% minimum threshold with HTML/XML reports
- **Fixtures**: 20+ reusable fixtures for database, API, and Celery testing
- **CI/CD**: GitHub Actions integration for automated testing
- See [PHASE_2.4_SUMMARY.md](PHASE_2.4_SUMMARY.md) for details

### Phase 3: Production Hardening
- **Caching**: Redis-based caching with decorators and connection pooling
- **Rate Limiting**: Token bucket and sliding window algorithms for API protection
- **Database Optimization**: 20+ indexes, connection pooling, batch operations
- **Application Server**: Gunicorn configuration with Nginx reverse proxy templates
- **Load Testing**: Locust-based load testing with multiple user behavior patterns
- See [PHASE_3_SUMMARY.md](PHASE_3_SUMMARY.md) for details

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

The system includes comprehensive NCBI SRA (Sequence Read Archive) integration for accessing thousands of publicly available eDNA datasets:

### Features
- **Search & Browse**: Search NCBI SRA with custom keywords and filters
- **Direct Download**: Download datasets using integrated SRA Toolkit
- **Batch Processing**: Queue and download multiple datasets efficiently
- **Auto-conversion**: Automatic SRA → FASTQ conversion
- **Web Interface**: Full integration in Streamlit dashboard
- **API Access**: Programmatic access via Python API

### Quick Start with SRA

1. **Using the Web Interface**
   ```bash
   # Launch dashboard
   streamlit run streamlit_app.py
   
   # Navigate to "SRA Browser" page
   # Search for datasets, download, and analyze
   ```

2. **Command Line**
   ```bash
   # Download a specific dataset
   python scripts/download_sra_data.py --accession SRR12345678 --output data/sra
   
   # Search and download
   python scripts/download_sra_data.py --search "marine eDNA" --max-results 10
   ```

3. **Python API**
   ```python
   from src.utils.sra_integration import SRAIntegrationUI
   
   # Initialize
   sra = SRAIntegrationUI()
   
   # Search datasets
   results = sra.search_sra_datasets(["eDNA", "18S rRNA"], max_results=50)
   
   # Download
   success, file_path = sra.download_sra_dataset("SRR12345678", output_dir)
   ```

For complete SRA integration documentation, see [SRA Integration Guide](docs/SRA_INTEGRATION_GUIDE.md).

## 🧪 Testing

```bash
pytest tests/
```

## Examples & Archived Tools

- Examples for common workflows are in the `examples/` folder. Notable examples:
    - `examples/ml_training_example.py` — a training example demonstrating model training steps.
    - `examples/sra_integration_example.py` — an example showing SRA integration usage.

- Embedding management scripts for multi-dataset workflows are in `scripts/`:
    - `scripts/consolidate_embeddings.py` — Build consolidated reference from all run embeddings
    - `scripts/compress_embeddings.py` — Compress embeddings to save storage space (~50%)
    - `scripts/search_reference.py` — Search across all embeddings for similar sequences
    - `scripts/manage_embedding_versions.py` — Track model versions and tag runs
    - See [EMBEDDING_MANAGEMENT.md](docs/EMBEDDING_MANAGEMENT.md) for detailed workflow

- Archived one-off or legacy scripts have been moved to `archive/removed_misc/`. If you need to restore a file, copy it back from that directory.

- The Windows helper `start_report_system.bat` is located at `scripts/windows/start_report_system.bat`.


## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
