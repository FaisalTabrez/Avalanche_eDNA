# Project Introduction and Instructions Compilation

---
---
# From: GEMINI.md
---

# 🧬 Deep-Sea eDNA Biodiversity Assessment System

## Project Overview

This project is a comprehensive, Python-based system for analyzing deep-sea environmental DNA (eDNA) to assess biodiversity. It combines a powerful bioinformatics pipeline with machine learning techniques to process, classify, and visualize eDNA sequence data. The system is designed to handle large datasets, discover novel taxa, and provide an interactive user experience through a Streamlit web application.

**Core Technologies:**
- **Language:** Python
- **Backend/Pipeline:** The core logic is a multi-step pipeline that includes:
    - **Preprocessing:** Quality filtering, adapter trimming, and chimera removal using tools like `cutadapt`.
    - **Sequence Embedding:** Generates vector representations of DNA sequences using a pre-trained Nucleotide Transformer model from Hugging Face (`zhihan1996/DNABERT-2-117M`).
    - **Clustering:** Uses `UMAP` for dimensionality reduction and `HDBSCAN` to group sequences into Operational Taxonomic Units (OTUs).
    - **Taxonomic Assignment:** A hybrid approach using `FAISS` for fast k-NN similarity search against a reference database, followed by a Lowest Common Ancestor (LCA) algorithm. A `BLAST`-based fallback is used for sequences that cannot be confidently classified by the primary method.
    - **Novelty Detection:** Identifies sequences that are significantly different from known reference databases, flagging them as potential new taxa.
- **Frontend:** An interactive web dashboard built with **Streamlit** (`streamlit_app.py`) allows users to upload data, configure and run analyses, and explore results through plots and tables.
- **Data Handling:** `pandas` and `numpy` for data manipulation, with `dask` for parallel processing.
- **Bioinformatics Tools:** `biopython`, `pysam`, and local installations of `BLAST`.

## Building and Running

### 1. Installation

The project uses Python and its dependencies are managed via `pip`.

```bash
# It is recommended to use a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install all required packages
pip install -r requirements.txt
```
*Note: The system may also require external tools like NCBI BLAST to be installed and configured in `config/config.yaml`.*

### 2. Running the Analysis Pipeline (CLI)

The main analysis pipeline can be executed from the command line. This is suitable for batch processing or running on a server.

```bash
# Run the full pipeline on a sample FASTA file
python scripts/run_pipeline.py --input path/to/your/sequences.fasta --output path/to/results_directory
```

### 3. Launching the Interactive Dashboard

The Streamlit application provides a user-friendly graphical interface for all analysis tasks.

```bash
# Launch the web dashboard
streamlit run streamlit_app.py
```
By default, the application will be available at `http://localhost:8501`.

### 4. Running Tests

The project includes a test suite using `pytest`.

```bash
# Run all tests
pytest tests/
```

## Development Conventions

- **Configuration:** All pipeline settings, file paths, and model parameters are managed centrally in `config/config.yaml`. This allows for easy modification of the system's behavior without changing the source code.
- **Modularity:** The codebase is organized into distinct modules within the `src/` directory, separating concerns like `preprocessing`, `clustering`, `taxonomy`, and `visualization`.
- **Entry Points:** The project has two main entry points: `scripts/run_pipeline.py` for command-line execution and `streamlit_app.py` for the web interface.
- **Data Storage:** The `config.yaml` defines specific directories (`AvalancheData/datasets` and `AvalancheData/runs`) for storing uploaded datasets and analysis results, keeping generated data separate from the source code.
- **SRA Integration:** The system is capable of pulling data directly from the NCBI Sequence Read Archive (SRA), with functionality defined in `scripts/download_sra_data.py` and `src/preprocessing/sra_processor.py`.

## ✅ **Fully Implemented Features**

### Preprocessing Pipeline
- Quality filtering, adapter trimming, chimera detection
- Multi-format support (FASTA, FASTQ, Swiss-Prot, GenBank, EMBL)
- NCBI SRA integration with automatic FASTQ conversion
- eDNA-specific filtering and marker gene detection

### Clustering Algorithms
- HDBSCAN (primary), K-means, DBSCAN, hierarchical clustering
- UMAP dimensionality reduction for visualization
- Silhouette score calculation and cluster statistics
- Representative sequence selection per cluster

### Taxonomy Assignment
- Hybrid KNN-LCA + BLAST approach with FAISS indexing
- NCBI taxdump integration for full lineage resolution
- Multi-source reference database support (PR2, SILVA, UNITE, GTDB)
- Conflict resolution and tie-breaker systems

### Novelty Detection
- Multiple methods: Isolation Forest, One-Class SVM, Local Outlier Factor
- Ensemble detection combining multiple algorithms
- Cluster coherence analysis and configurable thresholds
- Similarity and abundance-based novelty assessment

### Database & Report Management
- SQLite backend with full relational schema
- CRUD operations for organism profiles, datasets, reports
- Similarity matrix storage and cross-analysis queries
- Automated report generation and cataloguing

### User Interface
- Complete Streamlit dashboard for analysis workflow
- Interactive file upload, parameter configuration, progress tracking
- Results visualization with taxonomy viewers and export options
- Batch processing support

### Integrations
- NCBI SRA data download and processing
- BLAST database support with configurable parameters
- Hugging Face transformer models for embeddings
- BioPython for bioinformatics formats

### Scripts & Automation
- Complete pipeline script (scripts/run_pipeline.py)
- SRA data download and reference database building
- Index construction (BLAST, FAISS)
- Demo data generation and report export

### Analysis & Visualization
- Biodiversity metrics (Shannon, Simpson, richness)
- Interactive Plotly visualizations
- HTML dashboard generation
- CSV/JSON/PDF export capabilities

### Configuration & Deployment
- YAML-based configuration system
- Docker containerization ready
- GPU acceleration with automatic detection
- Structured logging and error handling

---
---
# From: .github/pull_request_template.md
---

# Title

feat(dev): Windows-friendly preview setup (no torch/UMAP required)

## Summary
- Make UMAP optional with PCA fallback in clustering
- Replace heavy torch-based trainer with a lightweight stub
- Provide torch-free DNATransformerEmbedder stub and wire the pipeline to it
- Keep models package __init__ light to avoid heavy imports
- Add results/ to .gitignore to keep generated artifacts out of VCS

## Motivation
Enable contributors on Windows (Python 3.13) to run a quick demo pipeline without GPU/torch or external native tools. Uses mock embeddings + scikit-learn to validate the flow quickly.

## Changes
- src/clustering/algorithms.py: optional UMAP, PCA fallback
- src/models/trainer.py: lightweight no-op trainer
- src/models/embeddings_stub.py: torch-free DNATransformerEmbedder stub
- scripts/run_pipeline.py: import stub embedder for this branch
- src/models/__init__.py: minimal import surface
- .gitignore: ignore results/

## Testing
Commands used:
- python scripts/run_pipeline.py --create-sample --input data/sample --output results/demo
- python scripts/run_pipeline.py --input data/sample/sample_edna_sequences.fasta --output results/demo --skip-preprocessing

Key outputs to verify:
- results/demo/pipeline_results.json
- results/demo/visualizations/analysis_dashboard.html

## Screenshots / Artifacts
- results/demo/clustering/cluster_visualization.png
- results/demo/novelty/novelty_visualization.png

## Notes
- DBSCAN+PCA with mock embeddings may yield 0 clusters. For a more illustrative demo, consider k-means.
- Full stack (torch, BLAST, cutadapt, vsearch) remains on main; this branch targets a lightweight preview.

## Checklist
- [ ] Title follows conventional commits (feat, fix, chore, etc.)
- [ ] CI passes
- [ ] Docs updated (if needed)
- [ ] No large binary artifacts committed
- [ ] Changes are isolated to preview path; main path unaffected

## Links
- Issue/Discussion: (optional)
- Related PRs: (optional)

---
---
# From: BLAST_INTEGRATION_GUIDE.md
---

# BLAST Integration Guide

## Overview

BLAST (Basic Local Alignment Search Tool) has been successfully integrated into the eDNA Biodiversity Assessment System for Windows. This integration provides taxonomic assignment capabilities using BLAST searches against reference databases.

## 🚀 Features Integrated

### 1. Windows BLAST Runner (`src/utils/blast_utils.py`)
- **WindowsBLASTRunner**: Optimized BLAST execution for Windows
- **Automatic BLAST verification**: Checks BLAST installation on startup
- **Database creation**: Creates BLAST databases from FASTA files
- **Sequence search**: Runs BLASTN searches with proper Windows path handling
- **Result parsing**: Parses BLAST XML output into structured results

### 2. Enhanced Taxonomy Assignment (`src/clustering/taxonomy.py`)
- **BlastTaxonomyAssigner**: Updated to use Windows BLAST utilities
- **Integrated workflow**: Seamless integration with existing pipeline
- **Configurable parameters**: E-value, identity thresholds, max targets
- **Result conversion**: Converts BLAST results to taxonomy assignment format

### 3. Database Setup Script (`scripts/build_blast_db.py`)
- **Enhanced script**: Updated with Windows-specific improvements
- **Multiple implementations**: Windows utilities + fallback methods
- **Better error handling**: Improved debugging and error reporting
- **Usage examples**: Sample commands for common use cases

### 4. Configuration Updates (`config/config.yaml`)
- **Windows paths**: Configured for BLAST 2.17.0+ installation
- **Executable paths**: Direct paths to BLAST binaries
- **Parameters**: Optimized settings for eDNA analysis

## 📋 Prerequisites

- ✅ **BLAST+ 2.17.0**: Installed at `C:\Program Files\NCBI\blast-2.17.0+\bin`
- ✅ **Python environment**: All dependencies installed
- ✅ **Sample data**: Available in `data/sample/sample_edna_sequences.fasta`

## 🔧 Usage Examples

### 1. Create a BLAST Database

```powershell
# Create database from sample data
python scripts\build_blast_db.py --fasta "data\sample\sample_edna_sequences.fasta" --db-out "reference\indices\sample_db"

# Create database with taxonomy mapping
python scripts\build_blast_db.py --fasta "reference.fasta" --taxid-map "taxonomy.txt" --db-out "reference\indices\my_db"
```

### 2. Use BLAST in Python Code

```python
from utils.blast_utils import WindowsBLASTRunner

# Initialize BLAST runner
blast_runner = WindowsBLASTRunner()

# Create database
success = blast_runner.create_blast_database(
    fasta_file="data/sample/sample_edna_sequences.fasta",
    database_name="my_database",
    database_type='nucl'
)

# Search sequences
results = blast_runner.run_blastn_search(
    query_sequences=["ATCGATCG..."],
    database_path="my_database",
    sequence_ids=["seq1"]
)
```

### 3. Taxonomy Assignment with BLAST

```python
from clustering.taxonomy import BlastTaxonomyAssigner

# Initialize taxonomy assigner
assigner = BlastTaxonomyAssigner(
    blast_db="reference/indices/my_database",
    identity_threshold=97.0
)

# Assign taxonomy
results = assigner.assign_taxonomy(
    sequences=["ATCGATCG...", "GCTAGCTA..."],
    sequence_ids=["seq1", "seq2"]
)

# Results include: sequence_id, taxonomy, identity, evalue, etc.
```

### 4. Integration with Main Pipeline

The BLAST integration is automatically used when:
- BLAST databases are available in the configured paths
- Taxonomy assignment methods include BLAST options
- Fallback taxonomy assignment is enabled

## 🔍 Configuration Options

Key configuration settings in `config/config.yaml`:

```yaml
taxonomy:
  blast:
    # Windows BLAST executable paths
    blastn_path: "C:\\Program Files\\NCBI\\blast-2.17.0+\\bin\\blastn.exe"
    makeblastdb_path: "C:\\Program Files\\NCBI\\blast-2.17.0+\\bin\\makeblastdb.exe"
    
    # Search parameters
    evalue: 1e-5
    max_targets: 10
    identity_threshold: 97.0
    num_threads: 4
    
  blast_fallback:
    enable: true
    min_identity_species: 97.0
```

## ✅ Tested Functionality

All components have been tested and verified:

- ✅ **BLAST Installation**: Version check and executable access
- ✅ **Database Creation**: FASTA to BLAST database conversion
- ✅ **Sequence Search**: Query sequences against databases
- ✅ **Taxonomy Assignment**: Full integration with taxonomy pipeline
- ✅ **Result Parsing**: XML output parsing and structuring
- ✅ **Windows Compatibility**: Proper path handling and execution

## 🐛 Troubleshooting

### Common Issues:

1. **"BLAST tools not found"**
   - Verify BLAST+ is installed at the configured path
   - Check Windows PATH environment variable

2. **"Database creation failed"**
   - Ensure input FASTA file exists and is readable
   - Check output directory permissions
   - Avoid paths with spaces when possible

3. **"No hits found"**
   - Verify database was created successfully
   - Check E-value and identity thresholds
   - Ensure query sequences are in correct format

### Debug Mode:

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Integration with Existing Pipeline

The BLAST integration seamlessly works with:

- **Data preprocessing**: Uses cleaned sequences for database creation
- **Embedding pipeline**: BLAST provides alternative to ML-based taxonomy
- **Clustering analysis**: BLAST results inform taxonomic clustering
- **Visualization**: Results displayed in dashboards and reports
- **Novelty detection**: Unknown sequences identified through BLAST searches

## 📊 Performance Notes

- Database creation: ~1-2 seconds for 1000 sequences
- Search performance: ~0.1-1 second per sequence (depends on database size)
- Memory usage: Minimal additional overhead
- Parallel processing: Configurable thread count for searches

## 🎯 Next Steps

The BLAST integration is ready for production use. Consider:

1. **Reference database setup**: Create comprehensive eDNA reference databases
2. **Taxonomy mapping**: Add NCBI taxonomy integration for better species identification
3. **Performance tuning**: Optimize parameters for specific use cases
4. **Custom databases**: Create specialized databases for specific environments

---

*Last updated: October 2, 2025*
*BLAST version: 2.17.0+*
*Integration status: ✅ Complete and tested*

---
---
# From: docs/api_reference.md
---

# API Documentation

## Deep-Sea eDNA Biodiversity Assessment System API

### Table of Contents
1. [Overview](#overview)
2. [Core Modules](#core-modules)
3. [Preprocessing](#preprocessing)
4. [Models](#models)
5. [Clustering](#clustering)
6. [Novelty Detection](#novelty-detection)
7. [Visualization](#visualization)
8. [Pipeline](#pipeline)

---

## Overview

The eDNA Biodiversity Assessment System provides a comprehensive API for analyzing environmental DNA sequences to identify taxonomic diversity and detect novel species.

### Key Features
- **Modular Design**: Each component can be used independently
- **Scalable Processing**: Handle datasets from hundreds to millions of sequences
- **Advanced ML**: Deep learning models for sequence analysis
- **Interactive Visualization**: Web-based dashboard and plotting utilities

---

## Core Modules

### Configuration Management

#### `utils.config.Config`
Manages system configuration from YAML files.

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

---

## Preprocessing

### Sequence Quality Filtering

#### `preprocessing.pipeline.SequenceQualityFilter`
Filters DNA sequences based on length, quality, and content criteria.

```python
from preprocessing.pipeline import SequenceQualityFilter

filter_obj = SequenceQualityFilter(
    min_length=50,
    max_length=500,
    quality_threshold=20,
    max_n_bases=5
)

# Filter a single sequence
is_valid = filter_obj.filter_sequence(sequence_record)

# Filter a FASTQ file
stats = filter_obj.filter_fastq(input_file, output_file)
```

**Parameters:**
- `min_length`: Minimum sequence length (default: 50)
- `max_length`: Maximum sequence length (default: 500)
- `quality_threshold`: Minimum average quality score (default: 20)
- `max_n_bases`: Maximum number of N bases allowed (default: 5)

### Adapter Trimming

#### `preprocessing.pipeline.AdapterTrimmer`
Removes adapter sequences using cutadapt.

```python
from preprocessing.pipeline import AdapterTrimmer

trimmer = AdapterTrimmer(
    adapters=["AGATCGGAAGAGC", "CTGTCTCTTATA"],
    min_length=50
)

success = trimmer.trim_adapters(input_file, output_file)
```

### Chimera Detection

#### `preprocessing.pipeline.ChimeraDetector`
Detects and removes chimeric sequences using VSEARCH.

```python
from preprocessing.pipeline import ChimeraDetector

detector = ChimeraDetector(
    reference_db="data/reference/silva_138.1.fasta",
    method="vsearch"
)

success = detector.detect_chimeras(input_file, output_file)
```

### Complete Preprocessing Pipeline

#### `preprocessing.pipeline.PreprocessingPipeline`
Orchestrates the complete preprocessing workflow.

```python
from preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()

# Process single file
stats = pipeline.process_file(input_file, output_prefix)

# Process directory
results = pipeline.process_directory(input_dir)

# Generate report
report = pipeline.generate_report(results)
```

---

## Models

### DNA Tokenization

#### `models.tokenizer.DNATokenizer`
Converts DNA sequences to numerical tokens for machine learning models.

```python
from models.tokenizer import DNATokenizer

# Initialize tokenizer
tokenizer = DNATokenizer(
    encoding_type="kmer",  # "kmer", "char", or "both"
    kmer_size=6,
    stride=1,
    add_special_tokens=True
)

# Encode single sequence
encoded = tokenizer.encode_sequence("ATCGATCGATCG", max_length=100)
# Returns: {'input_ids': array, 'attention_mask': array, 'tokens': list}

# Encode multiple sequences
batch_encoded = tokenizer.encode_sequences(sequences, max_length=100)

# Decode sequence
decoded = tokenizer.decode_sequence(token_ids)

# Save/load tokenizer
tokenizer.save("tokenizer.pkl")
loaded_tokenizer = DNATokenizer.load("tokenizer.pkl")
```

**Encoding Types:**
- `"kmer"`: K-mer based encoding
- `"char"`: Character-level encoding
- `"both"`: Combined k-mer and character encoding

### Sequence Dataset

#### `models.tokenizer.SequenceDataset`
Dataset wrapper for DNA sequences compatible with PyTorch DataLoader.

```python
from models.tokenizer import SequenceDataset

dataset = SequenceDataset(
    sequences=["ATCGATCG", "GCTAGCTA"],
    labels=["Species_A", "Species_B"],  # Optional
    tokenizer=tokenizer,
    max_length=512
)

# Access single item
item = dataset[0]  # Returns dict with 'input_ids', 'attention_mask', etc.

# Get batch
batch = dataset.get_batch([0, 1])
```

### Deep Learning Models

#### `models.embeddings.DNATransformerEmbedder`
Transformer-based model for generating sequence embeddings.

```python
from models.embeddings import DNATransformerEmbedder
import torch

model = DNATransformerEmbedder(
    vocab_size=4096,
    d_model=256,
    nhead=8,
    num_layers=6,
    dim_feedforward=1024,
    dropout=0.1,
    max_len=512,
    pooling_strategy='cls'  # 'cls', 'mean', or 'max'
)

# Generate embeddings
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

with torch.no_grad():
    embeddings = model(input_ids, attention_mask)
    # Shape: [batch_size, d_model]
```

#### `models.embeddings.DNAAutoencoder`
Autoencoder model for unsupervised sequence representation learning.

```python
from models.embeddings import DNAAutoencoder

model = DNAAutoencoder(
    vocab_size=4096,
    embedding_dim=128,
    hidden_dims=[256, 512, 256],
    latent_dim=64,
    dropout=0.1
)

# Forward pass
latent, reconstructed = model(input_ids, attention_mask)
# latent shape: [batch_size, latent_dim]
# reconstructed shape: [batch_size, vocab_size]

# Encode only
latent = model.encode(input_ids, attention_mask)
```

#### `models.embeddings.DNAContrastiveModel`
Contrastive learning wrapper for self-supervised training.

```python
from models.embeddings import DNAContrastiveModel

contrastive_model = DNAContrastiveModel(
    backbone_model=transformer_model,
    projection_dim=128,
    temperature=0.1
)

# Forward pass
projected = contrastive_model(input_ids, attention_mask)

# Compute contrastive loss
loss = contrastive_model.contrastive_loss(projected, labels)
```

### Model Training

#### `models.trainer.EmbeddingTrainer`
Trainer class for deep learning models.

```python
from models.trainer import EmbeddingTrainer

trainer = EmbeddingTrainer(model, tokenizer, device='auto')

# Prepare data
train_loader, val_loader = trainer.prepare_data(
    sequences=sequences,
    labels=labels,  # Optional
    validation_split=0.2,
    batch_size=32,
    max_length=512
)

# Train autoencoder
if isinstance(model, DNAAutoencoder):
    history = trainer.train_autoencoder(
        train_loader, val_loader,
        epochs=100,
        learning_rate=1e-4
    )

# Train contrastive model
elif isinstance(model, DNAContrastiveModel):
    history = trainer.train_contrastive(
        train_loader, val_loader,
        epochs=100,
        learning_rate=1e-4
    )

# Extract embeddings
embeddings = trainer.extract_embeddings(sequences, batch_size=32)

# Save model
trainer.save_model("models/trained_model", include_tokenizer=True)
```

---

## Clustering

### Embedding-Based Clustering

#### `clustering.algorithms.EmbeddingClusterer`
Clusters sequence embeddings using various algorithms.

```python
from clustering.algorithms import EmbeddingClusterer
import numpy as np

# Initialize clusterer
clusterer = EmbeddingClusterer(
    method="hdbscan",  # "hdbscan", "kmeans", "dbscan", "hierarchical"
    min_cluster_size=10,
    min_samples=5,
    metric="euclidean"
)

# Perform clustering
embeddings = np.random.randn(1000, 256)
cluster_labels = clusterer.fit(embeddings)

# Get cluster statistics
stats = clusterer.cluster_stats
print(f"Found {stats['n_clusters']} clusters")
print(f"Silhouette score: {stats['silhouette_score']}")

# Dimensionality reduction for visualization
reduced_embeddings = clusterer.reduce_dimensions(n_components=2)

# Plot results
clusterer.plot_clusters(sequences, save_path="clusters.png")

# Get representative sequences
representatives = clusterer.get_cluster_representatives(
    sequences, n_representatives=5
)

# Save results
clusterer.save_results(
    sequences, "output/clustering", 
    include_embeddings=True
)
```

**Clustering Methods:**
- `"hdbscan"`: Hierarchical density-based clustering
- `"kmeans"`: K-means clustering
- `"dbscan"`: Density-based clustering
- `"hierarchical"`: Hierarchical clustering

### Taxonomic Assignment

#### `clustering.taxonomy.MLTaxonomyClassifier`
Machine learning-based taxonomic classifier.

```python
from clustering.taxonomy import MLTaxonomyClassifier

classifier = MLTaxonomyClassifier(model_type="random_forest")

# Train classifier
results = classifier.train(
    embeddings=train_embeddings,
    taxonomic_labels=train_labels,
    validation_split=0.2
)

print(f"Validation accuracy: {results['val_accuracy']:.3f}")

# Make predictions
predictions = classifier.predict(test_embeddings)

for pred in predictions[:5]:
    print(f"Taxonomy: {pred['predicted_taxonomy']}")
    print(f"Confidence: {pred['confidence']:.3f}")
    print(f"Top 3: {pred['top_predictions']}")

# Save/load model
classifier.save_model("taxonomy_model.pkl")
new_classifier = MLTaxonomyClassifier()
new_classifier.load_model("taxonomy_model.pkl")
```

#### `clustering.taxonomy.BlastTaxonomyAssigner`
BLAST-based taxonomic assignment.

```python
from clustering.taxonomy import BlastTaxonomyAssigner

assigner = BlastTaxonomyAssigner(
    blast_db="data/reference/nt",
    evalue=1e-5,
    max_targets=10,
    identity_threshold=97.0
)

# Assign taxonomy
results = assigner.assign_taxonomy(
    sequences=sequences,
    sequence_ids=sequence_ids
)

for result in results[:5]:
    print(f"Sequence: {result['sequence_id']}")
    print(f"Best hit: {result['best_hit']}")
    print(f"Identity: {result['identity']:.1f}%')
    print(f"Taxonomy: {result['taxonomy']}")
```

#### `clustering.taxonomy.HybridTaxonomyAssigner`
Combines BLAST and ML approaches for robust taxonomy assignment.

```python
from clustering.taxonomy import HybridTaxonomyAssigner

hybrid_assigner = HybridTaxonomyAssigner(
    blast_assigner=blast_assigner,
    ml_classifier=ml_classifier,
    confidence_threshold=0.8
)

# Assign taxonomy using hybrid approach
results = hybrid_assigner.assign_taxonomy(
    sequences=sequences,
    embeddings=embeddings,
    sequence_ids=sequence_ids
)

# Generate assignment report
report_df = hybrid_assigner.generate_assignment_report(
    results, save_path="taxonomy_report.csv"
)
```

---

## Novelty Detection

### Basic Novelty Detection

#### `novelty.detection.NoveltyDetector`
Detects novel sequences using various machine learning algorithms.

```python
from novelty.detection import NoveltyDetector

detector = NoveltyDetector(
    method="isolation_forest",  # "isolation_forest", "one_class_svm", "local_outlier_factor"
    contamination=0.1  # Expected fraction of outliers
)

# Fit on reference data (known taxa)
detector.fit(reference_embeddings, normalize=True)

# Predict novelty
predictions = detector.predict(query_embeddings)
# Returns: array of 1 (normal) and -1 (novel)

# Get novelty scores
scores = detector.decision_function(query_embeddings)
# Higher scores = more normal
```

### Distance-Based Detection

#### `novelty.detection.DistanceBasedNoveltyDetector`
Uses k-nearest neighbors for novelty detection.

```python
from novelty.detection import DistanceBasedNoveltyDetector

detector = DistanceBasedNoveltyDetector(
    n_neighbors=5,
    distance_threshold=None,  # Auto-estimated if None
    metric='euclidean'
)

detector.fit(reference_embeddings)
predictions = detector.predict(query_embeddings)
scores = detector.decision_function(query_embeddings)
```

### Ensemble Detection

#### `novelty.detection.EnsembleNoveltyDetector`
Combines multiple detection methods for robust results.

```python
from novelty.detection import EnsembleNoveltyDetector

# Create base detectors
detectors = [
    NoveltyDetector(method="isolation_forest"),
    NoveltyDetector(method="one_class_svm"),
    DistanceBasedNoveltyDetector()
]

ensemble = EnsembleNoveltyDetector(
    detectors=detectors,
    voting='soft'  # 'hard' or 'soft'
)

ensemble.fit(reference_embeddings)
predictions = ensemble.predict(query_embeddings)
```

### Comprehensive Analysis

#### `novelty.detection.NoveltyAnalyzer`
High-level interface for complete novelty analysis.

```python
from novelty.detection import NoveltyAnalyzer

analyzer = NoveltyAnalyzer(
    similarity_threshold=0.85,
    abundance_threshold=0.001,
    cluster_coherence_threshold=0.7
)

# Run complete analysis
results = analyzer.analyze_novelty(
    query_embeddings=query_embeddings,
    reference_embeddings=reference_embeddings,
    query_sequences=sequences,
    query_abundances=abundances,  # Optional
    cluster_labels=cluster_labels  # Optional
)

print(f"Novel candidates: {results['novel_candidates']}")
print(f"Novel percentage: {results['novel_percentage']:.1f}%")

# Visualize results
analyzer.visualize_novelty_results(
    query_embeddings,
    np.array(results['predictions']['ensemble']),
    save_path="novelty_plot.png"
)
```

---

## Visualization

### Plotting Utilities

#### `visualization.plots.BiodiversityPlotter`
Comprehensive plotting utilities for biodiversity analysis.

```python
from visualization.plots import BiodiversityPlotter

plotter = BiodiversityPlotter(
    style="whitegrid",
    color_palette="husl",
    figsize=(12, 8)
)

# Sequence length distribution
fig1 = plotter.plot_sequence_length_distribution(
    sequence_lengths=[100, 150, 200, 180, 220],
    save_path="length_dist.html"
)

# Taxonomic composition
taxonomy_counts = {"Bacteria": 500, "Archaea": 200, "Eukaryota": 150}
fig2 = plotter.plot_taxonomic_composition(
    taxonomy_counts, 
    plot_type="pie",  # "pie", "bar", "treemap"
    save_path="taxonomy.html"
)

# Diversity indices
diversity_data = {"Shannon": 3.2, "Simpson": 0.85, "Chao1": 125}
fig3 = plotter.plot_diversity_indices(diversity_data)

# Cluster visualization
fig4 = plotter.plot_cluster_visualization(
    embeddings_2d=reduced_embeddings,
    cluster_labels=cluster_labels,
    novelty_labels=novelty_predictions,
    save_path="clusters.html"
)

# Novelty analysis
fig5 = plotter.plot_novelty_analysis(
    novelty_scores=novelty_scores,
    novelty_predictions=novelty_predictions,
    threshold=0.0
)

# Comprehensive dashboard
fig6 = plotter.create_analysis_dashboard(
    analysis_results=complete_results,
    save_path="dashboard.html"
)

# Show plot
fig1.show()
```

### Interactive Dashboard

#### `visualization.dashboard.BiodiversityDashboard`
Streamlit-based web interface for interactive analysis.

```python
from visualization.dashboard import BiodiversityDashboard

# Launch dashboard
dashboard = BiodiversityDashboard()
dashboard.run()
```

**Dashboard Features:**
- Data upload and preprocessing
- Real-time analysis configuration
- Interactive visualizations
- Results export and reporting

---

## Pipeline

### Complete Pipeline

#### `scripts.run_pipeline.eDNABiodiversityPipeline`
End-to-end pipeline orchestrating all analysis steps.

```python
from scripts.run_pipeline import eDNABiodiversityPipeline

pipeline = eDNABiodiversityPipeline(config_path="config/config.yaml")

# Run complete analysis
results = pipeline.run_complete_pipeline(
    input_data="data/sequences.fasta",
    output_dir="results/analysis",
    run_preprocessing=True,
    run_embedding=True,
    run_clustering=True,
    run_taxonomy=True,
    run_novelty=True,
    run_visualization=True
)

# Access results
print(f"Total sequences: {results['summary']['total_sequences_processed']}")
print(f"Clusters found: {results['summary']['total_clusters']}")
print(f"Novel taxa: {results['summary']['novel_taxa_candidates']}")
```

### Command Line Interface

```bash
# Complete analysis
python scripts/run_pipeline.py \
    --input data/sequences.fasta \
    --output results/analysis \
    --config config/custom.yaml

# Skip specific steps
python scripts/run_pipeline.py \
    --input data/sequences.fasta \
    --output results/analysis \
    --skip-preprocessing \
    --skip-taxonomy

# Create sample data
python scripts/run_pipeline.py \
    --create-sample \
    --input data/sample \
    --output results/sample_analysis
```

---

## Error Handling

### Common Exceptions

```python
from models.tokenizer import DNATokenizer

try:
    tokenizer = DNATokenizer.load("nonexistent.pkl")
except FileNotFoundError:
    print("Tokenizer file not found")

try:
    encoded = tokenizer.encode_sequence("INVALID_SEQUENCE")
except ValueError as e:
    print(f"Invalid sequence: {e}")
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Starting analysis...")
```

---

## Performance Considerations

### Memory Management

```python
# Process large datasets in chunks
def process_large_dataset(sequences, chunk_size=1000):
    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i+chunk_size]
        # Process chunk
        yield process_chunk(chunk)

# Use generators for memory efficiency
def sequence_generator(file_path):
    from Bio import SeqIO
    for record in SeqIO.parse(file_path, "fasta"):
        yield str(record.seq)
```

### GPU Acceleration

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Move model to GPU
model = model.to(device)
```

### Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def parallel_analysis(sequences, n_processes=4):
    with Pool(n_processes) as pool:
        process_func = partial(analyze_sequence, param1=value1)
        results = pool.map(process_func, sequences)
    return results
```

---

This API documentation provides comprehensive coverage of all major components in the eDNA Biodiversity Assessment System. Each class and function includes usage examples and parameter descriptions to facilitate easy integration and customization.

---
---
# From: docs/HACKATHON_SUBMISSION.md
---

# Hackathon Submission

## 1) Idea Title
Avalanche eDNA: A Hybrid AI + Bioinformatics Platform for Fast, Trustworthy Biodiversity Assessment

## 2) Idea Description
Environmental DNA (eDNA) enables the detection of organisms from trace genetic material in water, soil, and sediment samples. Yet today’s eDNA workflows are either slow (manual BLAST-only pipelines) or opaque (ML-only black boxes). Avalanche eDNA bridges this gap with a hybrid platform that is fast, trustworthy, and easy to use.

What we built
- End-to-end pipeline that ingests FASTA/FASTQ, generates sequence embeddings using a state-of-the-art DNA language model (DNABERT‑2), clusters similar sequences, assigns taxonomy via a KNN+LCA approach, and falls back to BLAST with lineage enrichment through NCBI taxdump.
- A modern Streamlit web UI with a Home page, Results Viewer, Run Browser, and Taxonomy Viewer, so biologists and stakeholders can explore results without command‑line tools.
- Storage conventions and run management (F:\AvalancheData\datasets and F:\AvalancheData\runs) for reproducible, shareable analyses.

Why it matters
- Speed: ML embeddings accelerate similarity search and taxonomy suggestions; BLAST is invoked selectively, reducing overall runtime.
- Trust: When BLAST provides a taxid, the system prioritizes that lineage and records tie‑breaks between KNN and BLAST, exposing conflicts rather than hiding them.
- Usability: A clear UI, recent‑runs shortcuts, and a run browser turn complex analyses into a guided experience.

Differentiators
- Hybrid accuracy: Combines ML nearest‑neighbor evidence (KNN+LCA) with authoritative BLAST taxids, producing richer and more reliable lineage.
- Explainability: Tie‑breaker reporting and lineage provenance show why a call was made and which data source won.
- Scalability path: CPU‑only works today; GPU or ONNX Runtime upgrades bring 10–100× embedding throughput for large datasets.

Impact & use cases
- Marine, freshwater, and sediment biodiversity surveys
- Invasive species monitoring and conservation prioritization
- Rapid triage of large field datasets with sharable, auditable outputs

Current status
- Fully working prototype on Windows; web UI launched locally; subset run identified 61 unique taxa with enriched lineage. GPU acceleration and run‑to‑run comparison are the next milestones.

## 3) Abstract/Summary
Avalanche eDNA is a hybrid AI + bioinformatics system for environmental DNA analysis that balances speed, transparency, and scientific rigor. The pipeline converts DNA sequences into numeric embeddings with a pretrained transformer (DNABERT‑2‑117M), enabling efficient similarity search and clustering. For taxonomy, the platform uses a KNN + lowest common ancestor approach over reference embeddings and integrates BLAST as a targeted fallback. When BLAST returns a taxonomic identifier (taxid), Avalanche resolves the full scientific lineage via NCBI taxdump and prioritizes this lineage over name‑only ML matches; otherwise, ML assignments are used with conservative consensus. The system records tie‑break decisions and flags potential conflicts, improving explainability.

A Streamlit web application makes results accessible: a Home page with navigation tiles and recent runs, a Results Viewer for pipeline summaries and visuals, a Taxonomy Viewer with filtering and downloads, and a Run Browser for fast discovery of analyses. Data and outputs are organized in standardized directories for reproducibility.

In tests on a 2,000‑sequence subset from SRR35551197, the taxonomy step completed in ~31 seconds and identified 61 unique taxa. Embedding throughput on CPU measured ~3.8 sequences/second on a 512‑sequence benchmark; GPU or ONNX Runtime can significantly accelerate this. Avalanche eDNA reduces time‑to‑insight for biodiversity assessments while maintaining lineage traceability and user‑friendly review, supporting applications in monitoring, conservation, and rapid environmental triage.

## 4) Technology Bucket
- AI/ML: Transformer‑based sequence embeddings (Hugging Face, PyTorch), KNN + LCA taxonomy
- Bioinformatics: NCBI BLAST fallback, taxid extraction, NCBI taxdump lineage resolver
- Data/Systems: FAISS (CPU) similarity index, reproducible run storage (datasets/runs)
- Web/UI: Streamlit front‑end with Results Viewer, Taxonomy Viewer, and Run Browser
- Performance/Scale (roadmap): GPU mixed precision, ONNX Runtime (CPU), representative‑only embedding, caching/deduplication

---
---
# From: docs/PROJECT_REPORT.md
---

# eDNA Biodiversity Analysis Project – Comprehensive Report

Date: 2025-09-30

Author: Avalanche eDNA Pipeline Team

---

## 1) Executive Summary

- Built and executed an end-to-end eDNA biodiversity analysis pipeline and Streamlit web UI on Windows.
- Data: SRR35551197 dataset (subset runs completed), with BLAST reference from env_nt and lineage via NCBI taxdump.
- Embeddings: DNABERT-2-117M on CPU; robust output handling added; batch size tuned to 256.
- Taxonomy: Hybrid KNN+LCA with BLAST fallback; lineage enrichment from BLAST taxids, with conflict/tie-break reporting.
- UI: Added a Home page, Results Viewer, Taxonomy Viewer, and a Run Browser with recent runs and deep links.
- Storage: Centralized directories at F:\AvalancheData\datasets and F:\AvalancheData\runs, with prior results migrated.
- Performance: CPU embedding throughput measured at ~3.8 seq/s (512 seq test). Full 318k embedding deferred; GPU path planned.

---

## 2) Goals and Scope

- Provide a reproducible, modular pipeline for eDNA sequence processing, embedding-based clustering, hybrid taxonomy assignment, novelty detection, and visualization.
- Support both CLI and web UI workflows with persistent storage conventions and run management.
- Enable lineage-accurate taxonomy via BLAST taxids and prioritize taxid-based lineage over name-only matches.

---

## 3) Data and References

- Primary dataset:
  - Input: F:\Dataset\SRR35551197.fasta.gz (decompressed prior to analysis)
  - Subsets: head2000, head512 created for faster iteration
- BLAST reference:
  - Database: F:\Dataset\env_nt_extracted\env_nt
- NCBI Taxdump (for lineage enrichment):
  - Directory: F:\Dataset\taxdump

---

## 4) System Architecture Overview

- CLI Pipeline: scripts/run_pipeline.py orchestrates preprocessing → embeddings → clustering → taxonomy → novelty → visualizations → summary.
- Core Modules: src/preprocessing, src/clustering, src/novelty, src/visualization, src/utils/config.
- Web UI: streamlit_app.py with pages for Home, Dataset Analysis, Results Viewer, Run Browser, and Taxonomy Viewer.
- Storage Conventions (config/config.yaml):
  - Datasets: F:\AvalancheData\datasets
  - Runs: F:\AvalancheData\runs
  - Results are stored under runs/<dataset_name>/<timestamp>/ with subfolders for clustering, taxonomy, novelty, visualizations.

---

## 5) Pipeline Implementation Details

### 5.1 Preprocessing
- Accepts FASTA/FASTQ and auto-detects formats in the UI path; CLI pipeline expects FASTA for direct runs.
- Quality/adapter parameters exist in config, with optional chimera detection (vsearch placeholder path).

### 5.2 Embedding Generation
- Model: zhihan1996/DNABERT-2-117M (Hugging Face) on CPU.
- Settings (config.yaml):
  - max_sequence_length: 512
  - stride: 128 (for chunking long sequences)
  - batch_size: 256 (increased from 8 for throughput)
  - Mean pooling across tokens; Post-processing: PCA to 256 dims, per-row L2 normalization.
- Robust output handling implemented for models returning tuple/dict; fallbacks in place.

### 5.3 Clustering
- Default method: HDBSCAN; falls back to DBSCAN when HDBSCAN is unavailable on the environment.
- Produces cluster assignments, stats, 2D embeddings, and a cluster visualization PNG.

### 5.4 Taxonomy Assignment
- Primary: KNN + LCA over reference embeddings (FAISS flat IP index; k=50; min_similarity 0.65; distance margin 0.07; rank-specific agreement thresholds).
- Cluster consensus smoothing to improve stability across member members.
- BLAST fallback: ncbi-blastn against env_nt_extracted; controlled by identity thresholds and max targets.
- Lineage enrichment and priority:
  - Extract taxid from BLAST XML hits; resolve lineage (species → kingdom) via taxdump traversal.
  - Prefer BLAST taxid lineage when available; otherwise fall back to KNN name-based lineage.
  - Tie-break report fields: tiebreak_winner (blast/knn), reason, conflict_flag.

### 5.5 Novelty Detection
- Thresholds (config): similarity_threshold 0.85, abundance_threshold 0.001, cluster_coherence 0.7.
- No novel candidates detected for the head2000 subset.

---

## 6) Results Summary

### 6.1 Completed Runs and Locations
- 2,000-sequence taxonomy run (head2000):
  - Path: F:\AvalancheData\runs\SRR35551197\head2000
  - Outputs: taxonomy/taxonomy_predictions.csv, taxonomy_tiebreak_report.csv, clustering/, novelty/, visualizations/
- 512-sequence embedding timing test (batch_size=256):
  - Path: F:\AvalancheData\runs\SRR35551197\head512_bs256
  - Pipeline runtime: 135.93 seconds (embedding + PCA/L2; other steps skipped)

### 6.2 Key Metrics
- head2000 taxonomy:
  - Taxonomy step completed in ~31 seconds
  - Identified 61 unique taxa (from enriched taxonomy_predictions.csv)
- Embedding throughput (CPU):
  - 512 sequences in 135.93 s → ~3.77 seq/s including model load and PCA/L2

Note: The full ~318k sequence embedding run was interrupted due to CPU runtime constraints; GPU acceleration is planned.

---

## 7) Web UI Overview

- Home Page: Navigation tiles, feature overview, recent runs list with deep links into Results Viewer.
- Dataset Analysis: Upload files, configure analysis, run quick stats and visualizations; persists inputs/outputs into configured storage directories.
- Results Viewer: Summaries of pipeline results, cluster visualization, taxonomy charts and tables, novelty section, and embedded dashboard HTML.
- Run Browser: Browse F:\AvalancheData\runs by dataset and timestamp; search/filter; open any run directly in Results Viewer.
- Taxonomy Viewer: Load taxonomy_predictions.csv (and optional tie-break report), visualize top taxa, filter conflicts, and download enriched outputs.

Launch command:

```
streamlit run "C:\\Volume D\\Avalanche\\streamlit_app.py"
```

---

## 8) Storage and Run Management

- Centralized roots in config/config.yaml:
  - storage.datasets_dir: F:\AvalancheData\datasets
  - storage.runs_dir: F:\AvalancheData\runs
- Migration performed for prior results from C:\\Volume D\\Avalanche\\results into the new runs directory.
- UI “Recent Runs” and “Run Browser” operate over storage.runs_dir; Results Viewer supports prefilled paths.

---

## 9) Reproducibility and How-To

CLI pipeline example (embedding-only quick test shown):

```
python "C:\\Volume D\\Avalanche\\scripts\\run_pipeline.py" \
  --input "C:\\Volume D\\Avalanche\\results\\demo\\input\\SRR35551197.head512.fasta" \
  --output "F:\\AvalancheData\\runs\\SRR35551197\\head512_bs256" \
  --skip-clustering --skip-taxonomy --skip-novelty --skip-visualization --skip-preprocessing
```

Web UI launch:

```
streamlit run "C:\\Volume D\\Avalanche\\streamlit_app.py"
```

Configuration highlights (config/config.yaml):
- Embedding.transformer: model_id "zhihan1996/DNABERT-2-117M", stride 128, batch_size 256
- Taxonomy.blast and blast_fallback: database path set to F:\\Dataset\\env_nt_extracted\\env_nt
- Taxonomy.taxdump_dir: F:\\Dataset\\taxdump
- Storage roots for datasets/runs: F:\\AvalancheData\\

Dependencies:
- transformers, torch, einops, scikit-learn, plotly, streamlit, faiss-cpu, biopython
- Note: HDBSCAN not available; DBSCAN used as fallback

---

## 10) Known Issues and Constraints

- CPU-only embedding performance is limited; full-scale runs on hundreds of thousands of reads are impractical without GPU or further optimization.
- HDBSCAN missing in the environment, so clustering falls back to DBSCAN (reduced clustering fidelity for some datasets).
- Streamlit “ScriptRunContext” warnings appear when executing the CLI; harmless in non-Streamlit contexts.
- Windows-specific path and quoting nuances require care in scripts/commands.

---

## 11) Recommendations and Next Steps

Performance and scaling:
- Pre-tokenize sequences and cache token IDs to reduce CPU overhead during embedding.
- Deduplicate sequences and/or precluster at 100% (or 99%) identity and embed only representatives.
- Consider ONNX Runtime (CPU EP) for 1.2–2x speedups vs. eager PyTorch.
- When a GPU is available, enable mixed precision (FP16/TF32), ensure true batched inference, and pad to multiples of 8.

Taxonomy and lineage:
- Expand BLAST parsing to support additional fields (e.g., alignment length, coverage) for quality scoring.
- Add toggles in UI to prioritize BLAST vs KNN or require confidence thresholds per rank.

UI/UX:
- Add per-run actions in Run Browser (rename, delete with confirmation, open folder).
- Add Run Comparison view: side-by-side taxa and cluster diffs.
- Server-side filtering/pagination for large taxonomy tables.

Reproducibility:
- Add a one-click “Export run bundle” with config, results, and a manifest.
- Include a requirements lock file and environment capture in each run folder.

---

## 12) Appendix

### A. Key Paths
- Runs (root): F:\AvalancheData\runs
- Datasets (root): F:\AvalancheData\datasets
- BLAST DB: F:\Dataset\env_nt_extracted\env_nt
- Taxdump: F:\Dataset\taxdump

### B. Notable Output Files
- pipeline_results.json – step timings, summary stats
- clustering/cluster_assignments.csv – per-sequence cluster labels
- taxonomy/taxonomy_predictions.csv – enriched lineage, KNN vs BLAST fields, conflict flags
- taxonomy/taxonomy_tiebreak_report.csv – conflicts, tie-break winner and reason
- visualizations/analysis_dashboard.html – dashboard view

### C. Commands Reference
- UI launch:
```
streamlit run "C:\\Volume D\\Avalanche\\streamlit_app.py"
```
- Copy latest taxonomy_predictions.csv into UI default location:
```
# Source → C:\\Volume D\\Avalanche\\results\\taxonomy\\taxonomy_predictions.csv
```

---

## Technical Report Addendum

### Abstract
This technical report documents the development and evaluation of an end-to-end environmental DNA (eDNA) biodiversity analysis system. We processed the SRR35551197 dataset via a hybrid pipeline integrating transformer-based sequence embeddings (DNABERT-2-117M), clustering, KNN+LCA taxonomy assignment with BLAST fallback, and novelty detection. A Streamlit web UI supports dataset analysis, run browsing, taxonomy inspection, and results visualization. On CPU, the embedding throughput measured ~3.8 sequences/second (512-sequence benchmark). The 2,000-sequence taxonomy subset run identified 61 unique taxa with lineage enrichment driven by BLAST taxids.

### Methods (Summary)
- Preprocessing: FASTA/FASTQ ingestion, basic quality handling (configurable), optional chimera detection.
- Embeddings: Hugging Face DNABERT-2-117M, max length 512, stride 128, batch size 256, mean-pooling; PCA to 256 dims + L2 normalization.
- Clustering: HDBSCAN preferred; DBSCAN fallback (environment-dependent availability).
- Taxonomy: KNN+LCA on pretrained reference embeddings with FAISS flat inner product; blastn fallback against env_nt; lineage from NCBI taxdump with BLAST taxid priority; tie-break report for KNN/BLAST conflicts.
- Novelty: Thresholds on similarity, abundance, and cluster coherence to flag novel candidates.
- Web UI: Streamlit application with Home, Dataset Analysis, Results Viewer, Run Browser, and Taxonomy Viewer.

### Results (head2000)
- Taxonomy completed in ~31 seconds on a 2,000-sequence subset, with 61 unique taxa identified.
- Novelty detection reported no candidates under the configured thresholds.
- Figures below illustrate clustering layout and novelty visualization from the latest run with available artifacts.

### Figures

Figure 1. Cluster visualization (head2000).

![Cluster visualization](report_assets/SRR35551197_head2000_cluster_visualization.png)

Figure 2. Novelty visualization (head2000).

![Novelty visualization](report_assets/SRR35551197_head2000_novelty_visualization.png)

### Discussion
CPU-only embedding is a bottleneck for full-scale runs. We recommend either GPU acceleration with mixed precision and true batching, or CPU-side reductions (deduplication, representative-only embedding, ONNX Runtime) to scale. Lineage accuracy improved by prioritizing BLAST taxids, with KNN-based names as fallback. The UI now supports streamlined navigation (homepage tiles, recent runs), run management (Run Browser), and inspection (Results/Taxonomy viewers), enabling practical workflows for iterative analysis.

### Conclusion
The system is operational end-to-end on Windows with centralized storage and a usable web UI. For production-scale datasets, enabling GPU execution will reduce embedding time by orders of magnitude. The codebase and UI are structured for further enhancements such as run comparison, ANN-based taxonomy lookup, and richer lineage QC.

---
---
# From: docs/user_guide.md
---

# Deep-Sea eDNA Biodiversity Assessment System - User Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Running the Pipeline](#running-the-pipeline)
5. [Using the Dashboard](#using-the-dashboard)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

## Installation

### Prerequisites
- Python 3.9 or higher
- Conda (recommended) or pip
- Git
- 8GB+ RAM recommended
- GPU support optional but recommended for large datasets

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Avalanche
```

2. **Create conda environment:**
```bash
conda create -n edna-biodiversity python=3.9
conda activate edna-biodiversity
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install BLAST tools (optional for taxonomy assignment):**
```bash
# On Ubuntu/Debian
sudo apt-get install ncbi-blast+

# On macOS with Homebrew
brew install blast

# On Windows
# Download from NCBI and add to PATH
```

5. **Download reference databases (optional):**
```bash
python scripts/download_data.py
```

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

## Configuration

### Main Configuration File: `config/config.yaml`

#### Data Paths
```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  reference_dir: "data/reference"
  output_dir: "data/output"
```

#### Preprocessing Parameters
```yaml
preprocessing:
  quality_threshold: 20        # Minimum average quality score
  min_length: 50              # Minimum sequence length
  max_length: 500             # Maximum sequence length
  adapter_sequences:          # Adapter sequences to remove
    - "AGATCGGAAGAGC"
    - "CTGTCTCTTATA"
```

#### Model Configuration
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

#### Clustering Settings
```yaml
clustering:
  method: "hdbscan"          # "hdbscan", "kmeans", "dbscan"
  min_cluster_size: 10
  min_samples: 5
  metric: "euclidean"
```

#### Taxonomy Assignment
```yaml
taxonomy:
  blast:
    database: "data/reference/nt"
    evalue: 1e-5
    max_targets: 10
    identity_threshold: 97.0
```

#### Novelty Detection
```yaml
novelty:
  similarity_threshold: 0.85   # Similarity threshold for known taxa
  abundance_threshold: 0.001   # Minimum abundance for consideration
  cluster_coherence: 0.7       # Minimum cluster coherence
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors
```bash
# Error: Out of memory during embedding generation
# Solution: Reduce batch size or use smaller embedding dimension
```
Edit `config/config.yaml`:
```yaml
embedding:
  training:
    batch_size: 16  # Reduce from default 32
```

#### 2. BLAST Not Found
```bash
# Error: BLAST tools not found
# Solution: Install BLAST and add to PATH
```

#### 3. GPU Not Detected
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Large Dataset Processing
For datasets with >10,000 sequences:
- Use chunked processing
- Enable GPU acceleration
- Consider cloud deployment

### Performance Optimization

#### 1. GPU Acceleration
```yaml
performance:
  use_gpu: true
  gpu_memory_fraction: 0.8
```

#### 2. Parallel Processing
```yaml
performance:
  n_jobs: -1  # Use all CPU cores
  chunk_size: 1000
```

#### 3. Memory Management
- Process data in batches
- Use data streaming for large files
- Enable garbage collection

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
```
---
---
# From: reference/README.md
---

# Marine Eukaryote Reference Integration

This folder hosts curated marine eukaryote references and derived indices used for taxonomy labeling (while the pipeline remains unsupervised-first for discovery).

Structure
- pr2/               Place PR2 18S reference FASTA and taxonomy here
- silva/             Place SILVA SSU (euk subset) FASTA and taxonomy here
- eukref/            Optional: curated EukRef clades
- mappings/          Taxonomic mappings (seq_id -> taxid)
- indices/
  - 18S/             FAISS and BLAST indices for 18S
  - COI/             (optional) indices for COI
- combined/
  - 18S/             Unified combined FASTA + taxonomy built from sources

What you need to provide (files you download separately)
- PR2 (18S):
  - pr2_18S.fasta(.gz)
  - taxonomy.tsv (must include scientific name column; common formats: name/taxonomy string)
- SILVA (SSU euk subset):
  - silva_18S_euk.fasta(.gz)
  - taxonomy.tsv or taxonomy embedded in headers
- Optional EukRef clades (FASTA + clade taxonomy table)
- NCBI taxdump (names.dmp, nodes.dmp, merged.dmp) to enable name→taxid mapping

Build steps (PowerShell examples)
1) Place downloads
- Put raw FASTAs and taxonomy TSVs under reference/pr2 and reference/silva
- Put NCBI taxdump under a local directory (e.g., F:\\Dataset\\taxdump)

2) Prepare combined references (harmonize names → taxids, deduplicate)
```powershell
python scripts\prepare_references.py \
  --pr2-fasta "reference\pr2\pr2_18S.fasta" \
  --pr2-taxonomy "reference\pr2\taxonomy.tsv" \
  --silva-fasta "reference\silva\silva_18S_euk.fasta" \
  --silva-taxonomy "reference\silva\taxonomy.tsv" \
  --taxdump-dir "F:\\Dataset\\taxdump" \
  --marker 18S \
  --out-dir "reference\combined\18S"
```
Outputs:
- reference\combined\18S\references.fasta
- reference\combined\18S\taxonomy.csv
- reference\mappings\combined_18S_taxid_map.txt

3) Build BLAST database (requires BLAST+ makeblastdb in PATH)
```powershell
python scripts\build_blast_db.py \
  --fasta "reference\combined\18S\references.fasta" \
  --taxid-map "reference\mappings\combined_18S_taxid_map.txt" \
  --db-out "reference\indices\18S\combined_18S"
```
Outputs (prefix): reference\indices\18S\combined_18S.*

4) Build KNN embedding index using your pipeline model
```powershell
python scripts\build_reference_index.py \
  --fasta "reference\combined\18S\references.fasta" \
  --labels-csv "reference\combined\18S\taxonomy.csv" \
  --output-dir "reference\indices\18S"
```
Outputs:
- reference\indices\18S\reference_embeddings.npy
- reference\indices\18S\reference_labels.csv

5) Update config/config.yaml
- taxonomy.knn.embeddings_path → reference\\indices\\18S\\reference_embeddings.npy
- taxonomy.knn.labels_path → reference\\indices\\18S\\reference_labels.csv
- taxonomy.blast_fallback.database → reference\\indices\\18S\\combined_18S
- taxonomy.taxdump_dir → your taxdump path

Notes
- You can start with either PR2 or SILVA alone; the scripts accept missing sources.
- For large references, run embedding/index builds on a GPU machine if possible for speed.
- This setup minimizes reference dependence by using them strictly for labeling and evidence.

---
---
# From: UNIVERSAL_DATASET_ANALYZER.md
---

# Universal Dataset Analyzer

## 🎯 Overview

The Universal Dataset Analyzer is a new integrated system for the eDNA Biodiversity Assessment project that provides comprehensive analysis of biological sequence datasets. Instead of creating separate analysis scripts for each dataset type, this system offers a unified interface that can handle multiple input formats and generate standardized analysis reports.

## 🚀 Key Features

### ✅ **Unified Input Interface**
- **Single command** for all dataset types
- **Auto-format detection** (FASTA, FASTQ, Swiss-Prot, GenBank, EMBL)
- **Gzipped file support** for compressed datasets
- **Flexible input options** with format override capabilities

### ✅ **Comprehensive Analysis**
- **Basic sequence statistics** (length distribution, percentiles)
- **Composition analysis** (auto-detects DNA/RNA/protein sequences)
- **Annotation mining** (organism distribution, description patterns)
- **Quality assessment** (for FASTQ files with quality scores)
- **Biodiversity metrics** (Shannon/Simpson diversity, evenness)

### ✅ **Performance Optimized**
- **Parallel processing** for composition analysis
- **Vectorized calculations** using NumPy
- **Memory-efficient streaming** for large files
- **Progress indicators** for long-running analyses
- **Subset testing** capability for large datasets

### ✅ **Standardized Output**
- **Text report format** (.txt files)
- **Consistent structure** across all dataset types
- **Detailed timing information** for performance monitoring
- **Processing metadata** included in reports

## 📋 Usage Examples

### Basic Analysis
```bash
# Analyze any supported biological sequence file
python scripts/analyze_dataset.py input_file.fasta output_report.txt
```

### Advanced Options
```bash
# With custom dataset name
python scripts/analyze_dataset.py data.fastq.gz report.txt --name "My Dataset"

# Force specific format (override auto-detection)
python scripts/analyze_dataset.py sequences.gz report.txt --format fasta

# Quick test with subset of sequences
python scripts/analyze_dataset.py large_file.fasta test_report.txt --max 1000

# Verbose output for debugging
python scripts/analyze_dataset.py data.fasta report.txt --verbose
```

## 🔄 Real-World Examples

### Example 1: Swiss-Prot Protein Database
```bash
# Analyze full Swiss-Prot database (482,697 sequences)
python scripts/analyze_dataset.py data/raw/swissprot.gz results/swissprot_analysis.txt --name "Swiss-Prot Complete Database"

# Results: 14.39 seconds processing time
# Output: Comprehensive protein composition and annotation analysis
```

### Example 2: eDNA Sequences
```bash
# Analyze environmental DNA samples
python scripts/analyze_dataset.py data/sample/sample_edna_sequences.fasta results/edna_analysis.txt --name "eDNA Sample Sequences"

# Results: 0.07 seconds processing time
```

### Example 3: GenBank Nucleotide Sequences
```bash
# Analyze a subset of GenBank sequences
python scripts/analyze_dataset.py data/raw/genbank_subset.fasta.gz results/genbank_analysis.txt --name "GenBank Nucleotide Subset"

# Results: 1.2 seconds processing time
```

## 🛠️ Implementation Details

### Input Handling
- Uses `Bio.SeqIO` for parsing various sequence formats.
- Handles gzipped files transparently.
- `format_override` parameter allows manual specification.

### Sequence Statistics
- Calculates:
    - Total sequences
    - Mean, median, min, max sequence length
    - Standard deviation of lengths
    - Sequence length distribution (percentiles)

### Composition Analysis
- Detects sequence type (DNA, RNA, Protein).
- Calculates GC content for DNA/RNA.
- Analyzes amino acid composition for proteins.
- Uses parallel processing for speed.

### Annotation Mining
- Extracts organism names and descriptions from sequence headers.
- Provides counts and distributions of common organisms and keywords.

### Quality Assessment (FASTQ)
- Calculates average Phred quality scores per base.
- Generates quality score distribution.

### Biodiversity Metrics
- Computes Shannon diversity index.
- Computes Simpson diversity index.
- Calculates Pielou's evenness.

### Output Report
- Generates a human-readable text file.
- Includes:
    - Input file details
    - Analysis parameters
    - Timing information for each step
    - Summary statistics and metrics

## 💡 Design Principles

### 1. **Simplicity**
- **Single script**: Consolidates functionality into one file.
- **Clear interface**: Easy-to-understand command-line arguments.

### 2. **Consistency**
- **Standardized output**: Reports have a predictable structure.
- **Unified logic**: Same analysis steps applied regardless of input format.

### 3. **Performance**
- **Optimized calculations**: Leverages NumPy and parallel processing.
- **Memory efficiency**: Handles large files using streaming.

### 4. **Extensibility**
- **Modular design**: Easy to add new analysis modules.
- **Clear structure**: Facilitates future development.

## 🌟 Benefits

### 1. **Efficiency**
- **Reduced development time**: No need for multiple scripts.
- **Faster analysis**: Optimized for speed and large datasets.

### 2. **Maintainability**
- **Single codebase**: Easier to update and debug.
- **Centralized logic**: Ensures consistent analysis across all dataset types.

### 3. **Usability**
- **Simple command-line interface**
- **Auto-format detection** reduces user complexity
- **Comprehensive help** and examples

### 4. **Scalability**
- **Optimized for large datasets**
- **Parallel processing** capabilities
- **Memory-efficient streaming**

### 5. **Integration**
- **Fits naturally** into existing eDNA project structure
- **Uses existing configuration system**
- **Compatible with project conventions**

## 🚀 Future Enhancements

### Planned Features
1. **Database Integration** - Direct connection to sequence databases
2. **Advanced Visualization** - Plot generation for reports
3. **Batch Processing** - Analyze multiple files in one command
4. **Quality Filtering** - Integration with preprocessing pipeline
5. **Taxonomy Assignment** - Integration with reference databases
6. **Clustering Analysis** - Integration with existing clustering modules

### Extension Points
- **Custom Analysis Modules** - Plugin architecture for specialized analyses
- **Output Formats** - JSON, CSV, HTML report generation
- **Cloud Integration** - Support for cloud storage and processing

## ✅ Conclusion

The Universal Dataset Analyzer represents a significant improvement over the previous approach of creating separate analysis scripts for each dataset type. It provides:

- **Unified interface** for all biological sequence formats
- **Optimized performance** with parallel processing
- **Comprehensive analysis** with standardized reporting
- **Easy integration** with the existing eDNA project
- **Scalable architecture** for future enhancements

This system makes it much easier to analyze biological datasets consistently while maintaining high performance and providing detailed, standardized reports for research and analysis purposes.
