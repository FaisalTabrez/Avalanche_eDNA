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
    print(f"Identity: {result['identity']:.1f}%")
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