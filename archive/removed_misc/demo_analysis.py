"""
Demo notebook showing eDNA biodiversity analysis workflow
"""

# Install required packages first (uncomment if needed)
# !pip install torch transformers scikit-learn streamlit plotly biopython

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path (adjust path as needed)
sys.path.append('../src')

# %% [markdown]
# # eDNA Biodiversity Assessment Demo
# 
# This notebook demonstrates the complete workflow for analyzing environmental DNA (eDNA) 
# sequences to assess biodiversity and identify potentially novel taxa.

# %% [markdown]
# ## 1. Setup and Configuration

from utils.config import config
from models.tokenizer import DNATokenizer
from models.embeddings import DNATransformerEmbedder
from clustering.algorithms import EmbeddingClusterer
from novelty.detection import NoveltyAnalyzer
from visualization.plots import BiodiversityPlotter

print("✅ All modules imported successfully!")
print(f"📁 Working directory: {Path.cwd()}")

# %% [markdown]
# ## 2. Create Sample eDNA Data

# Generate mock eDNA sequences
def create_sample_sequences(n_sequences=500, seq_length_range=(100, 400)):
    """Create mock eDNA sequences for demonstration"""
    nucleotides = ['A', 'T', 'G', 'C']
    sequences = []
    
    # Create sequences with some structure to simulate real data
    # Group 1: Marine bacteria-like sequences
    for i in range(n_sequences // 3):
        length = np.random.randint(*seq_length_range)
        # Add some bias toward certain patterns
        weights = [0.3, 0.2, 0.3, 0.2]  # Slight AT bias
        sequence = ''.join(np.random.choice(nucleotides, size=length, p=weights))
        sequences.append(sequence)
    
    # Group 2: Archaea-like sequences  
    for i in range(n_sequences // 3):
        length = np.random.randint(*seq_length_range)
        weights = [0.2, 0.3, 0.2, 0.3]  # Slight GC bias
        sequence = ''.join(np.random.choice(nucleotides, size=length, p=weights))
        sequences.append(sequence)
    
    # Group 3: Mixed/novel sequences
    for i in range(n_sequences - 2 * (n_sequences // 3)):
        length = np.random.randint(*seq_length_range)
        weights = [0.25, 0.25, 0.25, 0.25]  # Balanced
        sequence = ''.join(np.random.choice(nucleotides, size=length))
        sequences.append(sequence)
    
    return sequences

# Create sample data
sample_sequences = create_sample_sequences(n_sequences=500)
print(f"📊 Created {len(sample_sequences)} sample eDNA sequences")
print(f"📏 Sequence lengths: {min(len(s) for s in sample_sequences)} - {max(len(s) for s in sample_sequences)} bp")

# Display first few sequences
print("\n🔬 Sample sequences:")
for i, seq in enumerate(sample_sequences[:3]):
    print(f"Sequence {i+1}: {seq[:50]}... (length: {len(seq)})")

# %% [markdown]
# ## 3. Sequence Tokenization

# Initialize DNA tokenizer
tokenizer = DNATokenizer(
    encoding_type="kmer",
    kmer_size=6,
    stride=1,
    add_special_tokens=True
)

print(f"🔤 Tokenizer vocabulary size: {tokenizer.vocab_size}")
print(f"🧬 K-mer size: {tokenizer.kmer_size}")

# Example of tokenization
example_sequence = sample_sequences[0]
encoded = tokenizer.encode_sequence(example_sequence, max_length=100)

print(f"\n📝 Tokenization example:")
print(f"Original sequence: {example_sequence[:30]}...")
print(f"Encoded length: {len(encoded['input_ids'])}")
print(f"First 10 tokens: {encoded['input_ids'][:10]}")

# Batch encode all sequences
print("\n⚙️ Encoding all sequences...")
batch_encoded = tokenizer.encode_sequences(sample_sequences, max_length=150)
print(f"✅ Encoded {batch_encoded['input_ids'].shape[0]} sequences")
print(f"📊 Tensor shape: {batch_encoded['input_ids'].shape}")

# %% [markdown]
# ## 4. Generate Sequence Embeddings

# Create embedding model
embedding_model = DNATransformerEmbedder(
    vocab_size=tokenizer.vocab_size,
    d_model=128,  # Smaller for demo
    nhead=8,
    num_layers=4,
    dropout=0.1,
    pooling_strategy='cls'
)

print(f"🤖 Model parameters: {sum(p.numel() for p in embedding_model.parameters()):,}")

# Generate embeddings (using mock embeddings for demo)
print("\n🧠 Generating sequence embeddings...")

# For demo purposes, create structured mock embeddings
# In practice, you would train or use a pre-trained model
n_sequences = len(sample_sequences)
embedding_dim = 128

# Create embeddings with some cluster structure
embeddings = []
for i in range(n_sequences):
    if i < n_sequences // 3:
        # Cluster 1 (marine bacteria)
        center = np.array([1.0, 0.5, -0.2] * (embedding_dim // 3))
        noise = np.random.normal(0, 0.3, embedding_dim)
        embedding = center + noise
    elif i < 2 * n_sequences // 3:
        # Cluster 2 (archaea)
        center = np.array([-0.5, 1.2, 0.1] * (embedding_dim // 3))
        noise = np.random.normal(0, 0.3, embedding_dim)
        embedding = center + noise
    else:
        # Cluster 3 (mixed/novel)
        center = np.array([0.2, -0.8, 1.5] * (embedding_dim // 3))
        noise = np.random.normal(0, 0.5, embedding_dim)
        embedding = center + noise
    
    embeddings.append(embedding)

embeddings = np.array(embeddings)
print(f"✅ Generated embeddings with shape: {embeddings.shape}")

# %% [markdown]
# ## 5. Sequence Clustering

# Initialize clusterer
clusterer = EmbeddingClusterer(
    method="hdbscan",
    min_cluster_size=15,
    min_samples=5,
    metric="euclidean"
)

print("🔗 Performing sequence clustering...")

# Perform clustering
cluster_labels = clusterer.fit(embeddings)

# Display clustering results
stats = clusterer.cluster_stats
print(f"\n📊 Clustering Results:")
print(f"   • Number of clusters: {stats['n_clusters']}")
print(f"   • Number of noise points: {stats['n_noise_points']}")
print(f"   • Silhouette score: {stats.get('silhouette_score', 'N/A')}")

# Show cluster sizes
print("\n📈 Cluster sizes:")
for cluster_id, size in stats['cluster_sizes'].items():
    if cluster_id != '-1':
        print(f"   Cluster {cluster_id}: {size} sequences")

# Generate 2D visualization
reduced_embeddings = clusterer.reduce_dimensions(n_components=2)
print(f"\n🎨 Generated 2D projection: {reduced_embeddings.shape}")

# %% [markdown]
# ## 6. Taxonomic Assignment (Mock)

# Create mock taxonomy assignments
print("🏷️ Simulating taxonomy assignment...")

# Mock taxonomy based on clusters
taxonomies = []
for label in cluster_labels:
    if label == 0:
        taxonomy = np.random.choice(['Bacteria;Proteobacteria', 'Bacteria;Firmicutes'], p=[0.7, 0.3])
    elif label == 1:
        taxonomy = np.random.choice(['Archaea;Euryarchaeota', 'Archaea;Crenarchaeota'], p=[0.6, 0.4])
    elif label == 2:
        taxonomy = np.random.choice(['Eukaryota;Stramenopiles', 'Eukaryota;Alveolata', 'Unknown'], p=[0.4, 0.3, 0.3])
    else:  # Noise points
        taxonomy = 'Unknown'
    taxonomies.append(taxonomy)

# Count taxonomic groups
from collections import Counter
taxonomy_counts = Counter(taxonomies)

print("📋 Taxonomic composition:")
for taxonomy, count in taxonomy_counts.most_common():
    percentage = (count / len(taxonomies)) * 100
    print(f"   • {taxonomy}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ## 7. Novelty Detection

# Initialize novelty analyzer
novelty_analyzer = NoveltyAnalyzer(
    similarity_threshold=0.85,
    abundance_threshold=0.001,
    cluster_coherence_threshold=0.7
)

print("🆕 Detecting potentially novel taxa...")

# Create reference embeddings (simulating known taxa database)
# Take a subset of our embeddings as "known" reference
reference_indices = np.random.choice(len(embeddings), size=200, replace=False)
reference_embeddings = embeddings[reference_indices]

# Use remaining as query sequences
query_indices = np.setdiff1d(np.arange(len(embeddings)), reference_indices)
query_embeddings = embeddings[query_indices]
query_sequences = [sample_sequences[i] for i in query_indices]
query_cluster_labels = cluster_labels[query_indices]

# Run novelty analysis
novelty_results = novelty_analyzer.analyze_novelty(
    query_embeddings=query_embeddings,
    reference_embeddings=reference_embeddings,
    query_sequences=query_sequences,
    cluster_labels=query_cluster_labels
)

print(f"\n🔍 Novelty Detection Results:")
print(f"   • Total sequences analyzed: {novelty_results['total_sequences']}")
print(f"   • Novel candidates found: {novelty_results['novel_candidates']}")
print(f"   • Novelty percentage: {novelty_results['novel_percentage']:.1f}%")

# Show novel sequences
if novelty_results['novel_sequences']:
    print(f"\n🌟 Example novel sequences:")
    for i, seq in enumerate(novelty_results['novel_sequences'][:3]):
        print(f"   Novel {i+1}: {seq[:40]}...")

# %% [markdown]
# ## 8. Visualization

# Initialize plotter
plotter = BiodiversityPlotter()

print("📊 Creating visualizations...")

# 1. Sequence length distribution
sequence_lengths = [len(seq) for seq in sample_sequences]
fig1 = plotter.plot_sequence_length_distribution(sequence_lengths)
fig1.show()

# 2. Taxonomic composition
fig2 = plotter.plot_taxonomic_composition(
    taxonomy_counts, 
    plot_type="pie", 
    top_n=10
)
fig2.show()

# 3. Cluster visualization
ensemble_predictions = np.array(novelty_results['predictions']['ensemble'])
# Extend predictions to match all sequences
full_predictions = np.ones(len(embeddings))
full_predictions[query_indices] = ensemble_predictions

fig3 = plotter.plot_cluster_visualization(
    embeddings_2d=reduced_embeddings,
    cluster_labels=cluster_labels,
    novelty_labels=full_predictions
)
fig3.show()

# 4. Novelty analysis
fig4 = plotter.plot_novelty_analysis(
    novelty_scores=np.array(novelty_results['scores']['ensemble']),
    novelty_predictions=ensemble_predictions,
    threshold=0.0
)
fig4.show()

# 5. Diversity indices (mock)
diversity_indices = {
    'Shannon': np.random.uniform(2.5, 4.0),
    'Simpson': np.random.uniform(0.7, 0.95),
    'Chao1': np.random.uniform(80, 150),
    'ACE': np.random.uniform(90, 160)
}

fig5 = plotter.plot_diversity_indices(diversity_indices)
fig5.show()

print("✅ All visualizations generated!")

# %% [markdown]
# ## 9. Summary Report

print("\n" + "="*60)
print("🌊 eDNA BIODIVERSITY ASSESSMENT SUMMARY")
print("="*60)

print(f"\n📊 DATASET OVERVIEW:")
print(f"   • Total sequences processed: {len(sample_sequences)}")
print(f"   • Average sequence length: {np.mean(sequence_lengths):.1f} bp")
print(f"   • Sequence length range: {min(sequence_lengths)} - {max(sequence_lengths)} bp")

print(f"\n🔗 CLUSTERING RESULTS:")
print(f"   • Clustering method: HDBSCAN")
print(f"   • Number of clusters: {stats['n_clusters']}")
print(f"   • Largest cluster size: {max(int(size) for label, size in stats['cluster_sizes'].items() if label != '-1')}")
print(f"   • Unclustered sequences: {stats['n_noise_points']}")

print(f"\n🏷️ TAXONOMIC DIVERSITY:")
print(f"   • Major taxonomic groups: {len([t for t in taxonomy_counts.keys() if t != 'Unknown'])}")
print(f"   • Most abundant group: {taxonomy_counts.most_common(1)[0][0]}")
print(f"   • Unknown/unassigned: {taxonomy_counts['Unknown']} sequences")

print(f"\n🆕 NOVELTY DETECTION:")
print(f"   • Reference database size: {len(reference_embeddings)} sequences")
print(f"   • Query sequences analyzed: {len(query_sequences)}")
print(f"   • Novel candidates identified: {novelty_results['novel_candidates']}")
print(f"   • Novelty rate: {novelty_results['novel_percentage']:.1f}%")

print(f"\n📈 BIODIVERSITY METRICS:")
for metric, value in diversity_indices.items():
    print(f"   • {metric} index: {value:.2f}")

print(f"\n🎯 KEY FINDINGS:")
print(f"   • Detected {stats['n_clusters']} distinct sequence clusters")
print(f"   • Identified {len(taxonomy_counts)} taxonomic groups")
print(f"   • Found {novelty_results['novel_candidates']} potential novel taxa")
print(f"   • Overall diversity appears {'high' if diversity_indices['Shannon'] > 3 else 'moderate'}")

print("\n" + "="*60)
print("✅ Analysis completed successfully!")
print("🔬 Results ready for further investigation and validation")
print("="*60)

# %% [markdown]
# ## 10. Export Results

# Create results summary
results_summary = {
    'dataset': {
        'total_sequences': len(sample_sequences),
        'sequence_lengths': sequence_lengths,
        'avg_length': float(np.mean(sequence_lengths))
    },
    'clustering': {
        'method': 'hdbscan',
        'n_clusters': stats['n_clusters'],
        'n_noise': stats['n_noise_points'],
        'cluster_sizes': {k: int(v) for k, v in stats['cluster_sizes'].items()}
    },
    'taxonomy': {
        'composition': dict(taxonomy_counts),
        'n_groups': len(taxonomy_counts)
    },
    'novelty': {
        'total_analyzed': novelty_results['total_sequences'],
        'novel_candidates': novelty_results['novel_candidates'],
        'novel_percentage': novelty_results['novel_percentage']
    },
    'diversity': diversity_indices
}

print("💾 Results summary created and ready for export")
print("📁 Use the pipeline script for automated analysis of real data")

# %% [markdown]
# ## Next Steps
# 
# This demo showed the complete workflow for eDNA biodiversity assessment. For real data analysis:
# 
# 1. **Prepare your data**: Use FASTQ/FASTA files from sequencing
# 2. **Run preprocessing**: Quality filtering, adapter trimming, chimera removal
# 3. **Train or load models**: Use pre-trained embeddings or train custom models
# 4. **Configure parameters**: Adjust clustering and novelty detection thresholds
# 5. **Validate results**: Confirm novel taxa through targeted sequencing
# 
# ### Command Line Usage:
# ```bash
# # Complete analysis
# python scripts/run_pipeline.py --input your_sequences.fasta --output results/
# 
# # Launch interactive dashboard
# python scripts/launch_dashboard.py
# ```
# 
# ### Key Resources:
# - User Guide: `docs/user_guide.md`
# - API Reference: `docs/api_reference.md`  
# - Configuration: `config/config.yaml`
# - Tests: `python tests/test_system.py`

print("\n🎉 Demo completed! Check the documentation for more details.")