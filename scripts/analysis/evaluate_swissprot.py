"""
Evaluate SwissProt simulation results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_taxonomy_pipeline_v2 import TaxonomyClassificationPipeline
import numpy as np
import torch
import json
from datetime import datetime

output_dir = Path("swissprot_simulation_output")

# Load the trained pipeline
print("Loading trained pipeline...")
pipeline = TaxonomyClassificationPipeline(
    output_dir=str(output_dir),
    dnabert_model_path='./models/dnabert2_cpu',
    enable_dynamic_scaling=True,
    memory_budget_gb=4.0,
    target_accuracy=0.80,
    auto_adapt=True
)

# Load embeddings
embeddings = np.load(output_dir / 'embeddings' / 'dnabert2_embeddings.npy')

# Load the latest checkpoint to get the correct number of clusters
checkpoints = sorted((output_dir / 'checkpoints').glob('checkpoint_*.pt'))
if not checkpoints:
    print("ERROR: No checkpoints found!")
    sys.exit(1)

latest_checkpoint = checkpoints[-1]
print(f"Loading checkpoint: {latest_checkpoint.name}")

checkpoint_data = torch.load(latest_checkpoint, map_location=pipeline.device)

# Get the actual number of clusters from the model checkpoint
architecture = checkpoint_data.get('architecture', [256, 128])
# Find output layer size from the state dict
output_layer_key = [k for k in checkpoint_data['model_state_dict'].keys() if 'bias' in k][-1]
best_k = checkpoint_data['model_state_dict'][output_layer_key].shape[0]

print(f"Model trained with {best_k} clusters")

# Recreate clustering with the correct K
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print(f"Re-clustering with K={best_k}...")
clusterer = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = clusterer.fit_predict(embeddings)
silhouette = silhouette_score(embeddings, cluster_labels)
print(f"Silhouette score: {silhouette:.4f}")

print(f"Loaded {len(embeddings)} sequences with {best_k} clusters")

# Rebuild the model architecture
import torch.nn as nn
embedding_dim = embeddings.shape[1]
architecture = checkpoint_data.get('architecture', [256, 128])

layers = []
prev_dim = embedding_dim
for hidden_dim in architecture:
    layers.extend([
        nn.Linear(prev_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3)
    ])
    prev_dim = hidden_dim
layers.append(nn.Linear(prev_dim, best_k))

pipeline.classifier_model = nn.Sequential(*layers).to(pipeline.device)
pipeline.classifier_model.load_state_dict(checkpoint_data['model_state_dict'])

print(f"Model loaded: {embedding_dim} -> {' -> '.join(map(str, architecture))} -> {best_k}")

# Make predictions
print("Making predictions...")
pipeline.classifier_model.eval()
with torch.no_grad():
    embeddings_tensor = torch.FloatTensor(embeddings).to(pipeline.device)
    logits = pipeline.classifier_model(embeddings_tensor)
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

# Overall accuracy
overall_acc = np.mean(predictions == cluster_labels)

# Per-cluster accuracy breakdown
cluster_accs = []
for cluster_id in range(best_k):
    mask = cluster_labels == cluster_id
    if mask.sum() > 0:
        acc = np.mean(predictions[mask] == cluster_labels[mask])
        cluster_accs.append(acc)

# Early, middle, recent breakdown
n_clusters = best_k
early_clusters = list(range(0, n_clusters // 3))
middle_clusters = list(range(n_clusters // 3, 2 * n_clusters // 3))
recent_clusters = list(range(2 * n_clusters // 3, n_clusters))

early_mask = np.isin(cluster_labels, early_clusters)
middle_mask = np.isin(cluster_labels, middle_clusters)
recent_mask = np.isin(cluster_labels, recent_clusters)

early_acc = np.mean(predictions[early_mask] == cluster_labels[early_mask]) if early_mask.sum() > 0 else 0
middle_acc = np.mean(predictions[middle_mask] == cluster_labels[middle_mask]) if middle_mask.sum() > 0 else 0
recent_acc = np.mean(predictions[recent_mask] == cluster_labels[recent_mask]) if recent_mask.sum() > 0 else 0

# Recency bias
recency_bias = (recent_acc - early_acc) * 100

print("\n" + "="*80)
print("SWISSPROT EVALUATION RESULTS")
print("="*80)
print()
print(f"📊 Performance Metrics:")
print(f"   Overall Accuracy: {overall_acc*100:.1f}% ({int(overall_acc*len(embeddings))}/{len(embeddings):,})")
print(f"   Early clusters ({early_clusters[0]}-{early_clusters[-1]}): {early_acc*100:.1f}%")
print(f"   Middle clusters ({middle_clusters[0]}-{middle_clusters[-1]}): {middle_acc*100:.1f}%")
print(f"   Recent clusters ({recent_clusters[0]}-{recent_clusters[-1]}): {recent_acc*100:.1f}%")
print(f"   Recency bias: {recency_bias:+.1f}pp")
print()
print(f"Per-cluster accuracy:")
for i, acc in enumerate(cluster_accs):
    print(f"   Cluster {i:2d}: {acc*100:.1f}%")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'SwissProt',
    'sequences': len(embeddings),
    'n_clusters': best_k,
    'overall_accuracy': float(overall_acc),
    'early_accuracy': float(early_acc),
    'middle_accuracy': float(middle_acc),
    'recent_accuracy': float(recent_acc),
    'recency_bias_pp': float(recency_bias),
    'per_cluster_accuracy': [float(acc) for acc in cluster_accs]
}

results_file = output_dir / "evaluation_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print()
print(f"💾 Results saved to: {results_file}")
