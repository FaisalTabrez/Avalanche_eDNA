"""
DEPRECATED: SwissProt pipeline simulation

This script has been archived and moved to `experiments/swissprot/` and is no longer maintained.
The project is focused on eDNA workflows; protein experiments were exploratory and are stored for
historical reference only.

Archived copy: `experiments/swissprot/run_swissprot_simulation.py`
"""

print("This script has been archived. See experiments/swissprot/run_swissprot_simulation.py for the original content.")


def load_swissprot_sequences(filepath: str, max_sequences: int = 5000):
    """
    Load sequences from SwissProt gzipped FASTA file.
    
    Args:
        filepath: Path to swissprot.gz file
        max_sequences: Maximum number of sequences to load
        
    Returns:
        List of (header, sequence) tuples
    """
    sequences = []
    
    print(f"Loading sequences from {filepath}...")
    print(f"  Max sequences: {max_sequences:,}")
    
    with gzip.open(filepath, 'rt') as handle:
        for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
            if i >= max_sequences:
                break
            
            # Convert protein sequence to string
            seq_str = str(record.seq)
            
            # Only include sequences of reasonable length (50-1000 aa)
            if 50 <= len(seq_str) <= 1000:
                sequences.append((record.id, seq_str))
            
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i+1:,} sequences processed, {len(sequences):,} accepted")
    
    print(f"✓ Loaded {len(sequences):,} sequences")
    
    # Calculate length statistics
    lengths = [len(seq) for _, seq in sequences]
    print(f"  Length range: {min(lengths)}-{max(lengths)} aa")
    print(f"  Average length: {np.mean(lengths):.1f} aa")
    
    return sequences


def run_swissprot_simulation():
    """Run complete pipeline with dynamic scaling on SwissProt data."""
    
    print("="*80)
    print("SWISSPROT PIPELINE SIMULATION: Dynamic Scaling")
    print("="*80)
    print()
    
    # Configuration
    data_path = Path("data/raw/swissprot.gz")
    output_dir = Path("swissprot_simulation_output")
    max_sequences = 5000  # Limit for feasible processing
    memory_budget_gb = 4.0  # 4GB memory budget
    target_accuracy = 0.80
    
    print(f"Input: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Max sequences: {max_sequences:,}")
    print()
    
    # Step 1: Initialize pipeline with dynamic scaling
    print("="*80)
    print("STEP 1: Initialize Pipeline with Dynamic Scaling")
    print("="*80)
    print()
    
    pipeline = TaxonomyClassificationPipeline(
        output_dir=str(output_dir),
        dnabert_model_path='./models/dnabert2_cpu',
        enable_dynamic_scaling=True,
        memory_budget_gb=memory_budget_gb,
        target_accuracy=target_accuracy,
        auto_adapt=True
    )
    
    # Step 2: Load SwissProt sequences
    print("="*80)
    print("STEP 2: Load SwissProt Protein Sequences")
    print("="*80)
    print()
    
    sequences = load_swissprot_sequences(str(data_path), max_sequences)
    
    # Save to temporary FASTA for pipeline
    temp_fasta = output_dir / "temp_sequences.fasta"
    temp_fasta.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_fasta, 'w') as f:
        for header, seq in sequences:
            f.write(f">{header}\n{seq}\n")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total sequences: {len(sequences):,}")
    lengths = [len(seq) for _, seq in sequences]
    print(f"   Length range: {min(lengths)}-{max(lengths)} aa")
    print(f"   Average length: {np.mean(lengths):.1f} aa")
    
    # Step 3: Load sequences into pipeline
    pipeline.load_sequences(str(temp_fasta))
    
    # Step 4: Generate embeddings
    print("\n" + "="*80)
    print("STEP 3: Generate DNABERT-2 Embeddings")
    print("="*80)
    print()
    print("⚠️  Note: Using DNABERT-2 on protein sequences (not ideal but testing pipeline)")
    print()
    
    pipeline.generate_embeddings(batch_size=16)  # Smaller batch for proteins
    
    # Step 5: Auto-detect optimal K
    print("\n" + "="*80)
    print("STEP 4: Cluster Sequences (Auto-detect K)")
    print("="*80)
    print()
    
    # Test different cluster counts
    k_values = [10, 20, 30, 50, 75, 100]
    silhouette_scores = []
    
    print("Testing different cluster counts...")
    for k in k_values:
        labels = pipeline.cluster_sequences(n_clusters=k, method='kmeans')
        score = silhouette_score(pipeline.embeddings, labels)
        silhouette_scores.append(score)
        print(f"   K={k:3d}: Silhouette={score:.4f}")
    
    best_k = k_values[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    
    print(f"\n✓ Optimal K: {best_k} (silhouette={best_score:.4f})")
    
    # Cluster with optimal K
    cluster_labels = pipeline.cluster_sequences(
        method='kmeans',
        n_clusters=best_k
    )
    
    # Calculate clustering statistics
    cluster_sizes = np.bincount(cluster_labels)
    silhouette = silhouette_score(pipeline.embeddings, cluster_labels)
    
    print(f"\n📊 Clustering Results:")
    print(f"   Clusters: {best_k}")
    print(f"   Silhouette score: {silhouette:.4f}")
    print(f"   Cluster sizes: {cluster_sizes.tolist()}")
    
    # Step 6: Train classifier with dynamic scaling
    print("\n" + "="*80)
    print("STEP 5: Train Taxonomy Classifier (Dynamic Scaling)")
    print("="*80)
    print()
    
    training_results = pipeline.train_taxonomy_classifier(
        epochs_per_cluster=10,
        learning_rate=1e-3
    )
    
    # Step 7: Evaluate memory retention
    print("\n" + "="*80)
    print("STEP 6: Evaluate Model Memory Retention")
    print("="*80)
    print()
    
    # Test on all sequences using the classifier model
    import torch
    pipeline.classifier_model.eval()
    with torch.no_grad():
        embeddings_tensor = torch.FloatTensor(pipeline.embeddings).to(pipeline.device)
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
    
    # Memory usage
    buffer = training_results.get('dynamic_buffer')
    if buffer:
        total_samples = buffer.buffer.total_samples
        memory_mb = (total_samples * pipeline.embeddings.shape[1] * 4) / (1024 * 1024)
        memory_pct = (memory_mb / (memory_budget_gb * 1024)) * 100
        adaptations = len(buffer.adaptation_history)
        final_config = buffer.current_config
    else:
        total_samples = 0
        memory_mb = 0
        memory_pct = 0
        adaptations = 0
        final_config = None
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print()
    print(f"📊 Performance Metrics:")
    print(f"   Overall Accuracy: {overall_acc*100:.1f}% ({int(overall_acc*len(sequences))}/{len(sequences):,})")
    print(f"   Early clusters ({early_clusters[0]}-{early_clusters[-1]}): {early_acc*100:.1f}%")
    print(f"   Middle clusters ({middle_clusters[0]}-{middle_clusters[-1]}): {middle_acc*100:.1f}%")
    print(f"   Recent clusters ({recent_clusters[0]}-{recent_clusters[-1]}): {recent_acc*100:.1f}%")
    print(f"   Recency bias: {recency_bias:+.1f}pp")
    print()
    print(f"🔄 Dynamic Scaling:")
    print(f"   Adaptations: {adaptations}")
    print(f"   Memory usage: {memory_mb:.1f} MB / {memory_budget_gb*1024:.0f} MB")
    print(f"   Usage: {memory_pct:.1f}%")
    
    if final_config:
        print()
        print(f"⚙️  Final Configuration:")
        print(f"   Exemplars/cluster: {final_config.exemplars_per_cluster}")
        print(f"   Uncertainty buffer: {final_config.uncertainty_buffer_size:,}")
        print(f"   Recent buffer: {final_config.recent_buffer_size:,}")
        print(f"   Architecture: {final_config.architecture}")
        print(f"   Temperature: {final_config.temperature}")
        print(f"   Batch size: {final_config.batch_size}")
        print(f"   Replay ratio: {final_config.replay_ratio}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': str(data_path),
        'max_sequences': max_sequences,
        'sequences_loaded': len(sequences),
        'n_clusters': best_k,
        'silhouette_score': float(silhouette),
        'cluster_sizes': cluster_sizes.tolist(),
        'overall_accuracy': float(overall_acc),
        'early_accuracy': float(early_acc),
        'middle_accuracy': float(middle_acc),
        'recent_accuracy': float(recent_acc),
        'recency_bias_pp': float(recency_bias),
        'adaptations': adaptations,
        'memory_mb': float(memory_mb),
        'memory_budget_gb': memory_budget_gb,
        'memory_usage_pct': float(memory_pct),
        'per_cluster_accuracy': [float(acc) for acc in cluster_accs]
    }
    
    if final_config:
        results['final_config'] = {
            'exemplars_per_cluster': final_config.exemplars_per_cluster,
            'uncertainty_buffer_size': final_config.uncertainty_buffer_size,
            'recent_buffer_size': final_config.recent_buffer_size,
            'architecture': final_config.architecture,
            'temperature': final_config.temperature,
            'batch_size': final_config.batch_size,
            'replay_ratio': final_config.replay_ratio
        }
    
    results_file = output_dir / "simulation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"💾 Results saved to: {results_file}")
    
    # Final verdict
    print()
    print("="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print()
    print(f"📋 Verdict:")
    if overall_acc >= 0.70:
        print(f"   ✓ GOOD: Dynamic scaling achieved >{int(overall_acc*100)}% retention on SwissProt.")
    elif overall_acc >= 0.50:
        print(f"   ⚠️  FAIR: {int(overall_acc*100)}% retention - room for improvement.")
    else:
        print(f"   ✗ POOR: Only {int(overall_acc*100)}% retention - needs investigation.")
    
    if adaptations > 0:
        print(f"   ✓ Dynamic scaling adapted {adaptations} time(s) during training.")
    else:
        print(f"   ℹ️ No adaptations needed for this scale.")
    
    if memory_pct < 50:
        print(f"   ✓ Memory usage well within budget.")
    elif memory_pct < 80:
        print(f"   ⚠️  Memory usage moderate.")
    else:
        print(f"   ⚠️  Memory usage approaching limit.")


if __name__ == '__main__':
    run_swissprot_simulation()
