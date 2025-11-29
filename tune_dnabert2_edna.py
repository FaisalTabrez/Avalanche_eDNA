#!/usr/bin/env python3
"""
DNABERT-2 eDNA Hyperparameter Tuning Script

Systematic tuning of key hyperparameters for eDNA analysis:
- K-mer size (4-7)
- Projection dimension (64-256)
- Exemplars per cluster (25-200)
- Replay ratio (0.3-0.7)

Identifies optimal configurations and clusters showing forgetting.
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# UTF-8 encoding fix for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import json
import time
from itertools import product
from datetime import datetime

from src.models.tokenizer import DNATokenizer
from src.models.embeddings import DNAContrastiveModel
from src.models.trainer import EmbeddingTrainer
from src.models.dynamic_hybrid_buffer import ScalingConfig


def generate_edna_test_data(n_clusters=5, samples_per_cluster=100, seq_length=150):
    """
    Generate synthetic eDNA-like sequences for tuning.
    
    Args:
        n_clusters: Number of taxonomic clusters
        samples_per_cluster: Sequences per cluster
        seq_length: Length of each sequence
        
    Returns:
        sequences, labels
    """
    sequences = []
    labels = []
    
    # Generate cluster-specific motifs (simulate taxonomic signatures)
    cluster_motifs = [
        "ATCG" * 10,  # Cluster 0 motif
        "GCTA" * 10,  # Cluster 1 motif
        "CGAT" * 10,  # Cluster 2 motif
        "TAGC" * 10,  # Cluster 3 motif
        "CTAG" * 10,  # Cluster 4 motif
    ]
    
    for cluster_id in range(n_clusters):
        motif = cluster_motifs[cluster_id % len(cluster_motifs)]
        
        for _ in range(samples_per_cluster):
            # Start with motif
            seq = motif[:seq_length // 2]
            
            # Add random DNA to reach target length
            remaining = seq_length - len(seq)
            random_dna = ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=remaining))
            seq = seq + random_dna
            
            sequences.append(seq[:seq_length])
            labels.append(cluster_id)
    
    return sequences, np.array(labels)


def run_tuning_experiment(config_dict, sequences, labels, run_id):
    """
    Run a single tuning experiment with given configuration.
    
    Args:
        config_dict: Configuration dictionary with hyperparameters
        sequences: DNA sequences
        labels: Cluster labels
        run_id: Unique identifier for this run
        
    Returns:
        results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Run {run_id}: Testing configuration")
    print(f"{'='*80}")
    print(f"  K-mer size: {config_dict['kmer_size']}")
    print(f"  Projection dim: {config_dict['projection_dim']}")
    print(f"  Exemplars/cluster: {config_dict['exemplars_per_cluster']}")
    print(f"  Replay ratio: {config_dict['replay_ratio']}")
    print(f"  Memory budget: {config_dict['memory_budget_gb']} GB")
    
    start_time = time.time()
    
    try:
        # Create tokenizer with specified k-mer size
        tokenizer = DNATokenizer(
            encoding_type='kmer',
            kmer_size=config_dict['kmer_size']
        )
        
        # Create backbone model
        from src.models.embeddings import DNATransformerEmbedder
        backbone = DNATransformerEmbedder(
            vocab_size=tokenizer.vocab_size,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.1
        )
        
        # Wrap with contrastive learning
        model = DNAContrastiveModel(
            backbone_model=backbone,
            projection_dim=config_dict['projection_dim']
        )
        
        # Create scaling config
        n_clusters = len(np.unique(labels))
        scaling_config = ScalingConfig.auto_scale(
            n_clusters=n_clusters,
            dataset_size=len(sequences),
            memory_budget_gb=config_dict['memory_budget_gb'],
            target_accuracy=0.85
        )
        
        # Override specific parameters
        scaling_config.exemplars_per_cluster = config_dict['exemplars_per_cluster']
        scaling_config.replay_ratio = config_dict['replay_ratio']
        
        # Create trainer
        trainer = EmbeddingTrainer(
            model=model,
            tokenizer=tokenizer,
            device='cpu',
            scaling_config=scaling_config
        )
        
        # Train with dynamic scaling
        history = trainer.train_with_dynamic_scaling(
            sequences=sequences,
            labels=labels,
            epochs_per_task=3,  # Quick tuning runs
            learning_rate=1e-4,
            batch_size=16,
            validation_split=0.2
        )
        
        # Evaluate forgetting
        # Extract embeddings and make predictions
        all_embeddings = []
        for seq in sequences:
            emb = trainer.extract_embeddings([seq])
            all_embeddings.append(emb[0])
        
        embeddings = np.array(all_embeddings)
        
        # Simple prediction using nearest cluster centroid
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = embeddings[mask].mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        
        # Calculate distances and predict
        predictions = []
        for emb in embeddings:
            distances = np.linalg.norm(centroids - emb, axis=1)
            predictions.append(np.argmin(distances))
        predictions = np.array(predictions)
        
        # Overall accuracy
        overall_acc = np.mean(predictions == labels)
        
        # Per-cluster accuracy
        cluster_accs = []
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            if mask.sum() > 0:
                acc = np.mean(predictions[mask] == labels[mask])
                cluster_accs.append(float(acc))
            else:
                cluster_accs.append(0.0)
        
        # Early/Middle/Recent breakdown
        n_clusters = len(unique_labels)
        early_clusters = list(range(0, n_clusters // 3))
        middle_clusters = list(range(n_clusters // 3, 2 * n_clusters // 3))
        recent_clusters = list(range(2 * n_clusters // 3, n_clusters))
        
        early_mask = np.isin(labels, early_clusters)
        middle_mask = np.isin(labels, middle_clusters)
        recent_mask = np.isin(labels, recent_clusters)
        
        early_acc = np.mean(predictions[early_mask] == labels[early_mask]) if early_mask.sum() > 0 else 0
        middle_acc = np.mean(predictions[middle_mask] == labels[middle_mask]) if middle_mask.sum() > 0 else 0
        recent_acc = np.mean(predictions[recent_mask] == labels[recent_mask]) if recent_mask.sum() > 0 else 0
        
        # Forgetting metrics
        forgetting_score = (recent_acc - early_acc)  # Positive = recency bias
        avg_cluster_acc = np.mean(cluster_accs)
        min_cluster_acc = np.min(cluster_accs)
        
        runtime = time.time() - start_time
        
        results = {
            'run_id': run_id,
            'config': config_dict,
            'overall_accuracy': float(overall_acc),
            'early_accuracy': float(early_acc),
            'middle_accuracy': float(middle_acc),
            'recent_accuracy': float(recent_acc),
            'forgetting_score': float(forgetting_score),
            'avg_cluster_accuracy': float(avg_cluster_acc),
            'min_cluster_accuracy': float(min_cluster_acc),
            'cluster_accuracies': cluster_accs,
            'final_buffer_size': history['buffer_size'][-1] if 'buffer_size' in history else 0,
            'final_memory_mb': history['memory_mb'][-1] if 'memory_mb' in history else 0,
            'runtime_seconds': runtime,
            'success': True
        }
        
        print(f"\n✓ Run {run_id} completed:")
        print(f"  Overall accuracy: {overall_acc*100:.1f}%")
        print(f"  Early/Middle/Recent: {early_acc*100:.1f}% / {middle_acc*100:.1f}% / {recent_acc*100:.1f}%")
        print(f"  Forgetting score: {forgetting_score:+.3f} (lower = less forgetting)")
        print(f"  Min cluster acc: {min_cluster_acc*100:.1f}%")
        print(f"  Runtime: {runtime:.1f}s")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Run {run_id} FAILED: {e}")
        return {
            'run_id': run_id,
            'config': config_dict,
            'success': False,
            'error': str(e)
        }


def main():
    """Main tuning workflow"""
    
    print("="*80)
    print("DNABERT-2 eDNA HYPERPARAMETER TUNING")
    print("="*80)
    print()
    
    # Configuration space
    kmer_sizes = [4, 5, 6, 7]
    projection_dims = [64, 128, 256]
    exemplars_per_cluster = [25, 50, 100, 150]
    replay_ratios = [0.3, 0.5, 0.7]
    memory_budget = 1.0  # 1GB fixed
    
    # Generate grid
    all_configs = []
    for kmer, proj_dim, exemplars, replay in product(
        kmer_sizes, projection_dims, exemplars_per_cluster, replay_ratios
    ):
        all_configs.append({
            'kmer_size': kmer,
            'projection_dim': proj_dim,
            'exemplars_per_cluster': exemplars,
            'replay_ratio': replay,
            'memory_budget_gb': memory_budget
        })
    
    print(f"📊 Total configurations to test: {len(all_configs)}")
    print()
    
    # Generate test data
    print("Generating eDNA test data...")
    sequences, labels = generate_edna_test_data(
        n_clusters=5,
        samples_per_cluster=100,
        seq_length=150
    )
    print(f"✓ Generated {len(sequences)} sequences across {len(np.unique(labels))} clusters")
    print()
    
    # Run tuning experiments
    results = []
    for i, config in enumerate(all_configs[:12], 1):  # Run subset for testing (12 configs)
        result = run_tuning_experiment(config, sequences, labels, run_id=i)
        results.append(result)
        
        # Save intermediate results
        if i % 5 == 0:
            temp_df = pd.DataFrame([r for r in results if r['success']])
            temp_df.to_csv('tuning_results_partial.csv', index=False)
            print(f"\n💾 Saved partial results ({i} runs completed)")
    
    # Analyze results
    print("\n" + "="*80)
    print("TUNING RESULTS ANALYSIS")
    print("="*80)
    print()
    
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        print("❌ No successful runs!")
        return
    
    df = pd.DataFrame(successful_results)
    
    # Save full results
    output_dir = Path("tuning_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"dnabert2_edna_tuning_{timestamp}.csv"
    df.to_csv(results_file, index=False)
    
    # Save detailed JSON
    json_file = output_dir / f"dnabert2_edna_tuning_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Full results saved:")
    print(f"   CSV: {results_file}")
    print(f"   JSON: {json_file}")
    print()
    
    # Find best configurations
    print("📊 TOP 5 CONFIGURATIONS BY OVERALL ACCURACY:")
    print("-" * 80)
    top_by_accuracy = df.nlargest(5, 'overall_accuracy')
    for idx, row in top_by_accuracy.iterrows():
        print(f"\nRank {list(top_by_accuracy.index).index(idx) + 1}:")
        print(f"  Overall Accuracy: {row['overall_accuracy']*100:.1f}%")
        print(f"  K-mer: {row['config']['kmer_size']}, Proj Dim: {row['config']['projection_dim']}")
        print(f"  Exemplars: {row['config']['exemplars_per_cluster']}, Replay: {row['config']['replay_ratio']}")
        print(f"  Forgetting Score: {row['forgetting_score']:+.3f}")
        print(f"  Min Cluster Acc: {row['min_cluster_accuracy']*100:.1f}%")
    
    print("\n" + "="*80)
    print("📊 TOP 5 CONFIGURATIONS BY LOWEST FORGETTING:")
    print("-" * 80)
    top_by_forgetting = df.nsmallest(5, 'forgetting_score')
    for idx, row in top_by_forgetting.iterrows():
        print(f"\nRank {list(top_by_forgetting.index).index(idx) + 1}:")
        print(f"  Forgetting Score: {row['forgetting_score']:+.3f} (lower = better retention)")
        print(f"  Overall Accuracy: {row['overall_accuracy']*100:.1f}%")
        print(f"  K-mer: {row['config']['kmer_size']}, Proj Dim: {row['config']['projection_dim']}")
        print(f"  Exemplars: {row['config']['exemplars_per_cluster']}, Replay: {row['config']['replay_ratio']}")
        print(f"  Early/Recent: {row['early_accuracy']*100:.1f}% / {row['recent_accuracy']*100:.1f}%")
    
    print("\n" + "="*80)
    print("📊 BEST BALANCED CONFIGURATION (Accuracy + Low Forgetting):")
    print("-" * 80)
    
    # Composite score: maximize accuracy, minimize forgetting
    df['composite_score'] = df['overall_accuracy'] - abs(df['forgetting_score'])
    best_balanced = df.nlargest(1, 'composite_score').iloc[0]
    
    print(f"\n  Overall Accuracy: {best_balanced['overall_accuracy']*100:.1f}%")
    print(f"  Forgetting Score: {best_balanced['forgetting_score']:+.3f}")
    print(f"  K-mer size: {best_balanced['config']['kmer_size']}")
    print(f"  Projection dim: {best_balanced['config']['projection_dim']}")
    print(f"  Exemplars/cluster: {best_balanced['config']['exemplars_per_cluster']}")
    print(f"  Replay ratio: {best_balanced['config']['replay_ratio']}")
    print(f"  Min cluster accuracy: {best_balanced['min_cluster_accuracy']*100:.1f}%")
    
    # Recommended configuration
    print("\n" + "="*80)
    print("💡 RECOMMENDED CONFIGURATION FOR eDNA:")
    print("="*80)
    print()
    print("config.yaml updates:")
    print("```yaml")
    print(f"embedding:")
    print(f"  kmer_size: {best_balanced['config']['kmer_size']}")
    print(f"  training:")
    print(f"    projection_dim: {best_balanced['config']['projection_dim']}")
    print()
    print(f"# For ScalingConfig.auto_scale() or manual config:")
    print(f"# exemplars_per_cluster: {best_balanced['config']['exemplars_per_cluster']}")
    print(f"# replay_ratio: {best_balanced['config']['replay_ratio']}")
    print("```")
    print()
    
    # Save recommendation
    recommendation = {
        'timestamp': timestamp,
        'best_configuration': {
            'kmer_size': int(best_balanced['config']['kmer_size']),
            'projection_dim': int(best_balanced['config']['projection_dim']),
            'exemplars_per_cluster': int(best_balanced['config']['exemplars_per_cluster']),
            'replay_ratio': float(best_balanced['config']['replay_ratio'])
        },
        'performance': {
            'overall_accuracy': float(best_balanced['overall_accuracy']),
            'forgetting_score': float(best_balanced['forgetting_score']),
            'early_accuracy': float(best_balanced['early_accuracy']),
            'middle_accuracy': float(best_balanced['middle_accuracy']),
            'recent_accuracy': float(best_balanced['recent_accuracy']),
            'min_cluster_accuracy': float(best_balanced['min_cluster_accuracy'])
        }
    }
    
    recommendation_file = output_dir / f"recommended_config_{timestamp}.json"
    with open(recommendation_file, 'w') as f:
        json.dump(recommendation, f, indent=2)
    
    print(f"💾 Recommendation saved to: {recommendation_file}")
    print()
    print("="*80)
    print("✅ TUNING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
