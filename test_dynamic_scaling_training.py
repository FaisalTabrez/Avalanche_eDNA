#!/usr/bin/env python3
"""
Test Dynamic Scaling Integration with Model Training

This script verifies that dynamic scaling is properly integrated into the
EmbeddingTrainer and can be used for continual learning on DNA sequences.
"""

import sys
import io
from pathlib import Path
import numpy as np
import torch

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tokenizer import DNATokenizer
from src.models.embeddings import DNATransformerEmbedder, DNAContrastiveModel
from src.models.trainer import EmbeddingTrainer
from src.models.dynamic_hybrid_buffer import ScalingConfig


def generate_test_data(n_clusters=5, samples_per_cluster=50):
    """Generate synthetic DNA sequences with cluster labels."""
    print(f"\n{'='*70}")
    print(f"Generating Test Data: {n_clusters} clusters, {samples_per_cluster} samples each")
    print(f"{'='*70}")
    
    bases = ['A', 'T', 'G', 'C']
    sequences = []
    labels = []
    
    for cluster_id in range(n_clusters):
        # Each cluster has a characteristic pattern
        base_pattern = bases[cluster_id % 4] * 10
        
        for _ in range(samples_per_cluster):
            # Generate sequence with cluster-specific pattern
            seq_parts = [base_pattern]
            for _ in range(19):  # 20 total segments
                seq_parts.append(''.join(np.random.choice(bases, 10)))
            
            sequence = ''.join(seq_parts)[:200]  # 200bp sequences
            sequences.append(sequence)
            labels.append(cluster_id)
    
    print(f"✓ Generated {len(sequences)} sequences")
    print(f"✓ Clusters: {sorted(set(labels))}")
    print(f"✓ Example sequence: {sequences[0][:50]}...")
    
    return sequences, labels


def test_dynamic_scaling_training():
    """Test training with dynamic scaling enabled."""
    print("\n" + "="*70)
    print("TEST: Dynamic Scaling Training Integration")
    print("="*70)
    
    # Generate test data
    sequences, labels = generate_test_data(n_clusters=5, samples_per_cluster=50)
    
    # Initialize tokenizer
    print("\n[Step 1] Initializing tokenizer...")
    tokenizer = DNATokenizer(encoding_type='kmer', kmer_size=4)  # OPTIMIZED
    print(f"✓ Vocab size: {tokenizer.vocab_size}")
    
    # Create model
    print("\n[Step 2] Creating model...")
    backbone = DNATransformerEmbedder(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2
    )
    model = DNAContrastiveModel(
        backbone_model=backbone,
        projection_dim=64,
        temperature=0.1
    )
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create scaling config
    print("\n[Step 3] Creating scaling config...")
    scaling_config = ScalingConfig.auto_scale(
        n_clusters=5,
        dataset_size=len(sequences),
        memory_budget_gb=0.5,  # 500MB budget
        target_accuracy=0.80
    )
    print(f"✓ Scaling config:")
    print(f"  - Exemplars per cluster: {scaling_config.exemplars_per_cluster}")
    print(f"  - Uncertainty buffer: {scaling_config.uncertainty_buffer_size}")
    print(f"  - Recent buffer: {scaling_config.recent_buffer_size}")
    
    # Initialize trainer with scaling
    print("\n[Step 4] Initializing trainer with dynamic scaling...")
    trainer = EmbeddingTrainer(
        model=model,
        tokenizer=tokenizer,
        device='cpu',
        scaling_config=scaling_config
    )
    print("✓ Trainer initialized with dynamic buffer")
    
    # Train with dynamic scaling
    print("\n[Step 5] Training with dynamic scaling...")
    print("-" * 70)
    
    history = trainer.train_with_dynamic_scaling(
        sequences=sequences,
        labels=labels,
        epochs_per_task=3,  # 3 epochs per cluster
        learning_rate=1e-4,
        batch_size=16,
        max_length=256,
        replay_ratio=0.3,
        validation_split=0.2
    )
    
    print("-" * 70)
    print("✓ Training completed!")
    
    # Verify results
    print("\n[Step 6] Verifying results...")
    assert 'train_loss' in history, "Missing train_loss in history"
    assert 'val_loss' in history, "Missing val_loss in history"
    assert 'clusters' in history, "Missing clusters in history"
    assert 'buffer_size' in history, "Missing buffer_size in history"
    assert 'memory_mb' in history, "Missing memory_mb in history"
    assert 'exemplars_per_cluster' in history, "Missing exemplars_per_cluster in history"
    
    print(f"✓ Training history keys: {list(history.keys())}")
    print(f"✓ Trained on {len(history['train_loss'])} tasks")
    print(f"✓ Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"✓ Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"✓ Final buffer size: {history['buffer_size'][-1]} samples")
    print(f"✓ Final memory usage: {history['memory_mb'][-1]:.2f} MB")
    
    # Verify buffer metrics
    assert history['clusters'][-1] == 5, f"Expected 5 clusters, got {history['clusters'][-1]}"
    assert history['buffer_size'][-1] > 0, "Buffer size should be > 0"
    assert history['memory_mb'][-1] > 0, "Memory usage should be > 0"
    
    print("\n✓ All assertions passed!")
    
    # Test scaling metrics retrieval
    print("\n[Step 7] Retrieving scaling metrics...")
    scaling_metrics = trainer.get_scaling_metrics()
    print(f"✓ Scaling metrics: {list(scaling_metrics.keys())}")
    
    # Test embedding extraction
    print("\n[Step 8] Testing embedding extraction...")
    test_sequences = sequences[:10]
    embeddings = trainer.extract_embeddings(test_sequences, batch_size=5)
    print(f"✓ Extracted embeddings shape: {embeddings.shape}")
    assert embeddings.shape[0] == 10, "Should have 10 embeddings"
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nSummary:")
    print(f"  • Model trained on {len(sequences)} sequences across {len(set(labels))} clusters")
    print(f"  • Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  • Final buffer size: {history['buffer_size'][-1]} samples")
    print(f"  • Memory usage: {history['memory_mb'][-1]:.2f} MB")
    print(f"  • Dynamic scaling: ✓ Working")
    print(f"  • Continual learning: ✓ Working")
    print(f"  • Embedding extraction: ✓ Working")
    
    return True


if __name__ == "__main__":
    try:
        success = test_dynamic_scaling_training()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
