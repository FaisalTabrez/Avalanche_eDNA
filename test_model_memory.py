"""
Test Model Memory: Does the trained model remember sequences from simulation?

This script loads the final trained model from active replay simulation
and tests it on the same sequences to verify knowledge retention.
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def load_model_and_data():
    """Load the trained model and original data."""
    
    print("="*70)
    print("MODEL MEMORY TEST")
    print("="*70)
    
    # Load embeddings
    print("\n1. Loading embeddings from simulation...")
    emb_file = Path("pipeline_outputs_2500_active/embeddings/dnabert2_embeddings.npy")
    
    if not emb_file.exists():
        print(f"❌ Error: Embeddings not found at {emb_file}")
        print("   Run the active replay simulation first!")
        return None, None, None
    
    embeddings = np.load(emb_file)
    print(f"   ✓ Loaded embeddings: {embeddings.shape}")
    
    # Load clustering results
    print("\n2. Loading clustering results...")
    cluster_file = Path("pipeline_outputs_2500_active/clustering/results.json")
    
    if not cluster_file.exists():
        print(f"❌ Error: Clustering results not found at {cluster_file}")
        return None, None, None
    
    with open(cluster_file) as f:
        cluster_data = json.load(f)
    
    cluster_labels = np.array(cluster_data['cluster_labels'])
    n_clusters = cluster_data['n_clusters']
    print(f"   ✓ Loaded cluster labels: {n_clusters} clusters")
    print(f"   Cluster sizes: {cluster_data['cluster_sizes']}")
    
    # Find the latest checkpoint
    print("\n3. Loading trained model checkpoint...")
    checkpoint_dir = Path("pipeline_outputs_2500_active/checkpoints")
    
    if not checkpoint_dir.exists():
        print(f"❌ Error: Checkpoint directory not found")
        return None, None, None
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    
    if not checkpoints:
        print(f"❌ Error: No checkpoints found")
        return None, None, None
    
    # Load the LAST checkpoint (after training on all clusters)
    latest_checkpoint = checkpoints[-1]
    print(f"   Loading: {latest_checkpoint.name}")
    
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    # Recreate model architecture (simpler version from simulation)
    model = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, n_clusters)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Training metrics: {checkpoint['metrics']}")
    
    return model, embeddings, cluster_labels


def test_model_memory(model, embeddings, cluster_labels):
    """Test if model remembers the training data."""
    
    print("\n" + "="*70)
    print("MEMORY TEST RESULTS")
    print("="*70)
    
    n_clusters = len(np.unique(cluster_labels))
    
    # Test on all data
    print("\n📊 Testing model on ALL training sequences...")
    
    with torch.no_grad():
        X_all = torch.FloatTensor(embeddings)
        y_all = torch.LongTensor(cluster_labels)
        
        outputs = model(X_all)
        _, predicted = outputs.max(1)
        
        correct = predicted.eq(y_all).sum().item()
        total = len(cluster_labels)
        overall_accuracy = 100.0 * correct / total
        
        print(f"\n✓ Overall Memory Retention: {overall_accuracy:.1f}% ({correct}/{total})")
    
    # Per-cluster detailed analysis
    print("\n" + "─"*70)
    print("PER-CLUSTER MEMORY RETENTION")
    print("─"*70)
    
    cluster_results = {}
    confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        with torch.no_grad():
            X_cluster = torch.FloatTensor(embeddings[cluster_indices])
            y_cluster = torch.LongTensor(cluster_labels[cluster_indices])
            
            outputs = model(X_cluster)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            correct = predicted.eq(y_cluster).sum().item()
            accuracy = 100.0 * correct / len(cluster_indices)
            
            # Get average confidence
            correct_probs = probs[range(len(predicted)), predicted]
            avg_confidence = correct_probs.mean().item()
            
            # Count misclassifications
            for true_label, pred_label in zip(y_cluster.numpy(), predicted.numpy()):
                confusion_matrix[true_label, pred_label] += 1
            
            cluster_results[cluster_id] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': len(cluster_indices),
                'confidence': avg_confidence
            }
            
            status = "✓" if accuracy > 75 else "⚠️" if accuracy > 50 else "❌"
            print(f"\nCluster {cluster_id}: {status}")
            print(f"  Accuracy: {accuracy:>6.1f}% ({correct:>4}/{len(cluster_indices):>4})")
            print(f"  Avg Confidence: {avg_confidence:.3f}")
    
    # Show confusion matrix
    print("\n" + "─"*70)
    print("CONFUSION MATRIX (where sequences were misclassified)")
    print("─"*70)
    print("\n       Predicted →")
    print("True ↓  " + "  ".join([f"C{i}" for i in range(n_clusters)]))
    print("─"*70)
    
    for i in range(n_clusters):
        row_str = f"  C{i}    "
        for j in range(n_clusters):
            if i == j:
                row_str += f"{confusion_matrix[i,j]:>4}"  # Correct predictions
            else:
                row_str += f"{confusion_matrix[i,j]:>4}" if confusion_matrix[i,j] > 0 else "   ."
            row_str += "  "
        print(row_str)
    
    # Memory retention analysis
    print("\n" + "="*70)
    print("MEMORY RETENTION ANALYSIS")
    print("="*70)
    
    clusters_retained = sum(1 for r in cluster_results.values() if r['accuracy'] > 75)
    clusters_partial = sum(1 for r in cluster_results.values() if 50 < r['accuracy'] <= 75)
    clusters_forgotten = sum(1 for r in cluster_results.values() if r['accuracy'] <= 50)
    
    print(f"\n📊 Memory Status:")
    print(f"   ✓ Fully Retained (>75%):     {clusters_retained}/{n_clusters} clusters")
    print(f"   ⚠️  Partially Retained (50-75%): {clusters_partial}/{n_clusters} clusters")
    print(f"   ❌ Forgotten (<50%):         {clusters_forgotten}/{n_clusters} clusters")
    
    # Compare with expected results
    print("\n📈 Comparison with Expected Results:")
    print(f"   Expected (from simulation): 89.0%")
    print(f"   Actual (this test):         {overall_accuracy:.1f}%")
    
    if abs(overall_accuracy - 89.0) < 2.0:
        print(f"   ✅ MATCH! Model memory is intact.")
    elif overall_accuracy > 80:
        print(f"   ✓ Close match, minor variation is normal.")
    else:
        print(f"   ⚠️ Significant deviation - investigate.")
    
    # Test specific examples
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (random examples from each cluster)")
    print("="*70)
    
    np.random.seed(42)
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        # Sample 3 random sequences from this cluster
        sample_indices = np.random.choice(cluster_indices, min(3, len(cluster_indices)), replace=False)
        
        print(f"\nCluster {cluster_id} samples:")
        
        with torch.no_grad():
            for idx in sample_indices:
                X_sample = torch.FloatTensor(embeddings[idx:idx+1])
                output = model(X_sample)
                probs = torch.softmax(output, dim=1)[0]
                predicted_cluster = output.argmax(1).item()
                confidence = probs[predicted_cluster].item()
                
                is_correct = predicted_cluster == cluster_id
                status = "✓" if is_correct else "❌"
                
                print(f"  Sample {idx}: {status} Predicted C{predicted_cluster} "
                      f"(confidence: {confidence:.3f})")
                
                if not is_correct:
                    # Show top 3 predictions
                    top_probs, top_indices = probs.topk(3)
                    print(f"    Top predictions: ", end="")
                    for prob, cls in zip(top_probs, top_indices):
                        print(f"C{cls.item()}({prob:.2f}) ", end="")
                    print()
    
    return overall_accuracy, cluster_results


def test_incremental_forgetting(model, embeddings, cluster_labels):
    """Test memory retention for clusters learned earlier vs later."""
    
    print("\n" + "="*70)
    print("TEMPORAL FORGETTING ANALYSIS")
    print("="*70)
    print("\nDoes the model forget earlier clusters more than recent ones?")
    
    n_clusters = len(np.unique(cluster_labels))
    
    # Assume clusters were learned in order 0, 1, 2, 3, 4
    training_order = list(range(n_clusters))
    
    print(f"\nTraining order: {' → '.join([f'C{i}' for i in training_order])}")
    print("\nAccuracy by training position:")
    
    positions = []
    accuracies = []
    
    for position, cluster_id in enumerate(training_order):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        with torch.no_grad():
            X_cluster = torch.FloatTensor(embeddings[cluster_indices])
            y_cluster = torch.LongTensor(cluster_labels[cluster_indices])
            
            outputs = model(X_cluster)
            _, predicted = outputs.max(1)
            correct = predicted.eq(y_cluster).sum().item()
            accuracy = 100.0 * correct / len(cluster_indices)
            
            positions.append(position)
            accuracies.append(accuracy)
            
            time_ago = n_clusters - position - 1
            if time_ago == 0:
                time_str = "(most recent)"
            else:
                time_str = f"({time_ago} clusters ago)"
            
            print(f"  Position {position} (C{cluster_id}): {accuracy:>6.1f}% {time_str}")
    
    # Check for forgetting pattern
    print("\n📊 Forgetting Pattern Analysis:")
    
    early_avg = np.mean(accuracies[:2]) if len(accuracies) >= 2 else accuracies[0]
    late_avg = np.mean(accuracies[-2:]) if len(accuracies) >= 2 else accuracies[-1]
    
    print(f"   Early clusters (0-1) avg: {early_avg:.1f}%")
    print(f"   Late clusters ({n_clusters-2}-{n_clusters-1}) avg: {late_avg:.1f}%")
    
    if late_avg > early_avg + 5:
        print(f"   ⚠️ Recent bias detected: Later clusters have {late_avg - early_avg:.1f}pp higher accuracy")
        print(f"   This suggests some forgetting of earlier knowledge.")
    elif early_avg > late_avg + 5:
        print(f"   ℹ️ Early advantage: Earlier clusters have {early_avg - late_avg:.1f}pp higher accuracy")
        print(f"   This could indicate stronger consolidation of early knowledge.")
    else:
        print(f"   ✅ Balanced retention: Difference only {abs(late_avg - early_avg):.1f}pp")
        print(f"   Active replay successfully prevented temporal forgetting!")


def main():
    """Run complete memory test."""
    
    # Load model and data
    model, embeddings, cluster_labels = load_model_and_data()
    
    if model is None:
        print("\n❌ Cannot proceed without trained model and data.")
        print("   Please run the active replay simulation first:")
        print("   python run_complete_pipeline.py")
        return
    
    # Test overall memory
    overall_accuracy, cluster_results = test_model_memory(model, embeddings, cluster_labels)
    
    # Test temporal forgetting
    test_incremental_forgetting(model, embeddings, cluster_labels)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if overall_accuracy > 85:
        print("\n✅ EXCELLENT MEMORY RETENTION!")
        print(f"   Model remembers {overall_accuracy:.1f}% of training sequences.")
        print("   Active replay successfully prevented catastrophic forgetting.")
    elif overall_accuracy > 70:
        print("\n✓ GOOD MEMORY RETENTION")
        print(f"   Model remembers {overall_accuracy:.1f}% of training sequences.")
        print("   Some forgetting occurred but mostly retained.")
    elif overall_accuracy > 50:
        print("\n⚠️ MODERATE MEMORY RETENTION")
        print(f"   Model remembers {overall_accuracy:.1f}% of training sequences.")
        print("   Significant forgetting detected.")
    else:
        print("\n❌ POOR MEMORY RETENTION")
        print(f"   Model only remembers {overall_accuracy:.1f}% of training sequences.")
        print("   Severe catastrophic forgetting occurred.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
