"""
Real Pipeline Simulation: Dynamic Scaling on Actual eDNA Data

Runs the complete production pipeline with dynamic scaling on real eDNA sequences.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
from datetime import datetime

import numpy as np
from sklearn.metrics import silhouette_score

from scripts.run_taxonomy_pipeline_v2 import TaxonomyClassificationPipeline


def run_real_pipeline_simulation():
    """Run complete pipeline with dynamic scaling on real eDNA data."""

    print("=" * 80)
    print("REAL PIPELINE SIMULATION: Dynamic Scaling on eDNA Sequences")
    print("=" * 80)

    # Configuration
    fasta_file = "data/sample/sample_edna_sequences.fasta"
    output_dir = "real_pipeline_simulation_output"

    print(f"\nInput: {fasta_file}")
    print(f"Output: {output_dir}")

    # Initialize pipeline with dynamic scaling
    print("\n" + "=" * 80)
    print("STEP 1: Initialize Pipeline with Dynamic Scaling")
    print("=" * 80)

    pipeline = TaxonomyClassificationPipeline(
        output_dir=output_dir,
        dnabert_model_path="./models/dnabert2_cpu",
        device="cpu",
        # DYNAMIC SCALING CONFIGURATION
        enable_dynamic_scaling=True,
        memory_budget_gb=2.0,  # 2GB memory budget
        target_accuracy=0.80,  # 80% target accuracy
        auto_adapt=True,  # Auto-adjust as needed
        # Disable legacy and taxonomy features for simulation
        enable_continual_learning=False,
        use_blast=False,
        use_ml_classifier=False,
    )

    # Load sequences
    print("\n" + "=" * 80)
    print("STEP 2: Load eDNA Sequences")
    print("=" * 80)

    sequences = pipeline.load_sequences(fasta_file)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total sequences: {len(sequences):,}")

    lengths = [s["length"] for s in sequences]
    print(f"   Length range: {min(lengths)}-{max(lengths)} bp")
    print(f"   Average length: {sum(lengths)/len(lengths):.1f} bp")

    # Generate embeddings
    print("\n" + "=" * 80)
    print("STEP 3: Generate DNABERT-2 Embeddings")
    print("=" * 80)

    embeddings = pipeline.generate_embeddings(batch_size=32)

    # Auto-detect optimal cluster count
    print("\n" + "=" * 80)
    print("STEP 4: Cluster Sequences (Auto-detect K)")
    print("=" * 80)

    # Try different K values to find optimal
    import numpy as np
    from sklearn.metrics import silhouette_score

    print("\nTesting different cluster counts...")

    k_values = [5, 10, 15, 20, 25, 30]
    silhouette_scores = []

    for k in k_values:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
        print(f"   K={k:2d}: Silhouette={score:.4f}")

    # Select K with best silhouette score
    best_k = k_values[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)

    print(f"\n✓ Optimal K: {best_k} (silhouette={best_score:.4f})")

    # Cluster with optimal K
    cluster_labels = pipeline.cluster_sequences(method="kmeans", n_clusters=best_k)

    # Calculate clustering statistics
    cluster_sizes = np.bincount(cluster_labels)
    silhouette = silhouette_score(pipeline.embeddings, cluster_labels)

    print(f"\n📊 Clustering Results:")
    print(f"   Clusters: {best_k}")
    print(f"   Silhouette score: {silhouette:.4f}")
    print(f"   Cluster sizes: {cluster_sizes.tolist()}")

    # Train classifier with dynamic scaling
    print("\n" + "=" * 80)
    print("STEP 5: Train Taxonomy Classifier (Dynamic Scaling)")
    print("=" * 80)

    training_results = pipeline.train_taxonomy_classifier(
        epochs_per_cluster=10, learning_rate=1e-3
    )

    # Evaluate final model
    print("\n" + "=" * 80)
    print("STEP 6: Evaluate Model Memory Retention")
    print("=" * 80)

    import torch

    pipeline.classifier_model.eval()

    all_correct = 0
    all_total = 0
    cluster_accuracies = {}

    for cluster_id in range(best_k):
        cluster_mask = pipeline.cluster_labels == cluster_id
        cluster_embeddings = pipeline.embeddings[cluster_mask]

        with torch.no_grad():
            outputs = pipeline.classifier_model(
                torch.FloatTensor(cluster_embeddings).to(pipeline.device)
            )
            _, predicted = outputs.max(1)
            true_labels = torch.full(
                (len(cluster_embeddings),), cluster_id, dtype=torch.long
            )

            correct = predicted.eq(true_labels).sum().item()
            total = len(cluster_embeddings)
            accuracy = 100.0 * correct / total

            cluster_accuracies[cluster_id] = accuracy
            all_correct += correct
            all_total += total

    overall_accuracy = 100.0 * all_correct / all_total

    # Results summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n📊 Performance Metrics:")
    print(
        f"   Overall Accuracy: {overall_accuracy:.1f}% ({all_correct:,}/{all_total:,})"
    )

    # Per-cluster breakdown
    early_clusters = list(range(min(3, best_k)))
    middle_start = best_k // 3
    middle_end = 2 * best_k // 3
    middle_clusters = list(range(middle_start, middle_end))
    recent_clusters = list(range(max(0, best_k - 3), best_k))

    if early_clusters:
        early_acc = np.mean([cluster_accuracies[i] for i in early_clusters])
        print(f"   Early clusters (0-{early_clusters[-1]}): {early_acc:.1f}%")

    if middle_clusters:
        middle_acc = np.mean([cluster_accuracies[i] for i in middle_clusters])
        print(f"   Middle clusters ({middle_start}-{middle_end-1}): {middle_acc:.1f}%")

    if recent_clusters:
        recent_acc = np.mean([cluster_accuracies[i] for i in recent_clusters])
        print(
            f"   Recent clusters ({recent_clusters[0]}-{recent_clusters[-1]}): {recent_acc:.1f}%"
        )

    if early_clusters and recent_clusters:
        recency_bias = recent_acc - early_acc
        print(f"   Recency bias: {recency_bias:+.1f}pp")

    print(f"\n🔄 Dynamic Scaling:")
    print(f"   Adaptations: {training_results['adaptations']}")
    print(
        f"   Memory usage: {training_results['memory_mb']:.1f} MB / {pipeline.memory_budget_gb*1024:.0f} MB"
    )
    print(
        f"   Usage: {100*training_results['memory_mb']/(pipeline.memory_budget_gb*1024):.1f}%"
    )

    config = training_results["final_config"]
    print(f"\n⚙️  Final Configuration:")
    print(f"   Exemplars/cluster: {config['exemplars_per_cluster']}")
    print(f"   Uncertainty buffer: {config['uncertainty_buffer_size']:,}")
    print(f"   Recent buffer: {config['recent_buffer_size']:,}")
    print(f"   Architecture: {config['hidden_dims']}")
    print(f"   Temperature: {config['temperature']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Replay ratio: {config['replay_ratio']}")

    # Save results
    import json

    results_summary = {
        "dataset": fasta_file,
        "sequences": len(sequences),
        "clusters": best_k,
        "overall_accuracy": overall_accuracy,
        "early_accuracy": early_acc if early_clusters else None,
        "middle_accuracy": middle_acc if middle_clusters else None,
        "recent_accuracy": recent_acc if recent_clusters else None,
        "recency_bias": recency_bias if early_clusters and recent_clusters else None,
        "adaptations": training_results["adaptations"],
        "memory_mb": training_results["memory_mb"],
        "final_config": config,
    }

    results_file = Path(output_dir) / "simulation_results.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)

    # Verdict
    print("\n📋 Verdict:")

    if overall_accuracy > 90:
        print("   ✅ EXCELLENT: Dynamic scaling achieved >90% retention on real data!")
    elif overall_accuracy > 80:
        print("   ✅ VERY GOOD: Dynamic scaling achieved >80% retention on real data!")
    elif overall_accuracy > 70:
        print("   ✓ GOOD: Dynamic scaling achieved >70% retention on real data.")
    else:
        print("   ⚠️ MODERATE: Consider increasing memory budget or target accuracy.")

    if training_results["adaptations"] > 0:
        print(
            f"   ✓ System adapted {training_results['adaptations']} time(s) during training."
        )
    else:
        print("   ℹ️ No adaptations needed for this scale.")

    if training_results["memory_mb"] / (pipeline.memory_budget_gb * 1024) > 0.8:
        print(
            "   ⚠️ High memory usage - consider increasing budget for larger datasets."
        )
    else:
        print("   ✓ Memory usage well within budget.")

    return results_summary


if __name__ == "__main__":
    run_real_pipeline_simulation()
