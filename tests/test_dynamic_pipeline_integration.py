"""
Test: Dynamic Scaling in Production Pipeline

Quick test to verify the integrated dynamic scaling works in the main pipeline.
"""

import io
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_taxonomy_pipeline_v2 import TaxonomyClassificationPipeline


def create_test_fasta(n_clusters=25, samples_per_cluster=100):
    """Create a test FASTA file with synthetic sequences."""

    print(f"Creating test FASTA with {n_clusters} clusters...")

    fasta_file = Path("test_dynamic_scaling.fasta")

    records = []
    for cluster_id in range(n_clusters):
        # Generate cluster-specific pattern
        base_pattern = ["ATCG"[i] for i in np.random.randint(0, 4, 50)]

        for seq_id in range(samples_per_cluster):
            # Add variation to base pattern
            sequence = base_pattern.copy()
            for _ in range(5):  # 5 mutations
                pos = np.random.randint(0, len(sequence))
                sequence[pos] = np.random.choice(["A", "T", "C", "G"])

            record = SeqRecord(
                Seq("".join(sequence)),
                id=f"cluster{cluster_id}_seq{seq_id}",
                description=f"Cluster {cluster_id}",
            )
            records.append(record)

    # Write FASTA
    SeqIO.write(records, fasta_file, "fasta")

    print(f"✓ Created {len(records)} sequences in {fasta_file}")

    return fasta_file


def test_dynamic_scaling_pipeline():
    """Test the pipeline with dynamic scaling."""

    print("=" * 70)
    print("TESTING: Dynamic Scaling in Production Pipeline")
    print("=" * 70)

    # Create test data
    fasta_file = create_test_fasta(n_clusters=25, samples_per_cluster=100)

    # Initialize pipeline with dynamic scaling
    print("\n" + "=" * 70)
    print("Initializing Pipeline with Dynamic Scaling")
    print("=" * 70)

    pipeline = TaxonomyClassificationPipeline(
        output_dir="test_dynamic_pipeline_output",
        dnabert_model_path="./models/dnabert2_cpu",
        device="cpu",
        enable_dynamic_scaling=True,  # DYNAMIC SCALING
        memory_budget_gb=1.0,  # 1GB budget
        target_accuracy=0.80,  # 80% target
        auto_adapt=True,  # Auto adaptation
        enable_continual_learning=False,  # Disable legacy mode
        use_blast=False,  # Skip BLAST for test
        use_ml_classifier=False,  # Skip ML for test
    )

    # Load sequences
    print("\n" + "=" * 70)
    print("Step 1: Loading Sequences")
    print("=" * 70)

    sequences = pipeline.load_sequences(str(fasta_file))

    # Generate embeddings
    print("\n" + "=" * 70)
    print("Step 2: Generating Embeddings")
    print("=" * 70)

    embeddings = pipeline.generate_embeddings(batch_size=32)

    # Cluster sequences
    print("\n" + "=" * 70)
    print("Step 3: Clustering Sequences")
    print("=" * 70)

    clustering_results = pipeline.cluster_sequences(
        method="kmeans", n_clusters=25  # We know there are 25
    )

    # Train classifier with dynamic scaling
    print("\n" + "=" * 70)
    print("Step 4: Training Classifier (Dynamic Scaling)")
    print("=" * 70)

    training_results = pipeline.train_taxonomy_classifier(
        epochs_per_cluster=5, learning_rate=1e-3  # Fewer epochs for testing
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if "adaptations" in training_results:
        print(f"\n✅ Dynamic Scaling Worked!")
        print(f"   Total adaptations: {training_results['adaptations']}")
        print(f"   Memory usage: {training_results['memory_mb']:.1f} MB")

        config = training_results["final_config"]
        print(f"\n   Final configuration:")
        print(f"     Exemplars/cluster: {config['exemplars_per_cluster']}")
        print(f"     Uncertainty buffer: {config['uncertainty_buffer_size']:,}")
        print(f"     Recent buffer: {config['recent_buffer_size']:,}")
        print(f"     Architecture: {config['hidden_dims']}")
        print(f"     Temperature: {config['temperature']}")
    else:
        print("⚠️  No dynamic scaling results found")

    # Cleanup
    fasta_file.unlink()
    print(f"\n✓ Cleaned up {fasta_file}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return training_results


if __name__ == "__main__":
    test_dynamic_scaling_pipeline()
