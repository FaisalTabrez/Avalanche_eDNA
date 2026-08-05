"""
Complete eDNA Analysis and Continual Learning Pipeline
Uses real DNABERT-2 embeddings on synthetic 2500-sequence dataset

This combines:
1. DNABERT-2 embedding generation
2. Clustering analysis
3. Continual learning training
4. Performance evaluation
"""

import io
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Bio and ML imports
from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from transformers import AutoModel, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Our modules
sys.path.insert(0, str(Path(__file__).parent))
from src.models.checkpoint_manager import CheckpointManager
from src.models.continual_learning import ContinualLearner
from src.models.model_registry import ModelRegistry


class CompletePipeline:
    """End-to-end pipeline from sequences to trained model."""

    def __init__(
        self,
        fasta_file: str,
        output_dir: str = "pipeline_outputs",
        n_clusters: int = 5,
        device: str = "cpu",
    ):
        self.fasta_file = Path(fasta_file)
        self.output_dir = Path(output_dir)
        self.n_clusters = n_clusters
        self.device = device

        # Create output structure
        for subdir in [
            "embeddings",
            "clustering",
            "models",
            "checkpoints",
            "visualizations",
        ]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Load DNABERT-2
        print("Loading DNABERT-2-117M...")
        model_path = "./models/dnabert2_cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.dnabert_model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.dnabert_model.eval()
        print(
            f"✓ DNABERT-2 loaded ({sum(p.numel() for p in self.dnabert_model.parameters()):,} parameters)"
        )

        # Initialize components
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.output_dir / "checkpoints")
        )
        self.model_registry = ModelRegistry(
            registry_dir=str(self.output_dir / "models")
        )

        self.sequences = []
        self.embeddings = None
        self.cluster_labels = None

    def step1_load_sequences(self):
        """Load sequences from FASTA."""
        print(f"\n{'='*60}")
        print(f"STEP 1: Loading Sequences")
        print(f"{'='*60}")

        sequences = []
        for record in SeqIO.parse(str(self.fasta_file), "fasta"):
            # Parse header: >seq_0|marine_bacteria|length_450
            parts = record.id.split("|")
            org_type = parts[1] if len(parts) > 1 else "unknown"

            sequences.append(
                {
                    "id": record.id,
                    "sequence": str(record.seq),
                    "length": len(record.seq),
                    "organism": org_type,
                }
            )

        self.sequences = sequences
        print(f"✓ Loaded {len(sequences):,} sequences")
        print(
            f"  Length range: {min(s['length'] for s in sequences)}-{max(s['length'] for s in sequences)} bp"
        )
        print(f"  Total base pairs: {sum(s['length'] for s in sequences):,}")

        # Show organism distribution
        from collections import Counter

        org_counts = Counter(s["organism"] for s in sequences)
        print(f"\nOrganism Distribution:")
        for org, count in sorted(org_counts.items()):
            print(f"  {org:<30} {count:>5} sequences")

    def step2_generate_embeddings(self, batch_size: int = 32):
        """Generate DNABERT-2 embeddings."""
        print(f"\n{'='*60}")
        print(f"STEP 2: Generating DNABERT-2 Embeddings")
        print(f"{'='*60}")

        embeddings = []
        total = len(self.sequences)

        print(f"Processing {total:,} sequences in batches of {batch_size}...")
        start_time = time.time()

        for i in range(0, total, batch_size):
            batch = self.sequences[i : i + batch_size]
            batch_seqs = [s["sequence"] for s in batch]

            # Tokenize
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self.dnabert_model(**inputs)
                # Use [CLS] token embedding
                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                else:
                    hidden_states = outputs.last_hidden_state

                batch_embeddings = hidden_states[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

            if (i + batch_size) % 320 == 0 or i + batch_size >= total:
                elapsed = time.time() - start_time
                progress = min(i + batch_size, total)
                rate = progress / elapsed if elapsed > 0 else 0
                print(
                    f"  Progress: {progress:>5}/{total} ({100*progress/total:>5.1f}%) - {rate:.0f} seqs/sec"
                )

        self.embeddings = np.vstack(embeddings)
        elapsed = time.time() - start_time

        print(f"\n✓ Generated embeddings: {self.embeddings.shape}")
        print(f"  Total time: {elapsed:.1f} seconds")
        print(f"  Average: {elapsed/total*1000:.1f} ms per sequence")

        # Save embeddings
        emb_file = self.output_dir / "embeddings" / "dnabert2_embeddings.npy"
        np.save(emb_file, self.embeddings)
        print(f"  Saved to: {emb_file}")

    def step3_cluster_sequences(self):
        """Cluster sequences using k-means."""
        print(f"\n{'='*60}")
        print(f"STEP 3: Clustering Sequences")
        print(f"{'='*60}")

        print(f"Running k-means with k={self.n_clusters}...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)

        silhouette = silhouette_score(self.embeddings, self.cluster_labels)

        print(f"✓ Clustering complete")
        print(f"  Silhouette score: {silhouette:.3f}")
        print(f"  Cluster sizes: {np.bincount(self.cluster_labels).tolist()}")

        # Analyze cluster composition
        print(f"\nCluster Composition (by organism):")
        for cluster_id in range(self.n_clusters):
            cluster_seqs = [
                self.sequences[i]
                for i, l in enumerate(self.cluster_labels)
                if l == cluster_id
            ]
            orgs = [s["organism"] for s in cluster_seqs]
            from collections import Counter

            org_counts = Counter(orgs)
            print(f"\n  Cluster {cluster_id} ({len(cluster_seqs)} sequences):")
            for org, count in org_counts.most_common(3):
                pct = 100 * count / len(cluster_seqs)
                print(f"    {org:<30} {count:>4} ({pct:>5.1f}%)")

        # Save clustering results
        results = {
            "n_clusters": self.n_clusters,
            "silhouette_score": float(silhouette),
            "cluster_sizes": np.bincount(self.cluster_labels).tolist(),
            "cluster_labels": self.cluster_labels.tolist(),
        }

        with open(self.output_dir / "clustering" / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    def step4_train_continual_learning(self, epochs_per_cluster: int = 10):
        """Train with continual learning."""
        print(f"\n{'='*60}")
        print(f"STEP 4: Continual Learning Training")
        print(f"{'='*60}")

        # Create model
        model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.n_clusters),
        ).to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Continual learner with larger buffer and active replay
        continual_learner = ContinualLearner(
            strategy="combined",
            buffer_size=1000,  # Increased for better coverage
            ewc_lambda=100.0,  # Reduced to allow more plasticity
        )

        print(f"Model: 768 -> 256 -> 128 -> {self.n_clusters}")
        print(f"Strategy: Combined (Active Replay + EWC)")
        print(f"Buffer size: 1000")
        print(f"Replay ratio: 50% replay + 50% current cluster")
        print()

        training_history = {}

        # Train on each cluster sequentially
        for cluster_id in range(self.n_clusters):
            print(f"Training on Cluster {cluster_id}...")

            # Get cluster data
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            X_cluster = torch.FloatTensor(self.embeddings[cluster_indices]).to(
                self.device
            )
            y_cluster = torch.full(
                (len(cluster_indices),), cluster_id, dtype=torch.long
            ).to(self.device)

            # Training loop
            model.train()
            epoch_losses = []
            epoch_accs = []

            for epoch in range(epochs_per_cluster):
                total_loss = 0.0
                correct = 0
                total = 0
                replay_batches_used = 0

                # Shuffle
                indices = torch.randperm(len(cluster_indices))
                batch_size = 32

                for i in range(0, len(cluster_indices), batch_size):
                    batch_idx = indices[i : i + batch_size]
                    batch_X = X_cluster[batch_idx]
                    batch_y = y_cluster[batch_idx]

                    # ACTIVE REPLAY: Mix with replay buffer samples
                    if cluster_id > 0 and continual_learner.strategy in [
                        "replay",
                        "combined",
                    ]:
                        if len(continual_learner.replay_buffer.sequences) > 0:
                            # Sample from replay buffer (50% of batch)
                            replay_size = min(
                                batch_size // 2,
                                len(continual_learner.replay_buffer.sequences),
                            )
                            replay_samples = continual_learner.replay_buffer.sample(
                                replay_size
                            )

                            if (
                                replay_samples and replay_samples[0]
                            ):  # Check if we got valid samples
                                # Convert replay samples back to tensors
                                replay_X = torch.FloatTensor(
                                    [eval(seq) for seq in replay_samples[0]]
                                ).to(self.device)
                                replay_y = torch.LongTensor(replay_samples[1]).to(
                                    self.device
                                )

                                # Combine current batch with replay samples
                                batch_X = torch.cat([batch_X, replay_X], dim=0)
                                batch_y = torch.cat([batch_y, replay_y], dim=0)
                                replay_batches_used += 1

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = nn.CrossEntropyLoss()(outputs, batch_y)

                    # Add EWC regularization for clusters after the first
                    if cluster_id > 0 and continual_learner.strategy in [
                        "ewc",
                        "combined",
                    ]:
                        ewc_loss = continual_learner.compute_ewc_loss(model)
                        loss = loss + ewc_loss

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()

                    # Store in replay buffer
                    if continual_learner.strategy in ["replay", "combined"]:
                        batch_seqs_str = [
                            str(batch_X[j].cpu().numpy().tolist())
                            for j in range(len(batch_X))
                        ]
                        batch_labels = batch_y.cpu().tolist()
                        continual_learner.replay_buffer.add_samples(
                            batch_seqs_str, batch_labels
                        )

                avg_loss = total_loss / (len(cluster_indices) / batch_size)
                accuracy = 100.0 * correct / total
                epoch_losses.append(avg_loss)
                epoch_accs.append(accuracy)

                if (epoch + 1) % 2 == 0:
                    replay_info = (
                        f", Replay batches: {replay_batches_used}"
                        if cluster_id > 0
                        else ""
                    )
                    print(
                        f"  Epoch {epoch+1}/{epochs_per_cluster}: Loss={avg_loss:.4f}, Acc={accuracy:.1f}%{replay_info}"
                    )

            # Store Fisher information for EWC
            if continual_learner.strategy in ["ewc", "combined"]:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if name not in continual_learner.fisher_dict:
                            continual_learner.fisher_dict[name] = torch.zeros_like(
                                param
                            )
                        continual_learner.optimal_params[name] = param.clone().detach()

            # Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epochs_per_cluster,
                metrics={"loss": epoch_losses[-1], "accuracy": epoch_accs[-1]},
                dataset_info={"cluster": cluster_id, "size": len(cluster_indices)},
            )

            # Register model
            version = f"v1.{cluster_id}.0"
            self.model_registry.register_model(
                version=version,
                model_path=checkpoint_path,
                datasets=[f"Cluster {cluster_id}"],
                metrics={"loss": epoch_losses[-1], "accuracy": epoch_accs[-1]},
            )

            training_history[f"cluster_{cluster_id}"] = {
                "losses": epoch_losses,
                "accuracies": epoch_accs,
            }

            print(f"  ✓ Saved checkpoint: {Path(checkpoint_path).name}")
            print(f"  ✓ Registered model: {version}")

        # Evaluate on all clusters
        print(f"\nFinal Evaluation on All Clusters:")
        model.eval()

        cluster_results = {}
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for cluster_id in range(self.n_clusters):
                cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
                X_test = torch.FloatTensor(self.embeddings[cluster_indices]).to(
                    self.device
                )
                y_test = torch.full(
                    (len(cluster_indices),), cluster_id, dtype=torch.long
                ).to(self.device)

                outputs = model(X_test)
                _, predicted = outputs.max(1)
                correct = predicted.eq(y_test).sum().item()

                accuracy = 100.0 * correct / len(cluster_indices)
                cluster_results[cluster_id] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": len(cluster_indices),
                }

                total_correct += correct
                total_samples += len(cluster_indices)

                print(
                    f"  Cluster {cluster_id}: {accuracy:>6.1f}% ({correct:>4}/{len(cluster_indices):>4})"
                )

        overall_accuracy = 100.0 * total_correct / total_samples
        print(
            f"\n  Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_samples})"
        )

        return {
            "training_history": training_history,
            "cluster_results": cluster_results,
            "overall_accuracy": overall_accuracy,
        }

    def step5_visualize(self):
        """Create visualizations."""
        print(f"\n{'='*60}")
        print(f"STEP 5: Generating Visualizations")
        print(f"{'='*60}")

        # PCA projection
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.embeddings)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Clusters
        scatter = axes[0].scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=self.cluster_labels,
            cmap="viridis",
            alpha=0.6,
            s=20,
        )
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].set_title(f"eDNA Sequence Clusters (n={len(self.sequences):,})")
        plt.colorbar(scatter, ax=axes[0], label="Cluster")

        # Plot 2: Cluster sizes
        cluster_sizes = np.bincount(self.cluster_labels)
        axes[1].bar(range(len(cluster_sizes)), cluster_sizes, color="steelblue")
        axes[1].set_xlabel("Cluster ID")
        axes[1].set_ylabel("Number of Sequences")
        axes[1].set_title("Cluster Size Distribution")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "analysis.png", dpi=150)
        plt.close()

        print(f"✓ Saved visualizations")

    def run_complete_pipeline(self):
        """Execute all steps."""
        print("=" * 60)
        print("Complete eDNA Analysis & Continual Learning Pipeline")
        print("=" * 60)

        start_time = time.time()

        # Execute pipeline
        self.step1_load_sequences()
        self.step2_generate_embeddings()
        self.step3_cluster_sequences()
        results = self.step4_train_continual_learning()
        self.step5_visualize()

        total_time = time.time() - start_time

        # Save summary
        summary = {
            "date": datetime.now().isoformat(),
            "dataset": str(self.fasta_file),
            "total_sequences": len(self.sequences),
            "embedding_model": "DNABERT-2-117M",
            "n_clusters": self.n_clusters,
            "results": results,
            "total_time_seconds": total_time,
        }

        with open(self.output_dir / "pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✅ Pipeline Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Overall accuracy: {results['overall_accuracy']:.1f}%")
        print(f"Output directory: {self.output_dir}")


def main():
    # Use 2500-sequence dataset with ACTIVE replay
    pipeline = CompletePipeline(
        fasta_file="data/synthetic_edna/mixed_edna_2500.fasta",
        output_dir="pipeline_outputs_2500_active",
        n_clusters=5,
    )

    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
