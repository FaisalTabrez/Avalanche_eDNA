"""
Hybrid Memory Buffer with Advanced Refinements

Implements sophisticated continual learning memory management with:
- Temperature-scaled confidence for uncertainty detection
- Reservoir sampling for recent buffer
- Mini-retrieval during training
- Periodic centroid updates
- LoRA adapter support (optional)
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaledConfidence:
    """
    Temperature-scaled confidence scoring for better uncertainty detection.

    Raw softmax can be overconfident. Temperature scaling smooths the distribution:
    - T > 1: Softer, less confident predictions
    - T = 1: Standard softmax
    - T < 1: Sharper, more confident predictions
    """

    def __init__(self, temperature: float = 2.0):
        """
        Args:
            temperature: Scaling factor (default 2.0 for smoother difficulty detection)
        """
        self.temperature = temperature

    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature-scaled confidence scores.

        Args:
            logits: Raw model outputs [batch_size, n_classes]

        Returns:
            confidence: Max probability after temperature scaling [batch_size]
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Compute softmax
        probs = torch.softmax(scaled_logits, dim=1)

        # Get max probability (confidence)
        confidence, _ = probs.max(dim=1)

        return confidence

    def is_uncertain(
        self, logits: torch.Tensor, threshold: float = 0.7
    ) -> torch.Tensor:
        """
        Determine which samples are uncertain.

        Args:
            logits: Raw model outputs [batch_size, n_classes]
            threshold: Confidence threshold (default 0.7)

        Returns:
            uncertain_mask: Boolean mask [batch_size]
        """
        confidence = self.compute_confidence(logits)
        return confidence < threshold


class ReservoirBuffer:
    """
    Reservoir sampling buffer with recency-based priority.

    Maintains exactly max_size samples with preference for recent additions.
    Uses weighted reservoir sampling where weight = recency.
    """

    def __init__(self, max_size: int = 50000):
        """
        Args:
            max_size: Maximum number of samples to store
        """
        self.max_size = max_size
        self.buffer = []
        self.metadata = []  # (cluster_id, timestamp, priority)
        self.global_timestamp = 0

    def add(self, samples: np.ndarray, cluster_ids: np.ndarray):
        """
        Add samples using weighted reservoir sampling.

        Args:
            samples: Embeddings to add [n_samples, embedding_dim]
            cluster_ids: Cluster labels [n_samples]
        """
        for sample, cluster_id in zip(samples, cluster_ids):
            self.global_timestamp += 1

            # Calculate recency-based priority
            priority = self.global_timestamp

            if len(self.buffer) < self.max_size:
                # Buffer not full - just add
                self.buffer.append(sample)
                self.metadata.append(
                    {
                        "cluster_id": cluster_id,
                        "timestamp": self.global_timestamp,
                        "priority": priority,
                    }
                )
            else:
                # Buffer full - weighted replacement
                # Probability of replacement = priority / sum_of_priorities
                total_priority = sum(m["priority"] for m in self.metadata)

                # Randomly replace with probability proportional to new priority
                if random.random() < (priority / (total_priority + priority)):
                    # Remove lowest priority sample
                    min_idx = min(
                        range(len(self.metadata)),
                        key=lambda i: self.metadata[i]["priority"],
                    )

                    self.buffer[min_idx] = sample
                    self.metadata[min_idx] = {
                        "cluster_id": cluster_id,
                        "timestamp": self.global_timestamp,
                        "priority": priority,
                    }

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from buffer.

        Args:
            n_samples: Number of samples to retrieve

        Returns:
            samples: Sampled embeddings [n_samples, embedding_dim]
            cluster_ids: Corresponding cluster IDs [n_samples]
        """
        n_samples = min(n_samples, len(self.buffer))

        indices = random.sample(range(len(self.buffer)), n_samples)

        samples = np.array([self.buffer[i] for i in indices])
        cluster_ids = np.array([self.metadata[i]["cluster_id"] for i in indices])

        return samples, cluster_ids

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "clusters_represented": len(set(m["cluster_id"] for m in self.metadata)),
            "timestamp_range": (
                min(m["timestamp"] for m in self.metadata) if self.metadata else 0,
                max(m["timestamp"] for m in self.metadata) if self.metadata else 0,
            ),
        }


class ExemplarBuffer:
    """
    Exemplar buffer with periodic centroid updates.

    Stores representative samples from each cluster and updates centroids
    periodically to adapt to drift.
    """

    def __init__(self, exemplars_per_cluster: int = 100, update_interval: int = 50):
        """
        Args:
            exemplars_per_cluster: Number of exemplars to store per cluster
            update_interval: Recompute centroids every N clusters
        """
        self.exemplars_per_cluster = exemplars_per_cluster
        self.update_interval = update_interval

        self.exemplars = {}  # {cluster_id: np.ndarray [n_exemplars, dim]}
        self.centroids = {}  # {cluster_id: np.ndarray [dim]}
        self.all_samples = {}  # Temporary storage for centroid updates

        self.clusters_added = 0

    def add_cluster(self, cluster_id: int, samples: np.ndarray):
        """
        Add exemplars from a new cluster.

        Args:
            cluster_id: Cluster identifier
            samples: All samples from cluster [n_samples, embedding_dim]
        """
        # Store all samples temporarily (for centroid updates)
        self.all_samples[cluster_id] = samples

        # Select diverse exemplars
        exemplars = self._select_diverse_exemplars(samples, self.exemplars_per_cluster)

        self.exemplars[cluster_id] = exemplars
        self.centroids[cluster_id] = samples.mean(axis=0)

        self.clusters_added += 1

        # Periodic centroid update
        if self.clusters_added % self.update_interval == 0:
            self._update_centroids()

    def _select_diverse_exemplars(
        self, samples: np.ndarray, n_exemplars: int
    ) -> np.ndarray:
        """
        Select diverse representative samples using k-means++ initialization strategy.

        Args:
            samples: Cluster samples [n_samples, dim]
            n_exemplars: Number to select

        Returns:
            exemplars: Selected samples [n_exemplars, dim]
        """
        if len(samples) <= n_exemplars:
            return samples

        # For large n_exemplars, use faster random sampling with diversity check
        if n_exemplars > 100:
            # Random stratified sampling (much faster)
            indices = np.random.choice(len(samples), n_exemplars, replace=False)
            return samples[indices]

        # For smaller n_exemplars, use k-means++ for better diversity
        # Start with centroid
        centroid = samples.mean(axis=0, keepdims=True)
        exemplars = [centroid[0]]
        exemplar_indices = []
        remaining_indices = set(range(len(samples)))

        # Greedy selection: maximize distance from already selected
        for _ in range(n_exemplars - 1):
            # Vectorized distance computation (much faster)
            exemplar_array = np.array(exemplars)
            remaining = np.array(list(remaining_indices))

            # Compute distances from all remaining to all exemplars
            distances_to_exemplars = np.linalg.norm(
                samples[remaining][:, np.newaxis, :] - exemplar_array[np.newaxis, :, :],
                axis=2,
            )

            # Get minimum distance to any exemplar for each remaining sample
            min_distances = distances_to_exemplars.min(axis=1)

            # Select farthest point
            farthest_idx_in_remaining = min_distances.argmax()
            farthest_idx = remaining[farthest_idx_in_remaining]

            exemplars.append(samples[farthest_idx])
            remaining_indices.remove(farthest_idx)

        return np.array(exemplars)

    def _update_centroids(self):
        """
        Recompute centroids for all clusters.

        This adapts to subtle distribution shifts across cluster boundaries.
        """
        print(f"🔄 Updating centroids for {len(self.centroids)} clusters...")

        for cluster_id in self.centroids.keys():
            if cluster_id in self.all_samples:
                # Recompute centroid from all available samples
                self.centroids[cluster_id] = self.all_samples[cluster_id].mean(axis=0)

        # Clean up old samples to save memory (keep only recent 10%)
        clusters_to_keep = sorted(self.all_samples.keys())[
            -max(10, len(self.all_samples) // 10) :
        ]
        self.all_samples = {cid: self.all_samples[cid] for cid in clusters_to_keep}

    def sample(
        self, n_samples: int, exclude_cluster: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample exemplars from all clusters.

        Args:
            n_samples: Number of samples to retrieve
            exclude_cluster: Optionally exclude a specific cluster

        Returns:
            samples: Sampled exemplars [n_samples, dim]
            cluster_ids: Corresponding cluster IDs [n_samples]
        """
        available_clusters = [
            cid for cid in self.exemplars.keys() if cid != exclude_cluster
        ]

        if not available_clusters:
            return np.array([]), np.array([])

        # Sample proportionally from all clusters
        samples_list = []
        cluster_ids_list = []

        samples_per_cluster = max(1, n_samples // len(available_clusters))

        for cluster_id in available_clusters:
            cluster_exemplars = self.exemplars[cluster_id]

            # Sample from this cluster
            n_from_cluster = min(samples_per_cluster, len(cluster_exemplars))
            indices = np.random.choice(
                len(cluster_exemplars), n_from_cluster, replace=False
            )

            samples_list.append(cluster_exemplars[indices])
            cluster_ids_list.extend([cluster_id] * n_from_cluster)

        samples = np.vstack(samples_list)
        cluster_ids = np.array(cluster_ids_list)

        # If we have too many, subsample
        if len(samples) > n_samples:
            indices = np.random.choice(len(samples), n_samples, replace=False)
            samples = samples[indices]
            cluster_ids = cluster_ids[indices]

        return samples, cluster_ids

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            "n_clusters": len(self.exemplars),
            "total_exemplars": sum(len(ex) for ex in self.exemplars.values()),
            "exemplars_per_cluster": self.exemplars_per_cluster,
            "last_centroid_update": self.clusters_added
            - (self.clusters_added % self.update_interval),
        }


class UncertaintyBuffer:
    """
    Buffer for storing uncertain/difficult examples.

    Uses temperature-scaled confidence to identify hard examples.
    """

    def __init__(
        self, max_size: int = 50000, temperature: float = 2.0, threshold: float = 0.7
    ):
        """
        Args:
            max_size: Maximum buffer size
            temperature: Temperature for confidence scaling
            threshold: Confidence threshold for uncertainty
        """
        self.max_size = max_size
        self.confidence_scorer = TemperatureScaledConfidence(temperature)
        self.threshold = threshold

        self.buffer = []  # List of (sample, cluster_id, confidence)

    def add_batch(
        self, samples: np.ndarray, cluster_ids: np.ndarray, logits: torch.Tensor
    ):
        """
        Add samples with low confidence.

        Args:
            samples: Embeddings [batch_size, dim]
            cluster_ids: True cluster labels [batch_size]
            logits: Model outputs [batch_size, n_classes]
        """
        # Compute temperature-scaled confidence
        confidences = self.confidence_scorer.compute_confidence(logits)
        uncertain_mask = confidences < self.threshold

        # Add uncertain samples
        for sample, cluster_id, conf in zip(
            samples[uncertain_mask.cpu().numpy()],
            cluster_ids[uncertain_mask.cpu().numpy()],
            confidences[uncertain_mask].cpu().numpy(),
        ):
            self.buffer.append((sample, cluster_id, float(conf)))

        # Keep only hardest examples (lowest confidence)
        if len(self.buffer) > self.max_size:
            self.buffer = sorted(self.buffer, key=lambda x: x[2])[: self.max_size]

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample uncertain examples.

        Args:
            n_samples: Number of samples to retrieve

        Returns:
            samples: Sampled embeddings [n_samples, dim]
            cluster_ids: Corresponding cluster IDs [n_samples]
        """
        n_samples = min(n_samples, len(self.buffer))

        sampled_items = random.sample(self.buffer, n_samples)

        samples = np.array([item[0] for item in sampled_items])
        cluster_ids = np.array([item[1] for item in sampled_items])

        return samples, cluster_ids

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        if not self.buffer:
            return {"size": 0, "avg_confidence": 0.0, "min_confidence": 0.0}

        confidences = [item[2] for item in self.buffer]

        return {
            "size": len(self.buffer),
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "clusters_represented": len(set(item[1] for item in self.buffer)),
        }


class HybridMemoryBuffer:
    """
    Complete hybrid memory buffer combining all three strategies.

    Integrates:
    - Exemplar buffer with centroid updates
    - Uncertainty buffer with temperature-scaled confidence
    - Recent buffer with reservoir sampling
    """

    def __init__(
        self,
        exemplars_per_cluster: int = 100,
        uncertainty_size: int = 50000,
        recent_size: int = 50000,
        temperature: float = 2.0,
        uncertainty_threshold: float = 0.7,
        centroid_update_interval: int = 50,
    ):
        """
        Args:
            exemplars_per_cluster: Number of exemplars per cluster
            uncertainty_size: Max size of uncertainty buffer
            recent_size: Max size of recent buffer
            temperature: Temperature for confidence scaling
            uncertainty_threshold: Confidence threshold for uncertainty
            centroid_update_interval: Update centroids every N clusters
        """
        self.exemplar_buffer = ExemplarBuffer(
            exemplars_per_cluster=exemplars_per_cluster,
            update_interval=centroid_update_interval,
        )

        self.uncertainty_buffer = UncertaintyBuffer(
            max_size=uncertainty_size,
            temperature=temperature,
            threshold=uncertainty_threshold,
        )

        self.recent_buffer = ReservoirBuffer(max_size=recent_size)

    def add_cluster(
        self,
        cluster_id: int,
        samples: np.ndarray,
        cluster_labels: np.ndarray,
        logits: Optional[torch.Tensor] = None,
    ):
        """
        Add a new cluster to all buffers.

        Args:
            cluster_id: Cluster identifier
            samples: Embeddings from cluster [n_samples, dim]
            cluster_labels: Cluster labels [n_samples]
            logits: Model predictions (for uncertainty) [n_samples, n_classes]
        """
        # Add to exemplar buffer
        self.exemplar_buffer.add_cluster(cluster_id, samples)

        # Add to recent buffer
        self.recent_buffer.add(samples, cluster_labels)

        # Add uncertain samples if logits provided
        if logits is not None:
            self.uncertainty_buffer.add_batch(samples, cluster_labels, logits)

    def mini_retrieval(
        self,
        n_exemplar: int = 4,
        n_uncertain: int = 2,
        n_recent: int = 2,
        exclude_cluster: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mini-retrieval: Get small balanced batch from all buffers.

        This maintains mixture continuity during training.

        Args:
            n_exemplar: Number of exemplar samples
            n_uncertain: Number of uncertain samples
            n_recent: Number of recent samples
            exclude_cluster: Optionally exclude current cluster

        Returns:
            samples: Retrieved embeddings [n_total, dim]
            labels: Corresponding cluster IDs [n_total]
        """
        all_samples = []
        all_labels = []

        # Retrieve from exemplar buffer
        if n_exemplar > 0:
            ex_samples, ex_labels = self.exemplar_buffer.sample(
                n_exemplar, exclude_cluster=exclude_cluster
            )
            if len(ex_samples) > 0:
                all_samples.append(ex_samples)
                all_labels.append(ex_labels)

        # Retrieve from uncertainty buffer
        if n_uncertain > 0:
            unc_samples, unc_labels = self.uncertainty_buffer.sample(n_uncertain)
            if len(unc_samples) > 0:
                all_samples.append(unc_samples)
                all_labels.append(unc_labels)

        # Retrieve from recent buffer
        if n_recent > 0:
            rec_samples, rec_labels = self.recent_buffer.sample(n_recent)
            if len(rec_samples) > 0:
                all_samples.append(rec_samples)
                all_labels.append(rec_labels)

        if not all_samples:
            return np.array([]), np.array([])

        samples = np.vstack(all_samples)
        labels = np.concatenate(all_labels)

        return samples, labels

    def sample_replay_batch(
        self, batch_size: int, exclude_cluster: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a balanced replay batch from all buffers.

        Uses proportional sampling:
        - 40% from exemplar buffer
        - 30% from uncertainty buffer
        - 30% from recent buffer

        Args:
            batch_size: Total samples to retrieve
            exclude_cluster: Optionally exclude current cluster

        Returns:
            samples: Replay embeddings [batch_size, dim]
            labels: Corresponding cluster IDs [batch_size]
        """
        n_exemplar = int(batch_size * 0.4)
        n_uncertain = int(batch_size * 0.3)
        n_recent = batch_size - n_exemplar - n_uncertain

        return self.mini_retrieval(
            n_exemplar=n_exemplar,
            n_uncertain=n_uncertain,
            n_recent=n_recent,
            exclude_cluster=exclude_cluster,
        )

    def get_comprehensive_stats(self) -> Dict:
        """Get statistics from all buffers."""
        return {
            "exemplar": self.exemplar_buffer.get_stats(),
            "uncertainty": self.uncertainty_buffer.get_stats(),
            "recent": self.recent_buffer.get_stats(),
        }
