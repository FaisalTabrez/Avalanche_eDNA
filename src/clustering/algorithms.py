"""
Clustering algorithms for taxonomic grouping of eDNA sequences
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    import warnings
    warnings.warn("HDBSCAN not available. Using DBSCAN as fallback for clustering.")
# Try optional UMAP; fall back to PCA if unavailable
try:
    import umap  # type: ignore
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    from sklearn.decomposition import PCA

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingClusterer:
    """Clustering of sequence embeddings for taxonomic grouping"""
    
    def __init__(self, 
                 method: str = "hdbscan",
                 **kwargs):
        """
        Initialize clusterer
        
        Args:
            method: Clustering method ('hdbscan', 'kmeans', 'dbscan', 'hierarchical')
            **kwargs: Additional parameters for clustering algorithms
        """
        self.method = method
        self.kwargs = kwargs
        self.cluster_labels = None
        self.cluster_stats = None
        self.embeddings = None
        
        # Dimensionality reduction for visualization
        self.umap_reducer = None
        self.reduced_embeddings = None
        
        logger.info(f"Initialized clusterer with method: {method}")
    
    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform clustering on embeddings
        
        Args:
            embeddings: Array of sequence embeddings [n_sequences, embedding_dim]
            
        Returns:
            Cluster labels
        """
        self.embeddings = embeddings
        logger.info(f"Clustering {embeddings.shape[0]} sequences with {embeddings.shape[1]}-dim embeddings")
        
        if self.method == "hdbscan":
            self.cluster_labels = self._cluster_hdbscan(embeddings)
        elif self.method == "kmeans":
            self.cluster_labels = self._cluster_kmeans(embeddings)
        elif self.method == "dbscan":
            self.cluster_labels = self._cluster_dbscan(embeddings)
        elif self.method == "hierarchical":
            self.cluster_labels = self._cluster_hierarchical(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Compute cluster statistics
        self.cluster_stats = self._compute_cluster_stats()
        
        logger.info(f"Clustering complete: {len(np.unique(self.cluster_labels))} clusters found")
        return self.cluster_labels
    
    def _cluster_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """HDBSCAN clustering with fallback to DBSCAN"""
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available, using DBSCAN as fallback")
            return self._cluster_dbscan(embeddings)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.kwargs.get('min_cluster_size', 10),
            min_samples=self.kwargs.get('min_samples', 5),
            metric=self.kwargs.get('metric', 'euclidean'),
            cluster_selection_epsilon=self.kwargs.get('cluster_selection_epsilon', 0.0),
            alpha=self.kwargs.get('alpha', 1.0)
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Store additional information
        self.clusterer_info = {
            'probabilities': clusterer.probabilities_,
            'cluster_persistence': clusterer.cluster_persistence_,
            'outlier_scores': clusterer.outlier_scores_
        }
        
        return labels
    
    def _cluster_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """K-means clustering"""
        n_clusters = self.kwargs.get('n_clusters', self._estimate_n_clusters(embeddings))
        
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Store additional information
        self.clusterer_info = {
            'cluster_centers': clusterer.cluster_centers_,
            'inertia': clusterer.inertia_
        }
        
        return labels
    
    def _cluster_dbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """DBSCAN clustering"""
        clusterer = DBSCAN(
            eps=self.kwargs.get('eps', 0.5),
            min_samples=self.kwargs.get('min_samples', 5),
            metric=self.kwargs.get('metric', 'euclidean')
        )
        
        labels = clusterer.fit_predict(embeddings)
        return labels
    
    def _cluster_hierarchical(self, embeddings: np.ndarray) -> np.ndarray:
        """Hierarchical clustering"""
        # Compute distance matrix
        distances = pdist(embeddings, metric=self.kwargs.get('metric', 'euclidean'))
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method=self.kwargs.get('linkage_method', 'ward'))
        
        # Get clusters
        n_clusters = self.kwargs.get('n_clusters', self._estimate_n_clusters(embeddings))
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1  # Make 0-indexed
        
        # Store additional information
        self.clusterer_info = {
            'linkage_matrix': linkage_matrix,
            'distances': distances
        }
        
        return labels
    
    def _estimate_n_clusters(self, embeddings: np.ndarray) -> int:
        """Estimate optimal number of clusters using elbow method"""
        max_k = min(20, embeddings.shape[0] // 10)
        if max_k < 2:
            return 2
        
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simplified method)
        differences = np.diff(inertias)
        second_differences = np.diff(differences)
        elbow_idx = np.argmax(second_differences) + 2  # +2 because of double diff
        
        optimal_k = k_range[min(elbow_idx, len(k_range) - 1)]
        logger.info(f"Estimated optimal number of clusters: {optimal_k}")
        
        return optimal_k
    
    def _compute_cluster_stats(self) -> Dict[str, Any]:
        """Compute cluster statistics"""
        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise points (-1)
        n_noise = np.sum(self.cluster_labels == -1)
        
        stats = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'cluster_sizes': {},
            'silhouette_score': None,
            'calinski_harabasz_score': None
        }
        
        # Cluster sizes
        for label in unique_labels:
            stats['cluster_sizes'][str(label)] = np.sum(self.cluster_labels == label)
        
        # Silhouette score (only if we have more than 1 cluster and no single cluster)
        if n_clusters > 1 and len(np.unique(self.cluster_labels)) > 1:
            try:
                stats['silhouette_score'] = silhouette_score(self.embeddings, self.cluster_labels)
            except:
                pass
        
        # Calinski-Harabasz score
        if n_clusters > 1:
            try:
                stats['calinski_harabasz_score'] = calinski_harabasz_score(self.embeddings, self.cluster_labels)
            except:
                pass
        
        return stats
    
    def reduce_dimensions(self, 
                         n_components: int = 2,
                         n_neighbors: int = 15,
                         min_dist: float = 0.1) -> np.ndarray:
        """
        Reduce dimensionality for visualization.
        Uses UMAP if available; otherwise falls back to PCA.
        
        Args:
            n_components: Number of dimensions for reduced space
            n_neighbors: UMAP n_neighbors parameter (ignored for PCA)
            min_dist: UMAP min_dist parameter (ignored for PCA)
            
        Returns:
            Reduced embeddings
        """
        if self.embeddings is None:
            raise ValueError("Must fit clusterer first")
        
        logger.info(f"Reducing dimensionality to {n_components}D for visualization")
        
        if UMAP_AVAILABLE:
            self.umap_reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42
            )
            self.reduced_embeddings = self.umap_reducer.fit_transform(self.embeddings)
        else:
            logger.warning("UMAP not available; using PCA fallback for dimensionality reduction")
            pca = PCA(n_components=n_components, random_state=42)
            self.reduced_embeddings = pca.fit_transform(self.embeddings)
        
        return self.reduced_embeddings
    
    def plot_clusters(self, 
                     sequences: Optional[List[str]] = None,
                     save_path: Optional[Path] = None,
                     figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot cluster visualization
        
        Args:
            sequences: Optional list of sequences for labeling
            save_path: Optional path to save plot
            figsize: Figure size
        """
        if self.cluster_labels is None:
            raise ValueError("Must fit clusterer first")
        
        # Ensure we have reduced embeddings
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
        
        plt.figure(figsize=figsize)
        
        # Create color map
        unique_labels = np.unique(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = self.cluster_labels == label
            
            if label == -1:
                # Noise points
                plt.scatter(self.reduced_embeddings[mask, 0], 
                          self.reduced_embeddings[mask, 1],
                          c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                plt.scatter(self.reduced_embeddings[mask, 0], 
                          self.reduced_embeddings[mask, 1],
                          c=[colors[i]], s=60, alpha=0.7, label=f'Cluster {label}')
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title(f'Sequence Clustering ({self.method.upper()})\n'
                 f'{self.cluster_stats["n_clusters"]} clusters, '
                 f'{self.cluster_stats["n_noise_points"]} noise points')
        
        # Add legend (limit to reasonable number of entries)
        if len(unique_labels) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster plot saved to {save_path}")
        
        plt.show()
    
    def plot_cluster_stats(self, save_path: Optional[Path] = None) -> None:
        """Plot cluster statistics"""
        if self.cluster_stats is None:
            raise ValueError("Must fit clusterer first")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cluster size distribution
        cluster_sizes = [size for label, size in self.cluster_stats['cluster_sizes'].items() 
                        if label != '-1']  # Exclude noise
        
        axes[0].hist(cluster_sizes, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Cluster Size')
        axes[0].set_ylabel('Number of Clusters')
        axes[0].set_title('Cluster Size Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Top clusters by size
        cluster_data = [(int(label), size) for label, size in self.cluster_stats['cluster_sizes'].items() 
                       if label != '-1']
        cluster_data.sort(key=lambda x: x[1], reverse=True)
        
        top_clusters = cluster_data[:min(10, len(cluster_data))]
        if top_clusters:
            labels, sizes = zip(*top_clusters)
            
            axes[1].bar(range(len(labels)), sizes, alpha=0.7)
            axes[1].set_xlabel('Cluster ID')
            axes[1].set_ylabel('Cluster Size')
            axes[1].set_title('Top 10 Largest Clusters')
            axes[1].set_xticks(range(len(labels)))
            axes[1].set_xticklabels(labels, rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster statistics plot saved to {save_path}")
        
        plt.show()
    
    def get_cluster_representatives(self, 
                                  sequences: List[str], 
                                  n_representatives: int = 5) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get representative sequences for each cluster
        
        Args:
            sequences: List of sequences
            n_representatives: Number of representatives per cluster
            
        Returns:
            Dictionary mapping cluster ID to list of (sequence, distance_to_center) tuples
        """
        if self.cluster_labels is None or self.embeddings is None:
            raise ValueError("Must fit clusterer first")
        
        representatives = {}
        unique_labels = np.unique(self.cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            # Get sequences in this cluster
            cluster_mask = self.cluster_labels == label
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_sequences = [sequences[i] for i in np.where(cluster_mask)[0]]
            
            # Find cluster center
            cluster_center = np.mean(cluster_embeddings, axis=0)
            
            # Calculate distances to center
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            
            # Get top representatives (closest to center)
            top_indices = np.argsort(distances)[:n_representatives]
            
            representatives[int(label)] = [
                (cluster_sequences[i], float(distances[i])) 
                for i in top_indices
            ]
        
        return representatives
    
    def save_results(self, 
                    sequences: List[str],
                    save_dir: Path,
                    include_embeddings: bool = False) -> None:
        """
        Save clustering results
        
        Args:
            sequences: List of sequences
            save_dir: Directory to save results
            include_embeddings: Whether to save embeddings
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cluster assignments
        results_df = pd.DataFrame({
            'sequence_id': range(len(sequences)),
            'sequence': sequences,
            'cluster_id': self.cluster_labels
        })
        
        results_df.to_csv(save_dir / 'cluster_assignments.csv', index=False)
        
        # Save cluster statistics
        with open(save_dir / 'cluster_stats.txt', 'w') as f:
            f.write(f"Clustering Method: {self.method}\n")
            f.write(f"Number of Clusters: {self.cluster_stats['n_clusters']}\n")
            f.write(f"Number of Noise Points: {self.cluster_stats['n_noise_points']}\n")
            
            if self.cluster_stats['silhouette_score'] is not None:
                f.write(f"Silhouette Score: {self.cluster_stats['silhouette_score']:.4f}\n")
            
            if self.cluster_stats['calinski_harabasz_score'] is not None:
                f.write(f"Calinski-Harabasz Score: {self.cluster_stats['calinski_harabasz_score']:.4f}\n")
            
            f.write("\nCluster Sizes:\n")
            for label, size in self.cluster_stats['cluster_sizes'].items():
                f.write(f"Cluster {label}: {size} sequences\n")
        
        # Save embeddings if requested
        if include_embeddings and self.embeddings is not None:
            np.save(save_dir / 'embeddings.npy', self.embeddings)
            
            if self.reduced_embeddings is not None:
                np.save(save_dir / 'reduced_embeddings.npy', self.reduced_embeddings)
        
        # Save cluster representatives
        representatives = self.get_cluster_representatives(sequences)
        with open(save_dir / 'cluster_representatives.txt', 'w') as f:
            for cluster_id, reps in representatives.items():
                f.write(f"\nCluster {cluster_id}:\n")
                for i, (seq, dist) in enumerate(reps, 1):
                    f.write(f"  {i}. Distance: {dist:.4f}\n")
                    f.write(f"     Sequence: {seq[:100]}{'...' if len(seq) > 100 else ''}\n")
        
        logger.info(f"Clustering results saved to {save_dir}")

def main():
    """Main function for testing clustering"""
    # Create mock embeddings for testing
    np.random.seed(42)
    n_sequences = 1000
    embedding_dim = 256
    
    # Create embeddings with some structure
    cluster_centers = np.random.randn(5, embedding_dim) * 2
    embeddings = []
    sequences = []
    
    for i in range(n_sequences):
        # Assign to random cluster
        cluster_idx = np.random.randint(0, 5)
        # Add noise around cluster center
        embedding = cluster_centers[cluster_idx] + np.random.randn(embedding_dim) * 0.5
        embeddings.append(embedding)
        
        # Create mock sequence
        nucleotides = ['A', 'T', 'G', 'C']
        seq_length = np.random.randint(100, 300)
        sequence = ''.join(np.random.choice(nucleotides, size=seq_length))
        sequences.append(sequence)
    
    embeddings = np.array(embeddings)
    
    # Test clustering
    logger.info("Testing clustering algorithms...")
    
    # Test HDBSCAN
    clusterer = EmbeddingClusterer(method="hdbscan", min_cluster_size=20)
    labels = clusterer.fit(embeddings)
    
    # Visualize results
    clusterer.plot_clusters(sequences)
    clusterer.plot_cluster_stats()
    
    # Save results
    save_dir = Path("data/output/clustering_test")
    clusterer.save_results(sequences, save_dir, include_embeddings=True)
    
    logger.info("Clustering test complete!")

if __name__ == "__main__":
    main()