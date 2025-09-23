"""
Novelty detection system for identifying potentially novel taxa in eDNA datasets
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoveltyDetector:
    """Base class for novelty detection algorithms"""
    
    def __init__(self, method: str = "isolation_forest", **kwargs):
        """
        Initialize novelty detector
        
        Args:
            method: Detection method ('isolation_forest', 'one_class_svm', 'local_outlier_factor', 'elliptic_envelope')
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
        self.detector = None
        self.scaler = None
        self.is_fitted = False
        
        self._initialize_detector()
        logger.info(f"Novelty detector initialized with method: {method}")
    
    def _initialize_detector(self):
        """Initialize the detection algorithm"""
        if self.method == "isolation_forest":
            self.detector = IsolationForest(
                contamination=self.kwargs.get('contamination', 0.1),
                random_state=42,
                n_jobs=-1
            )
        elif self.method == "one_class_svm":
            self.detector = OneClassSVM(
                kernel=self.kwargs.get('kernel', 'rbf'),
                gamma=self.kwargs.get('gamma', 'scale'),
                nu=self.kwargs.get('nu', 0.1)
            )
        elif self.method == "local_outlier_factor":
            self.detector = LocalOutlierFactor(
                n_neighbors=self.kwargs.get('n_neighbors', 20),
                contamination=self.kwargs.get('contamination', 0.1),
                novelty=True,
                n_jobs=-1
            )
        elif self.method == "elliptic_envelope":
            self.detector = EllipticEnvelope(
                contamination=self.kwargs.get('contamination', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown novelty detection method: {self.method}")
    
    def fit(self, embeddings: np.ndarray, normalize: bool = True) -> 'NoveltyDetector':
        """
        Fit the novelty detector
        
        Args:
            embeddings: Training embeddings (known taxa)
            normalize: Whether to normalize embeddings
            
        Returns:
            Self
        """
        logger.info(f"Fitting novelty detector on {embeddings.shape[0]} samples")
        
        # Normalize if requested
        if normalize:
            self.scaler = StandardScaler()
            embeddings = self.scaler.fit_transform(embeddings)
        
        # Fit detector
        self.detector.fit(embeddings)
        self.is_fitted = True
        
        logger.info("Novelty detector fitted successfully")
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict novelty scores
        
        Args:
            embeddings: Query embeddings
            
        Returns:
            Novelty predictions (1 for normal, -1 for novel)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        # Normalize if scaler was used during fitting
        if self.scaler is not None:
            embeddings = self.scaler.transform(embeddings)
        
        return self.detector.predict(embeddings)
    
    def decision_function(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get novelty scores
        
        Args:
            embeddings: Query embeddings
            
        Returns:
            Novelty scores (higher = more normal)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        # Normalize if scaler was used during fitting
        if self.scaler is not None:
            embeddings = self.scaler.transform(embeddings)
        
        return self.detector.decision_function(embeddings)

class DistanceBasedNoveltyDetector:
    """Distance-based novelty detection using nearest neighbors"""
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 distance_threshold: float = None,
                 metric: str = 'euclidean'):
        """
        Initialize distance-based novelty detector
        
        Args:
            n_neighbors: Number of nearest neighbors to consider
            distance_threshold: Distance threshold for novelty (auto-estimated if None)
            metric: Distance metric
        """
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold
        self.metric = metric
        self.nn_model = None
        self.reference_embeddings = None
        self.is_fitted = False
        
        logger.info(f"Distance-based novelty detector initialized")
    
    def fit(self, embeddings: np.ndarray) -> 'DistanceBasedNoveltyDetector':
        """
        Fit the detector using reference embeddings
        
        Args:
            embeddings: Reference embeddings (known taxa)
            
        Returns:
            Self
        """
        logger.info(f"Fitting distance-based detector on {embeddings.shape[0]} reference samples")
        
        self.reference_embeddings = embeddings.copy()
        
        # Fit nearest neighbors model
        self.nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_jobs=-1
        )
        self.nn_model.fit(embeddings)
        
        # Estimate distance threshold if not provided
        if self.distance_threshold is None:
            self.distance_threshold = self._estimate_distance_threshold(embeddings)
        
        self.is_fitted = True
        logger.info(f"Distance threshold set to: {self.distance_threshold:.4f}")
        
        return self
    
    def _estimate_distance_threshold(self, embeddings: np.ndarray) -> float:
        """Estimate distance threshold using percentile of reference distances"""
        # Get distances to k-th nearest neighbor for all reference points
        distances, _ = self.nn_model.kneighbors(embeddings)
        kth_distances = distances[:, -1]  # Distance to k-th neighbor
        
        # Use 95th percentile as threshold
        threshold = np.percentile(kth_distances, 95)
        return threshold
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict novelty based on distance to nearest neighbors
        
        Args:
            embeddings: Query embeddings
            
        Returns:
            Novelty predictions (1 for normal, -1 for novel)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        # Get distances to nearest neighbors
        distances, _ = self.nn_model.kneighbors(embeddings)
        avg_distances = np.mean(distances, axis=1)
        
        # Predict based on threshold
        predictions = np.where(avg_distances <= self.distance_threshold, 1, -1)
        
        return predictions
    
    def decision_function(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get novelty scores based on average distance to k nearest neighbors
        
        Args:
            embeddings: Query embeddings
            
        Returns:
            Novelty scores (lower = more novel)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        # Get distances to nearest neighbors
        distances, _ = self.nn_model.kneighbors(embeddings)
        avg_distances = np.mean(distances, axis=1)
        
        # Convert to scores (negative of distance for consistency with sklearn)
        scores = -avg_distances
        
        return scores

class ClusterBasedNoveltyDetector:
    """Cluster-based novelty detection"""
    
    def __init__(self,
                 cluster_method: str = "dbscan",
                 eps: float = 0.5,
                 min_samples: int = 5,
                 outlier_threshold: float = 0.1):
        """
        Initialize cluster-based novelty detector
        
        Args:
            cluster_method: Clustering method
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter
            outlier_threshold: Threshold for considering points as outliers
        """
        self.cluster_method = cluster_method
        self.eps = eps
        self.min_samples = min_samples
        self.outlier_threshold = outlier_threshold
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.is_fitted = False
        
        logger.info("Cluster-based novelty detector initialized")
    
    def fit(self, embeddings: np.ndarray) -> 'ClusterBasedNoveltyDetector':
        """
        Fit the detector by clustering reference data
        
        Args:
            embeddings: Reference embeddings
            
        Returns:
            Self
        """
        logger.info(f"Clustering {embeddings.shape[0]} reference samples")
        
        if self.cluster_method == "dbscan":
            self.clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            self.cluster_labels = self.clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self.cluster_method}")
        
        # Compute cluster centers (exclude noise points)
        unique_labels = np.unique(self.cluster_labels)
        self.cluster_centers = {}
        
        for label in unique_labels:
            if label != -1:  # Exclude noise
                cluster_mask = self.cluster_labels == label
                cluster_points = embeddings[cluster_mask]
                self.cluster_centers[label] = np.mean(cluster_points, axis=0)
        
        self.reference_embeddings = embeddings
        self.is_fitted = True
        
        n_clusters = len(self.cluster_centers)
        n_noise = np.sum(self.cluster_labels == -1)
        
        logger.info(f"Found {n_clusters} clusters and {n_noise} noise points")
        
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict novelty based on cluster membership
        
        Args:
            embeddings: Query embeddings
            
        Returns:
            Novelty predictions
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        predictions = []
        
        for embedding in embeddings:
            # Find distance to nearest cluster center
            min_distance = float('inf')
            
            for center in self.cluster_centers.values():
                distance = np.linalg.norm(embedding - center)
                min_distance = min(min_distance, distance)
            
            # Compare with reference distances to determine novelty
            # This is a simplified approach - in practice you'd want more sophisticated scoring
            if min_distance > self._get_distance_threshold():
                predictions.append(-1)  # Novel
            else:
                predictions.append(1)   # Normal
        
        return np.array(predictions)
    
    def _get_distance_threshold(self) -> float:
        """Get distance threshold based on reference data clustering"""
        if not hasattr(self, '_distance_threshold'):
            # Compute average distance from points to their cluster centers
            distances = []
            
            for i, label in enumerate(self.cluster_labels):
                if label != -1:  # Exclude noise points
                    point = self.reference_embeddings[i]
                    center = self.cluster_centers[label]
                    distance = np.linalg.norm(point - center)
                    distances.append(distance)
            
            # Use 95th percentile as threshold
            self._distance_threshold = np.percentile(distances, 95)
        
        return self._distance_threshold

class EnsembleNoveltyDetector:
    """Ensemble novelty detector combining multiple methods"""
    
    def __init__(self, detectors: List[NoveltyDetector], voting: str = 'soft'):
        """
        Initialize ensemble detector
        
        Args:
            detectors: List of base detectors
            voting: Voting strategy ('hard' or 'soft')
        """
        self.detectors = detectors
        self.voting = voting
        self.is_fitted = False
        
        logger.info(f"Ensemble detector initialized with {len(detectors)} base detectors")
    
    def fit(self, embeddings: np.ndarray) -> 'EnsembleNoveltyDetector':
        """Fit all base detectors"""
        logger.info("Fitting ensemble detectors...")
        
        for i, detector in enumerate(self.detectors):
            logger.info(f"Fitting detector {i+1}/{len(self.detectors)}")
            detector.fit(embeddings)
        
        self.is_fitted = True
        logger.info("Ensemble fitting complete")
        
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Ensemble prediction"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        if self.voting == 'hard':
            # Majority voting
            predictions = np.array([detector.predict(embeddings) for detector in self.detectors])
            ensemble_pred = np.array([stats.mode(predictions[:, i])[0][0] for i in range(predictions.shape[1])])
        
        elif self.voting == 'soft':
            # Average of decision functions
            scores = np.array([detector.decision_function(embeddings) for detector in self.detectors])
            avg_scores = np.mean(scores, axis=0)
            ensemble_pred = np.where(avg_scores > 0, 1, -1)
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting}")
        
        return ensemble_pred
    
    def decision_function(self, embeddings: np.ndarray) -> np.ndarray:
        """Ensemble decision function"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        scores = np.array([detector.decision_function(embeddings) for detector in self.detectors])
        return np.mean(scores, axis=0)

class NoveltyAnalyzer:
    """High-level analyzer for novel taxa identification"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 abundance_threshold: float = 0.001,
                 cluster_coherence_threshold: float = 0.7):
        """
        Initialize novelty analyzer
        
        Args:
            similarity_threshold: Similarity threshold for known taxa
            abundance_threshold: Minimum abundance for consideration
            cluster_coherence_threshold: Minimum cluster coherence for novel taxa
        """
        self.similarity_threshold = similarity_threshold
        self.abundance_threshold = abundance_threshold
        self.cluster_coherence_threshold = cluster_coherence_threshold
        
        # Detectors
        self.novelty_detector = None
        self.distance_detector = None
        
        logger.info("Novelty analyzer initialized")
    
    def analyze_novelty(self,
                       query_embeddings: np.ndarray,
                       reference_embeddings: np.ndarray,
                       query_sequences: List[str],
                       query_abundances: Optional[np.ndarray] = None,
                       cluster_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive novelty analysis
        
        Args:
            query_embeddings: Embeddings of query sequences
            reference_embeddings: Embeddings of known reference sequences
            query_sequences: Query sequence strings
            query_abundances: Optional abundance data
            cluster_labels: Optional cluster labels for query sequences
            
        Returns:
            Novelty analysis results
        """
        logger.info(f"Analyzing novelty for {len(query_sequences)} sequences")
        
        # Initialize detectors
        self.novelty_detector = NoveltyDetector(method="isolation_forest")
        self.distance_detector = DistanceBasedNoveltyDetector()
        
        # Fit on reference data
        self.novelty_detector.fit(reference_embeddings)
        self.distance_detector.fit(reference_embeddings)
        
        # Get novelty predictions
        isolation_pred = self.novelty_detector.predict(query_embeddings)
        isolation_scores = self.novelty_detector.decision_function(query_embeddings)
        
        distance_pred = self.distance_detector.predict(query_embeddings)
        distance_scores = self.distance_detector.decision_function(query_embeddings)
        
        # Combine predictions
        ensemble_pred = np.where((isolation_pred == -1) | (distance_pred == -1), -1, 1)
        
        # Filter by abundance if provided
        if query_abundances is not None:
            abundance_mask = query_abundances >= self.abundance_threshold
            ensemble_pred = np.where(abundance_mask, ensemble_pred, 1)  # Not novel if too rare
        
        # Analyze cluster coherence if clusters provided
        cluster_coherence_scores = None
        if cluster_labels is not None:
            cluster_coherence_scores = self._analyze_cluster_coherence(
                query_embeddings, cluster_labels
            )
        
        # Identify novel candidates
        novel_indices = np.where(ensemble_pred == -1)[0]
        
        # Create results
        results = {
            'total_sequences': len(query_sequences),
            'novel_candidates': len(novel_indices),
            'novel_percentage': (len(novel_indices) / len(query_sequences)) * 100,
            'novel_indices': novel_indices.tolist(),
            'predictions': {
                'isolation_forest': isolation_pred.tolist(),
                'distance_based': distance_pred.tolist(),
                'ensemble': ensemble_pred.tolist()
            },
            'scores': {
                'isolation_forest': isolation_scores.tolist(),
                'distance_based': distance_scores.tolist()
            },
            'novel_sequences': [query_sequences[i] for i in novel_indices],
            'cluster_coherence_scores': cluster_coherence_scores
        }
        
        # Add detailed analysis for novel candidates
        if novel_indices.size > 0:
            results['novel_analysis'] = self._analyze_novel_candidates(
                query_embeddings[novel_indices],
                reference_embeddings,
                [query_sequences[i] for i in novel_indices],
                cluster_labels[novel_indices] if cluster_labels is not None else None
            )
        
        logger.info(f"Novelty analysis complete. Found {len(novel_indices)} potential novel taxa")
        
        return results
    
    def _analyze_cluster_coherence(self, 
                                 embeddings: np.ndarray, 
                                 cluster_labels: np.ndarray) -> Dict[int, float]:
        """Analyze coherence of clusters for novelty assessment"""
        coherence_scores = {}
        
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            cluster_mask = cluster_labels == label
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) < 2:
                coherence_scores[int(label)] = 0.0
                continue
            
            # Calculate intra-cluster distances
            distances = pdist(cluster_embeddings)
            avg_distance = np.mean(distances)
            
            # Calculate coherence score (inverse of average distance, normalized)
            coherence = 1.0 / (1.0 + avg_distance)
            coherence_scores[int(label)] = coherence
        
        return coherence_scores
    
    def _analyze_novel_candidates(self,
                                novel_embeddings: np.ndarray,
                                reference_embeddings: np.ndarray,
                                novel_sequences: List[str],
                                novel_cluster_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detailed analysis of novel candidates"""
        
        # Calculate distances to nearest known taxa
        distances = cdist(novel_embeddings, reference_embeddings)
        min_distances = np.min(distances, axis=1)
        
        # Group by clusters if available
        cluster_analysis = {}
        if novel_cluster_labels is not None:
            unique_clusters = np.unique(novel_cluster_labels)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:
                    continue
                
                cluster_mask = novel_cluster_labels == cluster_id
                cluster_sequences = [novel_sequences[i] for i in np.where(cluster_mask)[0]]
                cluster_distances = min_distances[cluster_mask]
                
                cluster_analysis[int(cluster_id)] = {
                    'n_sequences': len(cluster_sequences),
                    'avg_distance_to_known': float(np.mean(cluster_distances)),
                    'min_distance_to_known': float(np.min(cluster_distances)),
                    'max_distance_to_known': float(np.max(cluster_distances)),
                    'representative_sequences': cluster_sequences[:5]  # Top 5
                }
        
        analysis = {
            'n_novel_candidates': len(novel_sequences),
            'distances_to_known_taxa': {
                'mean': float(np.mean(min_distances)),
                'std': float(np.std(min_distances)),
                'min': float(np.min(min_distances)),
                'max': float(np.max(min_distances))
            },
            'cluster_analysis': cluster_analysis,
            'top_novel_sequences': novel_sequences[:10]  # Top 10
        }
        
        return analysis
    
    def visualize_novelty_results(self,
                                embeddings: np.ndarray,
                                novelty_predictions: np.ndarray,
                                save_path: Optional[Path] = None) -> None:
        """Visualize novelty detection results"""
        
        # Reduce dimensionality for visualization
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot normal points
        normal_mask = novelty_predictions == 1
        plt.scatter(reduced_embeddings[normal_mask, 0], 
                   reduced_embeddings[normal_mask, 1],
                   c='blue', alpha=0.6, label='Known Taxa', s=50)
        
        # Plot novel points
        novel_mask = novelty_predictions == -1
        plt.scatter(reduced_embeddings[novel_mask, 0], 
                   reduced_embeddings[novel_mask, 1],
                   c='red', alpha=0.8, label='Novel Candidates', s=100, marker='^')
        
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Novelty Detection Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Novelty visualization saved to {save_path}")
        
        plt.show()

def main():
    """Main function for testing novelty detection"""
    # Create mock data
    np.random.seed(42)
    
    # Reference data (known taxa)
    n_reference = 500
    reference_embeddings = np.random.randn(n_reference, 256)
    
    # Query data (mix of known and novel)
    n_query = 100
    query_embeddings = np.random.randn(n_query, 256)
    
    # Add some clearly novel points (outliers)
    novel_embeddings = np.random.randn(20, 256) * 3  # Further from origin
    query_embeddings = np.vstack([query_embeddings, novel_embeddings])
    
    query_sequences = [f"SEQUENCE_{i}" for i in range(len(query_embeddings))]
    
    # Test novelty detection
    logger.info("Testing novelty detection...")
    
    analyzer = NoveltyAnalyzer()
    results = analyzer.analyze_novelty(
        query_embeddings=query_embeddings,
        reference_embeddings=reference_embeddings,
        query_sequences=query_sequences
    )
    
    logger.info(f"Novelty analysis results:")
    logger.info(f"Total sequences: {results['total_sequences']}")
    logger.info(f"Novel candidates: {results['novel_candidates']}")
    logger.info(f"Novel percentage: {results['novel_percentage']:.1f}%")
    
    # Visualize results
    analyzer.visualize_novelty_results(
        query_embeddings,
        np.array(results['predictions']['ensemble'])
    )
    
    logger.info("Novelty detection testing complete!")

if __name__ == "__main__":
    main()