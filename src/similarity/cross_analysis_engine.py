"""
Cross-analysis similarity comparison engine.

This module provides sophisticated algorithms for comparing analysis results
across different reports, identifying patterns, and calculating similarity metrics.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime
from collections import defaultdict, Counter
from scipy.spatial.distance import cosine, jaccard
from scipy.stats import pearsonr, spearmanr
import json

from src.database.manager import DatabaseManager
from src.database.models import SimilarityMatrix, ReportComparison
from src.database.queries import ReportQueryEngine

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    Advanced similarity calculator for cross-analysis comparisons.
    """
    
    def __init__(self):
        """Initialize similarity calculator."""
        self.similarity_methods = {
            'jaccard': self._jaccard_similarity,
            'cosine': self._cosine_similarity,
            'overlap': self._overlap_similarity,
            'abundance_correlation': self._abundance_correlation,
            'taxonomic_similarity': self._taxonomic_similarity
        }
    
    def calculate_organism_overlap_similarity(self, 
                                           organisms_1: Set[str], 
                                           organisms_2: Set[str]) -> Dict[str, float]:
        """
        Calculate organism overlap similarity metrics.
        
        Args:
            organisms_1: Set of organism IDs from first report
            organisms_2: Set of organism IDs from second report
            
        Returns:
            Dictionary with similarity metrics
        """
        intersection = organisms_1.intersection(organisms_2)
        union = organisms_1.union(organisms_2)
        
        # Jaccard similarity
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        
        # Overlap coefficient (Szymkiewicz-Simpson coefficient)
        overlap_coeff = len(intersection) / min(len(organisms_1), len(organisms_2)) if min(len(organisms_1), len(organisms_2)) > 0 else 0.0
        
        # Sorensen-Dice coefficient
        dice_coeff = 2 * len(intersection) / (len(organisms_1) + len(organisms_2)) if (len(organisms_1) + len(organisms_2)) > 0 else 0.0
        
        return {
            'jaccard_similarity': jaccard_sim,
            'overlap_coefficient': overlap_coeff,
            'dice_coefficient': dice_coeff,
            'shared_organisms': len(intersection),
            'total_unique_organisms': len(union),
            'organisms_only_in_first': len(organisms_1 - organisms_2),
            'organisms_only_in_second': len(organisms_2 - organisms_1)
        }
    
    def calculate_abundance_similarity(self, 
                                     abundance_1: Dict[str, float], 
                                     abundance_2: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate abundance-based similarity metrics.
        
        Args:
            abundance_1: Organism abundance dictionary for first report
            abundance_2: Organism abundance dictionary for second report
            
        Returns:
            Dictionary with abundance similarity metrics
        """
        # Get all organisms from both reports
        all_organisms = set(abundance_1.keys()).union(set(abundance_2.keys()))
        
        # Create aligned abundance vectors
        vector_1 = np.array([abundance_1.get(org, 0.0) for org in all_organisms])
        vector_2 = np.array([abundance_2.get(org, 0.0) for org in all_organisms])
        
        # Calculate various similarity metrics
        similarities = {}
        
        # Cosine similarity
        if np.sum(vector_1) > 0 and np.sum(vector_2) > 0:
            similarities['cosine_similarity'] = 1 - cosine(vector_1, vector_2)
        else:
            similarities['cosine_similarity'] = 0.0
        
        # Pearson correlation
        if len(vector_1) > 1 and np.std(vector_1) > 0 and np.std(vector_2) > 0:
            corr, p_value = pearsonr(vector_1, vector_2)
            similarities['pearson_correlation'] = corr
            similarities['pearson_p_value'] = p_value
        else:
            similarities['pearson_correlation'] = 0.0
            similarities['pearson_p_value'] = 1.0
        
        # Spearman correlation
        if len(vector_1) > 1:
            spearman_corr, spearman_p = spearmanr(vector_1, vector_2)
            similarities['spearman_correlation'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
            similarities['spearman_p_value'] = spearman_p if not np.isnan(spearman_p) else 1.0
        else:
            similarities['spearman_correlation'] = 0.0
            similarities['spearman_p_value'] = 1.0
        
        # Bray-Curtis similarity
        bray_curtis = self._bray_curtis_similarity(vector_1, vector_2)
        similarities['bray_curtis_similarity'] = bray_curtis
        
        # Manhattan distance (normalized)
        manhattan_dist = np.sum(np.abs(vector_1 - vector_2))
        max_possible_dist = np.sum(np.maximum(vector_1, vector_2))
        similarities['manhattan_similarity'] = 1 - (manhattan_dist / max_possible_dist) if max_possible_dist > 0 else 1.0
        
        return similarities
    
    def calculate_taxonomic_composition_similarity(self, 
                                                 taxonomy_1: Dict[str, Dict[str, int]], 
                                                 taxonomy_2: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """
        Calculate taxonomic composition similarity at different levels.
        
        Args:
            taxonomy_1: Taxonomic composition for first report
            taxonomy_2: Taxonomic composition for second report
            
        Returns:
            Dictionary with taxonomic similarity metrics
        """
        similarities = {}
        
        taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
        for level in taxonomic_levels:
            level_comp_1 = taxonomy_1.get(level, {})
            level_comp_2 = taxonomy_2.get(level, {})
            
            if level_comp_1 or level_comp_2:
                # Calculate similarity for this taxonomic level
                level_similarity = self.calculate_abundance_similarity(level_comp_1, level_comp_2)
                similarities[f'{level}_similarity'] = level_similarity['cosine_similarity']
            else:
                similarities[f'{level}_similarity'] = 0.0
        
        # Calculate overall taxonomic similarity (weighted average)
        weights = {'kingdom': 0.1, 'phylum': 0.15, 'class': 0.15, 'order': 0.15, 
                  'family': 0.15, 'genus': 0.2, 'species': 0.3}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for level, weight in weights.items():
            if f'{level}_similarity' in similarities:
                weighted_sum += similarities[f'{level}_similarity'] * weight
                total_weight += weight
        
        similarities['overall_taxonomic_similarity'] = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return similarities
    
    def calculate_diversity_similarity(self, 
                                     diversity_1: Dict[str, float], 
                                     diversity_2: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate similarity between diversity metrics.
        
        Args:
            diversity_1: Diversity metrics for first report
            diversity_2: Diversity metrics for second report
            
        Returns:
            Dictionary with diversity similarity metrics
        """
        similarities = {}
        
        diversity_metrics = ['shannon_diversity', 'simpson_diversity', 'evenness', 'species_richness']
        
        for metric in diversity_metrics:
            val_1 = diversity_1.get(metric)
            val_2 = diversity_2.get(metric)
            
            # Handle None values
            if val_1 is None:
                val_1 = 0.0
            if val_2 is None:
                val_2 = 0.0
            
            if val_1 == 0.0 and val_2 == 0.0:
                similarities[f'{metric}_difference'] = 0.0
                similarities[f'{metric}_similarity'] = 1.0
            else:
                # Calculate relative difference
                max_val = max(val_1, val_2)
                difference = abs(val_1 - val_2)
                similarities[f'{metric}_difference'] = difference
                similarities[f'{metric}_similarity'] = 1 - (difference / max_val) if max_val > 0 else 1.0
        
        # Calculate overall diversity similarity
        metric_similarities = [similarities[f'{metric}_similarity'] for metric in diversity_metrics]
        similarities['overall_diversity_similarity'] = np.mean(metric_similarities)
        
        return similarities
    
    def calculate_environmental_similarity(self, 
                                         env_1: Dict[str, Any], 
                                         env_2: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate environmental context similarity.
        
        Args:
            env_1: Environmental context for first report
            env_2: Environmental context for second report
            
        Returns:
            Dictionary with environmental similarity metrics
        """
        similarities = {}
        
        # Location similarity (if coordinates available)
        if env_1.get('latitude') and env_1.get('longitude') and env_2.get('latitude') and env_2.get('longitude'):
            distance_km = self._haversine_distance(
                env_1['latitude'], env_1['longitude'],
                env_2['latitude'], env_2['longitude']
            )
            similarities['geographic_distance_km'] = distance_km
            # Similarity decreases with distance (using exponential decay)
            similarities['geographic_similarity'] = np.exp(-distance_km / 100)  # 100km scale
        else:
            similarities['geographic_distance_km'] = None
            similarities['geographic_similarity'] = 0.5  # Unknown
        
        # Depth similarity
        depth_1 = env_1.get('depth_meters')
        depth_2 = env_2.get('depth_meters')
        
        if depth_1 is not None and depth_2 is not None:
            depth_diff = abs(depth_1 - depth_2)
            similarities['depth_difference_m'] = depth_diff
            max_depth = max(depth_1, depth_2, 1)  # Avoid division by zero
            similarities['depth_similarity'] = 1 - (depth_diff / max_depth)
        else:
            similarities['depth_difference_m'] = None
            similarities['depth_similarity'] = 0.5  # Unknown
        
        # Temperature similarity
        temp_1 = env_1.get('temperature_celsius')
        temp_2 = env_2.get('temperature_celsius')
        if temp_1 is not None and temp_2 is not None:
            temp_diff = abs(temp_1 - temp_2)
            similarities['temperature_difference_c'] = temp_diff
            # Temperature similarity (assuming 20C range is very different)
            similarities['temperature_similarity'] = max(0, 1 - (temp_diff / 20))
        else:
            similarities['temperature_difference_c'] = None
            similarities['temperature_similarity'] = 0.5
        
        # Temporal similarity
        date_1 = env_1.get('collection_date')
        date_2 = env_2.get('collection_date')
        if date_1 and date_2:
            if isinstance(date_1, str):
                date_1 = datetime.fromisoformat(date_1)
            if isinstance(date_2, str):
                date_2 = datetime.fromisoformat(date_2)
            
            temporal_diff_days = abs((date_1 - date_2).days)
            similarities['temporal_difference_days'] = temporal_diff_days
            # Temporal similarity (assuming 365 days is very different)
            similarities['temporal_similarity'] = max(0, 1 - (temporal_diff_days / 365))
        else:
            similarities['temporal_difference_days'] = None
            similarities['temporal_similarity'] = 0.5
        
        return similarities
    
    def _bray_curtis_similarity(self, vector_1: np.ndarray, vector_2: np.ndarray) -> float:
        """Calculate Bray-Curtis similarity."""
        numerator = np.sum(np.minimum(vector_1, vector_2))
        denominator = np.sum(vector_1) + np.sum(vector_2)
        
        if denominator == 0:
            return 1.0 if np.array_equal(vector_1, vector_2) else 0.0
        
        return 2 * numerator / denominator
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in kilometers."""
        R = 6371  # Earth's radius in kilometers
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return 1 - cosine(vector1, vector2) if np.sum(vector1) > 0 and np.sum(vector2) > 0 else 0.0
    
    def _overlap_similarity(self, set1: set, set2: set) -> float:
        """Calculate overlap similarity (overlap coefficient)."""
        intersection = len(set1.intersection(set2))
        min_size = min(len(set1), len(set2))
        return intersection / min_size if min_size > 0 else 0.0
    
    def _abundance_correlation(self, abundance1: dict, abundance2: dict) -> float:
        """Calculate abundance correlation."""
        all_organisms = set(abundance1.keys()).union(set(abundance2.keys()))
        vector1 = np.array([abundance1.get(org, 0.0) for org in all_organisms])
        vector2 = np.array([abundance2.get(org, 0.0) for org in all_organisms])
        
        if len(vector1) > 1 and np.std(vector1) > 0 and np.std(vector2) > 0:
            corr, _ = pearsonr(vector1, vector2)
            return corr if not np.isnan(corr) else 0.0
        return 0.0
    
    def _taxonomic_similarity(self, taxonomy1: dict, taxonomy2: dict) -> float:
        """Calculate taxonomic similarity."""
        # This would implement sophisticated taxonomic comparison
        # For now, return a placeholder
        return 0.5


class CrossAnalysisEngine:
    """
    Main engine for cross-analysis similarity comparisons.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize cross-analysis engine.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.query_engine = ReportQueryEngine(self.db_manager)
        self.similarity_calculator = SimilarityCalculator()
    
    def compare_reports(self, report_id_1: str, report_id_2: str) -> Optional[SimilarityMatrix]:
        """
        Compare two analysis reports and calculate comprehensive similarity metrics.
        
        Args:
            report_id_1: First report ID
            report_id_2: Second report ID
            
        Returns:
            SimilarityMatrix instance with comparison results
        """
        logger.info(f"Comparing reports: {report_id_1} vs {report_id_2}")
        
        try:
            # Get reports from database
            report_1 = self.db_manager.get_analysis_report(report_id_1)
            report_2 = self.db_manager.get_analysis_report(report_id_2)
            
            if not report_1 or not report_2:
                logger.error("One or both reports not found")
                return None
            
            # Create basic comparison for now (simplified version)
            comparison_id = SimilarityMatrix.generate_comparison_id(report_id_1, report_id_2)
            
            # Calculate basic similarities with safe defaults
            shannon_1 = report_1.shannon_diversity or 0.0
            shannon_2 = report_2.shannon_diversity or 0.0
            shannon_diff = abs(shannon_1 - shannon_2)
            
            simpson_1 = report_1.simpson_diversity or 0.0
            simpson_2 = report_2.simpson_diversity or 0.0
            simpson_diff = abs(simpson_1 - simpson_2)
            
            evenness_1 = report_1.evenness or 0.0
            evenness_2 = report_2.evenness or 0.0
            evenness_diff = abs(evenness_1 - evenness_2)
            
            # Basic similarity calculation (can be improved later)
            diversity_similarity = 1.0 - (shannon_diff / max(shannon_1, shannon_2, 1.0))
            basic_similarity = max(0.0, min(1.0, diversity_similarity))
            
            # Get organism data for both reports (simplified)
            try:
                organisms_1 = self._get_report_organisms(report_id_1)
                organisms_2 = self._get_report_organisms(report_id_2)
                
                # Calculate organism overlap
                organisms_set_1 = set(organisms_1.keys()) if organisms_1 else set()
                organisms_set_2 = set(organisms_2.keys()) if organisms_2 else set()
                
                shared_organisms = len(organisms_set_1.intersection(organisms_set_2))
                total_organisms = len(organisms_set_1.union(organisms_set_2))
                
                jaccard_sim = shared_organisms / total_organisms if total_organisms > 0 else 0.0
                cosine_sim = jaccard_sim  # Simplified for now
                
            except Exception as e:
                logger.warning(f"Failed to get organism data: {e}")
                shared_organisms = 0
                jaccard_sim = 0.0
                cosine_sim = 0.0
            
            # Calculate overall similarity
            overall_similarity = (basic_similarity + jaccard_sim) / 2.0
            
            # Create similarity matrix with safe values
            similarity_matrix = SimilarityMatrix(
                comparison_id=comparison_id,
                report_id_1=report_id_1,
                report_id_2=report_id_2,
                organism_overlap_count=shared_organisms,
                organism_overlap_percentage=jaccard_sim * 100,
                jaccard_similarity=jaccard_sim,
                cosine_similarity=cosine_sim,
                kingdom_similarity=0.5,  # Placeholder
                phylum_similarity=0.5,   # Placeholder
                genus_similarity=0.5,    # Placeholder
                shannon_diversity_diff=shannon_diff,
                simpson_diversity_diff=simpson_diff,
                evenness_diff=evenness_diff,
                location_distance_km=None,
                depth_difference_m=None,
                temporal_difference_days=None,
                similarity_score=overall_similarity,
                comparison_method="simplified_cross_analysis"
            )
            
            # Store similarity matrix in database
            try:
                self.db_manager.store_similarity_matrix(similarity_matrix)
            except Exception as e:
                logger.warning(f"Failed to store similarity matrix: {e}")
            
            logger.info(f"Report comparison completed. Overall similarity: {overall_similarity:.3f}")
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Failed to compare reports: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def batch_compare_reports(self, report_ids: List[str]) -> List[SimilarityMatrix]:
        """
        Perform pairwise comparisons of multiple reports.
        
        Args:
            report_ids: List of report IDs to compare
            
        Returns:
            List of SimilarityMatrix instances
        """
        logger.info(f"Performing batch comparison of {len(report_ids)} reports")
        
        similarity_matrices = []
        
        # Perform pairwise comparisons
        for i in range(len(report_ids)):
            for j in range(i + 1, len(report_ids)):
                similarity_matrix = self.compare_reports(report_ids[i], report_ids[j])
                if similarity_matrix:
                    similarity_matrices.append(similarity_matrix)
        
        logger.info(f"Completed {len(similarity_matrices)} pairwise comparisons")
        return similarity_matrices
    
    def find_similar_reports(self, target_report_id: str, 
                           similarity_threshold: float = 0.7,
                           max_results: int = 10) -> List[Tuple[str, float]]:
        """
        Find reports similar to the target report.
        
        Args:
            target_report_id: ID of the target report
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results to return
            
        Returns:
            List of (report_id, similarity_score) tuples
        """
        return self.db_manager.find_similar_reports(
            target_report_id, similarity_threshold, max_results
        )
    
    def get_similarity_trends(self, time_period_days: int = 90) -> Dict[str, Any]:
        """
        Analyze similarity trends over time.
        
        Args:
            time_period_days: Time period to analyze in days
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            with self.db_manager.get_connection() as conn:
                cutoff_date = datetime.now() - timedelta(days=time_period_days)
                
                cursor = conn.execute("""
                    SELECT 
                        DATE(created_at) as comparison_date,
                        AVG(similarity_score) as avg_similarity,
                        COUNT(*) as comparison_count,
                        MIN(similarity_score) as min_similarity,
                        MAX(similarity_score) as max_similarity
                    FROM similarity_matrices 
                    WHERE created_at >= ?
                    GROUP BY DATE(created_at)
                    ORDER BY comparison_date
                """, (cutoff_date.isoformat(),))
                
                trends = []
                for row in cursor.fetchall():
                    trends.append({
                        'date': row[0],
                        'avg_similarity': row[1],
                        'comparison_count': row[2],
                        'min_similarity': row[3],
                        'max_similarity': row[4]
                    })
                
                # Calculate overall statistics
                if trends:
                    overall_avg = np.mean([t['avg_similarity'] for t in trends])
                    overall_std = np.std([t['avg_similarity'] for t in trends])
                else:
                    overall_avg = 0.0
                    overall_std = 0.0
                
                return {
                    'time_period_days': time_period_days,
                    'daily_trends': trends,
                    'overall_statistics': {
                        'average_similarity': overall_avg,
                        'similarity_std': overall_std,
                        'total_comparisons': sum(t['comparison_count'] for t in trends)
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get similarity trends: {str(e)}")
            return {}
    
    def _get_report_organisms(self, report_id: str) -> Dict[str, float]:
        """Get organism abundance data for a report."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT organism_id, COUNT(*) as abundance
                    FROM sequences 
                    WHERE report_id = ? AND organism_id IS NOT NULL
                    GROUP BY organism_id
                """, (report_id,))
                
                organisms = {}
                total_sequences = 0
                
                for row in cursor.fetchall():
                    organisms[row[0]] = row[1]
                    total_sequences += row[1]
                
                # Convert to relative abundance
                if total_sequences > 0:
                    organisms = {org_id: count / total_sequences for org_id, count in organisms.items()}
                
                return organisms
                
        except Exception as e:
            logger.error(f"Failed to get organisms for report {report_id}: {str(e)}")
            return {}
    
    def _get_report_taxonomy(self, report_id: str) -> Dict[str, Dict[str, int]]:
        """Get taxonomic composition for a report."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT kingdom, phylum, class, order_name, family, genus, species, COUNT(*) as count
                    FROM taxonomic_assignments ta
                    JOIN sequences s ON ta.sequence_id = s.sequence_id
                    WHERE s.report_id = ?
                    GROUP BY kingdom, phylum, class, order_name, family, genus, species
                """, (report_id,))
                
                taxonomy = defaultdict(lambda: defaultdict(int))
                
                levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
                
                for row in cursor.fetchall():
                    for i, level in enumerate(levels):
                        if row[i]:  # If this taxonomic level is not null
                            taxonomy[level][row[i]] += row[-1]  # Last column is count
                
                return dict(taxonomy)
                
        except Exception as e:
            logger.error(f"Failed to get taxonomy for report {report_id}: {str(e)}")
            return {}
    
    def _get_report_environmental_context(self, report_id: str) -> Dict[str, Any]:
        """Get environmental context for a report."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        ds.collection_date, ds.collection_location, ds.depth_meters,
                        ds.temperature_celsius, ds.ph_level, ds.salinity
                    FROM analysis_reports ar
                    JOIN datasets ds ON ar.dataset_id = ds.dataset_id
                    WHERE ar.report_id = ?
                """, (report_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'collection_date': row[0],
                        'collection_location': row[1],
                        'depth_meters': row[2],
                        'temperature_celsius': row[3],
                        'ph_level': row[4],
                        'salinity': row[5]
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get environmental context for report {report_id}: {str(e)}")
            return {}
    
    def _calculate_overall_similarity(self, 
                                    organism_sim: Dict[str, float],
                                    abundance_sim: Dict[str, float],
                                    taxonomic_sim: Dict[str, float],
                                    diversity_sim: Dict[str, float],
                                    environmental_sim: Dict[str, float]) -> float:
        """Calculate weighted overall similarity score."""
        
        # Define weights for different similarity components
        weights = {
            'organism_overlap': 0.25,
            'abundance_correlation': 0.25,
            'taxonomic_similarity': 0.20,
            'diversity_similarity': 0.15,
            'environmental_similarity': 0.15
        }
        
        # Extract key similarity scores
        scores = {
            'organism_overlap': organism_sim.get('jaccard_similarity', 0.0),
            'abundance_correlation': abundance_sim.get('cosine_similarity', 0.0),
            'taxonomic_similarity': taxonomic_sim.get('overall_taxonomic_similarity', 0.0),
            'diversity_similarity': diversity_sim.get('overall_diversity_similarity', 0.0),
            'environmental_similarity': np.mean([
                environmental_sim.get('geographic_similarity', 0.5),
                environmental_sim.get('depth_similarity', 0.5),
                environmental_sim.get('temporal_similarity', 0.5)
            ])
        }
        
        # Calculate weighted average
        weighted_sum = sum(scores[component] * weights[component] for component in weights.keys())
        
        return weighted_sum