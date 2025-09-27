"""
Query engine for eDNA analysis report management system.

This module provides high-level query operations for complex data retrieval,
analysis, and reporting across stored analysis results.
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from dataclasses import asdict

from .manager import DatabaseManager
from .models import OrganismProfile, AnalysisReport, SimilarityMatrix

logger = logging.getLogger(__name__)


class ReportQueryEngine:
    """
    Advanced query engine for analysis report management and cross-analysis.
    
    Provides sophisticated querying capabilities for organism profiling,
    similarity analysis, and trend detection.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize query engine.
        
        Args:
            db_manager: DatabaseManager instance. If None, creates new one.
        """
        self.db_manager = db_manager or DatabaseManager()
    
    def search_organisms(self, 
                        query: str = "",
                        kingdom: Optional[str] = None,
                        phylum: Optional[str] = None,
                        genus: Optional[str] = None,
                        is_novel: Optional[bool] = None,
                        min_confidence: Optional[float] = None,
                        limit: int = 100) -> List[OrganismProfile]:
        """
        Search organisms with flexible filtering criteria.
        
        Args:
            query: Text query for organism name or taxonomy
            kingdom: Filter by kingdom
            phylum: Filter by phylum  
            genus: Filter by genus
            is_novel: Filter by novelty status
            min_confidence: Minimum confidence score
            limit: Maximum results to return
            
        Returns:
            List of matching OrganismProfile instances
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Build dynamic query
                where_clauses = []
                params = []
                
                if query:
                    where_clauses.append("""
                        (organism_name LIKE ? OR taxonomic_lineage LIKE ? 
                         OR genus LIKE ? OR species LIKE ?)
                    """)
                    query_param = f"%{query}%"
                    params.extend([query_param, query_param, query_param, query_param])
                
                if kingdom:
                    where_clauses.append("kingdom = ?")
                    params.append(kingdom)
                
                if phylum:
                    where_clauses.append("phylum = ?")
                    params.append(phylum)
                
                if genus:
                    where_clauses.append("genus = ?")
                    params.append(genus)
                
                if is_novel is not None:
                    where_clauses.append("is_novel_candidate = ?")
                    params.append(is_novel)
                
                if min_confidence is not None:
                    where_clauses.append("confidence_score >= ?")
                    params.append(min_confidence)
                
                # Construct final query
                base_query = "SELECT * FROM organism_profiles"
                if where_clauses:
                    base_query += " WHERE " + " AND ".join(where_clauses)
                base_query += " ORDER BY detection_count DESC, confidence_score DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(base_query, params)
                
                organisms = []
                for row in cursor.fetchall():
                    organism = self.db_manager._row_to_organism_profile(row, cursor.description)
                    if organism:
                        organisms.append(organism)
                
                return organisms
                
        except Exception as e:
            logger.error(f"Failed to search organisms: {str(e)}")
            return []
    
    def get_organism_timeline(self, organism_id: str) -> Dict[str, Any]:
        """
        Get comprehensive timeline for an organism across all analyses.
        
        Args:
            organism_id: Organism identifier
            
        Returns:
            Dictionary with timeline data and trends
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Get detection timeline
                cursor = conn.execute("""
                    SELECT 
                        ar.created_at as analysis_date,
                        ar.report_id,
                        ar.report_name,
                        ds.dataset_name,
                        ds.collection_date,
                        ds.collection_location,
                        ds.depth_meters,
                        ta.confidence_score,
                        nd.novelty_score,
                        nd.is_novel_candidate,
                        COUNT(s.sequence_id) as sequence_count
                    FROM analysis_reports ar
                    JOIN datasets ds ON ar.dataset_id = ds.dataset_id
                    JOIN sequences s ON ar.report_id = s.report_id AND s.organism_id = ?
                    LEFT JOIN taxonomic_assignments ta ON s.sequence_id = ta.sequence_id
                    LEFT JOIN novelty_detections nd ON s.sequence_id = nd.sequence_id
                    GROUP BY ar.report_id
                    ORDER BY ar.created_at
                """, (organism_id,))
                
                detections = []
                confidence_trends = []
                novelty_trends = []
                location_detections = defaultdict(int)
                depth_detections = []
                
                for row in cursor.fetchall():
                    detection = {
                        'analysis_date': row[0],
                        'report_id': row[1],
                        'report_name': row[2],
                        'dataset_name': row[3],
                        'collection_date': row[4],
                        'collection_location': row[5],
                        'depth_meters': row[6],
                        'confidence_score': row[7],
                        'novelty_score': row[8],
                        'is_novel_candidate': bool(row[9]) if row[9] is not None else None,
                        'sequence_count': row[10]
                    }
                    detections.append(detection)
                    
                    # Collect trends
                    if row[7] is not None:  # confidence_score
                        confidence_trends.append((row[0], row[7]))
                    if row[8] is not None:  # novelty_score
                        novelty_trends.append((row[0], row[8]))
                    if row[5]:  # collection_location
                        location_detections[row[5]] += 1
                    if row[6] is not None:  # depth_meters
                        depth_detections.append(row[6])
                
                # Calculate trends
                confidence_trend = self._calculate_trend(confidence_trends)
                novelty_trend = self._calculate_trend(novelty_trends)
                
                # Geographic distribution
                most_common_location = max(location_detections.items(), 
                                         key=lambda x: x[1])[0] if location_detections else None
                
                # Depth analysis
                depth_stats = {}
                if depth_detections:
                    depth_stats = {
                        'min_depth': min(depth_detections),
                        'max_depth': max(depth_detections),
                        'mean_depth': np.mean(depth_detections),
                        'depth_range': max(depth_detections) - min(depth_detections)
                    }
                
                return {
                    'organism_id': organism_id,
                    'detection_count': len(detections),
                    'first_detection': detections[0]['analysis_date'] if detections else None,
                    'last_detection': detections[-1]['analysis_date'] if detections else None,
                    'detections': detections,
                    'trends': {
                        'confidence_trend': confidence_trend,
                        'novelty_trend': novelty_trend
                    },
                    'geographic_distribution': dict(location_detections),
                    'most_common_location': most_common_location,
                    'depth_distribution': depth_stats
                }
                
        except Exception as e:
            logger.error(f"Failed to get organism timeline: {str(e)}")
            return {}
    
    def find_co_occurring_organisms(self, organism_id: str, 
                                  min_co_occurrence: int = 2) -> List[Dict[str, Any]]:
        """
        Find organisms that frequently co-occur with the given organism.
        
        Args:
            organism_id: Reference organism ID
            min_co_occurrence: Minimum number of co-occurrences
            
        Returns:
            List of co-occurring organisms with statistics
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    WITH organism_reports AS (
                        SELECT DISTINCT report_id 
                        FROM sequences 
                        WHERE organism_id = ?
                    ),
                    co_occurring AS (
                        SELECT 
                            s.organism_id,
                            COUNT(*) as co_occurrence_count,
                            op.organism_name,
                            op.genus,
                            op.species,
                            op.kingdom
                        FROM sequences s
                        JOIN organism_reports or_ref ON s.report_id = or_ref.report_id
                        JOIN organism_profiles op ON s.organism_id = op.organism_id
                        WHERE s.organism_id != ?
                        GROUP BY s.organism_id
                        HAVING COUNT(*) >= ?
                    )
                    SELECT * FROM co_occurring
                    ORDER BY co_occurrence_count DESC
                """, (organism_id, organism_id, min_co_occurrence))
                
                co_occurring = []
                for row in cursor.fetchall():
                    co_occurring.append({
                        'organism_id': row[0],
                        'co_occurrence_count': row[1],
                        'organism_name': row[2],
                        'genus': row[3],
                        'species': row[4],
                        'kingdom': row[5]
                    })
                
                return co_occurring
                
        except Exception as e:
            logger.error(f"Failed to find co-occurring organisms: {str(e)}")
            return []
    
    def get_novelty_trends(self, time_period_days: int = 30) -> Dict[str, Any]:
        """
        Analyze novelty detection trends over time.
        
        Args:
            time_period_days: Time period to analyze
            
        Returns:
            Dictionary with novelty trends and statistics
        """
        try:
            with self.db_manager.get_connection() as conn:
                cutoff_date = datetime.now() - timedelta(days=time_period_days)
                
                # Novel candidates over time
                cursor = conn.execute("""
                    SELECT 
                        DATE(ar.created_at) as analysis_date,
                        COUNT(DISTINCT nd.organism_id) as novel_candidates,
                        COUNT(DISTINCT s.organism_id) as total_organisms,
                        AVG(nd.novelty_score) as avg_novelty_score
                    FROM analysis_reports ar
                    JOIN sequences s ON ar.report_id = s.report_id
                    LEFT JOIN novelty_detections nd ON s.sequence_id = nd.sequence_id 
                        AND nd.is_novel_candidate = 1
                    WHERE ar.created_at >= ?
                    GROUP BY DATE(ar.created_at)
                    ORDER BY analysis_date
                """, (cutoff_date.isoformat(),))
                
                daily_trends = []
                for row in cursor.fetchall():
                    daily_trends.append({
                        'date': row[0],
                        'novel_candidates': row[1],
                        'total_organisms': row[2],
                        'novelty_percentage': (row[1] / row[2] * 100) if row[2] > 0 else 0,
                        'avg_novelty_score': row[3]
                    })
                
                # Novel validation status
                cursor = conn.execute("""
                    SELECT 
                        nd.validation_status,
                        COUNT(*) as count
                    FROM novelty_detections nd
                    JOIN sequences s ON nd.sequence_id = s.sequence_id
                    JOIN analysis_reports ar ON s.report_id = ar.report_id
                    WHERE nd.is_novel_candidate = 1 AND ar.created_at >= ?
                    GROUP BY nd.validation_status
                """, (cutoff_date.isoformat(),))
                
                validation_status = dict(cursor.fetchall())
                
                # Top novel organisms
                cursor = conn.execute("""
                    SELECT 
                        op.organism_id,
                        op.organism_name,
                        op.genus,
                        op.species,
                        AVG(nd.novelty_score) as avg_novelty_score,
                        COUNT(DISTINCT ar.report_id) as detection_count
                    FROM organism_profiles op
                    JOIN sequences s ON op.organism_id = s.organism_id
                    JOIN novelty_detections nd ON s.sequence_id = nd.sequence_id
                    JOIN analysis_reports ar ON s.report_id = ar.report_id
                    WHERE nd.is_novel_candidate = 1 AND ar.created_at >= ?
                    GROUP BY op.organism_id
                    ORDER BY avg_novelty_score DESC, detection_count DESC
                    LIMIT 10
                """, (cutoff_date.isoformat(),))
                
                top_novel = []
                for row in cursor.fetchall():
                    top_novel.append({
                        'organism_id': row[0],
                        'organism_name': row[1],
                        'genus': row[2],
                        'species': row[3],
                        'avg_novelty_score': row[4],
                        'detection_count': row[5]
                    })
                
                return {
                    'time_period_days': time_period_days,
                    'daily_trends': daily_trends,
                    'validation_status': validation_status,
                    'top_novel_organisms': top_novel,
                    'summary': {
                        'total_novel_candidates': sum(validation_status.values()),
                        'avg_daily_novel_rate': np.mean([t['novelty_percentage'] for t in daily_trends]) if daily_trends else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get novelty trends: {str(e)}")
            return {}
    
    def get_taxonomic_diversity_analysis(self, report_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive taxonomic diversity analysis.
        
        Args:
            report_ids: Optional list of report IDs to analyze. If None, analyzes all reports.
            
        Returns:
            Dictionary with taxonomic diversity metrics and visualizations
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Build query with optional report filtering
                where_clause = ""
                params = []
                if report_ids:
                    placeholders = ",".join(["?" for _ in report_ids])
                    where_clause = f"WHERE ar.report_id IN ({placeholders})"
                    params.extend(report_ids)
                
                # Taxonomic distribution by level
                cursor = conn.execute(f"""
                    SELECT 
                        ta.kingdom,
                        ta.phylum, 
                        ta.class,
                        ta.order_name,
                        ta.family,
                        ta.genus,
                        ta.species,
                        COUNT(DISTINCT s.organism_id) as organism_count,
                        COUNT(s.sequence_id) as sequence_count,
                        AVG(ta.confidence_score) as avg_confidence
                    FROM analysis_reports ar
                    JOIN sequences s ON ar.report_id = s.report_id
                    LEFT JOIN taxonomic_assignments ta ON s.sequence_id = ta.sequence_id
                    {where_clause}
                    GROUP BY ta.kingdom, ta.phylum, ta.class, ta.order_name, ta.family, ta.genus, ta.species
                    ORDER BY organism_count DESC
                """, params)
                
                taxonomic_distribution = []
                kingdom_counts = defaultdict(int)
                phylum_counts = defaultdict(int)
                genus_counts = defaultdict(int)
                
                for row in cursor.fetchall():
                    entry = {
                        'kingdom': row[0],
                        'phylum': row[1],
                        'class': row[2],
                        'order': row[3],
                        'family': row[4],
                        'genus': row[5],
                        'species': row[6],
                        'organism_count': row[7],
                        'sequence_count': row[8],
                        'avg_confidence': row[9]
                    }
                    taxonomic_distribution.append(entry)
                    
                    # Aggregate counts
                    if row[0]:  # kingdom
                        kingdom_counts[row[0]] += row[7]
                    if row[1]:  # phylum
                        phylum_counts[row[1]] += row[7]
                    if row[5]:  # genus
                        genus_counts[row[5]] += row[7]
                
                # Calculate diversity indices at different taxonomic levels
                diversity_indices = {
                    'kingdom': self._calculate_diversity_indices(dict(kingdom_counts)),
                    'phylum': self._calculate_diversity_indices(dict(phylum_counts)),
                    'genus': self._calculate_diversity_indices(dict(genus_counts))
                }
                
                # Unknown/unassigned analysis
                cursor = conn.execute(f"""
                    SELECT 
                        COUNT(CASE WHEN ta.kingdom IS NULL THEN 1 END) as no_kingdom,
                        COUNT(CASE WHEN ta.phylum IS NULL THEN 1 END) as no_phylum,
                        COUNT(CASE WHEN ta.genus IS NULL THEN 1 END) as no_genus,
                        COUNT(CASE WHEN ta.species IS NULL THEN 1 END) as no_species,
                        COUNT(*) as total_sequences
                    FROM analysis_reports ar
                    JOIN sequences s ON ar.report_id = s.report_id
                    LEFT JOIN taxonomic_assignments ta ON s.sequence_id = ta.sequence_id
                    {where_clause}
                """, params)
                
                row = cursor.fetchone()
                unknown_stats = {
                    'no_kingdom': row[0],
                    'no_phylum': row[1], 
                    'no_genus': row[2],
                    'no_species': row[3],
                    'total_sequences': row[4],
                    'unknown_kingdom_pct': (row[0] / row[4] * 100) if row[4] > 0 else 0,
                    'unknown_genus_pct': (row[2] / row[4] * 100) if row[4] > 0 else 0
                }
                
                return {
                    'taxonomic_distribution': taxonomic_distribution,
                    'diversity_indices': diversity_indices,
                    'kingdom_summary': dict(kingdom_counts),
                    'phylum_summary': dict(phylum_counts),
                    'genus_summary': dict(genus_counts),
                    'unknown_assignments': unknown_stats,
                    'total_unique_organisms': len(taxonomic_distribution)
                }
                
        except Exception as e:
            logger.error(f"Failed to get taxonomic diversity analysis: {str(e)}")
            return {}
    
    def compare_report_pairs(self, report_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Perform detailed comparison analysis for multiple report pairs.
        
        Args:
            report_pairs: List of (report_id_1, report_id_2) tuples
            
        Returns:
            List of comparison results for each pair
        """
        comparisons = []
        
        for report_id_1, report_id_2 in report_pairs:
            comparison = self._compare_two_reports(report_id_1, report_id_2)
            if comparison:
                comparisons.append(comparison)
        
        return comparisons
    
    def _compare_two_reports(self, report_id_1: str, report_id_2: str) -> Optional[Dict[str, Any]]:
        """Compare two specific reports in detail."""
        try:
            with self.db_manager.get_connection() as conn:
                # Get basic report info
                cursor = conn.execute("""
                    SELECT report_id, dataset_id, report_name, created_at,
                           shannon_diversity, simpson_diversity, species_richness
                    FROM analysis_reports 
                    WHERE report_id IN (?, ?)
                """, (report_id_1, report_id_2))
                
                reports = {row[0]: dict(zip([
                    'report_id', 'dataset_id', 'report_name', 'created_at',
                    'shannon_diversity', 'simpson_diversity', 'species_richness'
                ], row)) for row in cursor.fetchall()}
                
                if len(reports) != 2:
                    return None
                
                # Get organism overlap
                cursor = conn.execute("""
                    SELECT 
                        s1.organism_id,
                        COUNT(s1.sequence_id) as count_report_1,
                        COUNT(s2.sequence_id) as count_report_2
                    FROM sequences s1
                    FULL OUTER JOIN sequences s2 ON s1.organism_id = s2.organism_id AND s2.report_id = ?
                    WHERE s1.report_id = ?
                    GROUP BY s1.organism_id
                    
                    UNION
                    
                    SELECT 
                        s2.organism_id,
                        COUNT(s1.sequence_id) as count_report_1,
                        COUNT(s2.sequence_id) as count_report_2
                    FROM sequences s2
                    LEFT JOIN sequences s1 ON s1.organism_id = s2.organism_id AND s1.report_id = ?
                    WHERE s2.report_id = ? AND s1.organism_id IS NULL
                    GROUP BY s2.organism_id
                """, (report_id_2, report_id_1, report_id_1, report_id_2))
                
                organism_data = []
                shared_organisms = 0
                total_organisms_1 = 0
                total_organisms_2 = 0
                
                for row in cursor.fetchall():
                    organism_id, count_1, count_2 = row
                    count_1 = count_1 or 0
                    count_2 = count_2 or 0
                    
                    organism_data.append({
                        'organism_id': organism_id,
                        'count_report_1': count_1,
                        'count_report_2': count_2,
                        'in_both': count_1 > 0 and count_2 > 0
                    })
                    
                    if count_1 > 0:
                        total_organisms_1 += 1
                    if count_2 > 0:
                        total_organisms_2 += 1
                    if count_1 > 0 and count_2 > 0:
                        shared_organisms += 1
                
                # Calculate similarity metrics
                jaccard_similarity = shared_organisms / (total_organisms_1 + total_organisms_2 - shared_organisms) if (total_organisms_1 + total_organisms_2 - shared_organisms) > 0 else 0
                overlap_percentage = (shared_organisms / max(total_organisms_1, total_organisms_2) * 100) if max(total_organisms_1, total_organisms_2) > 0 else 0
                
                # Diversity differences
                rep1 = reports[report_id_1]
                rep2 = reports[report_id_2]
                
                diversity_diff = {
                    'shannon_diff': abs((rep1['shannon_diversity'] or 0) - (rep2['shannon_diversity'] or 0)),
                    'simpson_diff': abs((rep1['simpson_diversity'] or 0) - (rep2['simpson_diversity'] or 0)),
                    'richness_diff': abs((rep1['species_richness'] or 0) - (rep2['species_richness'] or 0))
                }
                
                return {
                    'report_id_1': report_id_1,
                    'report_id_2': report_id_2,
                    'report_info': reports,
                    'organism_overlap': {
                        'shared_organisms': shared_organisms,
                        'total_organisms_1': total_organisms_1,
                        'total_organisms_2': total_organisms_2,
                        'jaccard_similarity': jaccard_similarity,
                        'overlap_percentage': overlap_percentage
                    },
                    'diversity_differences': diversity_diff,
                    'organism_details': organism_data
                }
                
        except Exception as e:
            logger.error(f"Failed to compare reports {report_id_1} and {report_id_2}: {str(e)}")
            return None
    
    def _calculate_trend(self, time_value_pairs: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Calculate trend statistics for time series data."""
        if len(time_value_pairs) < 2:
            return {'trend': 'insufficient_data', 'slope': 0, 'r_squared': 0}
        
        try:
            # Convert to numeric arrays
            times = [datetime.fromisoformat(t).timestamp() for t, v in time_value_pairs]
            values = [v for t, v in time_value_pairs]
            
            # Linear regression
            x = np.array(times)
            y = np.array(values)
            
            # Normalize x for numerical stability
            x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x
            
            # Calculate slope and correlation
            slope, intercept = np.polyfit(x_norm, y, 1)
            correlation = np.corrcoef(x_norm, y)[0, 1] if len(x_norm) > 1 else 0
            r_squared = correlation ** 2
            
            # Determine trend direction
            if abs(slope) < 0.01:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            return {
                'trend': trend,
                'slope': float(slope),
                'r_squared': float(r_squared),
                'correlation': float(correlation)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate trend: {str(e)}")
            return {'trend': 'error', 'slope': 0, 'r_squared': 0}
    
    def _calculate_diversity_indices(self, abundance_dict: Dict[str, int]) -> Dict[str, float]:
        """Calculate Shannon and Simpson diversity indices."""
        if not abundance_dict:
            return {'shannon': 0.0, 'simpson': 0.0, 'evenness': 0.0}
        
        total = sum(abundance_dict.values())
        proportions = [count / total for count in abundance_dict.values()]
        
        # Shannon diversity
        shannon = -sum(p * np.log(p) for p in proportions if p > 0)
        
        # Simpson diversity
        simpson = 1 - sum(p ** 2 for p in proportions)
        
        # Evenness
        num_species = len(abundance_dict)
        evenness = shannon / np.log(num_species) if num_species > 1 else 0
        
        return {
            'shannon': float(shannon),
            'simpson': float(simpson),
            'evenness': float(evenness),
            'richness': num_species
        }