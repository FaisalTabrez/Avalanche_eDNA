"""
Database manager for eDNA analysis report management system.

This module provides high-level database operations and transaction management
for storing and retrieving analysis data.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import json
from datetime import datetime

from .schema import DatabaseSchema
from .models import (
    OrganismProfile,
    DatasetInfo,
    AnalysisReport,
    SimilarityMatrix,
    ReportComparison,
    AnalysisStatus,
    SequenceType,
    serialize_to_json,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    High-level database manager for eDNA analysis reports.

    Provides CRUD operations and transaction management for all database entities.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.schema = DatabaseSchema(db_path)
        self.db_path = str(self.schema.db_path)

        # Initialize database if it doesn't exist
        if not self.schema.database_exists():
            logger.info("Creating new database")
            self.schema.create_database()
        else:
            logger.info("Using existing database")
            self.schema.migrate_database()

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections with transaction handling.

        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def store_organism_profile(self, profile: OrganismProfile) -> bool:
        """
        Store or update organism profile.

        Args:
            profile: OrganismProfile instance to store

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                # Check if organism already exists
                cursor = conn.execute(
                    """
                    SELECT organism_id FROM organism_profiles WHERE organism_id = ?
                """,
                    (profile.organism_id,),
                )

                if cursor.fetchone():
                    # Update existing organism
                    self._update_organism_profile(conn, profile)
                    logger.info(f"Updated organism profile: {profile.organism_id}")
                else:
                    # Insert new organism
                    self._insert_organism_profile(conn, profile)
                    logger.info(f"Created new organism profile: {profile.organism_id}")

                return True

        except Exception as e:
            logger.error(f"Failed to store organism profile: {str(e)}")
            return False

    def _insert_organism_profile(
        self, conn: sqlite3.Connection, profile: OrganismProfile
    ):
        """Insert new organism profile."""
        conn.execute(
            """
            INSERT INTO organism_profiles (
                organism_id, organism_name, taxonomic_lineage, kingdom, phylum, 
                class, order_name, family, genus, species, sequence_signature,
                first_detected, last_updated, detection_count, confidence_score,
                is_novel_candidate, novelty_score, reference_databases, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                profile.organism_id,
                profile.organism_name,
                profile.taxonomic_lineage,
                profile.kingdom,
                profile.phylum,
                profile.class_name,
                profile.order_name,
                profile.family,
                profile.genus,
                profile.species,
                profile.sequence_signature,
                profile.first_detected,
                profile.last_updated,
                profile.detection_count,
                profile.confidence_score,
                profile.is_novel_candidate,
                profile.novelty_score,
                json.dumps(profile.reference_databases)
                if profile.reference_databases
                else None,
                profile.notes,
            ),
        )

    def _update_organism_profile(
        self, conn: sqlite3.Connection, profile: OrganismProfile
    ):
        """Update existing organism profile."""
        conn.execute(
            """
            UPDATE organism_profiles SET
                organism_name = ?, taxonomic_lineage = ?, kingdom = ?, phylum = ?,
                class = ?, order_name = ?, family = ?, genus = ?, species = ?,
                sequence_signature = ?, last_updated = ?, detection_count = detection_count + 1,
                confidence_score = ?, is_novel_candidate = ?, novelty_score = ?,
                reference_databases = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
            WHERE organism_id = ?
        """,
            (
                profile.organism_name,
                profile.taxonomic_lineage,
                profile.kingdom,
                profile.phylum,
                profile.class_name,
                profile.order_name,
                profile.family,
                profile.genus,
                profile.species,
                profile.sequence_signature,
                profile.last_updated,
                profile.confidence_score,
                profile.is_novel_candidate,
                profile.novelty_score,
                json.dumps(profile.reference_databases)
                if profile.reference_databases
                else None,
                profile.notes,
                profile.organism_id,
            ),
        )

    def get_organism_profile(self, organism_id: str) -> Optional[OrganismProfile]:
        """
        Retrieve organism profile by ID.

        Args:
            organism_id: Unique organism identifier

        Returns:
            OrganismProfile instance or None if not found
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM organism_profiles WHERE organism_id = ?
                """,
                    (organism_id,),
                )

                row = cursor.fetchone()
                if row:
                    return self._row_to_organism_profile(row, cursor.description)
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve organism profile: {str(e)}")
            return None

    def store_dataset_info(self, dataset: DatasetInfo) -> bool:
        """
        Store dataset information.

        Args:
            dataset: DatasetInfo instance to store

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO datasets (
                        dataset_id, dataset_name, file_path, file_format, file_size_mb,
                        total_sequences, sequence_type, collection_date, collection_location,
                        depth_meters, temperature_celsius, ph_level, salinity,
                        environmental_conditions, preprocessing_params
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        dataset.dataset_id,
                        dataset.dataset_name,
                        dataset.file_path,
                        dataset.file_format,
                        dataset.file_size_mb,
                        dataset.total_sequences,
                        dataset.sequence_type.value if dataset.sequence_type else None,
                        dataset.collection_date,
                        dataset.collection_location,
                        dataset.depth_meters,
                        dataset.temperature_celsius,
                        dataset.ph_level,
                        dataset.salinity,
                        json.dumps(dataset.environmental_conditions)
                        if dataset.environmental_conditions
                        else None,
                        json.dumps(dataset.preprocessing_params)
                        if dataset.preprocessing_params
                        else None,
                    ),
                )

                logger.info(f"Stored dataset info: {dataset.dataset_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to store dataset info: {str(e)}")
            return False

    def store_analysis_report(self, report: AnalysisReport) -> bool:
        """
        Store analysis report.

        Args:
            report: AnalysisReport instance to store

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO analysis_reports (
                        report_id, dataset_id, report_name, analysis_type, status,
                        processing_time_seconds, min_length, max_length, mean_length,
                        median_length, std_length, sequence_type_detected, composition_data,
                        shannon_diversity, simpson_diversity, evenness, species_richness,
                        n_clusters, silhouette_score, cluster_coherence,
                        sequences_with_taxonomy, taxonomy_confidence_avg,
                        novel_candidates_count, novel_percentage, novelty_threshold,
                        full_report_path, results_json_path, visualizations_dir
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        report.report_id,
                        report.dataset_id,
                        report.report_name,
                        report.analysis_type,
                        report.status.value,
                        report.processing_time_seconds,
                        report.min_length,
                        report.max_length,
                        report.mean_length,
                        report.median_length,
                        report.std_length,
                        report.sequence_type_detected.value
                        if report.sequence_type_detected
                        else None,
                        json.dumps(report.composition_data)
                        if report.composition_data
                        else None,
                        report.shannon_diversity,
                        report.simpson_diversity,
                        report.evenness,
                        report.species_richness,
                        report.n_clusters,
                        report.silhouette_score,
                        report.cluster_coherence,
                        report.sequences_with_taxonomy,
                        report.taxonomy_confidence_avg,
                        report.novel_candidates_count,
                        report.novel_percentage,
                        report.novelty_threshold,
                        report.full_report_path,
                        report.results_json_path,
                        report.visualizations_dir,
                    ),
                )

                logger.info(f"Stored analysis report: {report.report_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to store analysis report: {str(e)}")
            return False

    def get_analysis_report(self, report_id: str) -> Optional[AnalysisReport]:
        """
        Retrieve analysis report by ID.

        Args:
            report_id: Unique report identifier

        Returns:
            AnalysisReport instance or None if not found
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM analysis_reports WHERE report_id = ?
                """,
                    (report_id,),
                )

                row = cursor.fetchone()
                if row:
                    return self._row_to_analysis_report(row, cursor.description)
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve analysis report: {str(e)}")
            return None

    def list_analysis_reports(
        self, dataset_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[AnalysisReport]:
        """
        List analysis reports with optional filtering.

        Args:
            dataset_id: Optional dataset ID filter
            limit: Maximum number of reports to return
            offset: Number of reports to skip

        Returns:
            List of AnalysisReport instances
        """
        try:
            with self.get_connection() as conn:
                if dataset_id:
                    cursor = conn.execute(
                        """
                        SELECT * FROM analysis_reports 
                        WHERE dataset_id = ? 
                        ORDER BY created_at DESC 
                        LIMIT ? OFFSET ?
                    """,
                        (dataset_id, limit, offset),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM analysis_reports 
                        ORDER BY created_at DESC 
                        LIMIT ? OFFSET ?
                    """,
                        (limit, offset),
                    )

                reports = []
                for row in cursor.fetchall():
                    report = self._row_to_analysis_report(row, cursor.description)
                    if report:
                        reports.append(report)

                return reports

        except Exception as e:
            logger.error(f"Failed to list analysis reports: {str(e)}")
            return []

    def store_similarity_matrix(self, similarity: SimilarityMatrix) -> bool:
        """
        Store similarity matrix for report comparison.

        Args:
            similarity: SimilarityMatrix instance to store

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO similarity_matrices (
                        comparison_id, report_id_1, report_id_2, organism_overlap_count,
                        organism_overlap_percentage, jaccard_similarity, cosine_similarity,
                        kingdom_similarity, phylum_similarity, genus_similarity,
                        shannon_diversity_diff, simpson_diversity_diff, evenness_diff,
                        cluster_structure_similarity, location_distance_km,
                        depth_difference_m, temporal_difference_days, similarity_score,
                        comparison_method
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        similarity.comparison_id,
                        similarity.report_id_1,
                        similarity.report_id_2,
                        similarity.organism_overlap_count,
                        similarity.organism_overlap_percentage,
                        similarity.jaccard_similarity,
                        similarity.cosine_similarity,
                        similarity.kingdom_similarity,
                        similarity.phylum_similarity,
                        similarity.genus_similarity,
                        similarity.shannon_diversity_diff,
                        similarity.simpson_diversity_diff,
                        similarity.evenness_diff,
                        similarity.cluster_structure_similarity,
                        similarity.location_distance_km,
                        similarity.depth_difference_m,
                        similarity.temporal_difference_days,
                        similarity.similarity_score,
                        similarity.comparison_method,
                    ),
                )

                logger.info(f"Stored similarity matrix: {similarity.comparison_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to store similarity matrix: {str(e)}")
            return False

    def find_similar_reports(
        self, report_id: str, similarity_threshold: float = 0.7, limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find reports similar to the given report.

        Args:
            report_id: Reference report ID
            similarity_threshold: Minimum similarity score
            limit: Maximum number of similar reports to return

        Returns:
            List of tuples (report_id, similarity_score)
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT 
                        CASE 
                            WHEN report_id_1 = ? THEN report_id_2 
                            ELSE report_id_1 
                        END as other_report_id,
                        similarity_score
                    FROM similarity_matrices 
                    WHERE (report_id_1 = ? OR report_id_2 = ?)
                        AND similarity_score >= ?
                    ORDER BY similarity_score DESC
                    LIMIT ?
                """,
                    (report_id, report_id, report_id, similarity_threshold, limit),
                )

                return [(row[0], row[1]) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to find similar reports: {str(e)}")
            return []

    def get_organism_detection_history(self, organism_id: str) -> List[Dict[str, Any]]:
        """
        Get detection history for an organism across all reports.

        Args:
            organism_id: Organism identifier

        Returns:
            List of detection records with report metadata
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT 
                        ar.report_id, ar.report_name, ar.created_at,
                        ds.dataset_name, ds.collection_date, ds.collection_location,
                        ta.confidence_score, ta.assignment_method,
                        nd.novelty_score, nd.is_novel_candidate
                    FROM analysis_reports ar
                    JOIN datasets ds ON ar.dataset_id = ds.dataset_id
                    LEFT JOIN sequences s ON ar.report_id = s.report_id 
                        AND s.organism_id = ?
                    LEFT JOIN taxonomic_assignments ta ON s.sequence_id = ta.sequence_id
                    LEFT JOIN novelty_detections nd ON s.sequence_id = nd.sequence_id
                    WHERE s.organism_id = ?
                    ORDER BY ar.created_at DESC
                """,
                    (organism_id, organism_id),
                )

                history = []
                for row in cursor.fetchall():
                    history.append(
                        {
                            "report_id": row[0],
                            "report_name": row[1],
                            "analysis_date": row[2],
                            "dataset_name": row[3],
                            "collection_date": row[4],
                            "collection_location": row[5],
                            "confidence_score": row[6],
                            "assignment_method": row[7],
                            "novelty_score": row[8],
                            "is_novel_candidate": bool(row[9])
                            if row[9] is not None
                            else None,
                        }
                    )

                return history

        except Exception as e:
            logger.error(f"Failed to get organism detection history: {str(e)}")
            return []

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.

        Returns:
            Dictionary with various database statistics
        """
        try:
            with self.get_connection() as conn:
                stats = {}

                # Table counts
                cursor = conn.execute("SELECT COUNT(*) FROM organism_profiles")
                stats["total_organisms"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM datasets")
                stats["total_datasets"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM analysis_reports")
                stats["total_reports"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM similarity_matrices")
                stats["total_comparisons"] = cursor.fetchone()[0]

                # Novel organisms
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM organism_profiles WHERE is_novel_candidate = 1
                """)
                stats["novel_organisms"] = cursor.fetchone()[0]

                # Recent activity
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM analysis_reports 
                    WHERE created_at >= datetime('now', '-7 days')
                """)
                stats["reports_last_7_days"] = cursor.fetchone()[0]

                # Top kingdoms
                cursor = conn.execute("""
                    SELECT kingdom, COUNT(*) as count 
                    FROM organism_profiles 
                    WHERE kingdom IS NOT NULL 
                    GROUP BY kingdom 
                    ORDER BY count DESC 
                    LIMIT 5
                """)
                stats["top_kingdoms"] = dict(cursor.fetchall())

                return stats

        except Exception as e:
            logger.error(f"Failed to get database statistics: {str(e)}")
            return {}

    def _row_to_organism_profile(
        self, row: tuple, description: Any
    ) -> Optional[OrganismProfile]:
        """Convert database row to OrganismProfile instance."""
        try:
            columns = [col[0] for col in description]
            data = dict(zip(columns, row))

            # Handle JSON fields
            if data.get("reference_databases"):
                data["reference_databases"] = json.loads(data["reference_databases"])

            # Handle datetime fields
            for field in ["first_detected", "last_updated", "created_at", "updated_at"]:
                if data.get(field):
                    data[field] = datetime.fromisoformat(data[field])

            # Handle class field (reserved keyword)
            if "class" in data:
                data["class_name"] = data.pop("class")

            return OrganismProfile(**data)

        except Exception as e:
            logger.error(f"Failed to convert row to OrganismProfile: {str(e)}")
            return None

    def _row_to_analysis_report(
        self, row: tuple, description: Any
    ) -> Optional[AnalysisReport]:
        """Convert database row to AnalysisReport instance."""
        try:
            columns = [col[0] for col in description]
            data = dict(zip(columns, row))

            # Handle JSON fields
            if data.get("composition_data"):
                data["composition_data"] = json.loads(data["composition_data"])

            # Handle enum fields
            if data.get("status"):
                data["status"] = AnalysisStatus(data["status"])
            if data.get("sequence_type_detected"):
                data["sequence_type_detected"] = SequenceType(
                    data["sequence_type_detected"]
                )

            # Handle datetime fields
            for field in ["created_at", "updated_at"]:
                if data.get(field):
                    data[field] = datetime.fromisoformat(data[field])

            return AnalysisReport(**data)

        except Exception as e:
            logger.error(f"Failed to convert row to AnalysisReport: {str(e)}")
            return None
