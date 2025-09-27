"""
Database schema definition for eDNA analysis report management system.

This module defines the complete database schema for storing organism profiles,
analysis reports, similarity matrices, and cross-analysis results.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseSchema:
    """
    Manages database schema creation and migration for the report management system.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database schema manager.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            default_db = Path(__file__).parent.parent.parent / "data" / "reports.db"
            self.db_path = default_db
        else:
            self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Database schema version for migrations
        self.schema_version = "1.0.0"
        
    def create_database(self) -> None:
        """
        Create complete database schema with all tables and indexes.
        """
        logger.info(f"Creating database schema at: {self.db_path}")
        
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Create all tables
            self._create_organism_profiles_table(conn)
            self._create_datasets_table(conn)
            self._create_analysis_reports_table(conn)
            self._create_sequences_table(conn)
            self._create_taxonomic_assignments_table(conn)
            self._create_clustering_results_table(conn)
            self._create_novelty_detections_table(conn)
            self._create_similarity_matrices_table(conn)
            self._create_report_comparisons_table(conn)
            self._create_analysis_metadata_table(conn)
            self._create_system_metadata_table(conn)
            
            # Create indexes for performance
            self._create_indexes(conn)
            
            # Store schema version
            self._store_schema_version(conn)
            
            conn.commit()
            
        logger.info("Database schema created successfully")
    
    def _create_organism_profiles_table(self, conn: sqlite3.Connection) -> None:
        """Create organism_profiles table for unique organism identification."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS organism_profiles (
                organism_id TEXT PRIMARY KEY,
                organism_name TEXT,
                taxonomic_lineage TEXT,
                kingdom TEXT,
                phylum TEXT,
                class TEXT,
                order_name TEXT,
                family TEXT,
                genus TEXT,
                species TEXT,
                sequence_signature TEXT,  -- Hash of representative sequences
                first_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detection_count INTEGER DEFAULT 1,
                confidence_score REAL,
                is_novel_candidate BOOLEAN DEFAULT FALSE,
                novelty_score REAL,
                reference_databases TEXT,  -- JSON list of databases
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_datasets_table(self, conn: sqlite3.Connection) -> None:
        """Create datasets table for dataset metadata."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                file_path TEXT,
                file_format TEXT,
                file_size_mb REAL,
                total_sequences INTEGER,
                sequence_type TEXT,  -- dna, rna, protein
                collection_date TIMESTAMP,
                collection_location TEXT,
                depth_meters REAL,
                temperature_celsius REAL,
                ph_level REAL,
                salinity REAL,
                environmental_conditions TEXT,  -- JSON
                preprocessing_params TEXT,  -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_analysis_reports_table(self, conn: sqlite3.Connection) -> None:
        """Create analysis_reports table for storing complete analysis results."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_reports (
                report_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                report_name TEXT,
                analysis_type TEXT,  -- full, clustering, taxonomy, novelty
                status TEXT DEFAULT 'completed',  -- pending, running, completed, failed
                processing_time_seconds REAL,
                
                -- Basic statistics
                min_length INTEGER,
                max_length INTEGER,
                mean_length REAL,
                median_length REAL,
                std_length REAL,
                
                -- Composition analysis
                sequence_type_detected TEXT,
                composition_data TEXT,  -- JSON
                
                -- Biodiversity metrics
                shannon_diversity REAL,
                simpson_diversity REAL,
                evenness REAL,
                species_richness INTEGER,
                
                -- Clustering results summary
                n_clusters INTEGER,
                silhouette_score REAL,
                cluster_coherence REAL,
                
                -- Taxonomy assignment summary
                sequences_with_taxonomy INTEGER,
                taxonomy_confidence_avg REAL,
                
                -- Novelty detection summary
                novel_candidates_count INTEGER,
                novel_percentage REAL,
                novelty_threshold REAL,
                
                -- File references
                full_report_path TEXT,
                results_json_path TEXT,
                visualizations_dir TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
            )
        """)
    
    def _create_sequences_table(self, conn: sqlite3.Connection) -> None:
        """Create sequences table for individual sequence data."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sequences (
                sequence_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                report_id TEXT NOT NULL,
                organism_id TEXT,  -- Links to organism_profiles
                sequence_hash TEXT UNIQUE,  -- MD5 hash of sequence
                sequence_length INTEGER,
                gc_content REAL,
                quality_score REAL,
                description TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id),
                FOREIGN KEY (report_id) REFERENCES analysis_reports (report_id),
                FOREIGN KEY (organism_id) REFERENCES organism_profiles (organism_id)
            )
        """)
    
    def _create_taxonomic_assignments_table(self, conn: sqlite3.Connection) -> None:
        """Create taxonomic_assignments table for taxonomy data."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS taxonomic_assignments (
                assignment_id TEXT PRIMARY KEY,
                sequence_id TEXT NOT NULL,
                report_id TEXT NOT NULL,
                organism_id TEXT,
                
                kingdom TEXT,
                phylum TEXT,
                class TEXT,
                order_name TEXT,
                family TEXT,
                genus TEXT,
                species TEXT,
                
                assignment_method TEXT,  -- blast, ml, hybrid
                confidence_score REAL,
                blast_evalue REAL,
                blast_identity REAL,
                blast_coverage REAL,
                reference_sequence_id TEXT,
                reference_database TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (sequence_id) REFERENCES sequences (sequence_id),
                FOREIGN KEY (report_id) REFERENCES analysis_reports (report_id),
                FOREIGN KEY (organism_id) REFERENCES organism_profiles (organism_id)
            )
        """)
    
    def _create_clustering_results_table(self, conn: sqlite3.Connection) -> None:
        """Create clustering_results table for cluster assignments."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clustering_results (
                cluster_assignment_id TEXT PRIMARY KEY,
                sequence_id TEXT NOT NULL,
                report_id TEXT NOT NULL,
                cluster_id INTEGER,
                cluster_label TEXT,
                cluster_size INTEGER,
                
                embedding_x REAL,  -- 2D projection coordinates
                embedding_y REAL,
                
                distance_to_centroid REAL,
                cluster_coherence REAL,
                is_outlier BOOLEAN DEFAULT FALSE,
                
                clustering_method TEXT,  -- hdbscan, kmeans, dbscan
                clustering_params TEXT,  -- JSON
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (sequence_id) REFERENCES sequences (sequence_id),
                FOREIGN KEY (report_id) REFERENCES analysis_reports (report_id)
            )
        """)
    
    def _create_novelty_detections_table(self, conn: sqlite3.Connection) -> None:
        """Create novelty_detections table for novel taxa detection results."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS novelty_detections (
                detection_id TEXT PRIMARY KEY,
                sequence_id TEXT NOT NULL,
                report_id TEXT NOT NULL,
                organism_id TEXT,
                
                is_novel_candidate BOOLEAN DEFAULT FALSE,
                novelty_score REAL,
                novelty_method TEXT,  -- isolation_forest, distance_based, cluster_based
                
                distance_to_nearest_known REAL,
                nearest_known_organism_id TEXT,
                cluster_coherence_score REAL,
                
                abundance_in_dataset REAL,
                abundance_threshold REAL,
                
                validation_status TEXT,  -- pending, validated, rejected
                validation_notes TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (sequence_id) REFERENCES sequences (sequence_id),
                FOREIGN KEY (report_id) REFERENCES analysis_reports (report_id),
                FOREIGN KEY (organism_id) REFERENCES organism_profiles (organism_id)
            )
        """)
    
    def _create_similarity_matrices_table(self, conn: sqlite3.Connection) -> None:
        """Create similarity_matrices table for cross-analysis comparisons."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS similarity_matrices (
                comparison_id TEXT PRIMARY KEY,
                report_id_1 TEXT NOT NULL,
                report_id_2 TEXT NOT NULL,
                
                organism_overlap_count INTEGER,
                organism_overlap_percentage REAL,
                jaccard_similarity REAL,
                cosine_similarity REAL,
                
                -- Taxonomic composition similarity
                kingdom_similarity REAL,
                phylum_similarity REAL,
                genus_similarity REAL,
                
                -- Diversity metric differences
                shannon_diversity_diff REAL,
                simpson_diversity_diff REAL,
                evenness_diff REAL,
                
                -- Clustering similarity
                cluster_structure_similarity REAL,
                
                -- Environmental context similarity
                location_distance_km REAL,
                depth_difference_m REAL,
                temporal_difference_days REAL,
                
                similarity_score REAL,  -- Overall composite score
                comparison_method TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (report_id_1) REFERENCES analysis_reports (report_id),
                FOREIGN KEY (report_id_2) REFERENCES analysis_reports (report_id)
            )
        """)
    
    def _create_report_comparisons_table(self, conn: sqlite3.Connection) -> None:
        """Create report_comparisons table for detailed comparison analysis."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS report_comparisons (
                comparison_detail_id TEXT PRIMARY KEY,
                comparison_id TEXT NOT NULL,
                organism_id TEXT NOT NULL,
                
                present_in_report_1 BOOLEAN DEFAULT FALSE,
                present_in_report_2 BOOLEAN DEFAULT FALSE,
                abundance_report_1 REAL,
                abundance_report_2 REAL,
                abundance_ratio REAL,
                
                novelty_score_report_1 REAL,
                novelty_score_report_2 REAL,
                novelty_status_changed BOOLEAN DEFAULT FALSE,
                
                taxonomy_agreement BOOLEAN DEFAULT TRUE,
                taxonomy_confidence_1 REAL,
                taxonomy_confidence_2 REAL,
                
                notes TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (comparison_id) REFERENCES similarity_matrices (comparison_id),
                FOREIGN KEY (organism_id) REFERENCES organism_profiles (organism_id)
            )
        """)
    
    def _create_analysis_metadata_table(self, conn: sqlite3.Connection) -> None:
        """Create analysis_metadata table for storing analysis parameters and settings."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_metadata (
                metadata_id TEXT PRIMARY KEY,
                report_id TEXT NOT NULL,
                
                -- Pipeline configuration
                pipeline_version TEXT,
                preprocessing_enabled BOOLEAN DEFAULT TRUE,
                embedding_model TEXT,
                clustering_algorithm TEXT,
                taxonomy_database TEXT,
                novelty_detection_method TEXT,
                
                -- Parameters (stored as JSON)
                preprocessing_params TEXT,
                embedding_params TEXT,
                clustering_params TEXT,
                taxonomy_params TEXT,
                novelty_params TEXT,
                
                -- System information
                system_info TEXT,  -- JSON with CPU, memory, OS info
                processing_node TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (report_id) REFERENCES analysis_reports (report_id)
            )
        """)
    
    def _create_system_metadata_table(self, conn: sqlite3.Connection) -> None:
        """Create system_metadata table for system-wide settings and metadata."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                data_type TEXT DEFAULT 'string',  -- string, integer, float, boolean, json
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for improved query performance."""
        indexes = [
            # Organism profiles indexes
            "CREATE INDEX IF NOT EXISTS idx_organism_name ON organism_profiles (organism_name)",
            "CREATE INDEX IF NOT EXISTS idx_organism_genus_species ON organism_profiles (genus, species)",
            "CREATE INDEX IF NOT EXISTS idx_organism_kingdom ON organism_profiles (kingdom)",
            "CREATE INDEX IF NOT EXISTS idx_organism_novel ON organism_profiles (is_novel_candidate)",
            "CREATE INDEX IF NOT EXISTS idx_organism_detection_count ON organism_profiles (detection_count)",
            
            # Datasets indexes
            "CREATE INDEX IF NOT EXISTS idx_dataset_name ON datasets (dataset_name)",
            "CREATE INDEX IF NOT EXISTS idx_dataset_collection_date ON datasets (collection_date)",
            "CREATE INDEX IF NOT EXISTS idx_dataset_location ON datasets (collection_location)",
            
            # Analysis reports indexes
            "CREATE INDEX IF NOT EXISTS idx_report_dataset ON analysis_reports (dataset_id)",
            "CREATE INDEX IF NOT EXISTS idx_report_type ON analysis_reports (analysis_type)",
            "CREATE INDEX IF NOT EXISTS idx_report_status ON analysis_reports (status)",
            "CREATE INDEX IF NOT EXISTS idx_report_created ON analysis_reports (created_at)",
            
            # Sequences indexes
            "CREATE INDEX IF NOT EXISTS idx_sequence_dataset ON sequences (dataset_id)",
            "CREATE INDEX IF NOT EXISTS idx_sequence_organism ON sequences (organism_id)",
            "CREATE INDEX IF NOT EXISTS idx_sequence_hash ON sequences (sequence_hash)",
            
            # Taxonomic assignments indexes
            "CREATE INDEX IF NOT EXISTS idx_taxonomy_sequence ON taxonomic_assignments (sequence_id)",
            "CREATE INDEX IF NOT EXISTS idx_taxonomy_organism ON taxonomic_assignments (organism_id)",
            "CREATE INDEX IF NOT EXISTS idx_taxonomy_genus_species ON taxonomic_assignments (genus, species)",
            "CREATE INDEX IF NOT EXISTS idx_taxonomy_confidence ON taxonomic_assignments (confidence_score)",
            
            # Clustering results indexes
            "CREATE INDEX IF NOT EXISTS idx_clustering_sequence ON clustering_results (sequence_id)",
            "CREATE INDEX IF NOT EXISTS idx_clustering_cluster ON clustering_results (cluster_id)",
            "CREATE INDEX IF NOT EXISTS idx_clustering_method ON clustering_results (clustering_method)",
            
            # Novelty detections indexes
            "CREATE INDEX IF NOT EXISTS idx_novelty_sequence ON novelty_detections (sequence_id)",
            "CREATE INDEX IF NOT EXISTS idx_novelty_organism ON novelty_detections (organism_id)",
            "CREATE INDEX IF NOT EXISTS idx_novelty_candidate ON novelty_detections (is_novel_candidate)",
            "CREATE INDEX IF NOT EXISTS idx_novelty_score ON novelty_detections (novelty_score)",
            
            # Similarity matrices indexes
            "CREATE INDEX IF NOT EXISTS idx_similarity_reports ON similarity_matrices (report_id_1, report_id_2)",
            "CREATE INDEX IF NOT EXISTS idx_similarity_score ON similarity_matrices (similarity_score)",
            "CREATE INDEX IF NOT EXISTS idx_similarity_created ON similarity_matrices (created_at)",
            
            # Report comparisons indexes
            "CREATE INDEX IF NOT EXISTS idx_comparison_detail ON report_comparisons (comparison_id)",
            "CREATE INDEX IF NOT EXISTS idx_comparison_organism ON report_comparisons (organism_id)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
    
    def _store_schema_version(self, conn: sqlite3.Connection) -> None:
        """Store current schema version in system_metadata table."""
        conn.execute("""
            INSERT OR REPLACE INTO system_metadata (key, value, data_type, description)
            VALUES (?, ?, ?, ?)
        """, (
            "schema_version", 
            self.schema_version,
            "string", 
            "Database schema version"
        ))
    
    def get_schema_version(self) -> Optional[str]:
        """
        Get current database schema version.
        
        Returns:
            Schema version string or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT value FROM system_metadata WHERE key = 'schema_version'
            """)
            result = cursor.fetchone()
            return result[0] if result else None
    
    def database_exists(self) -> bool:
        """
        Check if database file exists and has tables.
        
        Returns:
            True if database exists with tables, False otherwise
        """
        if not self.db_path.exists():
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM sqlite_master WHERE type='table'
                """)
                table_count = cursor.fetchone()[0]
                return table_count > 0
        except Exception:
            return False
    
    def migrate_database(self) -> None:
        """
        Migrate database to latest schema version.
        Currently a placeholder for future migration needs.
        """
        current_version = self.get_schema_version()
        
        if current_version is None:
            logger.info("No schema version found, creating new database")
            self.create_database()
        elif current_version != self.schema_version:
            logger.info(f"Migrating database from {current_version} to {self.schema_version}")
            # Future migration logic would go here
            self._store_schema_version(sqlite3.connect(self.db_path))
        else:
            logger.info("Database schema is up to date")
    
    def get_table_info(self) -> Dict[str, Any]:
        """
        Get information about all tables in the database.
        
        Returns:
            Dictionary with table information
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            table_info = {}
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                cursor = conn.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                table_info[table] = {
                    'row_count': row_count,
                    'columns': [col[1] for col in columns]  # Column names
                }
            
            return table_info