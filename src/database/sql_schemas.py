"""
SQL schema definitions for both SQLite and PostgreSQL
"""

# SQLite schema
SQLITE_SCHEMA = {
    'datasets': """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            sequence_type TEXT,
            file_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            metadata TEXT
        )
    """,
    
    'analysis_reports': """
        CREATE TABLE IF NOT EXISTS analysis_reports (
            report_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            parameters TEXT,
            results TEXT,
            error_message TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        )
    """,
    
    'organism_profiles': """
        CREATE TABLE IF NOT EXISTS organism_profiles (
            profile_id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL,
            organism_name TEXT NOT NULL,
            taxonomy_id TEXT,
            confidence REAL,
            abundance INTEGER,
            sequence_count INTEGER,
            metadata TEXT,
            FOREIGN KEY (report_id) REFERENCES analysis_reports(report_id) ON DELETE CASCADE
        )
    """,
    
    'similarity_matrices': """
        CREATE TABLE IF NOT EXISTS similarity_matrices (
            matrix_id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL,
            matrix_type TEXT NOT NULL,
            data BLOB NOT NULL,
            labels TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (report_id) REFERENCES analysis_reports(report_id) ON DELETE CASCADE
        )
    """,
    
    'report_comparisons': """
        CREATE TABLE IF NOT EXISTS report_comparisons (
            comparison_id TEXT PRIMARY KEY,
            report_id_1 TEXT NOT NULL,
            report_id_2 TEXT NOT NULL,
            comparison_type TEXT NOT NULL,
            similarity_score REAL,
            results TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (report_id_1) REFERENCES analysis_reports(report_id) ON DELETE CASCADE,
            FOREIGN KEY (report_id_2) REFERENCES analysis_reports(report_id) ON DELETE CASCADE
        )
    """
}

# PostgreSQL schema (with proper data types and indexes)
POSTGRES_SCHEMA = {
    'datasets': """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(500) NOT NULL,
            description TEXT,
            sequence_type VARCHAR(50),
            file_path TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            metadata JSONB
        )
    """,
    
    'analysis_reports': """
        CREATE TABLE IF NOT EXISTS analysis_reports (
            report_id VARCHAR(255) PRIMARY KEY,
            dataset_id VARCHAR(255) NOT NULL,
            analysis_type VARCHAR(100) NOT NULL,
            status VARCHAR(50) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            parameters JSONB,
            results JSONB,
            error_message TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        )
    """,
    
    'organism_profiles': """
        CREATE TABLE IF NOT EXISTS organism_profiles (
            profile_id VARCHAR(255) PRIMARY KEY,
            report_id VARCHAR(255) NOT NULL,
            organism_name VARCHAR(500) NOT NULL,
            taxonomy_id VARCHAR(100),
            confidence REAL,
            abundance INTEGER,
            sequence_count INTEGER,
            metadata JSONB,
            FOREIGN KEY (report_id) REFERENCES analysis_reports(report_id) ON DELETE CASCADE
        )
    """,
    
    'similarity_matrices': """
        CREATE TABLE IF NOT EXISTS similarity_matrices (
            matrix_id VARCHAR(255) PRIMARY KEY,
            report_id VARCHAR(255) NOT NULL,
            matrix_type VARCHAR(100) NOT NULL,
            data BYTEA NOT NULL,
            labels TEXT[],
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (report_id) REFERENCES analysis_reports(report_id) ON DELETE CASCADE
        )
    """,
    
    'report_comparisons': """
        CREATE TABLE IF NOT EXISTS report_comparisons (
            comparison_id VARCHAR(255) PRIMARY KEY,
            report_id_1 VARCHAR(255) NOT NULL,
            report_id_2 VARCHAR(255) NOT NULL,
            comparison_type VARCHAR(100) NOT NULL,
            similarity_score REAL,
            results JSONB,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (report_id_1) REFERENCES analysis_reports(report_id) ON DELETE CASCADE,
            FOREIGN KEY (report_id_2) REFERENCES analysis_reports(report_id) ON DELETE CASCADE
        )
    """
}

# PostgreSQL indexes for performance
POSTGRES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name)",
    "CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_reports_dataset_id ON analysis_reports(dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_reports_status ON analysis_reports(status)",
    "CREATE INDEX IF NOT EXISTS idx_reports_created_at ON analysis_reports(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_profiles_report_id ON organism_profiles(report_id)",
    "CREATE INDEX IF NOT EXISTS idx_profiles_organism ON organism_profiles(organism_name)",
    "CREATE INDEX IF NOT EXISTS idx_matrices_report_id ON similarity_matrices(report_id)",
    "CREATE INDEX IF NOT EXISTS idx_comparisons_report_1 ON report_comparisons(report_id_1)",
    "CREATE INDEX IF NOT EXISTS idx_comparisons_report_2 ON report_comparisons(report_id_2)",
]

# SQLite indexes (similar but simpler)
SQLITE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name)",
    "CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_reports_dataset_id ON analysis_reports(dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_reports_status ON analysis_reports(status)",
    "CREATE INDEX IF NOT EXISTS idx_reports_created_at ON analysis_reports(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_profiles_report_id ON organism_profiles(report_id)",
    "CREATE INDEX IF NOT EXISTS idx_profiles_organism ON organism_profiles(organism_name)",
    "CREATE INDEX IF NOT EXISTS idx_matrices_report_id ON similarity_matrices(report_id)",
    "CREATE INDEX IF NOT EXISTS idx_comparisons_report_1 ON report_comparisons(report_id_1)",
    "CREATE INDEX IF NOT EXISTS idx_comparisons_report_2 ON report_comparisons(report_id_2)",
]


def get_schema_for_database(db_type: str) -> dict:
    """
    Get appropriate schema for database type
    
    Args:
        db_type: 'sqlite' or 'postgresql'
        
    Returns:
        Dictionary of table creation statements
    """
    if db_type.lower() == 'postgresql':
        return POSTGRES_SCHEMA
    return SQLITE_SCHEMA


def get_indexes_for_database(db_type: str) -> list:
    """
    Get appropriate indexes for database type
    
    Args:
        db_type: 'sqlite' or 'postgresql'
        
    Returns:
        List of index creation statements
    """
    if db_type.lower() == 'postgresql':
        return POSTGRES_INDEXES
    return SQLITE_INDEXES
