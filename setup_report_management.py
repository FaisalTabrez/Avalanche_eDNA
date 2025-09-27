"""
Setup script for eDNA Report Management System.

This script initializes the database, creates necessary directories,
and sets up the complete report management system.
"""

import os
import sys
from pathlib import Path
import logging

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the system."""
    logger.info("Creating directory structure...")
    
    directories = [
        "data",
        "data/report_storage",
        "data/report_storage/reports",
        "data/report_storage/datasets", 
        "data/report_storage/results",
        "data/report_storage/visualizations",
        "data/report_storage/metadata",
        "data/report_storage/exports",
        "data/report_storage/backups",
        "data/report_storage/temp",
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {dir_path}")

def initialize_database():
    """Initialize the database with complete schema."""
    logger.info("Initializing database...")
    
    try:
        from src.database.manager import DatabaseManager
        
        # Initialize database manager (creates database and schema automatically)
        db_manager = DatabaseManager()
        
        # Verify database creation
        stats = db_manager.get_database_statistics()
        logger.info(f"✓ Database initialized successfully")
        logger.info(f"  - Tables created: {len(db_manager.schema.get_table_info())}")
        logger.info(f"  - Schema version: {db_manager.schema.get_schema_version()}")
        
        return db_manager
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {str(e)}")
        raise

def test_system_components():
    """Test all major system components."""
    logger.info("Testing system components...")
    
    try:
        # Test database manager
        from src.database.manager import DatabaseManager
        db_manager = DatabaseManager()
        logger.info("✓ Database manager: OK")
        
        # Test query engine
        from src.database.queries import ReportQueryEngine
        query_engine = ReportQueryEngine(db_manager)
        logger.info("✓ Query engine: OK")
        
        # Test report catalogue manager
        from src.report_management.catalogue_manager import ReportCatalogueManager
        catalogue_manager = ReportCatalogueManager(db_manager=db_manager)
        logger.info("✓ Report catalogue manager: OK")
        
        # Test similarity engine
        from src.similarity.cross_analysis_engine import CrossAnalysisEngine
        similarity_engine = CrossAnalysisEngine(db_manager)
        logger.info("✓ Cross-analysis engine: OK")
        
        # Test organism profiling
        from src.organism_profiling import OrganismIdentifier
        organism_identifier = OrganismIdentifier(db_manager)
        logger.info("✓ Organism profiling: OK")
        
        logger.info("✓ All system components initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component test failed: {str(e)}")
        return False

def create_sample_data():
    """Create sample data for demonstration."""
    logger.info("Creating sample data...")
    
    try:
        from src.database.models import DatasetInfo, OrganismProfile, SequenceType
        from src.database.manager import DatabaseManager
        from datetime import datetime
        
        db_manager = DatabaseManager()
        
        # Create sample dataset
        sample_dataset = DatasetInfo(
            dataset_id="DS_SAMPLE001",
            dataset_name="Sample Deep Sea eDNA",
            file_path="data/sample/sample_sequences.fasta",
            file_format="fasta",
            file_size_mb=0.5,
            sequence_type=SequenceType.DNA,
            collection_location="Mariana Trench",
            depth_meters=8000,
            temperature_celsius=2.1,
            collection_date=datetime(2025, 9, 15)
        )
        
        db_manager.store_dataset_info(sample_dataset)
        
        # Create sample organism profile
        sample_organism = OrganismProfile(
            organism_id="ORG_SAMPLE001",
            organism_name="Pyrococcus sample_sp",
            kingdom="Archaea",
            phylum="Euryarchaeota",
            genus="Pyrococcus",
            species="sample_sp",
            sequence_signature="abc123def456",
            detection_count=1,
            confidence_score=0.85,
            is_novel_candidate=True,
            novelty_score=0.75
        )
        
        db_manager.store_organism_profile(sample_organism)
        
        logger.info("✓ Sample data created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample data: {str(e)}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'fastapi', 
        'uvicorn',
        'pandas',
        'numpy',
        'plotly',
        'Bio',  # BioPython imports as 'Bio'
        'scipy',
        'sqlite3'  # Built into Python
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            elif package == 'Bio':
                import Bio  # BioPython
                logger.info(f"✓ biopython: installed")
                continue
            else:
                __import__(package)
            logger.info(f"✓ {package}: installed")
        except ImportError:
            if package == 'Bio':
                missing_packages.append('biopython')
                logger.warning(f"❌ biopython: missing")
            else:
                missing_packages.append(package)
                logger.warning(f"❌ {package}: missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✓ All dependencies satisfied")
    return True

def main():
    """Main setup function."""
    logger.info("🧬 Starting eDNA Report Management System Setup...")
    logger.info("="*60)
    
    try:
        # Check dependencies first
        if not check_dependencies():
            logger.error("❌ Setup failed: Missing dependencies")
            return False
        
        # Create directories
        setup_directories()
        
        # Initialize database
        db_manager = initialize_database()
        
        # Test components
        if not test_system_components():
            logger.error("❌ Setup failed: Component tests failed")
            return False
        
        # Create sample data
        create_sample_data()
        
        logger.info("="*60)
        logger.info("🎉 eDNA Report Management System setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Launch dashboard: python -m streamlit run src/dashboards/report_management_dashboard.py")
        logger.info("2. Start API server: python -m uvicorn src.api.report_management_api:app --reload")
        logger.info("3. View documentation: EDNA_REPORT_MANAGEMENT_SYSTEM_GUIDE.md")
        logger.info("")
        logger.info("System is ready for use! 🚀")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)