"""
Database initialization and optimization setup
Initialize indexes, configure connection pooling, and prepare for production use
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.optimization import (
    create_indexes,
    configure_pool_events,
    get_pool_config,
    vacuum_analyze,
    get_table_stats,
    get_index_usage
)
from src.utils.logger import get_logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = get_logger(__name__)


def init_database_optimizations(
    db_url: str = None,
    environment: str = 'production',
    create_idx: bool = True,
    run_vacuum: bool = False,
    show_stats: bool = False
):
    """
    Initialize database with optimizations
    
    Args:
        db_url: Database connection URL (defaults to env var)
        environment: Environment type ('production', 'development', 'testing')
        create_idx: Whether to create indexes
        run_vacuum: Whether to run VACUUM ANALYZE
        show_stats: Whether to display table/index statistics
    """
    logger.info("=" * 80)
    logger.info("Database Optimization Initialization")
    logger.info("=" * 80)
    
    # Get database URL
    if db_url is None:
        db_url = os.getenv('DATABASE_URL') or os.getenv('DB_URL')
        
        if db_url is None:
            # Construct from individual components
            db_type = os.getenv('DB_TYPE', 'postgresql')
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME', 'avalanche_edna')
            db_user = os.getenv('DB_USER', 'avalanche')
            db_password = os.getenv('DB_PASSWORD', 'password')
            
            db_url = f"{db_type}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    logger.info(f"Database URL: {db_url.split('@')[1] if '@' in db_url else db_url}")
    logger.info(f"Environment: {environment}")
    
    # Get pool configuration
    pool_config = get_pool_config(environment)
    logger.info(f"Connection pool config: {pool_config}")
    
    # Create engine with pooling
    try:
        engine = create_engine(db_url, **pool_config)
        logger.info("✓ Database engine created")
    except Exception as e:
        logger.error(f"✗ Failed to create database engine: {e}")
        return False
    
    # Configure pool events
    try:
        configure_pool_events(engine)
        logger.info("✓ Connection pool events configured")
    except Exception as e:
        logger.error(f"✗ Failed to configure pool events: {e}")
        return False
    
    # Create session maker
    Session = sessionmaker(bind=engine)
    
    # Create indexes
    if create_idx:
        try:
            logger.info("\nCreating database indexes...")
            create_indexes(engine)
            logger.info("✓ Database indexes created")
        except Exception as e:
            logger.error(f"✗ Failed to create indexes: {e}")
            return False
    
    # Run VACUUM ANALYZE
    if run_vacuum:
        try:
            logger.info("\nRunning VACUUM ANALYZE...")
            session = Session()
            vacuum_analyze(session)
            session.close()
            logger.info("✓ VACUUM ANALYZE completed")
        except Exception as e:
            logger.error(f"✗ Failed to run VACUUM ANALYZE: {e}")
            # Don't return False, this is optional
    
    # Show statistics
    if show_stats:
        try:
            logger.info("\nDatabase Statistics:")
            logger.info("-" * 80)
            
            session = Session()
            
            # Table statistics
            logger.info("\nTable Statistics:")
            table_stats = get_table_stats(session)
            for stat in table_stats[:10]:  # Show top 10
                logger.info(
                    f"  {stat['table']:<30} "
                    f"{stat['size']:>10} "
                    f"{stat['row_count']:>10,} rows "
                    f"({stat['dead_rows']:>6,} dead)"
                )
            
            # Index usage
            logger.info("\nIndex Usage (Low usage indexes):")
            index_stats = get_index_usage(session)
            low_usage = [s for s in index_stats if s['scans'] < 10][:10]
            for stat in low_usage:
                logger.info(
                    f"  {stat['index']:<40} "
                    f"scans: {stat['scans']:>6} "
                    f"size: {stat['size']:>10}"
                )
            
            session.close()
            logger.info("✓ Statistics displayed")
        
        except Exception as e:
            logger.warning(f"Could not retrieve statistics: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Database optimization initialization complete!")
    logger.info("=" * 80)
    
    return True


def quick_init():
    """
    Quick initialization for production use
    Creates indexes and configures pooling
    """
    return init_database_optimizations(
        environment='production',
        create_idx=True,
        run_vacuum=False,
        show_stats=False
    )


def full_init():
    """
    Full initialization with VACUUM and statistics
    Use for initial setup or major migrations
    """
    return init_database_optimizations(
        environment='production',
        create_idx=True,
        run_vacuum=True,
        show_stats=True
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize database optimizations')
    parser.add_argument(
        '--env',
        choices=['production', 'development', 'testing'],
        default='production',
        help='Environment type'
    )
    parser.add_argument(
        '--no-indexes',
        action='store_true',
        help='Skip index creation'
    )
    parser.add_argument(
        '--vacuum',
        action='store_true',
        help='Run VACUUM ANALYZE'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show table and index statistics'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        help='Database connection URL'
    )
    
    args = parser.parse_args()
    
    success = init_database_optimizations(
        db_url=args.db_url,
        environment=args.env,
        create_idx=not args.no_indexes,
        run_vacuum=args.vacuum,
        show_stats=args.stats
    )
    
    sys.exit(0 if success else 1)
