"""
Database performance optimization
Indexing, query optimization, and connection pooling
"""
from sqlalchemy import Index, text, event
from sqlalchemy.pool import QueuePool, NullPool
from src.database.models import (
    AnalysisRun, Dataset, Report, User,
    Sequence, TaxonomyPrediction, NoveltyDetection
)
from src.utils.logger import get_logger
import time

logger = get_logger(__name__)


# ============================================================================
# Database Indexes
# ============================================================================

# Analysis Run Indexes
analysis_run_status_idx = Index('idx_analysis_run_status', AnalysisRun.status)
analysis_run_user_idx = Index('idx_analysis_run_user_id', AnalysisRun.user_id)
analysis_run_dataset_idx = Index('idx_analysis_run_dataset_id', AnalysisRun.dataset_id)
analysis_run_created_idx = Index('idx_analysis_run_created_at', AnalysisRun.created_at.desc())
analysis_run_composite_idx = Index(
    'idx_analysis_run_user_status',
    AnalysisRun.user_id,
    AnalysisRun.status,
    AnalysisRun.created_at.desc()
)

# Dataset Indexes
dataset_user_idx = Index('idx_dataset_user_id', Dataset.user_id)
dataset_name_idx = Index('idx_dataset_name', Dataset.name)
dataset_created_idx = Index('idx_dataset_created_at', Dataset.created_at.desc())
dataset_composite_idx = Index(
    'idx_dataset_user_created',
    Dataset.user_id,
    Dataset.created_at.desc()
)

# Report Indexes
report_run_idx = Index('idx_report_analysis_run_id', Report.analysis_run_id)
report_created_idx = Index('idx_report_created_at', Report.created_at.desc())

# Sequence Indexes
sequence_dataset_idx = Index('idx_sequence_dataset_id', Sequence.dataset_id)
sequence_hash_idx = Index('idx_sequence_seq_hash', Sequence.seq_hash)

# Taxonomy Prediction Indexes
taxonomy_sequence_idx = Index('idx_taxonomy_sequence_id', TaxonomyPrediction.sequence_id)
taxonomy_confidence_idx = Index('idx_taxonomy_confidence', TaxonomyPrediction.confidence.desc())
taxonomy_rank_idx = Index('idx_taxonomy_rank_name', TaxonomyPrediction.rank, TaxonomyPrediction.predicted_taxon)

# Novelty Detection Indexes
novelty_sequence_idx = Index('idx_novelty_sequence_id', NoveltyDetection.sequence_id)
novelty_score_idx = Index('idx_novelty_score', NoveltyDetection.novelty_score.desc())
novelty_is_novel_idx = Index('idx_novelty_is_novel', NoveltyDetection.is_novel)


# ============================================================================
# Index Creation Functions
# ============================================================================

def create_indexes(engine):
    """
    Create all database indexes
    
    Args:
        engine: SQLAlchemy engine
    """
    logger.info("Creating database indexes...")
    
    indexes = [
        # Analysis Run
        analysis_run_status_idx,
        analysis_run_user_idx,
        analysis_run_dataset_idx,
        analysis_run_created_idx,
        analysis_run_composite_idx,
        
        # Dataset
        dataset_user_idx,
        dataset_name_idx,
        dataset_created_idx,
        dataset_composite_idx,
        
        # Report
        report_run_idx,
        report_created_idx,
        
        # Sequence
        sequence_dataset_idx,
        sequence_hash_idx,
        
        # Taxonomy Prediction
        taxonomy_sequence_idx,
        taxonomy_confidence_idx,
        taxonomy_rank_idx,
        
        # Novelty Detection
        novelty_sequence_idx,
        novelty_score_idx,
        novelty_is_novel_idx,
    ]
    
    created_count = 0
    skipped_count = 0
    
    with engine.begin() as conn:
        for idx in indexes:
            try:
                idx.create(conn, checkfirst=True)
                logger.debug(f"Created index: {idx.name}")
                created_count += 1
            except Exception as e:
                logger.warning(f"Skipped index {idx.name}: {e}")
                skipped_count += 1
    
    logger.info(f"Index creation complete: {created_count} created, {skipped_count} skipped")


def drop_indexes(engine):
    """
    Drop all custom indexes (for cleanup/rebuild)
    
    Args:
        engine: SQLAlchemy engine
    """
    logger.info("Dropping database indexes...")
    
    indexes = [
        analysis_run_status_idx, analysis_run_user_idx, analysis_run_dataset_idx,
        analysis_run_created_idx, analysis_run_composite_idx,
        dataset_user_idx, dataset_name_idx, dataset_created_idx, dataset_composite_idx,
        report_run_idx, report_created_idx,
        sequence_dataset_idx, sequence_hash_idx,
        taxonomy_sequence_idx, taxonomy_confidence_idx, taxonomy_rank_idx,
        novelty_sequence_idx, novelty_score_idx, novelty_is_novel_idx,
    ]
    
    dropped_count = 0
    
    with engine.begin() as conn:
        for idx in indexes:
            try:
                idx.drop(conn, checkfirst=True)
                logger.debug(f"Dropped index: {idx.name}")
                dropped_count += 1
            except Exception as e:
                logger.warning(f"Failed to drop index {idx.name}: {e}")
    
    logger.info(f"Index drop complete: {dropped_count} dropped")


def rebuild_indexes(engine):
    """
    Rebuild all indexes (drop then create)
    
    Args:
        engine: SQLAlchemy engine
    """
    logger.info("Rebuilding database indexes...")
    drop_indexes(engine)
    create_indexes(engine)
    logger.info("Index rebuild complete")


# ============================================================================
# Connection Pool Configuration
# ============================================================================

def get_pool_config(pool_type='production'):
    """
    Get connection pool configuration
    
    Args:
        pool_type: 'production', 'development', or 'testing'
    
    Returns:
        dict: Pool configuration
    """
    configs = {
        'production': {
            'poolclass': QueuePool,
            'pool_size': 20,  # Number of connections to maintain
            'max_overflow': 40,  # Additional connections when pool is full
            'pool_timeout': 30,  # Seconds to wait for connection
            'pool_recycle': 3600,  # Recycle connections after 1 hour
            'pool_pre_ping': True,  # Test connections before using
            'echo_pool': False,
        },
        'development': {
            'poolclass': QueuePool,
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'echo_pool': True,
        },
        'testing': {
            'poolclass': NullPool,  # No pooling for tests
        }
    }
    
    return configs.get(pool_type, configs['production'])


def configure_pool_events(engine):
    """
    Configure connection pool event listeners
    
    Args:
        engine: SQLAlchemy engine
    """
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        """Called when a new DB-API connection is created"""
        logger.debug("New database connection created")
        
        # Set connection-level configuration
        cursor = dbapi_conn.cursor()
        cursor.execute("SET SESSION statement_timeout = '300s'")  # 5 minutes
        cursor.execute("SET SESSION idle_in_transaction_session_timeout = '600s'")  # 10 minutes
        cursor.close()
    
    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_conn, connection_record, connection_proxy):
        """Called when a connection is retrieved from the pool"""
        connection_record.info['checkout_time'] = time.time()
    
    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_conn, connection_record):
        """Called when a connection is returned to the pool"""
        if 'checkout_time' in connection_record.info:
            checkout_duration = time.time() - connection_record.info['checkout_time']
            if checkout_duration > 60:  # Warn if connection held > 1 minute
                logger.warning(f"Connection held for {checkout_duration:.2f} seconds")
            del connection_record.info['checkout_time']
    
    logger.info("Connection pool event listeners configured")


# ============================================================================
# Query Optimization
# ============================================================================

def analyze_query(session, query):
    """
    Analyze query execution plan (PostgreSQL)
    
    Args:
        session: SQLAlchemy session
        query: SQLAlchemy query object
    
    Returns:
        str: Query execution plan
    """
    compiled = query.statement.compile(compile_kwargs={"literal_binds": True})
    sql = str(compiled)
    
    explain_query = f"EXPLAIN ANALYZE {sql}"
    result = session.execute(text(explain_query))
    
    plan = "\n".join([row[0] for row in result])
    return plan


def suggest_indexes(session, slow_query_threshold=1000):
    """
    Analyze slow queries and suggest indexes (PostgreSQL)
    
    Args:
        session: SQLAlchemy session
        slow_query_threshold: Queries slower than this (ms) are considered slow
    
    Returns:
        list: Suggested index definitions
    """
    # Query pg_stat_statements for slow queries
    query = text("""
        SELECT 
            query,
            calls,
            mean_exec_time,
            max_exec_time,
            stddev_exec_time
        FROM pg_stat_statements
        WHERE mean_exec_time > :threshold
        ORDER BY mean_exec_time DESC
        LIMIT 20
    """)
    
    try:
        result = session.execute(query, {"threshold": slow_query_threshold})
        slow_queries = result.fetchall()
        
        suggestions = []
        for row in slow_queries:
            suggestions.append({
                'query': row[0][:200],  # Truncate long queries
                'calls': row[1],
                'avg_time_ms': round(row[2], 2),
                'max_time_ms': round(row[3], 2),
                'stddev_ms': round(row[4], 2),
            })
        
        return suggestions
    except Exception as e:
        logger.warning(f"Could not analyze slow queries: {e}")
        return []


def vacuum_analyze(session, table_name=None):
    """
    Run VACUUM ANALYZE on table(s) to update statistics and reclaim space
    
    Args:
        session: SQLAlchemy session
        table_name: Specific table name, or None for all tables
    """
    if table_name:
        query = f"VACUUM ANALYZE {table_name}"
    else:
        query = "VACUUM ANALYZE"
    
    try:
        # VACUUM cannot run inside a transaction block
        session.connection().execution_options(isolation_level="AUTOCOMMIT")
        session.execute(text(query))
        logger.info(f"VACUUM ANALYZE completed for {table_name or 'all tables'}")
    except Exception as e:
        logger.error(f"VACUUM ANALYZE failed: {e}")


# ============================================================================
# Query Result Caching
# ============================================================================

def cache_query_result(cache_key, query_func, ttl=3600):
    """
    Cache query result with Redis
    
    Args:
        cache_key: Cache key
        query_func: Function that executes the query
        ttl: Time-to-live in seconds
    
    Returns:
        Query result (from cache or fresh)
    """
    from src.utils.cache import cache
    
    # Try cache first
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug(f"Query result cache hit: {cache_key}")
        return cached
    
    # Execute query
    logger.debug(f"Query result cache miss: {cache_key}")
    result = query_func()
    
    # Store in cache
    cache.set(cache_key, result, ttl=ttl)
    
    return result


# ============================================================================
# Batch Operations
# ============================================================================

def bulk_insert_optimized(session, model_class, data_list, batch_size=1000):
    """
    Optimized bulk insert with batching
    
    Args:
        session: SQLAlchemy session
        model_class: SQLAlchemy model class
        data_list: List of dictionaries with model data
        batch_size: Number of records per batch
    
    Returns:
        int: Number of records inserted
    """
    total_inserted = 0
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        
        try:
            session.bulk_insert_mappings(model_class, batch)
            session.commit()
            total_inserted += len(batch)
            logger.debug(f"Inserted batch {i // batch_size + 1}: {len(batch)} records")
        except Exception as e:
            session.rollback()
            logger.error(f"Batch insert failed at position {i}: {e}")
            raise
    
    logger.info(f"Bulk insert complete: {total_inserted} records")
    return total_inserted


def bulk_update_optimized(session, model_class, data_list, batch_size=1000):
    """
    Optimized bulk update with batching
    
    Args:
        session: SQLAlchemy session
        model_class: SQLAlchemy model class
        data_list: List of dictionaries with model data (must include primary key)
        batch_size: Number of records per batch
    
    Returns:
        int: Number of records updated
    """
    total_updated = 0
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        
        try:
            session.bulk_update_mappings(model_class, batch)
            session.commit()
            total_updated += len(batch)
            logger.debug(f"Updated batch {i // batch_size + 1}: {len(batch)} records")
        except Exception as e:
            session.rollback()
            logger.error(f"Batch update failed at position {i}: {e}")
            raise
    
    logger.info(f"Bulk update complete: {total_updated} records")
    return total_updated


# ============================================================================
# Database Statistics
# ============================================================================

def get_table_stats(session):
    """
    Get table size and row count statistics (PostgreSQL)
    
    Args:
        session: SQLAlchemy session
    
    Returns:
        list: Table statistics
    """
    query = text("""
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
            pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes,
            n_live_tup AS row_count,
            n_dead_tup AS dead_rows,
            last_vacuum,
            last_analyze
        FROM pg_stat_user_tables
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
    """)
    
    result = session.execute(query)
    stats = []
    
    for row in result:
        stats.append({
            'schema': row[0],
            'table': row[1],
            'size': row[2],
            'size_bytes': row[3],
            'row_count': row[4],
            'dead_rows': row[5],
            'last_vacuum': row[6],
            'last_analyze': row[7],
        })
    
    return stats


def get_index_usage(session):
    """
    Get index usage statistics (PostgreSQL)
    
    Args:
        session: SQLAlchemy session
    
    Returns:
        list: Index usage statistics
    """
    query = text("""
        SELECT 
            schemaname,
            tablename,
            indexname,
            idx_scan,
            idx_tup_read,
            idx_tup_fetch,
            pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
        FROM pg_stat_user_indexes
        ORDER BY idx_scan ASC
    """)
    
    result = session.execute(query)
    stats = []
    
    for row in result:
        stats.append({
            'schema': row[0],
            'table': row[1],
            'index': row[2],
            'scans': row[3],
            'tuples_read': row[4],
            'tuples_fetched': row[5],
            'size': row[6],
        })
    
    return stats
