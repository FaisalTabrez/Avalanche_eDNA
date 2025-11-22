"""
Maintenance tasks for system upkeep

This module contains Celery tasks for maintenance operations.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, Any
from celery import shared_task
from datetime import datetime, timedelta

from src.database.database import DatabaseManager
from src.database.backup_manager import BackupManager


logger = logging.getLogger(__name__)


@shared_task(name='src.tasks.maintenance_tasks.cleanup_old_results')
def cleanup_old_results(days: int = 30) -> Dict[str, Any]:
    """
    Clean up old analysis results
    
    Args:
        days: Delete results older than this many days
    
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        logger.info(f"Cleaning up results older than {days} days")
        
        db = DatabaseManager()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        deleted_runs = 0
        deleted_files = 0
        freed_space = 0
        
        with db.get_session() as session:
            from src.database.models import AnalysisRun
            
            # Find old completed runs
            old_runs = session.query(AnalysisRun).filter(
                AnalysisRun.completed_at < cutoff_date,
                AnalysisRun.status == 'completed'
            ).all()
            
            for run in old_runs:
                # Delete output files
                if run.output_path and Path(run.output_path).exists():
                    try:
                        # Calculate size before deletion
                        for path in Path(run.output_path).rglob('*'):
                            if path.is_file():
                                freed_space += path.stat().st_size
                                deleted_files += 1
                        
                        # Delete directory
                        shutil.rmtree(run.output_path)
                        logger.info(f"Deleted output directory: {run.output_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {run.output_path}: {e}")
                
                # Mark run as archived
                run.status = 'archived'
                deleted_runs += 1
            
            session.commit()
        
        return {
            'status': 'success',
            'deleted_runs': deleted_runs,
            'deleted_files': deleted_files,
            'freed_space_mb': freed_space / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}", exc_info=True)
        raise


@shared_task(name='src.tasks.maintenance_tasks.cleanup_temp_files')
def cleanup_temp_files() -> Dict[str, Any]:
    """
    Clean up temporary files
    
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        logger.info("Cleaning up temporary files")
        
        temp_dirs = [
            Path('data/report_storage/temp'),
            Path('/tmp/avalanche'),
        ]
        
        deleted_files = 0
        freed_space = 0
        
        for temp_dir in temp_dirs:
            if not temp_dir.exists():
                continue
            
            # Delete files older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for path in temp_dir.rglob('*'):
                if not path.is_file():
                    continue
                
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime)
                    if mtime < cutoff_time:
                        size = path.stat().st_size
                        path.unlink()
                        deleted_files += 1
                        freed_space += size
                except Exception as e:
                    logger.error(f"Failed to delete {path}: {e}")
        
        return {
            'status': 'success',
            'deleted_files': deleted_files,
            'freed_space_mb': freed_space / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Temp cleanup failed: {str(e)}", exc_info=True)
        raise


@shared_task(name='src.tasks.maintenance_tasks.backup_database')
def backup_database() -> Dict[str, Any]:
    """
    Perform database backup
    
    Returns:
        Dictionary with backup information
    """
    try:
        logger.info("Starting database backup")
        
        # Load backup configuration
        import yaml
        config_path = Path('config/backup.yaml')
        
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            # Use default configuration
            config = {
                'backup': {
                    'local': {'enabled': True, 'path': 'data/backups'},
                    'retention': {'daily': 7, 'weekly': 4, 'monthly': 12},
                    'compression': True
                }
            }
        
        # Perform backup
        backup_manager = BackupManager(config)
        backup_path = backup_manager.backup_database()
        
        return {
            'status': 'success',
            'backup_path': backup_path,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database backup failed: {str(e)}", exc_info=True)
        raise


@shared_task(name='src.tasks.maintenance_tasks.monitor_system_health')
def monitor_system_health() -> Dict[str, Any]:
    """
    Monitor system health and resource usage
    
    Returns:
        Dictionary with health metrics
    """
    try:
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get database stats
        db = DatabaseManager()
        with db.get_session() as session:
            from src.database.models import AnalysisRun
            
            total_runs = session.query(AnalysisRun).count()
            running_runs = session.query(AnalysisRun).filter_by(status='running').count()
            failed_runs = session.query(AnalysisRun).filter_by(status='failed').count()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024 ** 3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 ** 3)
            },
            'database': {
                'total_runs': total_runs,
                'running_runs': running_runs,
                'failed_runs': failed_runs
            }
        }
        
        # Log warnings for high resource usage
        if cpu_percent > 90:
            logger.warning(f"High CPU usage: {cpu_percent}%")
        if memory.percent > 90:
            logger.warning(f"High memory usage: {memory.percent}%")
        if disk.percent > 90:
            logger.warning(f"High disk usage: {disk.percent}%")
        
        return {
            'status': 'success',
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Health monitoring failed: {str(e)}", exc_info=True)
        raise


@shared_task(name='src.tasks.maintenance_tasks.cleanup_failed_tasks')
def cleanup_failed_tasks() -> Dict[str, Any]:
    """
    Clean up failed task records
    
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        logger.info("Cleaning up failed tasks")
        
        db = DatabaseManager()
        cutoff_date = datetime.now() - timedelta(days=7)
        
        cleaned_runs = 0
        cleaned_jobs = 0
        
        with db.get_session() as session:
            from src.database.models import AnalysisRun, DownloadJob
            
            # Clean failed analysis runs
            failed_runs = session.query(AnalysisRun).filter(
                AnalysisRun.status == 'failed',
                AnalysisRun.updated_at < cutoff_date
            ).all()
            
            for run in failed_runs:
                session.delete(run)
                cleaned_runs += 1
            
            # Clean failed download jobs
            failed_jobs = session.query(DownloadJob).filter(
                DownloadJob.status == 'failed',
                DownloadJob.updated_at < cutoff_date
            ).all()
            
            for job in failed_jobs:
                session.delete(job)
                cleaned_jobs += 1
            
            session.commit()
        
        return {
            'status': 'success',
            'cleaned_runs': cleaned_runs,
            'cleaned_jobs': cleaned_jobs
        }
        
    except Exception as e:
        logger.error(f"Failed task cleanup error: {str(e)}", exc_info=True)
        raise


@shared_task(name='src.tasks.maintenance_tasks.optimize_database')
def optimize_database() -> Dict[str, Any]:
    """
    Optimize database performance
    
    Returns:
        Dictionary with optimization results
    """
    try:
        logger.info("Optimizing database")
        
        db = DatabaseManager()
        
        # For PostgreSQL: VACUUM and ANALYZE
        if os.getenv('DB_TYPE') == 'postgresql':
            with db.engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.execute("VACUUM ANALYZE")
                logger.info("Executed VACUUM ANALYZE")
        
        # For SQLite: VACUUM
        elif os.getenv('DB_TYPE') == 'sqlite':
            with db.engine.connect() as conn:
                conn.execute("VACUUM")
                logger.info("Executed VACUUM")
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database optimization failed: {str(e)}", exc_info=True)
        raise
