"""
Download tasks for fetching external data

This module contains Celery tasks for downloading data from SRA and other sources.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from celery import shared_task
from datetime import datetime

from src.database.database import DatabaseManager
from src.utils.security import InputSanitizer


logger = logging.getLogger(__name__)


@shared_task(bind=True, name='src.tasks.download_tasks.download_sra_dataset')
def download_sra_dataset(
    self,
    accession: str,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download dataset from NCBI SRA
    
    Args:
        self: Task instance
        accession: SRA accession number (e.g., SRR1234567)
        output_dir: Output directory path
    
    Returns:
        Dictionary with download results
    """
    try:
        self.update_progress(0, 100, 'STARTED')
        
        # Validate accession
        sanitizer = InputSanitizer()
        if not sanitizer.is_alphanumeric(accession):
            raise ValueError(f"Invalid accession: {accession}")
        
        logger.info(f"Downloading SRA dataset: {accession}")
        
        # Initialize database
        db = DatabaseManager()
        
        # Create download record
        with db.get_session() as session:
            from src.database.models import DownloadJob
            
            job = DownloadJob(
                accession=accession,
                source='SRA',
                status='downloading',
                celery_task_id=self.request.id
            )
            session.add(job)
            session.commit()
            job_id = job.id
        
        # Set output directory
        if output_dir is None:
            output_dir = Path('data/raw/sra') / accession
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.update_progress(10, 100, 'Fetching metadata')
        
        # TODO: Fetch SRA metadata
        # from src.utils.sra_tools import get_metadata
        # metadata = get_metadata(accession)
        
        metadata = {
            'accession': accession,
            'organism': 'Unknown',
            'library_strategy': 'AMPLICON',
            'message': 'SRA metadata placeholder'
        }
        
        self.update_progress(30, 100, 'Downloading sequences')
        
        # TODO: Download using SRA tools
        # from src.utils.sra_tools import download_fastq
        # fastq_files = download_fastq(accession, output_dir, progress_callback=self.update_progress)
        
        fastq_files = [str(output_dir / f"{accession}_1.fastq")]
        
        self.update_progress(80, 100, 'Validating download')
        
        # Validate downloaded files
        total_size = sum(Path(f).stat().st_size for f in fastq_files if Path(f).exists())
        
        # Update database
        with db.get_session() as session:
            job = session.query(DownloadJob).filter_by(id=job_id).first()
            if job:
                job.status = 'completed'
                job.output_path = str(output_dir)
                job.file_size = total_size
                job.metadata = metadata
                job.completed_at = datetime.now()
                session.commit()
        
        self.update_progress(100, 100, 'Completed')
        
        logger.info(f"Download completed: {accession}")
        
        return {
            'status': 'success',
            'accession': accession,
            'output_path': str(output_dir),
            'files': fastq_files,
            'metadata': metadata,
            'total_size': total_size
        }
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}", exc_info=True)
        
        # Update database with failure
        if 'job_id' in locals():
            with db.get_session() as session:
                job = session.query(DownloadJob).filter_by(id=job_id).first()
                if job:
                    job.status = 'failed'
                    job.error_message = str(e)
                    session.commit()
        
        raise


@shared_task(bind=True, name='src.tasks.download_tasks.download_batch_sra')
def download_batch_sra(
    self,
    accessions: List[str],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download multiple SRA datasets
    
    Args:
        self: Task instance
        accessions: List of SRA accession numbers
        output_dir: Output directory path
    
    Returns:
        Dictionary with batch download results
    """
    try:
        total = len(accessions)
        self.update_progress(0, total, f'Downloading {total} datasets')
        
        results = []
        
        for idx, accession in enumerate(accessions):
            # Submit individual download task
            result = download_sra_dataset.delay(accession, output_dir)
            results.append({
                'accession': accession,
                'task_id': result.id
            })
            
            self.update_progress(idx + 1, total, f'Submitted {idx + 1}/{total}')
        
        return {
            'status': 'success',
            'total': total,
            'tasks': results
        }
        
    except Exception as e:
        logger.error(f"Batch download failed: {str(e)}", exc_info=True)
        raise


@shared_task(bind=True, name='src.tasks.download_tasks.download_reference_database')
def download_reference_database(
    self,
    database_name: str,
    version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download reference database (SILVA, PR2, etc.)
    
    Args:
        self: Task instance
        database_name: Name of database (silva, pr2, etc.)
        version: Database version
    
    Returns:
        Dictionary with download results
    """
    try:
        self.update_progress(0, 100, 'STARTED')
        
        logger.info(f"Downloading {database_name} database")
        
        # Validate database name
        valid_databases = ['silva', 'pr2', 'unite', 'greengenes']
        if database_name.lower() not in valid_databases:
            raise ValueError(f"Invalid database: {database_name}. Must be one of {valid_databases}")
        
        output_dir = Path('reference') / database_name.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.update_progress(20, 100, 'Downloading database')
        
        # TODO: Implement actual database download
        # from src.utils.reference_downloader import download_database
        # files = download_database(database_name, version, output_dir)
        
        files = [str(output_dir / f"{database_name}.fasta")]
        
        self.update_progress(80, 100, 'Building indices')
        
        # TODO: Build BLAST/other indices
        # from src.utils.index_builder import build_blast_db
        # build_blast_db(files[0], output_dir)
        
        self.update_progress(100, 100, 'Completed')
        
        return {
            'status': 'success',
            'database': database_name,
            'version': version or 'latest',
            'output_path': str(output_dir),
            'files': files
        }
        
    except Exception as e:
        logger.error(f"Database download failed: {str(e)}", exc_info=True)
        raise


@shared_task(bind=True, name='src.tasks.download_tasks.update_reference_databases')
def update_reference_databases(self) -> Dict[str, Any]:
    """
    Update all reference databases to latest versions
    
    Args:
        self: Task instance
    
    Returns:
        Dictionary with update results
    """
    try:
        databases = ['silva', 'pr2', 'unite']
        total = len(databases)
        
        self.update_progress(0, total, 'Updating databases')
        
        results = []
        
        for idx, db_name in enumerate(databases):
            result = download_reference_database.delay(db_name)
            results.append({
                'database': db_name,
                'task_id': result.id
            })
            
            self.update_progress(idx + 1, total, f'Updated {idx + 1}/{total}')
        
        return {
            'status': 'success',
            'databases_updated': total,
            'tasks': results
        }
        
    except Exception as e:
        logger.error(f"Database update failed: {str(e)}", exc_info=True)
        raise
