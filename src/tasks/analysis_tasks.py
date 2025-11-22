"""
Analysis tasks for background processing

This module contains Celery tasks for running analysis workflows.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from celery import shared_task
from datetime import datetime

from src.database.manager import DatabaseManager
from src.security.validators import FileValidator, InputSanitizer


logger = logging.getLogger(__name__)


@shared_task(bind=True, name='src.tasks.analysis_tasks.run_analysis')
def run_analysis(
    self,
    dataset_path: str,
    analysis_type: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run eDNA analysis workflow
    
    Args:
        self: Task instance
        dataset_path: Path to dataset file
        analysis_type: Type of analysis to run
        parameters: Analysis parameters
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Update progress
        self.update_progress(0, 100, 'STARTED')
        
        # Validate inputs
        validator = FileValidator()
        sanitizer = InputSanitizer()
        
        if not validator.is_safe_path(dataset_path):
            raise ValueError(f"Unsafe path: {dataset_path}")
        
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        logger.info(f"Starting {analysis_type} analysis on {dataset_path}")
        
        # Initialize database
        db = DatabaseManager()
        
        # Create analysis run record
        with db.get_session() as session:
            from src.database.models import AnalysisRun
            
            run = AnalysisRun(
                name=f"{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dataset_path=dataset_path,
                analysis_type=analysis_type,
                status='running',
                parameters=parameters or {},
                celery_task_id=self.request.id
            )
            session.add(run)
            session.commit()
            run_id = run.id
        
        self.update_progress(10, 100, 'Preprocessing data')
        
        # Load and preprocess data
        # TODO: Import and use actual analysis modules
        # from src.preprocessing.sequence_processor import preprocess_sequences
        # preprocessed_data = preprocess_sequences(dataset_path)
        
        self.update_progress(30, 100, 'Running analysis')
        
        # Run analysis based on type
        results = {}
        
        if analysis_type == 'taxonomic':
            # TODO: Import and use actual taxonomic analysis
            # from src.analysis.taxonomy import run_taxonomic_analysis
            # results = run_taxonomic_analysis(preprocessed_data, parameters)
            results = {'status': 'completed', 'message': 'Taxonomic analysis placeholder'}
            
        elif analysis_type == 'novelty':
            # TODO: Import and use actual novelty detection
            # from src.novelty.detection import detect_novel_sequences
            # results = detect_novel_sequences(preprocessed_data, parameters)
            results = {'status': 'completed', 'message': 'Novelty detection placeholder'}
            
        elif analysis_type == 'clustering':
            # TODO: Import and use actual clustering
            # from src.clustering.sequence_clustering import cluster_sequences
            # results = cluster_sequences(preprocessed_data, parameters)
            results = {'status': 'completed', 'message': 'Clustering placeholder'}
        
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        self.update_progress(80, 100, 'Saving results')
        
        # Save results
        output_dir = Path('analysis_outputs') / f"run_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update database
        with db.get_session() as session:
            run = session.query(AnalysisRun).filter_by(id=run_id).first()
            if run:
                run.status = 'completed'
                run.results = results
                run.output_path = str(output_dir)
                run.completed_at = datetime.now()
                session.commit()
        
        self.update_progress(100, 100, 'Completed')
        
        logger.info(f"Analysis completed: {run_id}")
        
        return {
            'status': 'success',
            'run_id': run_id,
            'results': results,
            'output_path': str(output_dir)
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        
        # Update database with failure
        if 'run_id' in locals():
            with db.get_session() as session:
                run = session.query(AnalysisRun).filter_by(id=run_id).first()
                if run:
                    run.status = 'failed'
                    run.error_message = str(e)
                    session.commit()
        
        raise


@shared_task(bind=True, name='src.tasks.analysis_tasks.run_blast_search')
def run_blast_search(
    self,
    query_path: str,
    database_path: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run BLAST search against reference database
    
    Args:
        self: Task instance
        query_path: Path to query sequences
        database_path: Path to BLAST database
        parameters: BLAST parameters
    
    Returns:
        Dictionary with BLAST results
    """
    try:
        self.update_progress(0, 100, 'STARTED')
        
        logger.info(f"Running BLAST search: {query_path} vs {database_path}")
        
        # Validate paths
        validator = FileValidator()
        if not validator.is_safe_path(query_path):
            raise ValueError(f"Unsafe query path: {query_path}")
        if not validator.is_safe_path(database_path):
            raise ValueError(f"Unsafe database path: {database_path}")
        
        self.update_progress(20, 100, 'Running BLAST')
        
        # TODO: Import and use actual BLAST implementation
        # from src.similarity.blast_search import run_blast
        # results = run_blast(query_path, database_path, parameters)
        
        results = {
            'query': query_path,
            'database': database_path,
            'hits': [],
            'message': 'BLAST search placeholder'
        }
        
        self.update_progress(100, 100, 'Completed')
        
        return {
            'status': 'success',
            'results': results
        }
        
    except Exception as e:
        logger.error(f"BLAST search failed: {str(e)}", exc_info=True)
        raise


@shared_task(bind=True, name='src.tasks.analysis_tasks.run_multiple_analyses')
def run_multiple_analyses(
    self,
    dataset_paths: list,
    analysis_type: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run analysis on multiple datasets in parallel
    
    Args:
        self: Task instance
        dataset_paths: List of dataset paths
        analysis_type: Type of analysis
        parameters: Analysis parameters
    
    Returns:
        Dictionary with all results
    """
    try:
        total = len(dataset_paths)
        self.update_progress(0, total, f'Processing {total} datasets')
        
        results = []
        
        for idx, dataset_path in enumerate(dataset_paths):
            # Submit individual analysis task
            result = run_analysis.delay(dataset_path, analysis_type, parameters)
            results.append({
                'dataset': dataset_path,
                'task_id': result.id
            })
            
            self.update_progress(idx + 1, total, f'Submitted {idx + 1}/{total}')
        
        return {
            'status': 'success',
            'total': total,
            'tasks': results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}", exc_info=True)
        raise
