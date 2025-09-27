"""
Analysis report storage and cataloguing system.

This module provides comprehensive functionality for storing, organizing, and
managing eDNA analysis reports with automatic cataloguing and metadata extraction.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import gzip

from src.database.manager import DatabaseManager
from src.database.models import (
    AnalysisReport, DatasetInfo, OrganismProfile, AnalysisStatus, SequenceType
)

logger = logging.getLogger(__name__)


class ReportCatalogueManager:
    """
    Manages the storage and cataloguing of analysis reports with automatic organization.
    """
    
    def __init__(self, 
                 storage_root: Optional[str] = None,
                 db_manager: Optional[DatabaseManager] = None):
        """
        Initialize report catalogue manager.
        
        Args:
            storage_root: Root directory for storing reports and data
            db_manager: Database manager instance
        """
        if storage_root is None:
            storage_root = Path(__file__).parent.parent.parent / "data" / "report_storage"
        
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        self.db_manager = db_manager or DatabaseManager()
        
        # Create organized directory structure
        self._initialize_storage_structure()
        
        # Storage configuration
        self.compress_large_files = True
        self.compression_threshold_mb = 10
        self.auto_cleanup_days = 365
    
    def _initialize_storage_structure(self):
        """Initialize organized storage directory structure."""
        directories = [
            "reports", "datasets", "results", "visualizations", 
            "metadata", "exports", "backups", "temp"
        ]
        
        for directory in directories:
            (self.storage_root / directory).mkdir(exist_ok=True)
        
        # Create year/month subdirectories
        current_year = datetime.now().year
        for year in range(current_year - 2, current_year + 2):
            for month in range(1, 13):
                month_dir = self.storage_root / "reports" / str(year) / f"{month:02d}"
                month_dir.mkdir(parents=True, exist_ok=True)
    
    def store_analysis_report(self, 
                            dataset_file_path: str,
                            analysis_results: Dict[str, Any],
                            report_name: Optional[str] = None,
                            environmental_context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Store complete analysis report with automatic cataloguing.
        
        Args:
            dataset_file_path: Path to the original dataset file
            analysis_results: Complete analysis results dictionary
            report_name: Optional custom report name
            environmental_context: Optional environmental metadata
            
        Returns:
            Tuple of (report_id, storage_path)
        """
        logger.info(f"Storing analysis report for dataset: {dataset_file_path}")
        
        # Generate unique report ID
        report_id = self._generate_report_id(dataset_file_path, analysis_results)
        
        # Create dataset info
        dataset_info = self._create_dataset_info(dataset_file_path, environmental_context)
        
        # Store dataset info in database
        self.db_manager.store_dataset_info(dataset_info)
        
        # Create organized storage path
        storage_path = self._create_storage_path(report_id)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Store original dataset file
        dataset_storage_path = self._store_dataset_file(dataset_file_path, storage_path)
        
        # Create analysis report object
        analysis_report = self._create_analysis_report(
            report_id, dataset_info.dataset_id, analysis_results, report_name
        )
        
        # Store analysis report in database
        self.db_manager.store_analysis_report(analysis_report)
        
        # Store detailed results files
        self._store_result_files(analysis_results, storage_path)
        
        # Generate and store comprehensive report
        report_file_path = self._generate_comprehensive_report(
            analysis_report, dataset_info, analysis_results, storage_path
        )
        
        # Update report with file paths
        analysis_report.full_report_path = str(report_file_path)
        analysis_report.results_json_path = str(storage_path / "results.json")
        analysis_report.visualizations_dir = str(storage_path / "visualizations")
        
        # Update database with file paths
        self.db_manager.store_analysis_report(analysis_report)
        
        # Create metadata file
        self._create_metadata_file(analysis_report, dataset_info, storage_path)
        
        logger.info(f"Analysis report stored successfully: {report_id}")
        return report_id, str(storage_path)
    
    def retrieve_analysis_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete analysis report.
        
        Args:
            report_id: Unique report identifier
            
        Returns:
            Complete analysis report dictionary or None if not found
        """
        # Get report from database
        report = self.db_manager.get_analysis_report(report_id)
        if not report:
            return None
        
        # Load detailed results
        results_path = Path(report.results_json_path) if report.results_json_path else None
        detailed_results = {}
        
        if results_path and results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    detailed_results = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load detailed results: {str(e)}")
        
        # Combine database report with detailed results
        complete_report = report.to_dict()
        complete_report['detailed_results'] = detailed_results
        
        return complete_report
    
    def list_reports(self, 
                    dataset_id: Optional[str] = None,
                    date_range: Optional[Tuple[datetime, datetime]] = None,
                    analysis_type: Optional[str] = None,
                    limit: int = 100,
                    offset: int = 0) -> List[Dict[str, Any]]:
        """List analysis reports with filtering options."""
        try:
            with self.db_manager.get_connection() as conn:
                # Build dynamic query
                where_clauses = []
                params = []
                
                if dataset_id:
                    where_clauses.append("ar.dataset_id = ?")
                    params.append(dataset_id)
                
                if date_range:
                    where_clauses.append("ar.created_at BETWEEN ? AND ?")
                    params.extend([date_range[0].isoformat(), date_range[1].isoformat()])
                
                if analysis_type:
                    where_clauses.append("ar.analysis_type = ?")
                    params.append(analysis_type)
                
                # Construct query
                base_query = """
                    SELECT 
                        ar.report_id, ar.report_name, ar.analysis_type, ar.status,
                        ar.created_at, ar.processing_time_seconds, ar.shannon_diversity,
                        ar.novel_candidates_count, ds.dataset_name, ds.collection_location
                    FROM analysis_reports ar
                    JOIN datasets ds ON ar.dataset_id = ds.dataset_id
                """
                
                if where_clauses:
                    base_query += " WHERE " + " AND ".join(where_clauses)
                
                base_query += " ORDER BY ar.created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor = conn.execute(base_query, params)
                
                reports = []
                for row in cursor.fetchall():
                    reports.append({
                        'report_id': row[0],
                        'report_name': row[1],
                        'analysis_type': row[2],
                        'status': row[3],
                        'created_at': row[4],
                        'processing_time_seconds': row[5],
                        'shannon_diversity': row[6],
                        'novel_candidates_count': row[7],
                        'dataset_name': row[8],
                        'collection_location': row[9]
                    })
                
                return reports
                
        except Exception as e:
            logger.error(f"Failed to list reports: {str(e)}")
            return []
    
    def search_reports(self, 
                      query: str,
                      search_fields: List[str] = None,
                      limit: int = 50) -> List[Dict[str, Any]]:
        """Search reports by text query."""
        if search_fields is None:
            search_fields = ['report_name', 'dataset_name', 'collection_location']
        
        try:
            with self.db_manager.get_connection() as conn:
                # Build search query
                search_conditions = []
                params = []
                
                for field in search_fields:
                    if field in ['report_name']:
                        search_conditions.append("ar.report_name LIKE ?")
                    elif field in ['dataset_name', 'collection_location']:
                        search_conditions.append(f"ds.{field} LIKE ?")
                    
                    params.append(f"%{query}%")
                
                if not search_conditions:
                    return []
                
                search_query = f"""
                    SELECT 
                        ar.report_id, ar.report_name, ar.analysis_type, ar.created_at,
                        ds.dataset_name, ds.collection_location, ar.novel_candidates_count
                    FROM analysis_reports ar
                    JOIN datasets ds ON ar.dataset_id = ds.dataset_id
                    WHERE ({' OR '.join(search_conditions)})
                    ORDER BY ar.created_at DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor = conn.execute(search_query, params)
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'report_id': row[0],
                        'report_name': row[1],
                        'analysis_type': row[2],
                        'created_at': row[3],
                        'dataset_name': row[4],
                        'collection_location': row[5],
                        'novel_candidates_count': row[6]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search reports: {str(e)}")
            return []
    
    def export_report(self, report_id: str, export_format: str = "json") -> Optional[str]:
        """Export report in specified format."""
        report = self.retrieve_analysis_report(report_id)
        if not report:
            return None
        
        export_dir = self.storage_root / "exports"
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format.lower() == "json":
            export_path = export_dir / f"{report_id}_{timestamp}.json"
            try:
                with open(export_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                return str(export_path)
            except Exception as e:
                logger.error(f"Failed to export report as JSON: {str(e)}")
                return None
        
        return None
    
    # Helper methods for internal use
    def _generate_report_id(self, dataset_file_path: str, analysis_results: Dict[str, Any]) -> str:
        """Generate unique report ID."""
        dataset_name = Path(dataset_file_path).stem
        timestamp = datetime.now().isoformat()
        analysis_hash = hashlib.md5(json.dumps(analysis_results, sort_keys=True, default=str).encode()).hexdigest()[:8]
        
        combined = f"{dataset_name}_{timestamp}_{analysis_hash}"
        report_hash = hashlib.md5(combined.encode()).hexdigest()[:12].upper()
        
        return f"RPT_{report_hash}"
    
    def _create_dataset_info(self, dataset_file_path: str, environmental_context: Optional[Dict[str, Any]]) -> DatasetInfo:
        """Create DatasetInfo object from file and context."""
        file_path = Path(dataset_file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        dataset_id = DatasetInfo.generate_dataset_id(file_path.stem, str(file_path))
        
        dataset_info = DatasetInfo(
            dataset_id=dataset_id,
            dataset_name=file_path.stem,
            file_path=str(file_path),
            file_format="fasta",  # Default, could be auto-detected
            file_size_mb=file_size_mb
        )
        
        if environmental_context:
            dataset_info.collection_date = environmental_context.get('collection_date')
            dataset_info.collection_location = environmental_context.get('collection_location')
            dataset_info.depth_meters = environmental_context.get('depth_meters')
            dataset_info.temperature_celsius = environmental_context.get('temperature_celsius')
            dataset_info.ph_level = environmental_context.get('ph_level')
            dataset_info.salinity = environmental_context.get('salinity')
        
        return dataset_info
    
    def _create_storage_path(self, report_id: str) -> Path:
        """Create organized storage path for report."""
        now = datetime.now()
        year_month_path = self.storage_root / "reports" / str(now.year) / f"{now.month:02d}"
        return year_month_path / report_id
    
    def _store_dataset_file(self, dataset_file_path: str, storage_path: Path) -> Path:
        """Store original dataset file in organized storage."""
        source_path = Path(dataset_file_path)
        target_path = storage_path / "dataset" / source_path.name
        target_path.parent.mkdir(exist_ok=True)
        
        if source_path.stat().st_size > (self.compression_threshold_mb * 1024 * 1024) and self.compress_large_files:
            compressed_path = target_path.with_suffix(target_path.suffix + '.gz')
            with open(source_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return compressed_path
        else:
            shutil.copy2(source_path, target_path)
            return target_path
    
    def _create_analysis_report(self, report_id: str, dataset_id: str, 
                              analysis_results: Dict[str, Any], report_name: Optional[str]) -> AnalysisReport:
        """Create AnalysisReport object from results."""
        basic_stats = analysis_results.get('basic_stats', {})
        composition = analysis_results.get('composition', {})
        diversity = analysis_results.get('diversity', {})
        processing_info = analysis_results.get('processing_info', {})
        
        # Detect sequence type
        sequence_type = None
        if composition.get('sequence_type'):
            try:
                sequence_type = SequenceType(composition['sequence_type'].lower())
            except ValueError:
                sequence_type = SequenceType.UNKNOWN
        
        analysis_report = AnalysisReport(
            report_id=report_id,
            dataset_id=dataset_id,
            report_name=report_name or f"Analysis Report {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            analysis_type="comprehensive",
            status=AnalysisStatus.COMPLETED,
            processing_time_seconds=processing_info.get('total_time'),
            min_length=basic_stats.get('min_length'),
            max_length=basic_stats.get('max_length'),
            mean_length=basic_stats.get('mean_length'),
            median_length=basic_stats.get('median_length'),
            std_length=basic_stats.get('std_length'),
            sequence_type_detected=sequence_type,
            composition_data=composition,
            shannon_diversity=diversity.get('shannon_diversity'),
            simpson_diversity=diversity.get('simpson_diversity'),
            evenness=diversity.get('evenness'),
            species_richness=diversity.get('species_richness')
        )
        
        return analysis_report
    
    def _store_result_files(self, analysis_results: Dict[str, Any], storage_path: Path):
        """Store detailed analysis result files."""
        results_dir = storage_path / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Store main results as JSON
        with open(results_dir / "results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
    
    def _generate_comprehensive_report(self, analysis_report: AnalysisReport, 
                                     dataset_info: DatasetInfo, analysis_results: Dict[str, Any],
                                     storage_path: Path) -> Path:
        """Generate comprehensive text report."""
        report_path = storage_path / "comprehensive_report.md"
        
        lines = []
        lines.append(f"# Comprehensive eDNA Analysis Report")
        lines.append(f"**Report ID:** {analysis_report.report_id}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Dataset Information
        lines.append("## Dataset Information")
        lines.append(f"- **Dataset Name:** {dataset_info.dataset_name}")
        lines.append(f"- **File Size:** {dataset_info.file_size_mb:.2f} MB")
        lines.append("")
        
        # Analysis Summary
        lines.append("## Analysis Summary")
        lines.append(f"- **Processing Time:** {analysis_report.processing_time_seconds:.2f} seconds")
        if analysis_report.shannon_diversity:
            lines.append(f"- **Shannon Diversity:** {analysis_report.shannon_diversity:.4f}")
        lines.append("")
        
        # Write report file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return report_path
    
    def _create_metadata_file(self, analysis_report: AnalysisReport, 
                            dataset_info: DatasetInfo, storage_path: Path):
        """Create metadata file for the report."""
        metadata = {
            'report_metadata': analysis_report.to_dict(),
            'dataset_metadata': dataset_info.to_dict(),
            'storage_info': {
                'storage_path': str(storage_path),
                'created_at': datetime.now().isoformat(),
                'storage_version': '1.0'
            }
        }
        
        metadata_path = storage_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)