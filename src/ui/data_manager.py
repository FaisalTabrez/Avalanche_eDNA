"""
Centralized Data Manager for Streamlit UI

Provides unified access to datasets, runs, and results across all pages.
Eliminates redundant directory navigation and provides caching for performance.
"""
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from dataclasses import dataclass
from src.utils.config import config as app_config


@dataclass
class RunInfo:
    """Metadata for a single run"""
    dataset: str
    run_id: str
    path: Path
    modified: str
    mtime: float
    has_pipeline: bool
    has_taxonomy: bool
    has_novelty: bool
    has_clustering: bool
    
    def to_dict(self):
        return {
            'dataset': self.dataset,
            'run_id': self.run_id,
            'path': str(self.path),
            'modified': self.modified,
            'mtime': self.mtime,
            'has_pipeline': self.has_pipeline,
            'has_taxonomy': self.has_taxonomy,
            'has_novelty': self.has_novelty,
            'has_clustering': self.has_clustering,
        }


class DataManager:
    """
    Centralized manager for datasets, runs, and results.
    
    Features:
    - Automatic discovery of runs and datasets
    - Caching for performance
    - Unified path resolution
    - Session state integration
    """
    
    def __init__(self):
        self.runs_root = Path(app_config.get('storage.runs_dir', 'consolidated_data/runs'))
        self.datasets_root = Path(app_config.get('storage.datasets_dir', 'consolidated_data/datasets'))
        self.results_root = Path(app_config.get('storage.results_dir', 'consolidated_data/results'))
        
    def get_current_run(self) -> Optional[Path]:
        """Get the currently selected run path from session state"""
        return st.session_state.get('current_run_path')
    
    def set_current_run(self, run_path: Path):
        """Set the current run and update all relevant session state"""
        st.session_state.current_run_path = run_path
        st.session_state.prefill_results_dir = str(run_path.resolve())
        
    def clear_current_run(self):
        """Clear current run selection"""
        if 'current_run_path' in st.session_state:
            del st.session_state.current_run_path
        if 'prefill_results_dir' in st.session_state:
            del st.session_state.prefill_results_dir
    
    @st.cache_data(ttl=60)
    def discover_runs(_self, force_refresh: bool = False) -> List[RunInfo]:
        """
        Discover all runs under the runs root directory.
        
        Structure: runs_root/dataset_name/run_timestamp/
        
        Returns:
            List of RunInfo objects sorted by modification time (newest first)
        """
        runs = []
        
        if not _self.runs_root.exists():
            return runs
        
        try:
            # Iterate through dataset folders
            for dataset_dir in _self.runs_root.iterdir():
                if not dataset_dir.is_dir():
                    continue
                    
                dataset_name = dataset_dir.name
                
                # Iterate through run folders
                for run_dir in dataset_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    
                    try:
                        # Get modification time
                        mtime = run_dir.stat().st_mtime
                        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                        
                        # Check for key files
                        has_pipeline = (run_dir / 'pipeline_results.json').exists()
                        has_taxonomy = (run_dir / 'taxonomy' / 'taxonomy_predictions.csv').exists()
                        has_novelty = (run_dir / 'novelty' / 'novelty_analysis.json').exists()
                        has_clustering = (run_dir / 'clustering' / 'cluster_assignments.csv').exists()
                        
                        runs.append(RunInfo(
                            dataset=dataset_name,
                            run_id=run_dir.name,
                            path=run_dir,
                            modified=mtime_str,
                            mtime=mtime,
                            has_pipeline=has_pipeline,
                            has_taxonomy=has_taxonomy,
                            has_novelty=has_novelty,
                            has_clustering=has_clustering,
                        ))
                    except Exception as e:
                        # Skip problematic runs
                        continue
                        
        except Exception as e:
            st.error(f"Error discovering runs: {e}")
            
        # Sort by modification time (newest first)
        runs.sort(key=lambda r: r.mtime, reverse=True)
        return runs
    
    def get_datasets(self) -> List[str]:
        """Get list of all dataset names"""
        runs = self.discover_runs()
        return sorted(list(set(r.dataset for r in runs)))
    
    def get_runs_for_dataset(self, dataset_name: str) -> List[RunInfo]:
        """Get all runs for a specific dataset"""
        runs = self.discover_runs()
        return [r for r in runs if r.dataset == dataset_name]
    
    def get_recent_runs(self, limit: int = 5) -> List[RunInfo]:
        """Get N most recent runs across all datasets"""
        runs = self.discover_runs()
        return runs[:limit]
    
    def search_runs(self, query: str, dataset_filter: Optional[str] = None) -> List[RunInfo]:
        """
        Search runs by dataset name or run ID.
        
        Args:
            query: Search string
            dataset_filter: Optional dataset name to filter by
            
        Returns:
            Filtered list of RunInfo objects
        """
        runs = self.discover_runs()
        
        # Apply dataset filter
        if dataset_filter and dataset_filter != "All":
            runs = [r for r in runs if r.dataset == dataset_filter]
        
        # Apply search query
        if query:
            query_lower = query.lower()
            runs = [r for r in runs if 
                   query_lower in r.dataset.lower() or 
                   query_lower in r.run_id.lower()]
        
        return runs
    
    def get_run_by_path(self, path: str) -> Optional[RunInfo]:
        """Get RunInfo for a specific path"""
        target = Path(path).resolve()
        runs = self.discover_runs()
        for run in runs:
            if run.path.resolve() == target:
                return run
        return None
    
    def load_pipeline_results(self, run_path: Path) -> Optional[Dict]:
        """Load pipeline_results.json for a run"""
        results_file = run_path / 'pipeline_results.json'
        if not results_file.exists():
            return None
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading pipeline results: {e}")
            return None
    
    def get_run_files(self, run_path: Path) -> Dict[str, Path]:
        """
        Get all standard file paths for a run.
        
        Returns:
            Dictionary mapping file types to their paths (if they exist)
        """
        files = {}
        
        # Top-level files
        if (run_path / 'pipeline_results.json').exists():
            files['pipeline_results'] = run_path / 'pipeline_results.json'
        if (run_path / 'analysis_report.txt').exists():
            files['analysis_report'] = run_path / 'analysis_report.txt'
        
        # Embeddings (check both .npy and .npz)
        if (run_path / 'embeddings.npy').exists():
            files['embeddings'] = run_path / 'embeddings.npy'
        elif (run_path / 'embeddings.npz').exists():
            files['embeddings'] = run_path / 'embeddings.npz'
        
        # Clustering files
        clustering_dir = run_path / 'clustering'
        if clustering_dir.exists():
            if (clustering_dir / 'cluster_visualization.png').exists():
                files['cluster_viz'] = clustering_dir / 'cluster_visualization.png'
            if (clustering_dir / 'cluster_stats.txt').exists():
                files['cluster_stats'] = clustering_dir / 'cluster_stats.txt'
            if (clustering_dir / 'cluster_assignments.csv').exists():
                files['cluster_assignments'] = clustering_dir / 'cluster_assignments.csv'
        
        # Taxonomy files
        taxonomy_dir = run_path / 'taxonomy'
        if taxonomy_dir.exists():
            if (taxonomy_dir / 'taxonomy_predictions.csv').exists():
                files['taxonomy_predictions'] = taxonomy_dir / 'taxonomy_predictions.csv'
            if (taxonomy_dir / 'taxonomy_tiebreak_report.csv').exists():
                files['taxonomy_tiebreak'] = taxonomy_dir / 'taxonomy_tiebreak_report.csv'
        
        # Novelty files
        novelty_dir = run_path / 'novelty'
        if novelty_dir.exists():
            if (novelty_dir / 'novelty_analysis.json').exists():
                files['novelty_analysis'] = novelty_dir / 'novelty_analysis.json'
            if (novelty_dir / 'novelty_candidates.txt').exists():
                files['novelty_candidates'] = novelty_dir / 'novelty_candidates.txt'
        
        # Visualization files
        viz_dir = run_path / 'visualizations'
        if viz_dir.exists():
            for viz_file in viz_dir.glob('*.png'):
                files[f'viz_{viz_file.stem}'] = viz_file
            for viz_file in viz_dir.glob('*.html'):
                files[f'viz_{viz_file.stem}'] = viz_file
        
        return files
    
    def search_across_embeddings(
        self,
        query_run: Path,
        query_seq_idx: int,
        top_k: int = 10,
        exclude_same_run: bool = True
    ) -> List[Dict]:
        """
        Search for similar sequences across all runs using embeddings.
        
        Args:
            query_run: Path to run containing query sequence
            query_seq_idx: Index of query sequence in that run
            top_k: Number of results to return
            exclude_same_run: Exclude results from the same run
            
        Returns:
            List of dicts with keys: dataset, run_id, seq_idx, distance, path
        """
        try:
            import numpy as np
            
            # Load query embedding
            query_emb_file = query_run / 'embeddings.npy'
            if not query_emb_file.exists():
                query_emb_file = query_run / 'embeddings.npz'
            
            if query_emb_file.suffix == '.npy':
                query_emb = np.load(query_emb_file, mmap_mode='r')[query_seq_idx]
            else:
                with np.load(query_emb_file) as data:
                    query_emb = data['embeddings'][query_seq_idx]
            
            # Search across all runs
            results = []
            runs = self.discover_runs()
            
            for run in runs:
                # Skip same run if requested
                if exclude_same_run and run.path == query_run:
                    continue
                
                # Load run embeddings
                emb_file = run.path / 'embeddings.npy'
                if not emb_file.exists():
                    emb_file = run.path / 'embeddings.npz'
                if not emb_file.exists():
                    continue
                
                if emb_file.suffix == '.npy':
                    run_emb = np.load(emb_file, mmap_mode='r')
                else:
                    with np.load(emb_file) as data:
                        run_emb = data['embeddings']
                
                # Compute distances
                distances = np.linalg.norm(run_emb - query_emb, axis=1)
                
                # Add all sequences from this run
                for seq_idx, dist in enumerate(distances):
                    results.append({
                        'dataset': run.dataset,
                        'run_id': run.run_id,
                        'seq_idx': seq_idx,
                        'distance': float(dist),
                        'path': run.path
                    })
            
            # Sort by distance and return top-k
            results.sort(key=lambda x: x['distance'])
            return results[:top_k]
            
        except Exception as e:
            st.error(f"Error searching embeddings: {e}")
            return []


# Singleton instance
_data_manager = None

def get_data_manager() -> DataManager:
    """Get or create the singleton DataManager instance"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager
