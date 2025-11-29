"""
Model Registry System
Tracks model versions, lineage, and performance across datasets
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Tracks model versions and their evolution over time
    
    Features:
    - Version tracking with semantic versioning
    - Dataset lineage (which datasets contributed to each version)
    - Performance metrics across versions
    - Model provenance and reproducibility
    - Comparison between versions
    """
    
    def __init__(self, registry_dir: str, backend: str = 'json'):
        """
        Initialize model registry
        
        Args:
            registry_dir: Directory to store registry data
            backend: Storage backend ('json' or 'sqlite')
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.backend = backend
        
        if backend == 'sqlite':
            self.db_path = self.registry_dir / 'registry.db'
            self._init_database()
        else:
            self.json_path = self.registry_dir / 'registry.json'
            self.registry_data = self._load_json_registry()
        
        logger.info(f"ModelRegistry initialized with {backend} backend at {self.registry_dir}")
    
    def register_model(self,
                      version: str,
                      model_path: str,
                      checkpoint_path: Optional[str] = None,
                      base_model: str = 'zhihan1996/DNABERT-2-117M',  # OPTIMIZED: k-mer=4, proj_dim=64, exemplars=50
                      parent_version: Optional[str] = None,
                      datasets: Optional[List[str]] = None,
                      metrics: Optional[Dict[str, float]] = None,
                      config: Optional[Dict[str, Any]] = None,
                      description: str = "") -> str:
        """
        Register a new model version
        
        Args:
            version: Version identifier (e.g., 'v1.0.0', 'marine_v2')
            model_path: Path to saved model
            checkpoint_path: Path to training checkpoint
            base_model: Base model identifier
            parent_version: Previous version this was derived from
            datasets: List of datasets used for training
            metrics: Performance metrics
            config: Training configuration
            description: Human-readable description
            
        Returns:
            Registered version ID
        """
        model_info = {
            'version': version,
            'model_path': str(model_path),
            'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
            'base_model': base_model,
            'parent_version': parent_version,
            'datasets': datasets or [],
            'metrics': metrics or {},
            'config': config or {},
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        if self.backend == 'sqlite':
            self._register_model_sqlite(model_info)
        else:
            self._register_model_json(model_info)
        
        logger.info(f"Registered model version: {version}")
        return version
    
    def get_model(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get model information by version
        
        Args:
            version: Version identifier
            
        Returns:
            Model information dictionary
        """
        if self.backend == 'sqlite':
            return self._get_model_sqlite(version)
        else:
            return self.registry_data.get('models', {}).get(version)
    
    def list_models(self, 
                   status: Optional[str] = None,
                   dataset: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered models
        
        Args:
            status: Filter by status ('active', 'archived', 'deprecated')
            dataset: Filter by dataset name
            
        Returns:
            List of model information dictionaries
        """
        if self.backend == 'sqlite':
            return self._list_models_sqlite(status, dataset)
        else:
            models = list(self.registry_data.get('models', {}).values())
            
            if status:
                models = [m for m in models if m.get('status') == status]
            
            if dataset:
                models = [m for m in models if dataset in m.get('datasets', [])]
            
            return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def get_lineage(self, version: str) -> List[Dict[str, Any]]:
        """
        Get the full lineage (ancestry) of a model version
        
        Args:
            version: Version to trace
            
        Returns:
            List of model versions from oldest ancestor to current
        """
        lineage = []
        current_version = version
        
        while current_version:
            model_info = self.get_model(current_version)
            if not model_info:
                break
            
            lineage.insert(0, model_info)
            current_version = model_info.get('parent_version')
        
        return lineage
    
    def get_children(self, version: str) -> List[Dict[str, Any]]:
        """
        Get all direct children of a model version
        
        Args:
            version: Parent version
            
        Returns:
            List of child model versions
        """
        all_models = self.list_models()
        children = [m for m in all_models if m.get('parent_version') == version]
        return sorted(children, key=lambda x: x.get('created_at', ''))
    
    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            Comparison dictionary
        """
        model1 = self.get_model(version1)
        model2 = self.get_model(version2)
        
        if not model1 or not model2:
            raise ValueError(f"One or both versions not found: {version1}, {version2}")
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'metric_differences': {},
            'dataset_differences': {
                'only_in_v1': list(set(model1.get('datasets', [])) - set(model2.get('datasets', []))),
                'only_in_v2': list(set(model2.get('datasets', [])) - set(model1.get('datasets', []))),
                'common': list(set(model1.get('datasets', [])) & set(model2.get('datasets', [])))
            },
            'created_at_diff_days': self._date_diff(model1.get('created_at'), model2.get('created_at'))
        }
        
        # Compare metrics
        metrics1 = model1.get('metrics', {})
        metrics2 = model2.get('metrics', {})
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            v1_val = metrics1.get(metric)
            v2_val = metrics2.get(metric)
            
            if v1_val is not None and v2_val is not None:
                comparison['metric_differences'][metric] = {
                    'v1': v1_val,
                    'v2': v2_val,
                    'change': v2_val - v1_val,
                    'percent_change': ((v2_val - v1_val) / v1_val * 100) if v1_val != 0 else 0
                }
        
        return comparison
    
    def get_best_model(self, metric: str = 'val_loss', minimize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the best model based on a specific metric
        
        Args:
            metric: Metric to compare
            minimize: True if lower is better
            
        Returns:
            Best model information
        """
        models = self.list_models(status='active')
        models_with_metric = [m for m in models if metric in m.get('metrics', {})]
        
        if not models_with_metric:
            return None
        
        if minimize:
            best = min(models_with_metric, key=lambda x: x['metrics'][metric])
        else:
            best = max(models_with_metric, key=lambda x: x['metrics'][metric])
        
        return best
    
    def update_model_status(self, version: str, status: str):
        """
        Update model status (active, archived, deprecated)
        
        Args:
            version: Model version
            status: New status
        """
        if self.backend == 'sqlite':
            self._update_status_sqlite(version, status)
        else:
            if version in self.registry_data.get('models', {}):
                self.registry_data['models'][version]['status'] = status
                self._save_json_registry()
        
        logger.info(f"Updated {version} status to: {status}")
    
    # JSON backend methods
    def _load_json_registry(self) -> Dict[str, Any]:
        """Load registry from JSON file"""
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                return json.load(f)
        return {'models': {}, 'metadata': {}}
    
    def _save_json_registry(self):
        """Save registry to JSON file"""
        with open(self.json_path, 'w') as f:
            json.dump(self.registry_data, f, indent=2)
    
    def _register_model_json(self, model_info: Dict[str, Any]):
        """Register model in JSON backend"""
        version = model_info['version']
        self.registry_data['models'][version] = model_info
        self._save_json_registry()
    
    # SQLite backend methods
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    version TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    checkpoint_path TEXT,
                    base_model TEXT,
                    parent_version TEXT,
                    description TEXT,
                    created_at TEXT,
                    status TEXT DEFAULT 'active',
                    config_json TEXT,
                    metrics_json TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    dataset_name TEXT,
                    FOREIGN KEY (model_version) REFERENCES models(version)
                )
            ''')
            
            conn.commit()
    
    def _register_model_sqlite(self, model_info: Dict[str, Any]):
        """Register model in SQLite backend"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO models 
                (version, model_path, checkpoint_path, base_model, parent_version, 
                 description, created_at, status, config_json, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_info['version'],
                model_info['model_path'],
                model_info.get('checkpoint_path'),
                model_info['base_model'],
                model_info.get('parent_version'),
                model_info['description'],
                model_info['created_at'],
                model_info['status'],
                json.dumps(model_info.get('config', {})),
                json.dumps(model_info.get('metrics', {}))
            ))
            
            # Insert datasets
            for dataset in model_info.get('datasets', []):
                conn.execute('''
                    INSERT INTO datasets (model_version, dataset_name)
                    VALUES (?, ?)
                ''', (model_info['version'], dataset))
            
            conn.commit()
    
    def _get_model_sqlite(self, version: str) -> Optional[Dict[str, Any]]:
        """Get model from SQLite backend"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM models WHERE version = ?
            ''', (version,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            model_info = dict(row)
            model_info['config'] = json.loads(model_info.pop('config_json'))
            model_info['metrics'] = json.loads(model_info.pop('metrics_json'))
            
            # Get datasets
            cursor = conn.execute('''
                SELECT dataset_name FROM datasets WHERE model_version = ?
            ''', (version,))
            model_info['datasets'] = [r[0] for r in cursor.fetchall()]
            
            return model_info
    
    def _list_models_sqlite(self, status: Optional[str], dataset: Optional[str]) -> List[Dict[str, Any]]:
        """List models from SQLite backend"""
        query = 'SELECT version FROM models'
        params = []
        
        if status:
            query += ' WHERE status = ?'
            params.append(status)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            versions = [row[0] for row in cursor.fetchall()]
        
        models = [self._get_model_sqlite(v) for v in versions]
        
        if dataset:
            models = [m for m in models if dataset in m.get('datasets', [])]
        
        return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def _update_status_sqlite(self, version: str, status: str):
        """Update model status in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE models SET status = ? WHERE version = ?
            ''', (status, version))
            conn.commit()
    
    @staticmethod
    def _date_diff(date1_str: str, date2_str: str) -> float:
        """Calculate difference in days between two ISO dates"""
        try:
            date1 = datetime.fromisoformat(date1_str)
            date2 = datetime.fromisoformat(date2_str)
            return abs((date2 - date1).days)
        except:
            return 0
