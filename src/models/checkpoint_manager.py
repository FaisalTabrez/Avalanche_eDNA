"""
Checkpoint Manager for Model Training
Handles saving, loading, and managing model checkpoints for continual learning
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints for training, fine-tuning, and continual learning
    
    Features:
    - Save/load full training state (model, optimizer, scheduler, epoch)
    - Track checkpoint metadata (datasets, metrics, timestamps)
    - Automatic best model tracking
    - Checkpoint versioning and lineage
    - Clean up old checkpoints
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5, keep_best: bool = True):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
            keep_best: Always keep the best checkpoint regardless of max_checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / 'checkpoints_metadata.json'
        self.metadata = self._load_metadata()
        
        logger.info(f"CheckpointManager initialized at {self.checkpoint_dir}")
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       scheduler: Optional[Any] = None,
                       dataset_info: Optional[Dict[str, Any]] = None,
                       model_config: Optional[Dict[str, Any]] = None,
                       checkpoint_name: Optional[str] = None) -> str:
        """
        Save a training checkpoint
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            epoch: Current epoch number
            metrics: Dictionary of metrics (loss, accuracy, etc.)
            scheduler: Optional learning rate scheduler
            dataset_info: Information about training dataset
            model_config: Model configuration
            checkpoint_name: Optional custom checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch{epoch}_{timestamp}"
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp,
            'dataset_info': dataset_info or {},
            'model_config': model_config or {}
        }
        
        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Update metadata
        self._update_metadata(checkpoint_name, epoch, metrics, dataset_info)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: Optional[str] = None,
                       checkpoint_name: Optional[str] = None,
                       load_best: bool = False) -> Dict[str, Any]:
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Full path to checkpoint file
            checkpoint_name: Name of checkpoint to load
            load_best: Load the best checkpoint based on metrics
            
        Returns:
            Checkpoint data dictionary
        """
        if load_best:
            checkpoint_path = self._get_best_checkpoint_path()
        elif checkpoint_name:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        elif checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint_path()
        
        if checkpoint_path is None:
            raise ValueError("No checkpoint found to load")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        return checkpoint_data
    
    def resume_training(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any] = None,
                       checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume training from a checkpoint
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Optional scheduler to load state into
            checkpoint_path: Path to checkpoint (uses latest if None)
            
        Returns:
            Checkpoint metadata (epoch, metrics, etc.)
        """
        checkpoint = self.load_checkpoint(checkpoint_path)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Resumed training from epoch {checkpoint['epoch']}")
        
        return {
            'start_epoch': checkpoint['epoch'] + 1,
            'metrics': checkpoint.get('metrics', {}),
            'dataset_info': checkpoint.get('dataset_info', {}),
            'model_config': checkpoint.get('model_config', {})
        }
    
    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """Get list of all checkpoints with metadata"""
        return sorted(
            self.metadata.get('checkpoints', []),
            key=lambda x: x.get('epoch', 0)
        )
    
    def get_best_checkpoint_info(self, metric: str = 'val_loss', minimize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get information about the best checkpoint
        
        Args:
            metric: Metric to use for comparison
            minimize: True if lower is better, False if higher is better
            
        Returns:
            Best checkpoint metadata
        """
        checkpoints = self.metadata.get('checkpoints', [])
        if not checkpoints:
            return None
        
        valid_checkpoints = [
            cp for cp in checkpoints 
            if metric in cp.get('metrics', {})
        ]
        
        if not valid_checkpoints:
            return None
        
        if minimize:
            best = min(valid_checkpoints, key=lambda x: x['metrics'][metric])
        else:
            best = max(valid_checkpoints, key=lambda x: x['metrics'][metric])
        
        return best
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'best_checkpoint': None}
    
    def _save_metadata(self):
        """Save checkpoint metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _update_metadata(self, 
                        checkpoint_name: str,
                        epoch: int,
                        metrics: Dict[str, float],
                        dataset_info: Optional[Dict[str, Any]]):
        """Update metadata with new checkpoint info"""
        checkpoint_info = {
            'name': checkpoint_name,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'dataset_info': dataset_info or {},
            'path': str(self.checkpoint_dir / f"{checkpoint_name}.pt")
        }
        
        self.metadata['checkpoints'].append(checkpoint_info)
        
        # Update best checkpoint
        if self.keep_best:
            best = self.get_best_checkpoint_info(metric='val_loss', minimize=True)
            if best:
                self.metadata['best_checkpoint'] = best['name']
        
        self._save_metadata()
    
    def _get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to most recent checkpoint"""
        checkpoints = self.metadata.get('checkpoints', [])
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda x: x.get('epoch', 0))
        return Path(latest['path'])
    
    def _get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        best_name = self.metadata.get('best_checkpoint')
        if not best_name:
            return None
        
        return self.checkpoint_dir / f"{best_name}.pt"
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only max_checkpoints most recent"""
        if self.max_checkpoints == 0:
            return
        
        checkpoints = self.metadata.get('checkpoints', [])
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by epoch
        sorted_checkpoints = sorted(checkpoints, key=lambda x: x.get('epoch', 0))
        
        # Determine which to keep
        best_name = self.metadata.get('best_checkpoint')
        to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        # Remove old checkpoints (but keep best)
        for checkpoint in to_remove:
            if self.keep_best and checkpoint['name'] == best_name:
                continue
            
            checkpoint_path = Path(checkpoint['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_path.name}")
            
            # Remove from metadata
            self.metadata['checkpoints'].remove(checkpoint)
        
        self._save_metadata()
    
    def export_checkpoint_info(self, output_file: str):
        """Export checkpoint metadata to JSON file"""
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Checkpoint metadata exported to {output_path}")
