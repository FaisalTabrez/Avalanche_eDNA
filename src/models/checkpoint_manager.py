"""
Checkpoint Manager for Model Training
Handles saving, loading, and managing model checkpoints for continual learning

This module provides comprehensive checkpoint management including:
- Saving/loading complete training state (model, optimizer, scheduler)
- Automatic best model tracking based on metrics
- Checkpoint history and metadata management
- Cleanup of old checkpoints with configurable retention
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
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
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir (str): Directory to store checkpoints. Will be created if it doesn't exist.
            max_checkpoints (int, optional): Maximum number of checkpoints to keep. Set to 0 for unlimited. Defaults to 5.
            keep_best (bool, optional): Always keep the best checkpoint regardless of max_checkpoints limit. Defaults to True.
        
        Example:
            >>> manager = CheckpointManager('checkpoints/', max_checkpoints=3)
            >>> # Saves checkpoints to 'checkpoints/' directory, keeping max 3
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
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       scheduler: Optional[Any] = None,
                       dataset_info: Optional[Dict[str, Any]] = None,
                       model_config: Optional[Dict[str, Any]] = None,
                       checkpoint_name: Optional[str] = None) -> str:
        """
        Save a complete training checkpoint with model state, optimizer, and metadata.
        
        Args:
            model (nn.Module): PyTorch model to save. The model's state_dict will be extracted.
            optimizer (optim.Optimizer): Optimizer whose state will be saved for resuming training.
            epoch (int): Current training epoch number (0-indexed).
            metrics (Dict[str, float]): Training/validation metrics (e.g., {'val_loss': 0.25, 'val_acc': 0.92}).
            scheduler (Optional[Any], optional): Learning rate scheduler to save. Defaults to None.
            dataset_info (Optional[Dict[str, Any]], optional): Metadata about dataset used (e.g., {'name': 'dataset1', 'size': 10000}). Defaults to None.
            model_config (Optional[Dict[str, Any]], optional): Model configuration dictionary. Defaults to None.
            checkpoint_name (Optional[str], optional): Custom checkpoint name. If None, auto-generated as 'checkpoint_epoch{epoch}_{timestamp}'. Defaults to None.
            
        Returns:
            str: Absolute path to the saved checkpoint file.
        
        Example:
            >>> manager = CheckpointManager('checkpoints/')
            >>> checkpoint_path = manager.save_checkpoint(
            ...     model=model,
            ...     optimizer=optimizer,
            ...     epoch=10,
            ...     metrics={'val_loss': 0.25, 'val_acc': 0.92},
            ...     dataset_info={'name': 'bacteria_16S'}
            ... )
        
        Note:
            - Automatically updates checkpoint metadata and history
            - Triggers cleanup of old checkpoints based on max_checkpoints setting
            - Updates best_checkpoint if current metrics are better
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
        Load a checkpoint from disk.
        
        Args:
            checkpoint_path (Optional[str], optional): Full path to checkpoint file. Takes priority if provided. Defaults to None.
            checkpoint_name (Optional[str], optional): Name of checkpoint to load (without .pt extension). Defaults to None.
            load_best (bool, optional): Load the best checkpoint based on tracked metrics. Defaults to False.
            
        Returns:
            Dict[str, Any]: Checkpoint data containing:
                - epoch (int): Training epoch
                - model_state_dict (Dict): Model weights
                - optimizer_state_dict (Dict): Optimizer state
                - scheduler_state_dict (Dict, optional): Scheduler state if saved
                - metrics (Dict[str, float]): Saved metrics
                - dataset_info (Dict): Dataset metadata
                - model_config (Dict): Model configuration
        
        Raises:
            ValueError: If no checkpoint is found or parameters are invalid.
            FileNotFoundError: If specified checkpoint file doesn't exist.
        
        Example:
            >>> manager = CheckpointManager('checkpoints/')
            >>> # Load best checkpoint
            >>> checkpoint = manager.load_checkpoint(load_best=True)
            >>> # Load specific checkpoint
            >>> checkpoint = manager.load_checkpoint(checkpoint_name='checkpoint_epoch10_20251126')
            >>> # Load from path
            >>> checkpoint = manager.load_checkpoint(checkpoint_path='checkpoints/my_checkpoint.pt')
        
        Note:
            Priority order: load_best > checkpoint_path > checkpoint_name > latest
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
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       scheduler: Optional[Any] = None,
                       checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume training from a checkpoint by loading states into provided objects.
        
        Args:
            model (nn.Module): Model to load weights into (modified in-place).
            optimizer (optim.Optimizer): Optimizer to load state into (modified in-place).
            scheduler (Optional[Any], optional): Scheduler to load state into if checkpoint contains scheduler state. Defaults to None.
            checkpoint_path (Optional[str], optional): Path to checkpoint. Uses latest if None. Defaults to None.
            
        Returns:
            Dict[str, Any]: Resume information containing:
                - start_epoch (int): Next epoch number to start from
                - metrics (Dict[str, float]): Metrics from checkpoint
                - dataset_info (Dict): Dataset metadata
                - model_config (Dict): Model configuration
        
        Example:
            >>> manager = CheckpointManager('checkpoints/')
            >>> model = MyModel()
            >>> optimizer = torch.optim.Adam(model.parameters())
            >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
            >>> 
            >>> resume_info = manager.resume_training(model, optimizer, scheduler)
            >>> start_epoch = resume_info['start_epoch']
            >>> print(f"Resuming from epoch {start_epoch}")
        
        Note:
            Model, optimizer, and scheduler are modified in-place.
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
