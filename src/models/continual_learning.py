"""
Continual Learning Strategies for DNABERT-2
Prevents catastrophic forgetting when training on sequential datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ContinualLearner:
    """
    Implements continual learning strategies to prevent catastrophic forgetting
    
    Strategies:
    1. Experience Replay - Store and replay samples from previous datasets
    2. Elastic Weight Consolidation (EWC) - Protect important weights
    3. Learning Without Forgetting (LwF) - Knowledge distillation
    """
    
    def __init__(self,
                 strategy: str = 'experience_replay',
                 buffer_size: int = 10000,
                 ewc_lambda: float = 0.4):
        """
        Initialize continual learner
        
        Args:
            strategy: Learning strategy ('experience_replay', 'ewc', 'lwf', 'combined')
            buffer_size: Size of experience replay buffer
            ewc_lambda: Importance of EWC regularization
        """
        self.strategy = strategy
        self.buffer_size = buffer_size
        self.ewc_lambda = ewc_lambda
        
        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(buffer_size)
        
        # EWC Fisher information matrix
        self.fisher_dict = {}
        self.optimal_params = {}
        
        # Previous model for LwF
        self.previous_model = None
        
        logger.info(f"ContinualLearner initialized with strategy: {strategy}")
    
    def store_samples(self, sequences: List[str], labels: Optional[List[int]] = None):
        """
        Store samples in experience replay buffer
        
        Args:
            sequences: DNA sequences to store
            labels: Optional labels for sequences
        """
        if self.strategy in ['experience_replay', 'combined']:
            self.replay_buffer.add_samples(sequences, labels)
            logger.info(f"Stored {len(sequences)} samples in replay buffer")
    
    def get_replay_samples(self, batch_size: int) -> Tuple[List[str], Optional[List[int]]]:
        """
        Sample from replay buffer
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            Tuple of (sequences, labels)
        """
        return self.replay_buffer.sample(batch_size)
    
    def compute_fisher_information(self,
                                   model: nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   device: str = 'cpu'):
        """
        Compute Fisher Information Matrix for EWC
        
        Args:
            model: Model to compute Fisher information for
            dataloader: DataLoader with samples from current task
            device: Device to use for computation
        """
        if self.strategy not in ['ewc', 'combined']:
            return
        
        logger.info("Computing Fisher Information Matrix...")
        model.eval()
        
        # Initialize Fisher dict
        fisher_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
        
        # Accumulate gradients
        num_samples = 0
        for batch in dataloader:
            model.zero_grad()
            
            # Forward pass
            if isinstance(batch, dict):
                outputs = model(**{k: v.to(device) for k, v in batch.items()})
            else:
                outputs = model(batch.to(device))
            
            # Get loss (assuming classification)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Sample from output distribution
            probs = F.softmax(logits, dim=1)
            sampled_labels = torch.multinomial(probs, 1).squeeze()
            
            # Compute loss
            loss = F.cross_entropy(logits, sampled_labels)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.pow(2)
            
            num_samples += batch['input_ids'].size(0) if isinstance(batch, dict) else batch.size(0)
        
        # Average
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
        
        self.fisher_dict = fisher_dict
        
        # Store optimal parameters
        self.optimal_params = {name: param.clone().detach() 
                              for name, param in model.named_parameters() 
                              if param.requires_grad}
        
        logger.info(f"Fisher Information computed from {num_samples} samples")
    
    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC regularization loss
        
        Args:
            model: Current model
            
        Returns:
            EWC loss term
        """
        if not self.fisher_dict:
            return torch.tensor(0.0)
        
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name].to(param.device)
                optimal = self.optimal_params[name].to(param.device)
                loss += (fisher * (param - optimal).pow(2)).sum()
        
        return self.ewc_lambda * loss
    
    def store_previous_model(self, model: nn.Module):
        """
        Store a copy of the current model for LwF
        
        Args:
            model: Model to store
        """
        if self.strategy in ['lwf', 'combined']:
            self.previous_model = type(model)(model.config).to(next(model.parameters()).device)
            self.previous_model.load_state_dict(model.state_dict())
            self.previous_model.eval()
            logger.info("Stored previous model for LwF")
    
    def compute_distillation_loss(self,
                                  current_outputs: torch.Tensor,
                                  inputs: torch.Tensor,
                                  temperature: float = 2.0) -> torch.Tensor:
        """
        Compute knowledge distillation loss (LwF)
        
        Args:
            current_outputs: Outputs from current model
            inputs: Input data
            temperature: Distillation temperature
            
        Returns:
            Distillation loss
        """
        if self.previous_model is None:
            return torch.tensor(0.0)
        
        with torch.no_grad():
            previous_outputs = self.previous_model(inputs)
        
        # Soft targets from previous model
        soft_targets = F.softmax(previous_outputs / temperature, dim=1)
        soft_predictions = F.log_softmax(current_outputs / temperature, dim=1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
        distillation_loss *= (temperature ** 2)
        
        return distillation_loss
    
    def get_combined_loss(self,
                         base_loss: torch.Tensor,
                         model: nn.Module,
                         current_outputs: Optional[torch.Tensor] = None,
                         inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute total loss with continual learning regularization
        
        Args:
            base_loss: Standard training loss
            model: Current model
            current_outputs: Model outputs (for LwF)
            inputs: Input data (for LwF)
            
        Returns:
            Combined loss
        """
        total_loss = base_loss
        
        # Add EWC loss
        if self.strategy in ['ewc', 'combined']:
            ewc_loss = self.compute_ewc_loss(model)
            total_loss += ewc_loss
        
        # Add distillation loss
        if self.strategy in ['lwf', 'combined'] and current_outputs is not None and inputs is not None:
            distill_loss = self.compute_distillation_loss(current_outputs, inputs)
            total_loss += distill_loss
        
        return total_loss


class ExperienceReplayBuffer:
    """
    Stores samples from previous tasks for experience replay
    Implements reservoir sampling for efficient storage
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            max_size: Maximum number of samples to store
        """
        self.max_size = max_size
        self.sequences = deque(maxlen=max_size)
        self.labels = deque(maxlen=max_size)
        self.num_seen = 0
    
    def add_samples(self, sequences: List[str], labels: Optional[List[int]] = None):
        """
        Add samples using reservoir sampling
        
        Args:
            sequences: DNA sequences to add
            labels: Optional labels
        """
        if labels is None:
            labels = [None] * len(sequences)
        
        for seq, label in zip(sequences, labels):
            if len(self.sequences) < self.max_size:
                # Buffer not full, add directly
                self.sequences.append(seq)
                self.labels.append(label)
            else:
                # Reservoir sampling
                idx = np.random.randint(0, self.num_seen + 1)
                if idx < self.max_size:
                    self.sequences[idx] = seq
                    self.labels[idx] = label
            
            self.num_seen += 1
    
    def sample(self, batch_size: int) -> Tuple[List[str], Optional[List[int]]]:
        """
        Sample random batch from buffer
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            Tuple of (sequences, labels)
        """
        if len(self.sequences) == 0:
            return [], None
        
        indices = np.random.choice(len(self.sequences), 
                                  size=min(batch_size, len(self.sequences)),
                                  replace=False)
        
        sampled_seqs = [self.sequences[i] for i in indices]
        sampled_labels = [self.labels[i] for i in indices] if self.labels[0] is not None else None
        
        return sampled_seqs, sampled_labels
    
    def __len__(self):
        return len(self.sequences)
    
    def get_all_samples(self) -> Tuple[List[str], Optional[List[int]]]:
        """Get all samples in buffer"""
        return list(self.sequences), list(self.labels) if self.labels[0] is not None else None


class DatasetMemory:
    """
    Tracks which datasets have been used for training
    Helps organize continual learning across multiple datasets
    """
    
    def __init__(self):
        self.datasets = []
        self.dataset_stats = {}
    
    def add_dataset(self, name: str, num_samples: int, metadata: Optional[Dict] = None):
        """
        Register a new dataset
        
        Args:
            name: Dataset name
            num_samples: Number of samples in dataset
            metadata: Additional metadata
        """
        self.datasets.append(name)
        self.dataset_stats[name] = {
            'num_samples': num_samples,
            'order': len(self.datasets),
            'metadata': metadata or {}
        }
        logger.info(f"Registered dataset: {name} ({num_samples} samples)")
    
    def get_training_history(self) -> List[str]:
        """Get ordered list of datasets used for training"""
        return self.datasets.copy()
    
    def get_total_samples(self) -> int:
        """Get total number of samples seen across all datasets"""
        return sum(stats['num_samples'] for stats in self.dataset_stats.values())
