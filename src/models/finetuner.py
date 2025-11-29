"""
DNABERT-2 Fine-Tuning Module
Provides fine-tuning capabilities for DNABERT-2 on domain-specific eDNA data.

This module enables:
- Layer-wise freezing strategies for efficient fine-tuning
- Parameter-efficient training with selective layer updates
- Contrastive learning for sequence representation
- Learning rate scheduling with warmup
- Mixed precision training support (future)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DNABERTFineTuner:
    """
    Fine-tune DNABERT-2 for eDNA-specific tasks
    
    Features:
    - Layer freezing strategies
    - Learning rate scheduling
    - Parameter-efficient fine-tuning (LoRA optional)
    - Gradient accumulation
    - Mixed precision training
    """
    
    def __init__(self,
                 model_id: str = 'zhihan1996/DNABERT-2-117M',
                 freeze_layers: Optional[List[int]] = None,
                 freeze_embeddings: bool = True,
                 device: str = 'auto'):
        """
        Initialize DNABERT-2 fine-tuner with configurable freezing strategy.
        
        Args:
            model_id (str, optional): Hugging Face model identifier for DNABERT-2 model. 
                Defaults to 'zhihan1996/DNABERT-2-117M'.
            freeze_layers (Optional[List[int]], optional): List of transformer layer indices to freeze (0-indexed). 
                If None, all layers are trainable. Example: [0, 1, 2] freezes first 3 layers. Defaults to None.
            freeze_embeddings (bool, optional): Whether to freeze the embedding layer. 
                Recommended for small datasets. Defaults to True.
            device (str, optional): Device for model ('auto', 'cuda', 'cpu'). 
                'auto' selects GPU if available. Defaults to 'auto'.
        
        Example:
            >>> # Full fine-tuning (all layers trainable)
            >>> finetuner = DNABERTFineTuner(model_id='zhihan1996/DNABERT-2-117M', freeze_layers=None, freeze_embeddings=False)
            >>> 
            >>> # Conservative fine-tuning (freeze bottom 6 layers)
            >>> finetuner = DNABERTFineTuner(freeze_layers=[0, 1, 2, 3, 4, 5], freeze_embeddings=True)
        
        Note:
            - Freezing layers reduces memory usage and training time
            - Typically freeze lower layers for domain adaptation
            - Use freeze_embeddings=True for small datasets to prevent overfitting
        """
        self.model_id = model_id
        self.freeze_layers = freeze_layers or []
        self.freeze_embeddings = freeze_embeddings
        
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing DNABERTFineTuner on device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(self.device)
        
        # Apply freezing strategy
        self._apply_freezing_strategy()
        
        # Count trainable parameters
        self._log_parameter_stats()
    
    def _apply_freezing_strategy(self):
        """Apply layer freezing based on configuration"""
        # Freeze embeddings if requested
        if self.freeze_embeddings:
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
            logger.info("Frozen embedding layer")
        
        # Freeze specific layers
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            for layer_idx in self.freeze_layers:
                if layer_idx < len(self.model.encoder.layer):
                    for param in self.model.encoder.layer[layer_idx].parameters():
                        param.requires_grad = False
                    logger.info(f"Frozen layer {layer_idx}")
    
    def _log_parameter_stats(self):
        """Log statistics about trainable vs frozen parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logger.info(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    def prepare_optimizer(self,
                         learning_rate: float = 2e-5,
                         weight_decay: float = 0.01,
                         adam_epsilon: float = 1e-8) -> torch.optim.Optimizer:
        """
        Prepare optimizer with parameter groups
        
        Args:
            learning_rate: Learning rate for fine-tuning
            weight_decay: Weight decay for regularization
            adam_epsilon: Epsilon for Adam optimizer
            
        Returns:
            Configured optimizer
        """
        # Separate parameters with/without weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_epsilon
        )
        
        return optimizer
    
    def prepare_scheduler(self,
                         optimizer: torch.optim.Optimizer,
                         num_training_steps: int,
                         num_warmup_steps: Optional[int] = None,
                         warmup_ratio: float = 0.1):
        """
        Prepare learning rate scheduler with warmup
        
        Args:
            optimizer: Optimizer to schedule
            num_training_steps: Total number of training steps
            num_warmup_steps: Number of warmup steps (overrides warmup_ratio)
            warmup_ratio: Ratio of total steps to use for warmup
            
        Returns:
            Learning rate scheduler
        """
        if num_warmup_steps is None:
            num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Scheduler: {num_warmup_steps} warmup steps, {num_training_steps} total steps")
        return scheduler
    
    def extract_embeddings(self,
                          sequences: List[str],
                          batch_size: int = 8,
                          max_length: int = 512) -> torch.Tensor:
        """
        Extract embeddings from fine-tuned model
        
        Args:
            sequences: List of DNA sequences
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Tensor of embeddings (num_sequences, embedding_dim)
        """
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_seqs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Extract [CLS] token embedding or mean pool
                if hasattr(outputs, 'last_hidden_state'):
                    # Mean pooling with attention mask
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    embeddings = outputs[0][:, 0, :]  # CLS token
                
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def save_model(self, output_dir: str, save_tokenizer: bool = True):
        """
        Save fine-tuned model
        
        Args:
            output_dir: Directory to save model
            save_tokenizer: Whether to save tokenizer as well
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")
        
        # Save tokenizer
        if save_tokenizer:
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Tokenizer saved to {output_path}")
    
    def load_model(self, model_dir: str):
        """
        Load fine-tuned model from directory
        
        Args:
            model_dir: Directory containing saved model
        """
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        
        # Try to load tokenizer if available
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {model_path}: {e}")
    
    def get_freezing_strategy(self, strategy: str = 'gradual') -> List[int]:
        """
        Get layer freezing configuration based on strategy
        
        Args:
            strategy: Freezing strategy name
                - 'all': Freeze all layers (only classification head trainable)
                - 'none': Freeze no layers (full fine-tuning)
                - 'half': Freeze first half of layers
                - 'gradual': Freeze bottom 75% of layers
                - 'top3': Only train top 3 layers
                
        Returns:
            List of layer indices to freeze
        """
        if not hasattr(self.model, 'encoder') or not hasattr(self.model.encoder, 'layer'):
            return []
        
        num_layers = len(self.model.encoder.layer)
        
        strategies = {
            'all': list(range(num_layers)),
            'none': [],
            'half': list(range(num_layers // 2)),
            'gradual': list(range(int(num_layers * 0.75))),
            'top3': list(range(num_layers - 3)) if num_layers > 3 else []
        }
        
        return strategies.get(strategy, [])


class ContrastiveLearningHead(nn.Module):
    """
    Contrastive learning projection head for DNABERT-2 fine-tuning.
    
    Projects high-dimensional embeddings to a lower-dimensional space optimized
    for contrastive learning (e.g., SimCLR, NT-Xent loss). This head can be
    attached to DNABERT-2 for self-supervised or semi-supervised learning.
    
    Attributes:
        projection (nn.Sequential): MLP projection network.
        temperature (float): Temperature parameter for contrastive loss scaling.
    """
    
    def __init__(self, input_dim: int, projection_dim: int = 64, temperature: float = 0.07):  # OPTIMIZED
        """
        Initialize contrastive learning head.
        
        Args:
            input_dim (int): Dimension of input embeddings from DNABERT-2 (typically 768).
            projection_dim (int, optional): Dimension of projected space. Lower dimensions
                encourage learning of more general features. Defaults to 128.
            temperature (float, optional): Temperature for NT-Xent loss. Lower values
                increase sensitivity to hard negatives. Defaults to 0.07.
        
        Example:
            >>> head = ContrastiveLearningHead(input_dim=768, projection_dim=128)
            >>> embeddings = model(sequences)  # Shape: [batch_size, 768]
            >>> projections = head(embeddings)  # Shape: [batch_size, 128]
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings to contrastive space"""
        return nn.functional.normalize(self.projection(embeddings), dim=1)
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
        
        This is the standard contrastive learning loss used in SimCLR and related methods.
        Also known as InfoNCE loss. Maximizes agreement between differently augmented
        views of the same sequence while pushing apart different sequences.
        
        Args:
            z1 (torch.Tensor): Projected embeddings from augmented view 1. Shape: [batch_size, projection_dim].
            z2 (torch.Tensor): Projected embeddings from augmented view 2. Shape: [batch_size, projection_dim].
            
        Returns:
            torch.Tensor: Scalar contrastive loss value (lower is better).
        
        Example:
            >>> head = ContrastiveLearningHead(input_dim=768)
            >>> # Get embeddings for two augmented views of same batch
            >>> z1 = head(embeddings_aug1)  # [32, 128]
            >>> z2 = head(embeddings_aug2)  # [32, 128]
            >>> loss = head.contrastive_loss(z1, z2)
            >>> loss.backward()
        
        Note:
            - Batch size should be sufficiently large (>= 32) for effective contrastive learning
            - Both z1 and z2 must have the same shape
            - Method name is 'contrastive_loss' (not 'nt_xent_loss')
        """
        batch_size = z1.shape[0]
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create labels
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Compute loss
        loss = nn.functional.cross_entropy(sim_matrix, labels)
        
        return loss
