"""
Model training utilities for DNA sequence embeddings.

Provides comprehensive training functionality for DNATransformerEmbedder,
DNAAutoencoder, and DNAContrastiveModel.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. EmbeddingTrainer will not function.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingTrainer:
    """
    Trainer class for deep learning models.
    
    Supports training DNATransformerEmbedder, DNAAutoencoder, and DNAContrastiveModel
    with flexible training configurations.
    """
    
    def __init__(self, 
                 model: Any,
                 tokenizer: Any,
                 device: str = 'auto'):
        """
        Initialize trainer
        
        Args:
            model: Model to train (DNATransformerEmbedder, DNAAutoencoder, or DNAContrastiveModel)
            tokenizer: DNA tokenizer instance
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for EmbeddingTrainer")
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self,
                    sequences: List[str],
                    labels: Optional[List[Union[str, int]]] = None,
                    validation_split: float = 0.2,
                    batch_size: int = 32,
                    max_length: int = 512) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training
        
        Args:
            sequences: List of DNA sequences
            labels: Optional labels for sequences
            validation_split: Fraction of data to use for validation
            batch_size: Batch size for training
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info(f"Preparing data: {len(sequences)} sequences")
        
        # Encode sequences
        encoded = self.tokenizer.encode_sequences(sequences, max_length=max_length)
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
        
        # Prepare dataset
        if labels is not None:
            # Convert labels to integers if they're strings
            if isinstance(labels[0], str):
                unique_labels = list(set(labels))
                label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
                labels_tensor = torch.tensor([label_to_id[label] for label in labels], dtype=torch.long)
            else:
                labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
        else:
            dataset = TensorDataset(input_ids, attention_mask)
        
        # Split into train and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Train size: {train_size}, Validation size: {val_size}")
        
        return train_loader, val_loader
    
    def train_autoencoder(self,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         epochs: int = 100,
                         learning_rate: float = 1e-4,
                         save_best: bool = True,
                         save_path: Optional[Path] = None) -> Dict[str, List[float]]:
        """
        Train autoencoder model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            save_best: Whether to save best model
            save_path: Path to save model
            
        Returns:
            Training history dictionary
        """
        from src.models.embeddings import DNAAutoencoder
        
        if not isinstance(self.model, DNAAutoencoder):
            raise ValueError("Model must be DNAAutoencoder for autoencoder training")
        
        logger.info("Starting autoencoder training...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                optimizer.zero_grad()
                
                latent, reconstructed = self.model(input_ids, attention_mask)
                
                # Reconstruction loss (reconstruct input_ids)
                target = input_ids.float()
                loss = criterion(reconstructed, F.one_hot(input_ids, num_classes=self.model.vocab_size).float())
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    
                    latent, reconstructed = self.model(input_ids, attention_mask)
                    loss = criterion(reconstructed, F.one_hot(input_ids, num_classes=self.model.vocab_size).float())
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if save_best and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Best model saved with val_loss: {best_val_loss:.4f}")
        
        logger.info("Training complete!")
        return history
    
    def train_contrastive(self,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         epochs: int = 100,
                         learning_rate: float = 1e-4,
                         save_best: bool = True,
                         save_path: Optional[Path] = None) -> Dict[str, List[float]]:
        """
        Train contrastive learning model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            save_best: Whether to save best model
            save_path: Path to save model
            
        Returns:
            Training history dictionary
        """
        from src.models.embeddings import DNAContrastiveModel
        
        if not isinstance(self.model, DNAContrastiveModel):
            raise ValueError("Model must be DNAContrastiveModel for contrastive training")
        
        logger.info("Starting contrastive learning training...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device) if len(batch) > 2 else None
                
                optimizer.zero_grad()
                
                projections = self.model(input_ids, attention_mask)
                loss = self.model.contrastive_loss(projections, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device) if len(batch) > 2 else None
                    
                    projections = self.model(input_ids, attention_mask)
                    loss = self.model.contrastive_loss(projections, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if save_best and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Best model saved with val_loss: {best_val_loss:.4f}")
        
        logger.info("Training complete!")
        return history
    
    def extract_embeddings(self,
                          sequences: List[str],
                          batch_size: int = 32,
                          max_length: int = 512) -> np.ndarray:
        """
        Extract embeddings from sequences
        
        Args:
            sequences: List of DNA sequences
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Extracting embeddings for {len(sequences)} sequences")
        
        self.model.eval()
        embeddings_list = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # Encode batch
            encoded = self.tokenizer.encode_sequences(batch_sequences, max_length=max_length)
            input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                # Get embeddings based on model type
                from src.models.embeddings import DNAContrastiveModel, DNAAutoencoder
                
                if isinstance(self.model, DNAContrastiveModel):
                    embeddings = self.model.get_embeddings(input_ids, attention_mask)
                elif isinstance(self.model, DNAAutoencoder):
                    embeddings, _ = self.model(input_ids, attention_mask)
                else:
                    embeddings = self.model(input_ids, attention_mask)
                
                embeddings_list.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        all_embeddings = np.concatenate(embeddings_list, axis=0)
        logger.info(f"Extracted embeddings shape: {all_embeddings.shape}")
        
        return all_embeddings
    
    def save_model(self, save_path: Union[str, Path], include_tokenizer: bool = True) -> None:
        """
        Save model and optionally tokenizer
        
        Args:
            save_path: Path to save model
            include_tokenizer: Whether to save tokenizer
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_path / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save tokenizer
        if include_tokenizer:
            tokenizer_path = save_path / "tokenizer.pkl"
            self.tokenizer.save(tokenizer_path)
            logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, load_path: Union[str, Path]) -> None:
        """
        Load model weights
        
        Args:
            load_path: Path to load model from
        """
        load_path = Path(load_path)
        model_path = load_path / "model.pt"
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"Model loaded from {model_path}")
