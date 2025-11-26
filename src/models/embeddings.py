"""PyTorch-based minimal embedding & autoencoder implementations used by tests.

These classes are intentionally lightweight to make the test-suite runnable
without requiring large model checkpoints. They implement the small API
expected by the unit tests in `tests/test_system.py`.
"""

from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.models.dnabert import DNABERTEmbedder
    DNABERT_AVAILABLE = True
except ImportError:
    DNABERT_AVAILABLE = False


class DNATransformerEmbedder(nn.Module):
    """Simple Transformer-like encoder that returns a fixed-size embedding.

    This implementation uses an embedding layer followed by a small
    TransformerEncoder and mean-pooling over the token dimension.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 512, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))

        # Simple projection head (identity by default)
        self.projector = nn.Linear(d_model, d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return embeddings with shape (batch_size, d_model)."""
        # input_ids: (batch, seq_len)
        x = self.token_embedding(input_ids.long())  # (batch, seq_len, d_model)

        # Apply transformer encoder
        # Transformer expects padding mask of shape (batch, seq_len) where True indicates padding
        src_key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: 1 for tokens, 0 for padding
            src_key_padding_mask = (attention_mask == 0)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling over valid tokens
        if attention_mask is not None:
            attn = attention_mask.unsqueeze(-1).float()
            summed = (x * attn).sum(dim=1)
            denom = attn.sum(dim=1).clamp(min=1e-9)
            pooled = summed / denom
        else:
            pooled = x.mean(dim=1)

        out = self.projector(pooled)
        return out


class DNAAutoencoder(nn.Module):
    """Simple autoencoder producing a latent vector and a reconstructed output.

    The reconstructed output is a dense vector per sample (not a full sequence reconstruction),
    which is sufficient for the unit tests that only check shapes.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dims: Optional[List[int]] = None,
                 latent_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [64, 128, 64]
        self.vocab_size = vocab_size

        # Token embedding to convert token ids to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Encoder MLP
        encoder_layers = []
        in_dim = embedding_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder MLP (reconstruct to a vector of size vocab_size for simplicity)
        decoder_layers = [nn.Linear(latent_dim, hidden_dims[-1]), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = hidden_dims[-1]
        for h in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, vocab_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (latent, reconstructed) where latent.shape=(batch, latent_dim) and reconstructed.shape=(batch, vocab_size)."""
        # Embed tokens and mean-pool
        x = self.embedding(input_ids.long())  # (batch, seq_len, embedding_dim)
        if attention_mask is not None:
            attn = attention_mask.unsqueeze(-1).float()
            summed = (x * attn).sum(dim=1)
            denom = attn.sum(dim=1).clamp(min=1e-9)
            pooled = summed / denom
        else:
            pooled = x.mean(dim=1)

        latent = self.encoder(pooled)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class DNAContrastiveModel(nn.Module):
    """Contrastive learning wrapper for self-supervised training.
    
    This model wraps a backbone encoder (e.g., DNATransformerEmbedder) with
    a projection head for contrastive learning.
    """
    
    def __init__(self, 
                 backbone_model: nn.Module,
                 projection_dim: int = 128,
                 temperature: float = 0.1):
        """
        Initialize contrastive model
        
        Args:
            backbone_model: Encoder model (e.g., DNATransformerEmbedder)
            projection_dim: Dimension of projection head output
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()
        self.backbone = backbone_model
        self.temperature = temperature
        
        # Projection head: 2-layer MLP
        # Determine backbone output dimension
        if hasattr(backbone_model, 'd_model'):
            backbone_dim = backbone_model.d_model
        else:
            # Default to common dimension
            backbone_dim = 256
        
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, projection_dim)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through backbone and projection head
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Projected embeddings (batch_size, projection_dim)
        """
        # Get embeddings from backbone
        embeddings = self.backbone(input_ids, attention_mask)
        
        # Project to lower dimension
        projected = self.projection_head(embeddings)
        
        # L2 normalize for contrastive learning
        projected = F.normalize(projected, dim=1)
        
        return projected
    
    def contrastive_loss(self, 
                        projections: torch.Tensor,
                        labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
        
        Args:
            projections: Normalized projections (batch_size, projection_dim)
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            Contrastive loss value
        """
        batch_size = projections.shape[0]
        device = projections.device
        eps = 1e-8
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create mask to exclude self-similarity
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        if labels is not None:
            # Supervised contrastive loss
            # Positive pairs: same label
            labels = labels.contiguous().view(-1, 1)
            mask_positive = torch.eq(labels, labels.T).float().to(device)
            mask_positive = mask_positive.masked_fill(torch.eye(batch_size, dtype=torch.bool, device=device), 0)
            
            # Check if there are any positive pairs
            num_positives = mask_positive.sum(1)
            if num_positives.min() == 0:
                # Handle case where some samples have no positive pairs
                # Return a default loss
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # Compute loss with numerical stability
            max_sim = similarity_matrix.max(dim=1, keepdim=True)[0]
            exp_sim = torch.exp(similarity_matrix - max_sim)
            log_prob = (similarity_matrix - max_sim) - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)
            
            # Mean of log-likelihood over positive pairs
            mean_log_prob_pos = (mask_positive * log_prob).sum(1) / (num_positives + eps)
            loss = -mean_log_prob_pos.mean()
        else:
            # Self-supervised contrastive loss (SimCLR-style)
            # Assume batch contains augmented pairs (first half and second half are pairs)
            if batch_size % 2 != 0:
                raise ValueError("Batch size must be even for self-supervised contrastive learning")
            
            # Create positive pair mask
            labels = torch.cat([torch.arange(batch_size // 2), torch.arange(batch_size // 2)]).to(device)
            labels = labels.contiguous().view(-1, 1)
            mask_positive = torch.eq(labels, labels.T).float()
            mask_positive = mask_positive.masked_fill(torch.eye(batch_size, dtype=torch.bool, device=device), 0)
            
            # Compute loss with numerical stability
            max_sim = similarity_matrix.max(dim=1, keepdim=True)[0]
            exp_sim = torch.exp(similarity_matrix - max_sim)
            log_prob = (similarity_matrix - max_sim) - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)
            num_positives = mask_positive.sum(1)
            mean_log_prob_pos = (mask_positive * log_prob).sum(1) / (num_positives + eps)
            loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get backbone embeddings without projection
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Backbone embeddings
        """
        return self.backbone(input_ids, attention_mask)


class ModelFactory:
    @staticmethod
    def create_transformer(vocab_size: int, config: Dict[str, Any]) -> DNATransformerEmbedder:
        return DNATransformerEmbedder(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6),
            dim_feedforward=config.get('dim_feedforward', 512),
            dropout=config.get('dropout', 0.1),
            max_len=config.get('max_len', 512)
        )

    @staticmethod
    def create_autoencoder(vocab_size: int, config: Dict[str, Any]) -> DNAAutoencoder:
        return DNAAutoencoder(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 32),
            hidden_dims=config.get('hidden_dims', [64, 128, 64]),
            latent_dim=config.get('latent_dim', 16),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_contrastive(backbone_model: nn.Module, config: Dict[str, Any]) -> DNAContrastiveModel:
        return DNAContrastiveModel(
            backbone_model=backbone_model,
            projection_dim=config.get('projection_dim', 128),
            temperature=config.get('temperature', 0.1)
        )
    
    @staticmethod
    def create_dnabert(config: Dict[str, Any]) -> Optional['DNABERTEmbedder']:
        """Create DNABERT pre-trained embedder
        
        Args:
            config: Configuration dict with DNABERT settings
                - model_size: Model variant (dnabert, dnabert-3, dnabert-6)
                - device: Device to use (auto, cuda, cpu)
                - cache_dir: Directory to cache model weights
                - max_length: Maximum sequence length
        
        Returns:
            DNABERTEmbedder instance or None if not available
        """
        if not DNABERT_AVAILABLE:
            raise ImportError(
                "DNABERT not available. Please ensure transformers is installed: "
                "pip install transformers sentencepiece accelerate einops"
            )
        
        return DNABERTEmbedder(
            model_size=config.get('model_size', 'dnabert'),
            device=config.get('device'),
            cache_dir=config.get('cache_dir', 'models/pretrained/dna_models'),
            max_length=config.get('max_length', 512)
        )