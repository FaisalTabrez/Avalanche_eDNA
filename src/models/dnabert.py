"""
DNABERT Integration
Wrapper for DNABERT pre-trained models from HuggingFace
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class DNABERTEmbedder:
    """
    Wrapper for pre-trained DNABERT models from HuggingFace.
    Supports DNABERT v1 models:
    - DNA_bert_6: 6-mer tokenization (recommended, 768-dim embeddings)
    - DNA_bert_3: 3-mer tokenization (alternative)
    """

    AVAILABLE_MODELS = {
        # DNABERT original (works on CPU, uses k-mer tokenization)
        "dnabert": "zhihan1996/DNA_bert_6",
        "dnabert-3": "zhihan1996/DNA_bert_3",
        # DNABERT-2 (requires GPU - uses Flash Attention)
        "dnabert2": "zhihan1996/DNABERT-2-117M",
        "dnabert2-117m": "zhihan1996/DNABERT-2-117M",
    }

    def __init__(
        self,
        model_size: str = "dnabert2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_length: int = 1000,
    ):
        """
        Initialize DNABERT or DNABERT-2

        Args:
            model_size: Model identifier (nt-250m, dnabert2, etc)
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache model weights
            max_length: Maximum sequence length (default 1000bp)
        """
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model size must be one of {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_name = self.AVAILABLE_MODELS[model_size]
        self.model_size = model_size
        self.max_length = min(max_length, 6000)  # Model's maximum context length

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing DNABERT model ({model_size})")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Device: {self.device}")

        # Cache directory
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer and model
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=cache_dir, trust_remote_code=True
            )

            # Ensure padding token is set (required for ESM-based models like DNABERT-2)
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    # Add a new padding token
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            logger.info("Loading model (this may take a few minutes on first run)...")
            # Try to load with revision pinned to avoid compatibility issues
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    revision="main",
                    # Disable Flash Attention for CPU compatibility
                    attn_implementation="eager",
                )
            except Exception as e:
                logger.warning(f"Failed to load with AutoModel: {e}")
                logger.info("Trying alternative loading method...")
                # Fallback: Try loading EsmModel directly
                from transformers import EsmModel

                self.model = EsmModel.from_pretrained(
                    self.model_name, cache_dir=cache_dir, trust_remote_code=True
                )

            # Resize embeddings if we added a new padding token
            if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
                self.model.resize_token_embeddings(len(self.tokenizer))

            self.model.to(self.device)
            self.model.eval()

            # Get embedding dimension from model config
            if hasattr(self.model.config, "hidden_size"):
                self.embedding_dim = self.model.config.hidden_size
            elif hasattr(self.model.config, "d_model"):
                self.embedding_dim = self.model.config.d_model
            else:
                # Fallback: infer from model
                try:
                    self.embedding_dim = (
                        self.model.embeddings.word_embeddings.embedding_dim
                    )
                except:
                    self.embedding_dim = 512  # Default for NT models

            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Embedding dimension: {self.embedding_dim}")
            logger.info(f"  Max sequence length: {self.max_length} bp")
            logger.info(f"  Parameters: ~{model_size.upper()}")

        except Exception as e:
            logger.error(f"Failed to load DNABERT: {str(e)}")
            raise

    def _prepare_sequences(self, sequences: List[str]) -> List[str]:
        """
        Prepare DNA sequences for tokenization

        Args:
            sequences: List of DNA sequences

        Returns:
            Prepared sequences
        """
        prepared = []
        for seq in sequences:
            # Convert to uppercase
            seq = seq.upper().strip()

            # Remove any non-ACGT characters (keep only valid bases)
            seq = "".join(c for c in seq if c in "ACGT")

            # DNABERT uses k-mer tokenization (overlapping k-mers)
            # For DNA_bert_6, uses 6-mers with stride 1
            k = 6 if "bert_6" in self.model_name else 3
            kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
            prepared.append(" ".join(kmers))

        return prepared

    def encode(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 8,
        pool_mode: str = "mean",
        return_attention: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate embeddings for DNA sequences

        Args:
            sequences: Single sequence or list of sequences
            batch_size: Batch size for processing
            pool_mode: Pooling strategy ('mean', 'max', 'cls')
            return_attention: Whether to return attention weights

        Returns:
            Embeddings array (num_sequences, embedding_dim) or dict with embeddings and attention
        """
        # Handle single sequence
        if isinstance(sequences, str):
            sequences = [sequences]

        logger.info(f"Encoding {len(sequences)} sequences...")

        # Prepare sequences
        prepared_seqs = self._prepare_sequences(sequences)

        # Process in batches
        all_embeddings = []
        all_attention = [] if return_attention else None

        for i in range(0, len(prepared_seqs), batch_size):
            batch = prepared_seqs[i : i + batch_size]

            # Tokenize
            tokens = self.tokenizer.batch_encode_plus(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            # Move to device
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=return_attention,
                )

            # Pool embeddings
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            if pool_mode == "mean":
                # Mean pooling over sequence length (excluding padding)
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            elif pool_mode == "max":
                # Max pooling
                embeddings = torch.max(hidden_states, dim=1)[0]
            elif pool_mode == "cls":
                # Use [CLS] token (first token)
                embeddings = hidden_states[:, 0, :]
            else:
                raise ValueError(f"Invalid pool_mode: {pool_mode}")

            all_embeddings.append(embeddings.cpu().numpy())

            if return_attention:
                # Average attention across all layers and heads
                attention = outputs.attentions
                avg_attention = (
                    torch.stack(attention).mean(dim=0).mean(dim=1)
                )  # (batch, seq_len, seq_len)
                all_attention.append(avg_attention.cpu().numpy())

        # Concatenate batches
        embeddings = np.concatenate(all_embeddings, axis=0)

        logger.info(f"✓ Generated embeddings: {embeddings.shape}")

        if return_attention:
            attention = np.concatenate(all_attention, axis=0)
            return {"embeddings": embeddings, "attention": attention}
        else:
            return embeddings

    def get_similarity(
        self,
        seq1: Union[str, np.ndarray],
        seq2: Union[str, np.ndarray],
        metric: str = "cosine",
    ) -> float:
        """
        Compute similarity between two sequences

        Args:
            seq1: First sequence (string or embedding)
            seq2: Second sequence (string or embedding)
            metric: Similarity metric ('cosine', 'euclidean', 'dot')

        Returns:
            Similarity score
        """
        # Get embeddings if sequences are strings
        if isinstance(seq1, str):
            emb1 = self.encode(seq1)
        else:
            emb1 = seq1

        if isinstance(seq2, str):
            emb2 = self.encode(seq2)
        else:
            emb2 = seq2

        # Compute similarity
        if metric == "cosine":
            similarity = np.dot(emb1.flatten(), emb2.flatten()) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
        elif metric == "euclidean":
            similarity = -np.linalg.norm(emb1 - emb2)  # Negative for similarity
        elif metric == "dot":
            similarity = np.dot(emb1.flatten(), emb2.flatten())
        else:
            raise ValueError(f"Invalid metric: {metric}")

        return float(similarity)

    def batch_similarity(
        self, sequences: List[str], query: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute pairwise similarities or similarities to query

        Args:
            sequences: List of sequences
            query: Optional query sequence (if None, compute pairwise)

        Returns:
            Similarity matrix or similarity vector
        """
        embeddings = self.encode(sequences)

        if query is not None:
            # Compute similarity to query
            query_emb = self.encode(query)
            similarities = np.dot(embeddings, query_emb.T) / (
                np.linalg.norm(embeddings, axis=1, keepdims=True)
                * np.linalg.norm(query_emb)
            )
            return similarities.flatten()
        else:
            # Compute pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity

            return cosine_similarity(embeddings)

    def save_embeddings(
        self,
        sequences: List[str],
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate and save embeddings to file

        Args:
            sequences: List of sequences
            output_path: Path to save embeddings
            metadata: Optional metadata to save with embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate embeddings
        embeddings = self.encode(sequences)

        # Save
        data = {
            "embeddings": embeddings,
            "model_name": self.model_name,
            "model_size": self.model_size,
            "embedding_dim": self.embedding_dim,
            "num_sequences": len(sequences),
        }

        if metadata:
            data["metadata"] = metadata

        np.savez_compressed(output_path, **data)
        logger.info(f"✓ Saved embeddings to {output_path}")

    @staticmethod
    def load_embeddings(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load saved embeddings

        Args:
            file_path: Path to embeddings file

        Returns:
            Dictionary with embeddings and metadata
        """
        data = np.load(file_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    def __repr__(self) -> str:
        return (
            f"DNABERTEmbedder("
            f"model={self.model_size}, "
            f"embedding_dim={self.embedding_dim}, "
            f"device={self.device})"
        )


class DNABERTClassifier(torch.nn.Module):
    """
    Classifier head for DNABERT
    Useful for taxonomy classification or other supervised tasks
    """

    def __init__(
        self,
        embedder: DNABERTEmbedder,
        num_classes: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        freeze_backbone: bool = True,
    ):
        """
        Initialize classifier

        Args:
            embedder: DNABERT embedder
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()

        self.embedder = embedder
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.embedder.model.parameters():
                param.requires_grad = False

        # Build classifier head
        layers = []
        in_dim = embedder.embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    torch.nn.Linear(in_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, num_classes))

        self.classifier = torch.nn.Sequential(*layers)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        Forward pass

        Args:
            sequences: List of DNA sequences

        Returns:
            Logits (batch_size, num_classes)
        """
        # Get embeddings
        embeddings = self.embedder.encode(sequences)
        embeddings = torch.from_numpy(embeddings).to(self.embedder.device)

        # Classify
        logits = self.classifier(embeddings)
        return logits
