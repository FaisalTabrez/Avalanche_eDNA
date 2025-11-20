#!/usr/bin/env python3
"""
Train custom DNA embedding models for eDNA analysis

This script provides a complete workflow for training DNA sequence embedding models
using contrastive learning or autoencoder architectures.

Usage:
    # Train contrastive model
    python scripts/train_model.py --input data/training_sequences.fasta \\
                                  --labels data/labels.csv \\
                                  --model-type contrastive \\
                                  --output models/my_model

    # Train autoencoder model
    python scripts/train_model.py --input data/sequences.fasta \\
                                  --model-type autoencoder \\
                                  --output models/autoencoder_model

    # Resume training from checkpoint
    python scripts/train_model.py --input data/sequences.fasta \\
                                  --resume models/checkpoint \\
                                  --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.tokenizer import DNATokenizer, SequenceDataset
from src.models.embeddings import DNATransformerEmbedder, DNAAutoencoder, DNAContrastiveModel
from src.models.trainer import EmbeddingTrainer
from src.utils.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_sequences(input_path: str) -> List[str]:
    """
    Load DNA sequences from FASTA file
    
    Args:
        input_path: Path to FASTA file
        
    Returns:
        List of DNA sequences
    """
    sequences = []
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading sequences from {input_path}")
    
    try:
        from Bio import SeqIO
        for record in SeqIO.parse(input_file, 'fasta'):
            sequences.append(str(record.seq).upper())
    except ImportError:
        # Fallback to simple parsing if BioPython not available
        logger.warning("BioPython not available, using simple FASTA parser")
        with open(input_file, 'r') as f:
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq).upper())
                        current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:
                sequences.append(''.join(current_seq).upper())
    
    logger.info(f"Loaded {len(sequences)} sequences")
    return sequences


def load_labels(labels_path: Optional[str], sequences: List[str]) -> Optional[List[str]]:
    """
    Load labels for supervised training
    
    Args:
        labels_path: Path to labels file (CSV or TXT)
        sequences: List of sequences (for validation)
        
    Returns:
        List of labels or None if not provided
    """
    if labels_path is None:
        return None
    
    labels_file = Path(labels_path)
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    logger.info(f"Loading labels from {labels_path}")
    
    # Try CSV format first
    try:
        df = pd.read_csv(labels_file)
        if 'label' in df.columns:
            labels = df['label'].tolist()
        elif 'class' in df.columns:
            labels = df['class'].tolist()
        elif 'taxonomy' in df.columns:
            labels = df['taxonomy'].tolist()
        else:
            # Use first column
            labels = df.iloc[:, 0].tolist()
    except:
        # Fallback to simple text file (one label per line)
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
    
    if len(labels) != len(sequences):
        raise ValueError(f"Number of labels ({len(labels)}) does not match number of sequences ({len(sequences)})")
    
    logger.info(f"Loaded {len(labels)} labels ({len(set(labels))} unique classes)")
    return labels


def create_model(model_type: str, vocab_size: int, config_dict: Dict) -> Tuple:
    """
    Create model and tokenizer
    
    Args:
        model_type: Type of model ('transformer', 'autoencoder', 'contrastive')
        vocab_size: Vocabulary size
        config_dict: Configuration dictionary
        
    Returns:
        Tuple of (model, model_name)
    """
    embed_config = config_dict.get('embedding', {})
    training_config = config_dict.get('training', {})
    
    d_model = embed_config.get('embedding_dim', 256)
    num_layers = embed_config.get('transformer', {}).get('num_layers', 6)
    num_heads = embed_config.get('transformer', {}).get('num_heads', 8)
    dropout = embed_config.get('transformer', {}).get('dropout', 0.1)
    
    if model_type == 'transformer':
        model = DNATransformerEmbedder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        model_name = 'DNATransformerEmbedder'
        
    elif model_type == 'autoencoder':
        model = DNAAutoencoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        model_name = 'DNAAutoencoder'
        
    elif model_type == 'contrastive':
        # Create backbone first
        backbone = DNATransformerEmbedder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Wrap with contrastive learning
        projection_dim = training_config.get('projection_dim', 128)
        temperature = training_config.get('temperature', 0.1)
        
        model = DNAContrastiveModel(
            backbone_model=backbone,
            projection_dim=projection_dim,
            temperature=temperature
        )
        model_name = 'DNAContrastiveModel'
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Created {model_name} with vocab_size={vocab_size}, d_model={d_model}")
    return model, model_name


def train_model(args):
    """Main training function"""
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("Starting DNA Embedding Model Training")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
    
    # Override with command-line arguments
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            custom_config = yaml.safe_load(f)
            config_dict.update(custom_config)
    
    # Load sequences
    sequences = load_sequences(args.input)
    
    # Load labels if provided
    labels = load_labels(args.labels, sequences) if args.labels else None
    
    # Create tokenizer
    logger.info(f"Creating tokenizer with k-mer size {args.kmer_size}")
    tokenizer = DNATokenizer(
        encoding_type='kmer',
        kmer_size=args.kmer_size,
        stride=args.kmer_stride
    )
    
    # Resume from checkpoint or create new model
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # Load existing model
        checkpoint_path = Path(args.resume)
        
        # Determine model type from checkpoint
        if args.model_type:
            model, model_name = create_model(args.model_type, tokenizer.vocab_size, config_dict)
        else:
            # Try to infer from checkpoint metadata
            logger.warning("Model type not specified, defaulting to contrastive")
            model, model_name = create_model('contrastive', tokenizer.vocab_size, config_dict)
        
        trainer = EmbeddingTrainer(model, tokenizer, device=args.device)
        
        # Load checkpoint
        try:
            trainer.load_model(str(checkpoint_path))
            logger.info("Successfully loaded checkpoint")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting with fresh model")
    else:
        # Create new model
        model, model_name = create_model(args.model_type, tokenizer.vocab_size, config_dict)
        trainer = EmbeddingTrainer(model, tokenizer, device=args.device)
    
    # Prepare data
    logger.info("Preparing training data...")
    train_loader, val_loader = trainer.prepare_data(
        sequences=sequences,
        labels=labels,
        validation_split=args.val_split,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    if args.model_type == 'autoencoder' or isinstance(model, DNAAutoencoder):
        history = trainer.train_autoencoder(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
    else:
        # Contrastive or transformer (use contrastive training)
        history = trainer.train_contrastive(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir / "model"), include_tokenizer=True)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    # Save training metadata
    metadata = {
        'model_type': args.model_type,
        'model_class': model_name,
        'num_sequences': len(sequences),
        'num_classes': len(set(labels)) if labels else None,
        'kmer_size': args.kmer_size,
        'vocab_size': tokenizer.vocab_size,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'validation_split': args.val_split,
        'device': args.device,
        'training_time_seconds': time.time() - start_time,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Extract and save embeddings from training data
    if args.save_embeddings:
        logger.info("Extracting embeddings from training data...")
        embeddings = trainer.extract_embeddings(sequences, batch_size=args.batch_size)
        
        np.save(output_dir / "embeddings.npy", embeddings)
        
        # Save labels if available
        if labels:
            labels_df = pd.DataFrame({'label': labels})
            labels_df.to_csv(output_dir / "labels.csv", index=False)
        
        logger.info(f"Saved embeddings: {embeddings.shape}")
    
    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Final training loss: {metadata['final_train_loss']:.4f}")
    logger.info(f"Final validation loss: {metadata['final_val_loss']:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Train DNA embedding models for eDNA analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', required=True,
                        help='Input FASTA file with DNA sequences')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for trained model')
    parser.add_argument('--labels', '-l', default=None,
                        help='Labels file (CSV or TXT) for supervised training')
    parser.add_argument('--config', '-c', default=None,
                        help='Custom configuration YAML file')
    
    # Model configuration
    parser.add_argument('--model-type', choices=['transformer', 'autoencoder', 'contrastive'],
                        default='contrastive',
                        help='Type of model to train (default: contrastive)')
    parser.add_argument('--kmer-size', type=int, default=6,
                        help='K-mer size for tokenization (default: 6)')
    parser.add_argument('--kmer-stride', type=int, default=1,
                        help='K-mer stride for tokenization (default: 1)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split fraction (default: 0.2)')
    
    # Device configuration
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'],
                        default='auto',
                        help='Device for training (default: auto)')
    
    # Resume training
    parser.add_argument('--resume', default=None,
                        help='Resume training from checkpoint directory')
    
    # Additional options
    parser.add_argument('--save-embeddings', action='store_true',
                        help='Save embeddings of training data')
    
    args = parser.parse_args()
    
    try:
        train_model(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
