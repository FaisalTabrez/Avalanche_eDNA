"""
Example: Training DNA Sequence Embeddings with Contrastive Learning

This example demonstrates how to use the ML training infrastructure to train
DNA sequence embeddings using contrastive learning.
"""

import numpy as np
from pathlib import Path

from src.models.tokenizer import DNATokenizer, SequenceDataset
from src.models.embeddings import DNATransformerEmbedder, DNAContrastiveModel, ModelFactory
from src.models.trainer import EmbeddingTrainer


def example_contrastive_training():
    """Example of training a contrastive model for DNA sequences"""
    
    print("=" * 80)
    print("DNA Sequence Contrastive Learning Example")
    print("=" * 80)
    
    # 1. Prepare sample DNA sequences with labels
    sequences = [
        "ATCGATCGATCGATCGATCG" * 5,  # Class A (100bp)
        "ATCGATCGATCGATCGATCG" * 5,  # Class A
        "GCTAGCTAGCTAGCTAGCTA" * 5,  # Class B
        "GCTAGCTAGCTAGCTAGCTA" * 5,  # Class B
        "TTAATTAATTAATTAATTAA" * 5,  # Class C
        "TTAATTAATTAATTAATTAA" * 5,  # Class C
        "CCGGCCGGCCGGCCGGCCGG" * 5,  # Class D
        "CCGGCCGGCCGGCCGGCCGG" * 5,  # Class D
    ] * 3  # Replicate for more training data
    
    labels = ["gene_A", "gene_A", "gene_B", "gene_B", 
              "gene_C", "gene_C", "gene_D", "gene_D"] * 3
    
    print(f"\n📊 Dataset: {len(sequences)} sequences, {len(set(labels))} classes")
    
    # 2. Create tokenizer
    print("\n🔤 Creating DNA tokenizer...")
    tokenizer = DNATokenizer(
        encoding_type='kmer',
        kmer_size=6,
        stride=1,
        add_special_tokens=True
    )
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # 3. Create backbone and contrastive model
    print("\n🧠 Creating contrastive model...")
    backbone_config = {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'max_len': 512
    }
    
    backbone = ModelFactory.create_transformer(
        vocab_size=tokenizer.vocab_size,
        config=backbone_config
    )
    
    contrastive_config = {
        'projection_dim': 64,
        'temperature': 0.1
    }
    
    model = ModelFactory.create_contrastive(
        backbone_model=backbone,
        config=contrastive_config
    )
    
    print(f"   Backbone: {sum(p.numel() for p in backbone.parameters()):,} parameters")
    print(f"   Projection: {sum(p.numel() for p in model.projection_head.parameters()):,} parameters")
    
    # 4. Create trainer
    print("\n🎯 Initializing trainer...")
    trainer = EmbeddingTrainer(model, tokenizer, device='cpu')
    
    # 5. Prepare data loaders
    print("\n📦 Preparing data loaders...")
    train_loader, val_loader = trainer.prepare_data(
        sequences=sequences,
        labels=labels,
        validation_split=0.2,
        batch_size=8,
        max_length=200
    )
    
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    
    # 6. Train the model
    print("\n🚀 Training contrastive model...")
    print("   (This will take a few minutes on CPU)")
    
    history = trainer.train_contrastive(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        learning_rate=1e-4,
        save_best=False
    )
    
    print("\n📈 Training History:")
    for epoch in range(len(history['train_loss'])):
        print(f"   Epoch {epoch+1}: Train Loss={history['train_loss'][epoch]:.4f}, "
              f"Val Loss={history['val_loss'][epoch]:.4f}")
    
    # 7. Extract embeddings
    print("\n🔍 Extracting embeddings...")
    test_sequences = sequences[:4]  # First 4 sequences
    embeddings = trainer.extract_embeddings(test_sequences, batch_size=2, max_length=200)
    
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Embeddings mean: {embeddings.mean():.4f}")
    print(f"   Embeddings std: {embeddings.std():.4f}")
    
    # 8. Compute similarity between sequences
    print("\n📊 Computing pairwise similarities...")
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity(embeddings)
    print(f"   Similarity matrix:")
    for i in range(len(test_sequences)):
        print(f"   Seq {i}: {' '.join([f'{s:.3f}' for s in similarities[i]])}")
    
    print("\n   Expected: Same class pairs should have higher similarity!")
    print(f"   Same class (0,1): {similarities[0,1]:.3f}")
    print(f"   Different class (0,2): {similarities[0,2]:.3f}")
    
    # 9. Save model
    print("\n💾 Saving model...")
    save_path = Path("models/contrastive_dna_model")
    trainer.save_model(save_path, include_tokenizer=True)
    print(f"   Model saved to: {save_path}")
    
    print("\n✅ Example completed successfully!")
    print("=" * 80)


def example_embedding_extraction():
    """Example of using a trained model for embedding extraction"""
    
    print("\n" + "=" * 80)
    print("Embedding Extraction Example")
    print("=" * 80)
    
    # Create a simple model for demonstration
    tokenizer = DNATokenizer(encoding_type='kmer', kmer_size=4)
    
    model_config = {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'max_len': 256
    }
    
    model = ModelFactory.create_transformer(
        vocab_size=tokenizer.vocab_size,
        config=model_config
    )
    
    trainer = EmbeddingTrainer(model, tokenizer, device='cpu')
    
    # Extract embeddings from sequences
    sequences = [
        "ATCGATCGATCG",
        "GCTAGCTAGCTA",
        "TTAATTAATTAA",
    ]
    
    print(f"\n🔍 Extracting embeddings for {len(sequences)} sequences...")
    embeddings = trainer.extract_embeddings(sequences, batch_size=2, max_length=50)
    
    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"   Sequence 1 embedding (first 10 dims): {embeddings[0, :10]}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Run contrastive training example
    example_contrastive_training()
    
    # Run embedding extraction example
    example_embedding_extraction()
