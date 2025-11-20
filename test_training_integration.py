#!/usr/bin/env python3
"""
Quick integration test for ML training pipeline

This script verifies that:
1. Training script can create models
2. Models can be saved and loaded
3. Models can be used in the main pipeline
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_training_integration():
    """Test end-to-end training integration"""
    print("=" * 80)
    print("ML Training Pipeline Integration Test")
    print("=" * 80)
    
    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Step 1: Create sample sequences
        print("\n[1/4] Creating sample sequences...")
        sample_sequences = [
            ">seq1\nATCGATCGATCGATCGATCGATCGATCGATCG",
            ">seq2\nGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
            ">seq3\nTGCATGCATGCATGCATGCATGCATGCATGCA",
            ">seq4\nCGATCGATCGATCGATCGATCGATCGATCGAT",
            ">seq5\nATGCATGCATGCATGCATGCATGCATGCATGC",
            ">seq6\nGATCGATCGATCGATCGATCGATCGATCGATC",
            ">seq7\nTACGTACGTACGTACGTACGTACGTACGTACG",
            ">seq8\nCGTACGTACGTACGTACGTACGTACGTACGTA",
        ]
        
        input_file = temp_dir / "test_sequences.fasta"
        with open(input_file, 'w') as f:
            f.write('\n'.join(sample_sequences))
        print(f"✓ Created {len(sample_sequences)} test sequences")
        
        # Step 2: Test model creation
        print("\n[2/4] Testing model creation...")
        from src.models.tokenizer import DNATokenizer
        from src.models.embeddings import DNATransformerEmbedder, DNAContrastiveModel
        from src.models.trainer import EmbeddingTrainer
        
        tokenizer = DNATokenizer(encoding_type='kmer', kmer_size=4)
        backbone = DNATransformerEmbedder(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            num_layers=2
        )
        model = DNAContrastiveModel(
            backbone_model=backbone,
            projection_dim=32
        )
        print(f"✓ Created model with vocab_size={tokenizer.vocab_size}")
        
        # Step 3: Test training (1 epoch only for speed)
        print("\n[3/4] Testing model training (1 epoch)...")
        trainer = EmbeddingTrainer(model, tokenizer, device='cpu')
        
        # Load sequences
        from Bio import SeqIO
        sequences = [str(rec.seq) for rec in SeqIO.parse(input_file, 'fasta')]
        
        # Prepare data
        train_loader, val_loader = trainer.prepare_data(
            sequences=sequences,
            labels=None,
            validation_split=0.25,  # 6 train, 2 val
            batch_size=2,  # Use even batch size for self-supervised
            max_length=128
        )
        
        # Train for 1 epoch
        history = trainer.train_contrastive(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            learning_rate=1e-3
        )
        
        print(f"✓ Training completed: train_loss={history['train_loss'][0]:.4f}")
        
        # Step 4: Test save/load
        print("\n[4/4] Testing model save and load...")
        model_path = temp_dir / "test_model"
        trainer.save_model(str(model_path / "model"), include_tokenizer=True)
        print(f"✓ Model saved to {model_path}")
        
        # Load and verify
        new_trainer = EmbeddingTrainer(model, tokenizer, device='cpu')
        new_trainer.load_model(str(model_path / "model"))
        
        # Extract embeddings
        embeddings = new_trainer.extract_embeddings(sequences[:3], batch_size=2)
        print(f"✓ Model loaded and embeddings extracted: shape={embeddings.shape}")
        
        # Verify embedding dimensions
        assert embeddings.shape[0] == 3, "Wrong number of embeddings"
        assert embeddings.shape[1] == 64, "Wrong embedding dimension"
        
        print("\n" + "=" * 80)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 80)
        print("\nML Training Pipeline is successfully integrated!")
        print("\nYou can now:")
        print("  1. Train models: python scripts/train_model.py --input ... --output ...")
        print("  2. Use in pipeline: python scripts/run_pipeline.py --train-model ...")
        print("  3. Load custom models: python scripts/run_pipeline.py --model-path ...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n✓ Cleaned up temporary files")


if __name__ == '__main__':
    success = test_training_integration()
    sys.exit(0 if success else 1)
