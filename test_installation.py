#!/usr/bin/env python
"""
Test script to verify installation and core functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all core imports"""
    print("🧪 Testing Core Imports...")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
        
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly: {e}")
        return False
    
    try:
        import umap
        print(f"✅ UMAP-learn")
    except ImportError as e:
        print(f"❌ UMAP-learn: {e}")
        return False
    
    return True

def test_custom_modules():
    """Test custom eDNA modules"""
    print("\n🧬 Testing eDNA Modules...")
    
    try:
        from models.tokenizer import DNATokenizer
        tokenizer = DNATokenizer(encoding_type="kmer", kmer_size=3)
        encoded = tokenizer.encode_sequence("ATCGATCG")
        print("✅ DNA Tokenizer works!")
    except Exception as e:
        print(f"❌ DNA Tokenizer: {e}")
        return False
    
    try:
        from clustering.algorithms import EmbeddingClusterer
        clusterer = EmbeddingClusterer(method="kmeans", n_clusters=3)
        print("✅ Clustering algorithms work!")
    except Exception as e:
        print(f"❌ Clustering: {e}")
        return False
    
    try:
        from novelty.detection import NoveltyDetector
        detector = NoveltyDetector(method="isolation_forest")
        print("✅ Novelty detection works!")
    except Exception as e:
        print(f"❌ Novelty detection: {e}")
        return False
    
    try:
        from visualization.plots import BiodiversityPlotter
        plotter = BiodiversityPlotter()
        print("✅ Visualization works!")
    except Exception as e:
        print(f"❌ Visualization: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic end-to-end functionality"""
    print("\n🔬 Testing Basic Functionality...")
    
    try:
        # Test tokenization
        from models.tokenizer import DNATokenizer
        import numpy as np
        
        sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA", "TTAATTAATTAA"]
        tokenizer = DNATokenizer(encoding_type="kmer", kmer_size=3)
        encoded = tokenizer.encode_sequences(sequences, max_length=10)
        
        print(f"✅ Tokenized {len(sequences)} sequences")
        print(f"   Shape: {encoded['input_ids'].shape}")
        
        # Test clustering with mock embeddings
        from clustering.algorithms import EmbeddingClusterer
        
        embeddings = np.random.randn(100, 64)
        clusterer = EmbeddingClusterer(method="kmeans", n_clusters=3)
        labels = clusterer.fit(embeddings)
        
        print(f"✅ Clustered {len(embeddings)} embeddings into {len(np.unique(labels))} clusters")
        
        # Test novelty detection
        from novelty.detection import NoveltyDetector
        
        reference_embeddings = np.random.randn(50, 64)
        query_embeddings = np.random.randn(20, 64)
        
        detector = NoveltyDetector(method="isolation_forest")
        detector.fit(reference_embeddings)
        predictions = detector.predict(query_embeddings)
        
        novel_count = np.sum(predictions == -1)
        print(f"✅ Novelty detection: {novel_count}/{len(predictions)} novel sequences detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🌊" + "="*50)
    print("  eDNA Biodiversity System - Installation Test")
    print("="*50 + "🌊")
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Core imports failed. Please install missing dependencies.")
        return False
    
    # Test custom modules
    modules_ok = test_custom_modules()
    
    if not modules_ok:
        print("\n❌ Custom modules failed. Please check the installation.")
        return False
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    if not functionality_ok:
        print("\n❌ Basic functionality test failed.")
        return False
    
    # Success
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("="*60)
    print("\n✅ Your eDNA Biodiversity Assessment System is ready!")
    print("\n🚀 Next steps:")
    print("   • Run demo: python scripts/run_demo.py")
    print("   • Launch dashboard: python scripts/launch_dashboard.py")
    print("   • Run analysis: python scripts/run_pipeline.py --help")
    print("\n📚 Documentation:")
    print("   • User Guide: docs/user_guide.md")
    print("   • API Reference: docs/api_reference.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)