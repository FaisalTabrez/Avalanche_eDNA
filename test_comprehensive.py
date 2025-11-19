#!/usr/bin/env python3
"""
Comprehensive test to verify the fix for the AttributeError in dataset analyzer
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.dataset_analyzer import DatasetAnalyzer
from src.analysis.advanced_taxonomic_analyzer import AdvancedTaxonomicAnalyzer

def test_with_mock_data():
    """Test with mock sequence data to simulate the full analysis pipeline."""
    
    # Create a mock analyzer
    analyzer = DatasetAnalyzer()
    
    # Mock sequence data
    class MockSeqRecord:
        def __init__(self, id, seq="", description=""):
            self.id = id
            self.seq = seq
            self.description = description
            self.annotations = {}
    
    # Create some mock sequences
    sequences = [
        MockSeqRecord("seq1", "ATCG", "test sequence 1"),
        MockSeqRecord("seq2", "GGCC", "test sequence 2"),
        MockSeqRecord("seq3", "AATT", "test sequence 3"),
    ]
    
    # Test the full analysis pipeline
    try:
        # This will trigger the advanced taxonomic analysis
        if analyzer.advanced_taxonomic_analyzer:
            taxonomic_results = analyzer.advanced_taxonomic_analyzer.analyze_taxonomic_composition(sequences)
            
            # Verify the structure of novel_candidates
            print(f"novel_candidates type: {type(taxonomic_results.get('novel_candidates', []))}")
            print(f"novel_candidates content: {taxonomic_results.get('novel_candidates', [])[:2]}")  # First 2 items
            
            # Test the report generation section
            lines = []
            analyzer._add_advanced_taxonomic_section(lines, taxonomic_results)
            
            print("✅ Full pipeline test passed")
            assert taxonomic_results is not None
            print("✅ Test passed: successfully processed mock eDNA data")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    success = test_with_mock_data()
    sys.exit(0 if success else 1)