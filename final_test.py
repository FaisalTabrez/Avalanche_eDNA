#!/usr/bin/env python3
"""
Final comprehensive test to verify the fix for the AttributeError
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.dataset_analyzer import DatasetAnalyzer

def test_all_scenarios():
    """Test all possible scenarios for novel_candidates data."""
    
    analyzer = DatasetAnalyzer()
    
    # Test 1: List of dictionaries (normal case)
    print("Test 1: List of dictionaries (normal case)")
    taxonomic_data1 = {
        'novel_candidates': [
            {
                'sequence_id': 'seq_1',
                'novelty_score': 0.85,
                'potential_rank': 'species'
            },
            {
                'sequence_id': 'seq_2',
                'novelty_score': 0.72,
                'potential_rank': 'genus'
            }
        ],
        'total_sequences': 100
    }
    
    lines1 = []
    try:
        analyzer._add_advanced_taxonomic_section(lines1, taxonomic_data1)
        print("✅ Test 1 passed")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Empty list
    print("\nTest 2: Empty list")
    taxonomic_data2 = {
        'novel_candidates': [],
        'total_sequences': 50
    }
    
    lines2 = []
    try:
        analyzer._add_advanced_taxonomic_section(lines2, taxonomic_data2)
        print("✅ Test 2 passed")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Dictionary (fallback case)
    print("\nTest 3: Dictionary (fallback case)")
    taxonomic_data3 = {
        'novel_candidates': {
            'candidate_count': 3,
            'novel_percentage': 6.0
        }
    }
    
    lines3 = []
    try:
        analyzer._add_advanced_taxonomic_section(lines3, taxonomic_data3)
        print("✅ Test 3 passed")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    # Test 4: Missing novel_candidates key
    print("\nTest 4: Missing novel_candidates key")
    taxonomic_data4 = {
        'other_key': 'value'
    }
    
    lines4 = []
    try:
        analyzer._add_advanced_taxonomic_section(lines4, taxonomic_data4)
        print("✅ Test 4 passed")
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The fix is working correctly.")
    return True

if __name__ == "__main__":
    success = test_all_scenarios()
    sys.exit(0 if success else 1)