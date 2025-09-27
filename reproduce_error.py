#!/usr/bin/env python3
"""
Script to reproduce the exact error from the traceback
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.dataset_analyzer import DatasetAnalyzer

def reproduce_error():
    """Try to reproduce the exact error."""
    
    # Create analyzer instance
    analyzer = DatasetAnalyzer()
    
    # Create test data that mimics what might cause the error
    # This is what the advanced_taxonomic_analyzer actually returns
    taxonomic_data = {
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
    
    lines = []
    
    try:
        # This should work correctly with the current implementation
        analyzer._add_advanced_taxonomic_section(lines, taxonomic_data)
        print("✅ No error occurred - the fix is working correctly")
        return True
    except AttributeError as e:
        if "'list' object has no attribute 'get'" in str(e):
            print(f"❌ The error still occurs: {e}")
            print("This suggests there's still an issue with the code")
            return False
        else:
            print(f"❌ A different AttributeError occurred: {e}")
            return False
    except Exception as e:
        print(f"❌ A different error occurred: {e}")
        return False

if __name__ == "__main__":
    success = reproduce_error()
    sys.exit(0 if success else 1)