#!/usr/bin/env python3
"""
Debug script to check line numbers and identify where the error might be occurring
"""

import sys
import traceback
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.dataset_analyzer import DatasetAnalyzer

def debug_with_precise_error():
    """Debug with precise error tracking."""
    
    # Create analyzer instance
    analyzer = DatasetAnalyzer()
    
    # Create test data that might cause the error
    taxonomic_data = {
        'novel_candidates': [
            {
                'sequence_id': 'seq_1',
                'novelty_score': 0.85,
                'potential_rank': 'species'
            }
        ],
        'total_sequences': 100
    }
    
    lines = []
    
    try:
        # Call the method that's causing the error
        analyzer._add_advanced_taxonomic_section(lines, taxonomic_data)
        print("✅ No error occurred")
        return True
    except AttributeError as e:
        print(f"❌ AttributeError occurred: {e}")
        # Print the full traceback to see exact line numbers
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Other error occurred: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_with_precise_error()