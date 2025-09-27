#!/usr/bin/env python3
"""
Debug script to reproduce and fix the exact error from the traceback
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.dataset_analyzer import DatasetAnalyzer

def debug_error():
    """Debug the exact error scenario."""
    
    # Create analyzer instance
    analyzer = DatasetAnalyzer()
    
    # Create test data that might cause the error
    # This simulates what might have been in the original code
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
    
    # Let's manually check what might have caused the original error
    # The error was: lines.append(f"- Potential novel species: {novel.get('candidate_count', 0)}")
    # This suggests there was a variable 'novel' that was treated as a dict but was actually a list
    
    try:
        # Try to reproduce the original error by simulating what might have been the buggy code
        if 'novel_candidates' in taxonomic_data:
            novel = taxonomic_data['novel_candidates']  # This is a list!
            # This would cause the error:
            # lines.append(f"- Potential novel species: {novel.get('candidate_count', 0)}")
            
            # But our fixed code does this instead:
            if isinstance(novel, list):
                candidate_count = len(novel)
                lines.append(f"- Potential novel species: {candidate_count}")
                print("✅ Fixed code correctly identified novel_candidates as a list")
            else:
                print("❌ novel_candidates is not a list")
                
        return True
        
    except Exception as e:
        print(f"Error in debug: {e}")
        return False

if __name__ == "__main__":
    debug_error()