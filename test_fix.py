#!/usr/bin/env python3
"""
Test script to verify the fix for the AttributeError in _add_advanced_taxonomic_section
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analysis.dataset_analyzer import DatasetAnalyzer

def test_advanced_taxonomic_section():
    """Test the _add_advanced_taxonomic_section method with list data."""
    
    # Create a mock analyzer instance
    analyzer = DatasetAnalyzer()
    
    # Test data that mimics the actual output from advanced_taxonomic_analyzer
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
    
    # Test lines
    lines = []
    
    # This should not raise an AttributeError
    try:
        analyzer._add_advanced_taxonomic_section(lines, taxonomic_data)
        print("✅ Test passed: _add_advanced_taxonomic_section handled list correctly")
        print(f"Generated {len(lines)} lines of output")
        for line in lines:
            print(f"  {line}")
        assert len(lines) > 0
        print("✅ Advanced taxonomic features working correctly")
    except Exception as e:
        print(f"Error in advanced taxonomy section: {str(e)}")
        raise
    except AttributeError as e:
        print(f"❌ Test failed with AttributeError: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_advanced_taxonomic_section()
    sys.exit(0 if success else 1)