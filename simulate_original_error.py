#!/usr/bin/env python3
"""
Script to simulate what the original buggy code might have looked like
"""

def simulate_original_buggy_code():
    """Simulate the original buggy code that caused the error."""
    
    # This is what the original code might have looked like (buggy version)
    taxonomic_data = {
        'novel_candidates': [
            {
                'sequence_id': 'seq_1',
                'novelty_score': 0.85,
                'potential_rank': 'species'
            }
        ]
    }
    
    lines = []
    
    try:
        # Original buggy code that would cause the error:
        if 'novel_candidates' in taxonomic_data:
            novel = taxonomic_data['novel_candidates']  # This is a LIST, not a dict!
            # This line would cause the AttributeError:
            lines.append(f"- Potential novel species: {novel.get('candidate_count', 0)}")
            
        print("❌ This should have caused an error!")
        return False
    except AttributeError as e:
        if "'list' object has no attribute 'get'" in str(e):
            print(f"✅ Successfully reproduced the original error: {e}")
            return True
        else:
            print(f"❌ Different error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def simulate_fixed_code():
    """Simulate the current fixed code."""
    
    taxonomic_data = {
        'novel_candidates': [
            {
                'sequence_id': 'seq_1',
                'novelty_score': 0.85,
                'potential_rank': 'species'
            }
        ]
    }
    
    lines = []
    
    try:
        # Current fixed code:
        if 'novel_candidates' in taxonomic_data:
            novel_candidates = taxonomic_data['novel_candidates']  # This is a list
            
            # Check the type before calling methods
            if isinstance(novel_candidates, list):
                candidate_count = len(novel_candidates)  # Use len() for lists
                lines.append(f"- Potential novel species: {candidate_count}")
            elif isinstance(novel_candidates, dict):
                # Fallback for dictionaries
                candidate_count = novel_candidates.get('candidate_count', 0)
                lines.append(f"- Potential novel species: {candidate_count}")
            
        print("✅ Fixed code works correctly")
        return True
    except Exception as e:
        print(f"❌ Error in fixed code: {e}")
        return False

if __name__ == "__main__":
    print("Testing original buggy code:")
    simulate_original_buggy_code()
    
    print("\nTesting current fixed code:")
    simulate_fixed_code()