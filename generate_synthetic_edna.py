"""
Generate Synthetic eDNA Dataset for Continual Learning Testing

Creates a larger, more realistic eDNA dataset with multiple organism groups:
- Marine bacteria (16S rRNA)
- Freshwater algae (18S rRNA) 
- Soil fungi (ITS)
- River microbes (COI)
- Lake zooplankton (12S rRNA)

This allows testing continual learning at larger scale with diverse sequences.
"""

import random
import numpy as np
from pathlib import Path
from typing import List, Tuple


# DNA base frequencies for different organism types
ORGANISM_PROFILES = {
    'marine_bacteria': {
        'name': 'Marine Bacteria (16S rRNA)',
        'gc_content': 0.55,  # Higher GC in marine bacteria
        'avg_length': 450,
        'length_std': 50,
        'motifs': ['TACG', 'GTAC', 'CAGT']  # Common 16S motifs
    },
    'freshwater_algae': {
        'name': 'Freshwater Algae (18S rRNA)',
        'gc_content': 0.50,
        'avg_length': 400,
        'length_std': 40,
        'motifs': ['TGCA', 'ACGT', 'CGTA']
    },
    'soil_fungi': {
        'name': 'Soil Fungi (ITS)',
        'gc_content': 0.48,
        'avg_length': 350,
        'length_std': 60,
        'motifs': ['ATGC', 'GCTA', 'TAGC']
    },
    'river_microbes': {
        'name': 'River Microbes (COI)',
        'gc_content': 0.45,
        'avg_length': 300,
        'length_std': 45,
        'motifs': ['CATG', 'GTCA', 'ATCG']
    },
    'lake_zooplankton': {
        'name': 'Lake Zooplankton (12S rRNA)',
        'gc_content': 0.42,
        'avg_length': 250,
        'length_std': 35,
        'motifs': ['GCAT', 'TACG', 'CGTG']
    }
}


def generate_sequence(gc_content: float, length: int, motifs: List[str]) -> str:
    """
    Generate a realistic DNA sequence with given GC content and motifs.
    
    Args:
        gc_content: Target GC content (0-1)
        length: Sequence length
        motifs: List of motifs to include
        
    Returns:
        DNA sequence string
    """
    # Calculate base frequencies
    gc = gc_content / 2  # Split between G and C
    at = (1 - gc_content) / 2  # Split between A and T
    
    bases = ['A', 'T', 'G', 'C']
    weights = [at, at, gc, gc]
    
    # Generate random sequence
    sequence = ''.join(random.choices(bases, weights=weights, k=length))
    
    # Insert motifs at random positions
    for motif in motifs[:2]:  # Use 2 motifs per sequence
        if len(sequence) > len(motif) + 10:
            pos = random.randint(5, len(sequence) - len(motif) - 5)
            sequence = sequence[:pos] + motif + sequence[pos+len(motif):]
    
    return sequence


def generate_dataset(
    organism_type: str,
    num_sequences: int,
    output_file: Path,
    start_id: int = 0
) -> int:
    """
    Generate synthetic eDNA sequences for one organism type.
    
    Args:
        organism_type: Type of organism (key in ORGANISM_PROFILES)
        num_sequences: Number of sequences to generate
        output_file: Output FASTA file path
        start_id: Starting sequence ID
        
    Returns:
        Next available sequence ID
    """
    profile = ORGANISM_PROFILES[organism_type]
    
    print(f"Generating {num_sequences} sequences for {profile['name']}...")
    
    sequences = []
    for i in range(num_sequences):
        # Random length around average
        length = int(np.random.normal(profile['avg_length'], profile['length_std']))
        length = max(100, min(600, length))  # Clip to reasonable range
        
        # Generate sequence
        seq = generate_sequence(
            profile['gc_content'],
            length,
            profile['motifs']
        )
        
        # Create FASTA entry
        seq_id = start_id + i
        header = f">seq_{seq_id}|{organism_type}|length_{length}"
        sequences.append((header, seq))
    
    # Write to file
    with open(output_file, 'w') as f:
        for header, seq in sequences:
            f.write(f"{header}\n{seq}\n")
    
    print(f"  ✓ Wrote {num_sequences} sequences to {output_file.name}")
    print(f"    Length range: {min(len(s[1]) for s in sequences)}-{max(len(s[1]) for s in sequences)} bp")
    print(f"    GC content: {profile['gc_content']:.1%}")
    
    return start_id + num_sequences


def create_mixed_dataset(output_file: Path, sequences_per_type: int = 500):
    """
    Create a mixed dataset with all organism types.
    
    Args:
        output_file: Output FASTA file path
        sequences_per_type: Number of sequences per organism type
    """
    print(f"\n{'='*60}")
    print(f"Creating Mixed eDNA Dataset")
    print(f"{'='*60}\n")
    
    all_sequences = []
    seq_id = 0
    
    for org_type, profile in ORGANISM_PROFILES.items():
        print(f"Generating {sequences_per_type} sequences for {profile['name']}...")
        
        for i in range(sequences_per_type):
            # Random length
            length = int(np.random.normal(profile['avg_length'], profile['length_std']))
            length = max(100, min(600, length))
            
            # Generate sequence
            seq = generate_sequence(
                profile['gc_content'],
                length,
                profile['motifs']
            )
            
            # Create entry
            header = f">seq_{seq_id}|{org_type}|length_{length}"
            all_sequences.append((header, seq, org_type))
            seq_id += 1
        
        print(f"  ✓ Generated {sequences_per_type} sequences")
    
    # Shuffle to mix organism types
    random.shuffle(all_sequences)
    
    # Write to file
    print(f"\nWriting mixed dataset to {output_file}...")
    with open(output_file, 'w') as f:
        for header, seq, _ in all_sequences:
            f.write(f"{header}\n{seq}\n")
    
    # Statistics
    total_seqs = len(all_sequences)
    total_bp = sum(len(s[1]) for s in all_sequences)
    avg_length = total_bp / total_seqs
    
    print(f"\n{'='*60}")
    print(f"✅ Dataset Created Successfully!")
    print(f"{'='*60}")
    print(f"Total sequences: {total_seqs:,}")
    print(f"Total base pairs: {total_bp:,}")
    print(f"Average length: {avg_length:.0f} bp")
    print(f"Organism types: {len(ORGANISM_PROFILES)}")
    print(f"Sequences per type: {sequences_per_type}")
    print(f"Output file: {output_file}")
    print(f"File size: ~{output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Show organism distribution
    print(f"\nOrganism Distribution:")
    from collections import Counter
    org_counts = Counter(s[2] for s in all_sequences)
    for org_type, count in sorted(org_counts.items()):
        profile = ORGANISM_PROFILES[org_type]
        print(f"  {profile['name']:<40} {count:>5} sequences")


def main():
    """Generate synthetic eDNA datasets."""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Output directory
    output_dir = Path('data/synthetic_edna')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Option 1: Create separate files per organism type (for sequential training)
    print("Option 1: Separate files per organism type")
    print("-" * 60)
    
    seq_id = 0
    for org_type in ORGANISM_PROFILES.keys():
        output_file = output_dir / f"{org_type}.fasta"
        seq_id = generate_dataset(org_type, num_sequences=500, output_file=output_file, start_id=seq_id)
    
    # Option 2: Create mixed dataset (for overall testing)
    print(f"\n{'='*60}")
    print("Option 2: Mixed dataset with all organisms")
    print("-" * 60)
    
    mixed_file = output_dir / "mixed_edna_2500.fasta"
    create_mixed_dataset(mixed_file, sequences_per_type=500)
    
    # Option 3: Larger dataset
    print(f"\n{'='*60}")
    print("Option 3: Larger mixed dataset (5000 sequences)")
    print("-" * 60)
    
    large_file = output_dir / "mixed_edna_5000.fasta"
    create_mixed_dataset(large_file, sequences_per_type=1000)
    
    print(f"\n{'='*60}")
    print("✅ All Datasets Generated!")
    print(f"{'='*60}")
    print(f"\nGenerated files in {output_dir}:")
    for f in sorted(output_dir.glob('*.fasta')):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:<30} {size_mb:>6.1f} MB")
    
    print(f"\nNext steps:")
    print(f"  1. Run clustering: python edna_analysis_pipeline.py")
    print(f"  2. Train with continual learning: python train_edna_continual.py")
    print(f"  3. Compare performance on different dataset sizes")


if __name__ == "__main__":
    main()
