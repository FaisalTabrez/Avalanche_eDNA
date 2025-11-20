#!/usr/bin/env python3
"""
Build a BLAST nucleotide database from a reference FASTA with an optional seq_id→taxid map.
Optimized for Windows BLAST+ integration.

Requires BLAST+ (makeblastdb) installed.

Example (PowerShell):
python scripts\\build_blast_db.py `
  --fasta "reference\\combined\\18S\\references.fasta" `
  --taxid-map "reference\\mappings\\combined_18S_taxid_map.txt" `
  --db-out "reference\\indices\\18S\\combined_18S"

Example with sample data:
python scripts\\build_blast_db.py `
  --fasta "data\\sample\\sample_edna_sequences.fasta" `
  --db-out "reference\\indices\\sample_db"
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from utils.blast_utils import WindowsBLASTRunner, get_blast_runner
    BLAST_UTILS_AVAILABLE = True
except ImportError:
    BLAST_UTILS_AVAILABLE = False
    print("Warning: BLAST utilities not available, using basic implementation")


def build_db_with_utils(fasta: Path, db_out: Path, taxid_map: Path | None) -> None:
    """Build BLAST database using Windows BLAST utilities"""
    if not BLAST_UTILS_AVAILABLE:
        raise RuntimeError("BLAST utilities not available")
    
    blast_runner = get_blast_runner()
    
    # Ensure output directory exists
    db_out.parent.mkdir(parents=True, exist_ok=True)
    
    success = blast_runner.create_blast_database(
        fasta_file=str(fasta),
        database_name=str(db_out),
        database_type='nucl'
    )
    
    if not success:
        raise RuntimeError("Failed to create BLAST database using Windows BLAST utilities")


def build_db_basic(fasta: Path, db_out: Path, taxid_map: Path | None) -> None:
    """Basic BLAST database creation (fallback)"""
    import subprocess
    
    # Try to find makeblastdb in common Windows locations
    makeblastdb_paths = [
        'makeblastdb',  # If in PATH
        r'C:\Program Files\NCBI\blast-2.17.0+\bin\makeblastdb.exe'
    ]
    
    makeblastdb_cmd = None
    for path in makeblastdb_paths:
        try:
            result = subprocess.run([path, '-version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                makeblastdb_cmd = path
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    if not makeblastdb_cmd:
        raise RuntimeError('makeblastdb not found. Please install BLAST+ or check PATH.')
    
    # Ensure output directory exists
    db_out.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [makeblastdb_cmd, '-in', str(fasta), '-dbtype', 'nucl', '-parse_seqids', '-out', str(db_out)]
    if taxid_map and taxid_map.exists():
        cmd.extend(['-taxid_map', str(taxid_map)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"makeblastdb failed: {result.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser(description='Build BLAST DB from FASTA with optional taxid map')
    ap.add_argument('--fasta', required=True, type=str, help='Input FASTA file path')
    ap.add_argument('--db-out', required=True, type=str, help='Output DB prefix path (no extension)')
    ap.add_argument('--taxid-map', type=str, help='TSV with columns: sequence_id<tab>taxid')
    ap.add_argument('--force-basic', action='store_true', help='Force use of basic implementation')
    args = ap.parse_args()

    fasta = Path(args.fasta)
    db_out = Path(args.db_out)
    taxid_map = Path(args.taxid_map) if args.taxid_map else None

    if not fasta.exists():
        raise FileNotFoundError(f'FASTA not found: {fasta}')
    
    if taxid_map and not taxid_map.exists():
        print(f"Warning: Taxid map not found: {taxid_map}")
        taxid_map = None

    print(f"Building BLAST database...")
    print(f"  Input FASTA: {fasta}")
    print(f"  Output DB: {db_out}")
    if taxid_map:
        print(f"  Taxid map: {taxid_map}")
    
    try:
        if BLAST_UTILS_AVAILABLE and not args.force_basic:
            print("Using Windows BLAST utilities...")
            build_db_with_utils(fasta, db_out, taxid_map)
        else:
            print("Using basic BLAST implementation...")
            build_db_basic(fasta, db_out, taxid_map)
        
        print(f'[DONE] BLAST DB built successfully at: {db_out}')
        
        # List created files
        db_files = list(db_out.parent.glob(f"{db_out.name}.*"))
        if db_files:
            print(f"Created {len(db_files)} database files:")
            for f in sorted(db_files):
                print(f"  - {f.name}")
                
    except Exception as e:
        print(f"[ERROR] Failed to build BLAST database: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
