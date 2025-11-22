#!/usr/bin/env python
"""
Compress Run Embeddings to Save Storage

Converts .npy embeddings to compressed .npz format (typically 50-60% smaller).
Verifies compressed files before deleting originals.

Usage:
    python scripts/compress_embeddings.py                    # Dry run
    python scripts/compress_embeddings.py --execute          # Actually compress
    python scripts/compress_embeddings.py --older-than 30    # Only compress runs older than 30 days
"""

import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


def get_file_age_days(path: Path) -> float:
    """Get age of file in days"""
    mtime = path.stat().st_mtime
    age_seconds = time.time() - mtime
    return age_seconds / (60 * 60 * 24)


def compress_embedding(npy_path: Path, verify: bool = True, delete_original: bool = False) -> bool:
    """
    Compress a .npy embedding file to .npz.
    
    Args:
        npy_path: Path to .npy file
        verify: Verify compressed file before deleting original
        delete_original: Delete original after successful compression
        
    Returns:
        True if compression successful
    """
    npz_path = npy_path.with_suffix('.npz')
    
    # Skip if already compressed
    if npz_path.exists():
        print(f"   SKIP: {npy_path.name} (already compressed)")
        return False
    
    try:
        # Load original
        original = np.load(npy_path, mmap_mode='r')
        original_size = npy_path.stat().st_size
        
        # Compress
        np.savez_compressed(npz_path, embeddings=original)
        compressed_size = npz_path.stat().st_size
        
        # Verify if requested
        if verify:
            with np.load(npz_path) as data:
                compressed = data['embeddings']
            
            if compressed.shape != original.shape or compressed.dtype != original.dtype:
                print(f"   ERROR: Verification failed for {npy_path.name}")
                npz_path.unlink()
                return False
            
            # Spot check a few values
            if not np.allclose(original[0], compressed[0], rtol=1e-6):
                print(f"   ERROR: Data mismatch for {npy_path.name}")
                npz_path.unlink()
                return False
        
        # Report compression ratio
        ratio = (1 - compressed_size / original_size) * 100
        savings_mb = (original_size - compressed_size) / 1024 / 1024
        
        print(f"   ✓ {npy_path.name}: {original_size/1024/1024:.1f}MB → {compressed_size/1024/1024:.1f}MB "
              f"(saved {ratio:.1f}%, {savings_mb:.1f}MB)")
        
        # Delete original if requested
        if delete_original:
            npy_path.unlink()
            print(f"     Deleted original")
        
        return True
        
    except Exception as e:
        print(f"   ERROR compressing {npy_path.name}: {e}")
        if npz_path.exists():
            npz_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compress run embeddings to save storage"
    )
    parser.add_argument(
        '--runs-root',
        type=Path,
        default=Path('AvalancheData/runs'),
        help='Root directory containing runs (default: AvalancheData/runs)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually compress files (default: dry run)'
    )
    parser.add_argument(
        '--delete-original',
        action='store_true',
        help='Delete original .npy files after successful compression'
    )
    parser.add_argument(
        '--older-than',
        type=int,
        default=0,
        help='Only compress files older than N days (default: 0, all files)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Embedding Compression Tool")
    print("=" * 60)
    
    if not args.execute:
        print("\n⚠️  DRY RUN MODE - no files will be modified")
        print("   Add --execute to actually compress files\n")
    
    # Discover .npy files
    print(f"\nScanning {args.runs_root} for .npy embeddings...")
    npy_files = list(args.runs_root.rglob("embeddings.npy"))
    
    if not npy_files:
        print("No .npy embedding files found.")
        return
    
    print(f"Found {len(npy_files)} embedding files\n")
    
    # Filter by age if requested
    if args.older_than > 0:
        cutoff = datetime.now() - timedelta(days=args.older_than)
        npy_files = [f for f in npy_files if get_file_age_days(f) >= args.older_than]
        print(f"Filtered to {len(npy_files)} files older than {args.older_than} days\n")
    
    # Process files
    total_original_size = 0
    total_compressed_size = 0
    compressed_count = 0
    skipped_count = 0
    
    for npy_file in npy_files:
        dataset = npy_file.parent.parent.name
        run_id = npy_file.parent.name
        age_days = get_file_age_days(npy_file)
        
        print(f"\n{dataset}/{run_id} (age: {age_days:.1f} days)")
        
        if args.execute:
            success = compress_embedding(
                npy_file,
                verify=True,
                delete_original=args.delete_original
            )
            
            if success:
                compressed_count += 1
                # Update totals
                npz_file = npy_file.with_suffix('.npz')
                if npz_file.exists():
                    total_compressed_size += npz_file.stat().st_size
                total_original_size += npy_file.stat().st_size if npy_file.exists() else 0
            else:
                skipped_count += 1
        else:
            # Dry run - just report size
            size_mb = npy_file.stat().st_size / 1024 / 1024
            print(f"   Would compress: {npy_file.name} ({size_mb:.1f}MB)")
            total_original_size += npy_file.stat().st_size
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if args.execute:
        print(f"Compressed:       {compressed_count}")
        print(f"Skipped:          {skipped_count}")
        if compressed_count > 0:
            savings_mb = (total_original_size - total_compressed_size) / 1024 / 1024
            ratio = (1 - total_compressed_size / total_original_size) * 100
            print(f"Total savings:    {savings_mb:.1f}MB ({ratio:.1f}%)")
        if args.delete_original:
            print(f"\n✓ Original .npy files deleted")
        else:
            print(f"\n⚠️  Original .npy files kept (use --delete-original to remove)")
    else:
        print(f"Files to compress: {len(npy_files)}")
        print(f"Total size:        {total_original_size/1024/1024:.1f}MB")
        print(f"Est. savings:      ~{total_original_size*0.5/1024/1024:.1f}MB (50%)")
        print(f"\nRun with --execute to compress files")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
