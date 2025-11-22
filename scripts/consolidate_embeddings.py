#!/usr/bin/env python
"""
Consolidate Run Embeddings into Reference Index

Build a consolidated reference index from all run embeddings for:
- Cross-dataset similarity search
- Model training and fine-tuning
- Reference database building
- Incremental updates as new datasets are processed

This is separate from build_reference_index.py which builds taxonomy references from FASTA.

Usage:
    python scripts/consolidate_embeddings.py
    python scripts/consolidate_embeddings.py --incremental  # Update existing index
    python scripts/consolidate_embeddings.py --compress     # Compress source embeddings after indexing
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: FAISS not installed. Install with: pip install faiss-cpu")


def discover_embeddings(runs_root: Path) -> List[Tuple[Path, str, str]]:
    """
    Discover all embedding files in runs directory.
    
    Returns:
        List of (embedding_path, dataset_name, run_id)
    """
    embeddings = []
    
    if not runs_root.exists():
        print(f"Runs directory not found: {runs_root}")
        return embeddings
    
    # Structure: runs_root/dataset_name/run_id/embeddings.npy
    for emb_file in runs_root.rglob("embeddings.npy"):
        if emb_file.is_file():
            run_dir = emb_file.parent
            dataset_dir = run_dir.parent
            
            dataset_name = dataset_dir.name
            run_id = run_dir.name
            
            embeddings.append((emb_file, dataset_name, run_id))
    
    # Also check for compressed embeddings
    for emb_file in runs_root.rglob("embeddings.npz"):
        if emb_file.is_file():
            run_dir = emb_file.parent
            dataset_dir = run_dir.parent
            
            dataset_name = dataset_dir.name
            run_id = run_dir.name
            
            embeddings.append((emb_file, dataset_name, run_id))
    
    return sorted(embeddings, key=lambda x: (x[1], x[2]))


def load_embedding(path: Path) -> np.ndarray:
    """Load embedding from .npy or .npz file"""
    if path.suffix == '.npy':
        return np.load(path, mmap_mode='r')
    elif path.suffix == '.npz':
        with np.load(path) as data:
            return data['embeddings']
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_existing_reference(ref_dir: Path) -> Tuple[np.ndarray, List[Dict], set]:
    """
    Load existing reference index and metadata.
    
    Returns:
        (embeddings array, metadata list, set of processed run_ids)
    """
    emb_path = ref_dir / "consolidated_embeddings.npy"
    meta_path = ref_dir / "consolidated_metadata.json"
    
    if not emb_path.exists() or not meta_path.exists():
        return None, [], set()
    
    embeddings = np.load(emb_path)
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract processed run IDs
    processed = set(f"{m['dataset']}/{m['run_id']}" for m in metadata)
    
    return embeddings, metadata, processed


def build_consolidated_index(
    runs_root: Path,
    output_dir: Path,
    incremental: bool = False,
    compress_source: bool = False
):
    """
    Build consolidated reference index from all run embeddings.
    
    Args:
        runs_root: Root directory containing runs
        output_dir: Directory to save consolidated index
        incremental: Only add new runs to existing index
        compress_source: Compress source .npy files to .npz after indexing
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Building Consolidated Embedding Index")
    print("=" * 60)
    
    # Discover all embeddings
    print(f"\n1. Discovering embeddings in {runs_root}...")
    discovered = discover_embeddings(runs_root)
    print(f"   Found {len(discovered)} embedding files")
    
    if not discovered:
        print("   No embeddings found. Nothing to do.")
        return
    
    # Load existing reference if incremental
    existing_embeddings = None
    all_metadata = []
    processed_runs = set()
    
    if incremental:
        print("\n2. Loading existing consolidated index...")
        existing_embeddings, all_metadata, processed_runs = load_existing_reference(output_dir)
        if existing_embeddings is not None:
            print(f"   Existing index: {len(existing_embeddings)} sequences from {len(processed_runs)} runs")
        else:
            print("   No existing index found. Building from scratch.")
    
    # Process embeddings
    print(f"\n3. Processing embeddings...")
    new_embeddings = []
    new_metadata = []
    skipped = 0
    
    for emb_path, dataset, run_id in discovered:
        run_key = f"{dataset}/{run_id}"
        
        # Skip if already processed (incremental mode)
        if incremental and run_key in processed_runs:
            skipped += 1
            continue
        
        try:
            print(f"   Loading: {dataset}/{run_id} ({emb_path.name})")
            emb = load_embedding(emb_path)
            
            # Track metadata for each sequence
            for seq_idx in range(len(emb)):
                new_metadata.append({
                    'dataset': dataset,
                    'run_id': run_id,
                    'seq_idx': seq_idx,
                    'source_file': str(emb_path.relative_to(runs_root))
                })
            
            new_embeddings.append(np.array(emb))
            
            # Compress source file if requested
            if compress_source and emb_path.suffix == '.npy':
                compressed_path = emb_path.with_suffix('.npz')
                print(f"      Compressing to {compressed_path.name}...")
                np.savez_compressed(compressed_path, embeddings=emb)
                # Verify compression worked
                test = np.load(compressed_path)['embeddings']
                if test.shape == emb.shape:
                    emb_path.unlink()  # Delete original
                    print(f"      ✓ Compressed and removed original")
                else:
                    print(f"      ✗ Compression verification failed, keeping original")
            
        except Exception as e:
            print(f"   ERROR loading {emb_path}: {e}")
            continue
    
    print(f"   Processed: {len(new_embeddings)} new runs")
    if incremental and skipped > 0:
        print(f"   Skipped: {skipped} already indexed runs")
    
    # Combine with existing if incremental
    if existing_embeddings is not None and incremental:
        print("\n4. Merging with existing index...")
        all_embeddings = np.vstack([existing_embeddings] + new_embeddings)
        all_metadata = all_metadata + new_metadata
    else:
        print("\n4. Stacking all embeddings...")
        all_embeddings = np.vstack(new_embeddings)
        all_metadata = new_metadata
    
    print(f"   Total sequences: {len(all_embeddings):,}")
    print(f"   Embedding dimension: {all_embeddings.shape[1]}")
    print(f"   Total size: {all_embeddings.nbytes / 1024 / 1024:.1f} MB")
    
    # Save consolidated embeddings
    print(f"\n5. Saving consolidated embeddings...")
    emb_output = output_dir / "consolidated_embeddings.npy"
    np.save(emb_output, all_embeddings)
    print(f"   ✓ Saved to {emb_output}")
    
    # Save metadata
    print(f"\n6. Saving metadata...")
    meta_output = output_dir / "consolidated_metadata.json"
    metadata_summary = {
        'total_sequences': len(all_metadata),
        'total_runs': len(set(f"{m['dataset']}/{m['run_id']}" for m in all_metadata)),
        'datasets': sorted(list(set(m['dataset'] for m in all_metadata))),
        'embedding_dim': int(all_embeddings.shape[1]),
        'created_at': datetime.now().isoformat(),
        'incremental': incremental,
        'sequences': all_metadata
    }
    
    with open(meta_output, 'w') as f:
        json.dump(metadata_summary, f, indent=2)
    
    print(f"   ✓ Saved to {meta_output}")
    
    # Build FAISS index
    if FAISS_AVAILABLE:
        print(f"\n7. Building FAISS index...")
        d = all_embeddings.shape[1]
        
        # Use appropriate index based on size
        if len(all_embeddings) < 10000:
            # Small dataset: use exact search
            index = faiss.IndexFlatL2(d)
            index_type = "Flat (exact search)"
        else:
            # Large dataset: use IVF for faster approximate search
            nlist = min(100, len(all_embeddings) // 100)
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            # Train index
            print(f"   Training IVF index with {nlist} clusters...")
            index.train(all_embeddings.astype(np.float32))
            index_type = f"IVF{nlist} (approximate search)"
        
        # Add vectors
        index.add(all_embeddings.astype(np.float32))
        
        # Save index
        index_output = output_dir / "consolidated_index.faiss"
        faiss.write_index(index, str(index_output))
        
        print(f"   ✓ Built {index_type}")
        print(f"   ✓ Saved to {index_output}")
    else:
        print(f"\n7. Skipping FAISS index (not installed)")
    
    # Save summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total sequences:  {len(all_embeddings):,}")
    print(f"Total runs:       {metadata_summary['total_runs']}")
    print(f"Datasets:         {', '.join(metadata_summary['datasets'])}")
    print(f"Embedding dim:    {d}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - consolidated_embeddings.npy  ({all_embeddings.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  - consolidated_metadata.json")
    if FAISS_AVAILABLE:
        print(f"  - consolidated_index.faiss")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build consolidated index from all run embeddings"
    )
    parser.add_argument(
        '--runs-root',
        type=Path,
        default=Path('AvalancheData/runs'),
        help='Root directory containing runs (default: AvalancheData/runs)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/reference'),
        help='Output directory for consolidated index (default: data/reference)'
    )
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Only add new runs to existing index'
    )
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Compress source .npy files to .npz after indexing'
    )
    
    args = parser.parse_args()
    
    build_consolidated_index(
        runs_root=args.runs_root,
        output_dir=args.output,
        incremental=args.incremental,
        compress_source=args.compress
    )


if __name__ == '__main__':
    main()
