#!/usr/bin/env python
"""
Search Reference Database for Similar Sequences

Query the consolidated reference index to find similar sequences across all datasets.

Usage:
    python scripts/search_reference.py --query "ATCGATCG..."
    python scripts/search_reference.py --query-file sequence.fasta --top-k 10
    python scripts/search_reference.py --run My_Dataset/2024-11-22_10-30-45 --seq-idx 42
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: FAISS not installed. Falling back to numpy search (slower).")


def load_reference(ref_dir: Path) -> Tuple[np.ndarray, List[Dict], object]:
    """
    Load reference embeddings, metadata, and FAISS index.
    
    Returns:
        (embeddings, metadata, faiss_index or None)
    """
    # Try consolidated index first, fall back to taxonomy reference
    emb_path = ref_dir / "consolidated_embeddings.npy"
    meta_path = ref_dir / "consolidated_metadata.json"
    index_path = ref_dir / "consolidated_index.faiss"
    
    if not emb_path.exists():
        # Fall back to taxonomy reference
        emb_path = ref_dir / "reference_embeddings.npy"
        meta_path = ref_dir / "reference_metadata.json"
        index_path = ref_dir / "reference_index.faiss"
    
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Reference not found in {ref_dir}")
    
    print(f"Loading reference from {ref_dir}...")
    print(f"  Using: {emb_path.name}")
    embeddings = np.load(emb_path)
    
    with open(meta_path, 'r') as f:
        data = json.load(f)
        # Handle both formats
        metadata = data.get('sequences', data)
    
    print(f"  Loaded {len(embeddings):,} sequences")
    
    # Load FAISS index if available
    index = None
    if FAISS_AVAILABLE and index_path.exists():
        index = faiss.read_index(str(index_path))
        print(f"  Loaded FAISS index from {index_path.name}")
    
    return embeddings, metadata, index


def search_with_faiss(index, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Search using FAISS index"""
    distances, indices = index.search(query.astype(np.float32), k)
    return distances[0], indices[0]


def search_with_numpy(embeddings: np.ndarray, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback search using numpy (slower)"""
    # Compute L2 distances
    distances = np.linalg.norm(embeddings - query, axis=1)
    # Get top-k indices
    indices = np.argsort(distances)[:k]
    return distances[indices], indices


def get_query_embedding(
    query_str: str = None,
    query_file: Path = None,
    run_path: Path = None,
    seq_idx: int = None
) -> np.ndarray:
    """
    Get query embedding from various sources.
    
    Priority: run_path/seq_idx > query_file > query_str
    """
    if run_path and seq_idx is not None:
        # Load from specific run
        emb_file = run_path / "embeddings.npy"
        if not emb_file.exists():
            emb_file = run_path / "embeddings.npz"
        
        if emb_file.suffix == '.npy':
            emb = np.load(emb_file, mmap_mode='r')
        else:
            with np.load(emb_file) as data:
                emb = data['embeddings']
        
        if seq_idx >= len(emb):
            raise IndexError(f"Sequence index {seq_idx} out of range (max: {len(emb)-1})")
        
        return emb[seq_idx:seq_idx+1]
    
    elif query_file:
        # Load from FASTA file and embed (requires embedding model)
        raise NotImplementedError("Embedding new sequences not yet implemented")
    
    elif query_str:
        # Direct embedding from sequence string (requires embedding model)
        raise NotImplementedError("Embedding new sequences not yet implemented")
    
    else:
        raise ValueError("Must provide query_str, query_file, or run_path + seq_idx")


def format_results(
    distances: np.ndarray,
    indices: np.ndarray,
    metadata: List[Dict],
    show_sequences: bool = False
) -> str:
    """Format search results for display"""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("SEARCH RESULTS")
    lines.append("=" * 80)
    
    for rank, (dist, idx) in enumerate(zip(distances, indices), 1):
        meta = metadata[idx]
        lines.append(f"\nRank {rank}: Distance = {dist:.4f}")
        lines.append(f"  Dataset:  {meta['dataset']}")
        lines.append(f"  Run ID:   {meta['run_id']}")
        lines.append(f"  Seq idx:  {meta['seq_idx']}")
        lines.append(f"  Source:   {meta['source_file']}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Search reference database for similar sequences"
    )
    parser.add_argument(
        '--reference',
        type=Path,
        default=Path('data/reference'),
        help='Reference directory (default: data/reference)'
    )
    
    # Query sources (mutually exclusive)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        '--query',
        type=str,
        help='Query sequence string'
    )
    query_group.add_argument(
        '--query-file',
        type=Path,
        help='Query sequence FASTA file'
    )
    query_group.add_argument(
        '--run',
        type=str,
        help='Run path (e.g., My_Dataset/2024-11-22_10-30-45)'
    )
    
    parser.add_argument(
        '--seq-idx',
        type=int,
        help='Sequence index in run (required with --run)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate run/seq-idx pairing
    if args.run and args.seq_idx is None:
        parser.error("--seq-idx required when using --run")
    if args.seq_idx is not None and not args.run:
        parser.error("--run required when using --seq-idx")
    
    # Load reference
    embeddings, metadata, index = load_reference(args.reference)
    
    # Get query embedding
    print("\nPreparing query...")
    
    if args.run:
        run_path = Path('consolidated_data/runs') / args.run
        query_emb = get_query_embedding(run_path=run_path, seq_idx=args.seq_idx)
        print(f"  Query: {args.run} sequence #{args.seq_idx}")
    else:
        # For now, these require embedding model (not implemented)
        print("  ERROR: Only --run queries are currently supported")
        print("  Embedding new sequences requires the embedding model (coming soon)")
        return
    
    # Search
    print(f"\nSearching for top {args.top_k} matches...")
    
    if index:
        distances, indices = search_with_faiss(index, query_emb, args.top_k)
    else:
        distances, indices = search_with_numpy(embeddings, query_emb, args.top_k)
    
    # Display results
    results_text = format_results(distances, indices, metadata)
    print(results_text)
    
    # Save if requested
    if args.output:
        results_data = {
            'query': {
                'run': args.run,
                'seq_idx': args.seq_idx,
            },
            'results': [
                {
                    'rank': rank,
                    'distance': float(dist),
                    'dataset': metadata[idx]['dataset'],
                    'run_id': metadata[idx]['run_id'],
                    'seq_idx': metadata[idx]['seq_idx'],
                    'source_file': metadata[idx]['source_file']
                }
                for rank, (dist, idx) in enumerate(zip(distances, indices), 1)
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n✓ Results saved to {args.output}")


if __name__ == '__main__':
    main()
