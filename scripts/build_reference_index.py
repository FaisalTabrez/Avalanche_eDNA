#!/usr/bin/env python3
"""
Build a KNN reference index for taxonomy assignment.

Given a labeled reference FASTA (and optionally a labels CSV), this script:
- embeds sequences using the current embedding backend (as configured in config/config.yaml)
- applies the same post-processing (PCA to 256, L2 normalization)
- writes reference_embeddings.npy and reference_labels.csv under the output directory
- optionally writes a small metadata JSON file describing the build

Usage examples (PowerShell / Windows):

python scripts\\build_reference_index.py \
  --fasta F:\\Dataset\\reference.fasta \
  --output-dir data\\reference

# With labels CSV mapping (must include a 'sequence_id' column matching FASTA IDs)
python scripts\\build_reference_index.py \
  --fasta F:\\Dataset\\reference.fasta \
  --labels-csv F:\\Dataset\\reference_labels.csv \
  --output-dir data\\reference

Outputs (by default):
- data/reference/reference_embeddings.npy
- data/reference/reference_labels.csv
- data/reference/reference_meta.json

Note: This uses the same Nucleotide Transformer configuration as the main pipeline.
It will be slow on CPU for large reference sets; consider running on a machine with a GPU.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from Bio import SeqIO

# Ensure we can import local modules
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_pipeline import eDNABiodiversityPipeline  # type: ignore


def read_fasta_sequences(fasta_path: Path, max_seqs: Optional[int] = None) -> List[SeqIO.SeqRecord]:
    records = []
    count = 0
    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            records.append(rec)
            count += 1
            if max_seqs is not None and count >= max_seqs:
                break
    return records


def build_labels_from_records(records: List[SeqIO.SeqRecord]) -> pd.DataFrame:
    # Minimal labels: use description as species placeholder; leave higher ranks empty
    rows = []
    for rec in records:
        species = str(rec.description).strip()
        rows.append(
            {
                "sequence_id": rec.id,
                "species": species,
                "genus": None,
                "family": None,
                "order": None,
                "class": None,
                "phylum": None,
                "kingdom": None,
            }
        )
    return pd.DataFrame(rows)


def merge_labels(records: List[SeqIO.SeqRecord], labels_csv: Optional[Path]) -> pd.DataFrame:
    if labels_csv is None:
        return build_labels_from_records(records)
    df = pd.read_csv(labels_csv)
    if "sequence_id" not in df.columns:
        raise ValueError("labels CSV must contain a 'sequence_id' column matching FASTA record IDs")
    # Keep only known taxonomy columns (create if missing)
    wanted = ["sequence_id", "species", "genus", "family", "order", "class", "phylum", "kingdom"]
    for col in wanted:
        if col not in df.columns:
            df[col] = None
    # Filter and reorder
    df = df[wanted]
    # Align to FASTA order
    id_order = [rec.id for rec in records]
    df_indexed = df.set_index("sequence_id").reindex(id_order)
    missing = [sid for sid in id_order if sid not in df_indexed.index]
    if missing:
        print(f"[WARN] {len(missing)} sequence IDs missing in labels CSV; filling minimal labels.")
        fallback = build_labels_from_records(records)
        fallback = fallback.set_index("sequence_id").reindex(id_order)
        df_indexed = df_indexed.combine_first(fallback)
    return df_indexed.reset_index()


def embed_sequences(records: List[SeqIO.SeqRecord], out_dir: Path) -> np.ndarray:
    # Use the pipeline's embedding step so we inherit the configured model + post-processing
    pipeline = eDNABiodiversityPipeline()
    seqs = [str(rec.seq) for rec in records]
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = pipeline._run_embedding_step(seqs, out_dir)  # returns np.ndarray
    return embeddings.astype(np.float32)


def write_outputs(embeddings: np.ndarray, labels: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "reference_embeddings.npy"
    lab_path = out_dir / "reference_labels.csv"
    meta_path = out_dir / "reference_meta.json"

    np.save(emb_path, embeddings)
    labels.to_csv(lab_path, index=False)

    meta = {
        "num_sequences": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "paths": {
            "embeddings": str(emb_path),
            "labels": str(lab_path),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"embeddings": str(emb_path), "labels": str(lab_path), "meta": str(meta_path)}


def main() -> None:
    p = argparse.ArgumentParser(description="Build KNN reference index for taxonomy assignment")
    p.add_argument("--fasta", required=True, help="Path to reference FASTA")
    p.add_argument("--labels-csv", help="Optional CSV mapping 'sequence_id' to taxonomy columns")
    p.add_argument("--output-dir", default="data/reference", help="Output directory for reference files")
    p.add_argument("--max-seqs", type=int, help="Optional cap on number of sequences to embed (for testing)")

    args = p.parse_args()

    fasta_path = Path(args.fasta)
    labels_csv = Path(args.labels_csv) if args.labels_csv else None
    out_dir = Path(args.output_dir)

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    print(f"[INFO] Reading FASTA: {fasta_path}")
    records = read_fasta_sequences(fasta_path, max_seqs=args.max_seqs)
    if not records:
        raise ValueError("No sequences found in FASTA")
    print(f"[INFO] Loaded {len(records)} sequences")

    print("[INFO] Preparing labels...")
    labels = merge_labels(records, labels_csv)

    print("[INFO] Generating embeddings (using pipeline configuration)...")
    embeddings = embed_sequences(records, out_dir)
    if embeddings.shape[0] != len(records):
        raise RuntimeError("Embeddings count does not match number of sequences")

    print("[INFO] Writing reference artifacts...")
    paths = write_outputs(embeddings, labels, out_dir)

    print("[DONE]")
    print("Embeddings:", paths["embeddings"]) 
    print("Labels:", paths["labels"]) 
    print("Meta:", paths["meta"]) 
    print("\nUpdate config taxonomy.knn to:")
    print(f"  embeddings_path: \"{paths['embeddings']}\"")
    print(f"  labels_path: \"{paths['labels']}\"")


if __name__ == "__main__":
    main()
