#!/usr/bin/env python3
"""
Prepare marine eukaryote references by harmonizing PR2/SILVA inputs into a unified
FASTA + taxonomy table with NCBI taxids (when resolvable).

Inputs (any subset is acceptable):
- --pr2-fasta, --pr2-taxonomy
- --silva-fasta, --silva-taxonomy
- --taxdump-dir (required for name→taxid mapping)

Outputs (under --out-dir):
- references.fasta                (deduplicated, cleaned)
- taxonomy.csv                    (sequence_id + ranks + source + name + taxid)
- Also writes taxid map to reference/mappings/combined_18S_taxid_map.txt by default

Example (PowerShell):
python scripts\\prepare_references.py `
  --pr2-fasta "reference\\pr2\\pr2_18S.fasta" `
  --pr2-taxonomy "reference\\pr2\\taxonomy.tsv" `
  --silva-fasta "reference\\silva\\silva_18S_euk.fasta" `
  --silva-taxonomy "reference\\silva\\taxonomy.tsv" `
  --taxdump-dir "F:\\Dataset\\taxdump" `
  --marker 18S `
  --out-dir "reference\\combined\\18S"
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio import SeqIO

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from clustering.taxonomy import TaxdumpResolver  # type: ignore


def load_name_to_taxid(taxdump_dir: Path) -> Dict[str, int]:
    """Build a simple name→taxid map from NCBI names.dmp (scientific names only)."""
    names_path = taxdump_dir / 'names.dmp'
    name_to_taxid: Dict[str, int] = {}
    with open(names_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 4:
                continue
            taxid_str, name_txt, _unique, name_class = parts[:4]
            if name_class == 'scientific name':
                try:
                    taxid = int(taxid_str)
                except ValueError:
                    continue
                key = name_txt.strip().lower()
                if key not in name_to_taxid:
                    name_to_taxid[key] = taxid
    return name_to_taxid


def read_taxonomy_table(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not path.exists():
        return None
    # Try auto-detect: TSV/CSV
    sep = '\t' if path.suffix.lower() in ['.tsv', '.txt'] else ','
    df = pd.read_csv(path, sep=sep)
    return df


def extract_name_from_header(header: str) -> Optional[str]:
    """Best-effort organism name extraction from FASTA header."""
    # Common patterns: Genus species; [Organism]; or full taxonomy; keep simple
    import re
    for pat in [r'\b([A-Z][a-z]+\s+[a-z][a-z\-]+)\b', r'\[([^\]]+)\]']:
        m = re.search(pat, header)
        if m:
            return m.group(1).strip()
    # Fallback: first token if looks taxon-like
    toks = header.split()
    if len(toks) >= 2 and toks[0][0].isupper() and toks[1].islower():
        return f"{toks[0]} {toks[1]}"
    return None


def iter_fasta_with_names(fasta: Optional[Path], taxonomy_df: Optional[pd.DataFrame], source: str) -> List[Tuple[str, str, Optional[str]]]:
    """Yield (seq_id, sequence, name) from a FASTA and optional taxonomy table."""
    if not fasta or not fasta.exists():
        return []
    records = []
    with open(fasta, 'r', encoding='utf-8', errors='ignore') as handle:
        for rec in SeqIO.parse(handle, 'fasta'):
            name: Optional[str] = None
            if taxonomy_df is not None:
                # Heuristic joins: sequence_id or header substring
                # Try exact on sequence ID
                if 'sequence_id' in taxonomy_df.columns:
                    row = taxonomy_df.loc[taxonomy_df['sequence_id'] == rec.id]
                    if not row.empty:
                        # Prefer a 'name' or 'scientific_name' column if present; else any taxonomy string
                        for col in ['name', 'scientific_name', 'organism', 'taxonomy', 'taxon']:
                            if col in taxonomy_df.columns and pd.notna(row.iloc[0].get(col)):
                                name = str(row.iloc[0][col])
                                break
                # If not found, try matching on rec.description
                if name is None and 'name' in taxonomy_df.columns:
                    # crude contains match; avoid O(N^2) for big tables by sampling; acceptable here for moderate files
                    cand = taxonomy_df[taxonomy_df['name'].astype(str).str.contains(rec.id, na=False)]
                    if not cand.empty:
                        name = str(cand.iloc[0]['name'])
            if name is None:
                name = extract_name_from_header(rec.description)
            records.append((rec.id, str(rec.seq), name))
    return records


def filter_and_dedupe(records: List[Tuple[str, str, Optional[str]]], max_n_fraction: float = 0.1, min_length: int = 60) -> List[Tuple[str, str, Optional[str]]]:
    out: List[Tuple[str, str, Optional[str]]] = []
    seen_seq: Dict[str, str] = {}
    for sid, seq, name in records:
        if len(seq) < min_length:
            continue
        n_frac = seq.upper().count('N') / max(len(seq), 1)
        if n_frac > max_n_fraction:
            continue
        # Deduplicate by exact sequence; keep first occurrence
        if seq in seen_seq:
            continue
        seen_seq[seq] = sid
        out.append((sid, seq, name))
    return out


def make_lineage_labels(resolver: TaxdumpResolver, name_to_taxid: Dict[str, int], name: Optional[str]) -> Tuple[Optional[int], Dict[str, Optional[str]]]:
    taxid: Optional[int] = None
    if name:
        taxid = name_to_taxid.get(name.lower())
    lineage = resolver.lineage_by_taxid(taxid) if taxid is not None else resolver.lineage_by_name(name if isinstance(name, str) else None)
    return taxid, lineage


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare unified marine eukaryote references (PR2 + SILVA)')
    ap.add_argument('--pr2-fasta', type=str)
    ap.add_argument('--pr2-taxonomy', type=str)
    ap.add_argument('--silva-fasta', type=str)
    ap.add_argument('--silva-taxonomy', type=str)
    ap.add_argument('--taxdump-dir', required=True, type=str)
    ap.add_argument('--marker', default='18S', choices=['18S','COI','OTHER'])
    ap.add_argument('--out-dir', required=True, type=str)
    ap.add_argument('--max-n-fraction', type=float, default=0.1)
    ap.add_argument('--min-length', type=int, default=60)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir = Path('reference') / 'mappings'
    mappings_dir.mkdir(parents=True, exist_ok=True)

    taxdump_dir = Path(args.taxdump_dir)
    resolver = TaxdumpResolver(str(taxdump_dir))
    if not resolver.available():
        raise RuntimeError(f"Taxdump not available at {taxdump_dir} (names.dmp and nodes.dmp are required)")
    name2taxid = load_name_to_taxid(taxdump_dir)

    pr2_tax = read_taxonomy_table(Path(args.pr2_taxonomy)) if args.pr2_taxonomy else None
    silva_tax = read_taxonomy_table(Path(args.silva_taxonomy)) if args.silva_taxonomy else None

    recs: List[Tuple[str, str, Optional[str]]] = []
    recs += iter_fasta_with_names(Path(args.pr2_fasta) if args.pr2_fasta else None, pr2_tax, source='pr2')
    recs += iter_fasta_with_names(Path(args.silva_fasta) if args.silva_fasta else None, silva_tax, source='silva')

    if not recs:
        raise RuntimeError('No reference sequences found. Provide at least one FASTA.')

    recs = filter_and_dedupe(recs, max_n_fraction=float(args.max_n_fraction), min_length=int(args.min_length))

    # Build outputs
    fasta_out = out_dir / 'references.fasta'
    tax_out = out_dir / 'taxonomy.csv'
    taxid_map_out = Path('reference') / 'mappings' / f"combined_{args.marker}_taxid_map.txt"

    rows: List[Dict[str, Optional[str]]] = []
    with open(fasta_out, 'w') as fw:
        for sid, seq, name in recs:
            taxid, lineage = make_lineage_labels(resolver, name2taxid, name)
            # FASTA header: keep original ID; BLAST taxid map will be used separately
            fw.write(f">{sid}\n{seq}\n")
            row: Dict[str, Optional[str]] = {
                'sequence_id': sid,
                'source': 'pr2_silva',
                'marker': args.marker,
                'name': name,
                'taxid': int(taxid) if isinstance(taxid, int) else None,
            }
            # Merge lineage ranks
            for r in ['kingdom','phylum','class','order','family','genus','species']:
                row[r] = lineage.get(r) if isinstance(lineage, dict) else None
            rows.append(row)

    df = pd.DataFrame(rows)
    # Prefer species name to fill missing name if absent
    df['name'] = df.apply(lambda x: x['name'] if pd.notna(x['name']) and x['name'] else x.get('species'), axis=1)
    df.to_csv(tax_out, index=False)

    # Write taxid map (unknowns get 0)
    with open(taxid_map_out, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        for _, r in df.iterrows():
            tid = int(r['taxid']) if pd.notna(r['taxid']) else 0
            w.writerow([r['sequence_id'], tid])

    print('[DONE] Prepared combined references:')
    print('  FASTA   :', fasta_out)
    print('  Taxonomy:', tax_out)
    print('  TaxID map:', taxid_map_out)


if __name__ == '__main__':
    main()
