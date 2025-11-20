# Marine Eukaryote Reference Integration

This folder hosts curated marine eukaryote references and derived indices used for taxonomy labeling (while the pipeline remains unsupervised-first for discovery).

Structure
- pr2/               Place PR2 18S reference FASTA and taxonomy here
- silva/             Place SILVA SSU (euk subset) FASTA and taxonomy here
- eukref/            Optional: curated EukRef clades
- mappings/          Taxonomic mappings (seq_id -> taxid)
- indices/
  - 18S/             FAISS and BLAST indices for 18S
  - COI/             (optional) indices for COI
- combined/
  - 18S/             Unified combined FASTA + taxonomy built from sources

What you need to provide (files you download separately)
- PR2 (18S):
  - pr2_18S.fasta(.gz)
  - taxonomy.tsv (must include scientific name column; common formats: name/taxonomy string)
- SILVA (SSU euk subset):
  - silva_18S_euk.fasta(.gz)
  - taxonomy.tsv or taxonomy embedded in headers
- Optional EukRef clades (FASTA + clade taxonomy table)
- NCBI taxdump (names.dmp, nodes.dmp, merged.dmp) to enable name→taxid mapping

Build steps (PowerShell examples)
1) Place downloads
- Put raw FASTAs and taxonomy TSVs under reference/pr2 and reference/silva
- Put NCBI taxdump under a local directory (e.g., F:\\Dataset\\taxdump)

2) Prepare combined references (harmonize names → taxids, deduplicate)
```powershell
python scripts\prepare_references.py `
  --pr2-fasta "reference\pr2\pr2_18S.fasta" `
  --pr2-taxonomy "reference\pr2\taxonomy.tsv" `
  --silva-fasta "reference\silva\silva_18S_euk.fasta" `
  --silva-taxonomy "reference\silva\taxonomy.tsv" `
  --taxdump-dir "F:\\Dataset\\taxdump" `
  --marker 18S `
  --out-dir "reference\combined\18S"
```
Outputs:
- reference\combined\18S\references.fasta
- reference\combined\18S\taxonomy.csv
- reference\mappings\combined_18S_taxid_map.txt

3) Build BLAST database (requires BLAST+ makeblastdb in PATH)
```powershell
python scripts\build_blast_db.py `
  --fasta "reference\combined\18S\references.fasta" `
  --taxid-map "reference\mappings\combined_18S_taxid_map.txt" `
  --db-out "reference\indices\18S\combined_18S"
```
Outputs (prefix): reference\indices\18S\combined_18S.*

4) Build KNN embedding index using your pipeline model
```powershell
python scripts\build_reference_index.py `
  --fasta "reference\combined\18S\references.fasta" `
  --labels-csv "reference\combined\18S\taxonomy.csv" `
  --output-dir "reference\indices\18S"
```
Outputs:
- reference\indices\18S\reference_embeddings.npy
- reference\indices\18S\reference_labels.csv

5) Update config/config.yaml
- taxonomy.knn.embeddings_path → reference\\indices\\18S\\reference_embeddings.npy
- taxonomy.knn.labels_path → reference\\indices\\18S\\reference_labels.csv
- taxonomy.blast_fallback.database → reference\\indices\\18S\\combined_18S
- taxonomy.taxdump_dir → your taxdump path

Notes
- You can start with either PR2 or SILVA alone; the scripts accept missing sources.
- For large references, run embedding/index builds on a GPU machine if possible for speed.
- This setup minimizes reference dependence by using them strictly for labeling and evidence.
