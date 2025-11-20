# Title

feat(dev): Windows-friendly preview setup (no torch/UMAP required)

## Summary
- Make UMAP optional with PCA fallback in clustering
- Replace heavy torch-based trainer with a lightweight stub
- Provide torch-free DNATransformerEmbedder stub and wire the pipeline to it
- Keep models package __init__ light to avoid heavy imports
- Add results/ to .gitignore to keep generated artifacts out of VCS

## Motivation
Enable contributors on Windows (Python 3.13) to run a quick demo pipeline without GPU/torch or external native tools. Uses mock embeddings + scikit-learn to validate the flow quickly.

## Changes
- src/clustering/algorithms.py: optional UMAP, PCA fallback
- src/models/trainer.py: lightweight no-op trainer
- src/models/embeddings_stub.py: torch-free DNATransformerEmbedder stub
- scripts/run_pipeline.py: import stub embedder for this branch
- src/models/__init__.py: minimal import surface
- .gitignore: ignore results/

## Testing
Commands used:
- python scripts/run_pipeline.py --create-sample --input data/sample --output results/demo
- python scripts/run_pipeline.py --input data/sample/sample_edna_sequences.fasta --output results/demo --skip-preprocessing

Key outputs to verify:
- results/demo/pipeline_results.json
- results/demo/visualizations/analysis_dashboard.html

## Screenshots / Artifacts
- results/demo/clustering/cluster_visualization.png
- results/demo/novelty/novelty_visualization.png

## Notes
- DBSCAN+PCA with mock embeddings may yield 0 clusters. For a more illustrative demo, consider k-means.
- Full stack (torch, BLAST, cutadapt, vsearch) remains on main; this branch targets a lightweight preview.

## Checklist
- [ ] Title follows conventional commits (feat, fix, chore, etc.)
- [ ] CI passes
- [ ] Docs updated (if needed)
- [ ] No large binary artifacts committed
- [ ] Changes are isolated to preview path; main path unaffected

## Links
- Issue/Discussion: (optional)
- Related PRs: (optional)
