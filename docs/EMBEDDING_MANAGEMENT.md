# Embedding Management Guide

This guide explains how to manage embeddings across multiple datasets for model training and database building.

## Overview

The Avalanche eDNA system generates vector embeddings for sequences, which are used for:
- **Similarity search** - Find similar sequences across all datasets
- **Model training** - Train and fine-tune classification models
- **Reference database** - Build a growing reference of known taxa
- **Novelty detection** - Identify potential new species
- **Cross-dataset analysis** - Compare sequences across studies

## Storage Structure

```
consolidated_data/runs/
├── Dataset_A/
│   ├── 2024-11-20_10-30-00/
│   │   ├── embeddings.npy          # Original embeddings (large)
│   │   └── embeddings.npz          # Compressed embeddings (smaller)
│   └── 2024-11-22_15-45-00/
│       └── embeddings.npy
└── Dataset_B/
    └── 2024-11-21_12-00-00/
        └── embeddings.npz

data/reference/
├── reference_embeddings.npy        # Consolidated embeddings
├── reference_index.faiss           # Fast search index
└── reference_metadata.json         # Source tracking
```

## Workflow

### 1. Run Analysis → Generate Embeddings
When you run analysis, embeddings are automatically saved:
```bash
python scripts/run_pipeline.py --input sequences.fasta --output consolidated_data/runs/My_Dataset/2024-11-22
```

This creates `consolidated_data/runs/My_Dataset/2024-11-22/embeddings.npy`

### 2. Build Consolidated Reference
Periodically merge all run embeddings into a single reference database:

```bash
# First time - build from scratch
python scripts/consolidate_embeddings.py

# Subsequent runs - only add new data
python scripts/consolidate_embeddings.py --incremental

# Also compress source files to save space
python scripts/consolidate_embeddings.py --incremental --compress
```

**Output:**
- `data/reference/consolidated_embeddings.npy` - All embeddings combined
- `data/reference/consolidated_index.faiss` - Fast search index
- `data/reference/consolidated_metadata.json` - Track which sequences came from which run

### 3. Compress Old Embeddings
Save ~50% storage by compressing .npy → .npz:

```bash
# Dry run to see what would be compressed
python scripts/compress_embeddings.py

# Compress files older than 30 days
python scripts/compress_embeddings.py --older-than 30 --execute

# Compress and delete originals
python scripts/compress_embeddings.py --execute --delete-original
```

### 4. Search Across All Data
Find similar sequences across all datasets:

```bash
# Search using a sequence from an existing run
python scripts/search_reference.py --run My_Dataset/2024-11-22_10-30-45 --seq-idx 42 --top-k 10

# Save results to JSON
python scripts/search_reference.py --run My_Dataset/2024-11-22 --seq-idx 42 --output results.json
```

## Best Practices

### For Active Research
✅ **Keep run embeddings** - Don't delete, they're your historical record  
✅ **Build reference monthly** - Run `build_reference_index.py --incremental` regularly  
✅ **Compress old runs** - After 30 days, compress to .npz  
✅ **Version reference** - Tag reference builds (e.g., `reference_2024-11.npy`)  

### For Production Database
✅ **Use FAISS index** - Much faster search than brute-force numpy  
✅ **Backup reference** - Reference is critical, back up to cloud/external drive  
✅ **Document sources** - Keep metadata.json to track provenance  
✅ **Monitor size** - Plan storage: ~10MB per 10K sequences at 256D  

### Storage Management
When disk space is tight:
1. **Compress first** - Run `compress_embeddings.py` (~50% savings)
2. **Archive old runs** - Move >6 month old runs to external storage
3. **Keep reference** - Always keep `data/reference/` intact
4. **Delete carefully** - Only delete runs after they're in reference

## File Formats

### .npy (Original)
```python
import numpy as np
embeddings = np.load("embeddings.npy")  # Fast, uncompressed
print(embeddings.shape)  # (N_sequences, embedding_dim)
```

### .npz (Compressed)
```python
import numpy as np
with np.load("embeddings.npz") as data:
    embeddings = data['embeddings']  # ~50% smaller
```

### FAISS Index (Fast Search)
```python
import faiss
index = faiss.read_index("reference_index.faiss")
distances, indices = index.search(query, k=10)  # Much faster than numpy
```

## Integration with Data Manager

The UI automatically uses embeddings through the data manager:

```python
from src.ui.data_manager import get_data_manager

dm = get_data_manager()

# Get files for a run (includes embeddings)
files = dm.get_run_files(run_path)
if 'embeddings' in files:
    print(f"Embeddings available: {files['embeddings']}")

# Search across all runs
results = dm.search_across_embeddings(
    query_run=Path("consolidated_data/runs/My_Dataset/2024-11-22"),
    query_seq_idx=42,
    top_k=10
)
```

## Troubleshooting

### "No embeddings found"
- Check `consolidated_data/runs/` structure
- Verify analysis completed successfully
- Look for both `.npy` and `.npz` files

### "Reference build failed"
- Ensure FAISS installed: `pip install faiss-cpu`
- Check sufficient disk space
- Verify no corrupt embedding files

### "Search is slow"
- Use FAISS index instead of numpy search
- Rebuild index with `build_reference_index.py`
- Consider IVF index for >100K sequences

### "Out of memory"
- Use memory-mapped loading: `mmap_mode='r'`
- Process in batches
- Compress embeddings to save RAM

## Advanced: Embedding Version Management

Track which model version generated which embeddings:

```bash
# Register a new model version
python scripts/manage_embedding_versions.py register --version v1.0 --model "nt-500m-1000g" --description "Initial baseline"

# Tag existing runs with version
python scripts/manage_embedding_versions.py auto-tag --version v1.0

# Later, with improved model
python scripts/manage_embedding_versions.py register --version v2.0 --model "nt-500m-2000g" --description "Fine-tuned on eDNA"

# Tag new runs
python scripts/manage_embedding_versions.py tag-run My_Dataset/2024-12-01 --version v2.0

# List all versions
python scripts/manage_embedding_versions.py list

# Compare two versions
python scripts/manage_embedding_versions.py compare v1.0 v2.0
```

Build version-specific references:

```bash
# Build consolidated reference for v1.0 runs
python scripts/consolidate_embeddings.py --output data/reference_v1.0

# Build for v2.0 runs
python scripts/consolidate_embeddings.py --output data/reference_v2.0

# Search against specific version
python scripts/search_reference.py --reference data/reference_v1.0 --run ... 
python scripts/search_reference.py --reference data/reference_v2.0 --run ...
```

## Related Documentation

- [Data Manager API](../src/ui/data_manager.py) - Programmatic access
- [Run Pipeline](../scripts/run_pipeline.py) - Generate embeddings
- [Configuration](configuration.md) - Embedding settings
