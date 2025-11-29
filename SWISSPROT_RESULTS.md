# SWISSPROT_RESULTS.md — ARCHIVED

This file has been archived and replaced with a maintained copy in the `experiments/swissprot/` folder.

Please see `experiments/swissprot/SWISSPROT_RESULTS.md` for the full historical SwissProt experiment report.

DEPRECATED: SwissProt/protein experiments are not maintained in this eDNA-first project.

## Dataset
- **Source**: `data/raw/swissprot.gz`
- **Sequences loaded**: 4,703 protein sequences
- **Length range**: 50-1,000 amino acids
- **Average length**: 343.1 aa
- **Processing time**: ~42 minutes (embedding generation)

## Pipeline Configuration
- **Dynamic scaling**: Enabled
- **Memory budget**: 4.0 GB
- **Target accuracy**: 80%
- **Auto-adaptation**: True
- **Optimal clusters**: 10 (silhouette score: 0.1496)

## Clustering Results
Cluster sizes (10 clusters):
```
[504, 461, 326, 559, 654, 415, 229, 811, 641, 103]
```

## Training Configuration
- **Architecture**: 768 → 256 → 128 → 10
- **Exemplars/cluster**: 235
- **Uncertainty buffer**: 1,000 samples
- **Recent buffer**: 1,000 samples
- **Temperature**: 1.5
- **Batch size**: 32
- **Replay ratio**: 0.5
- **Epochs per cluster**: 10

## Performance Results

### Overall Metrics
- **Overall Accuracy**: 60.7% (2,854/4,703)
- **Early clusters (0-2)**: 68.2%
- **Middle clusters (3-5)**: 36.2%
- **Recent clusters (6-9)**: 77.5%
- **Recency bias**: +9.3 percentage points

### Per-Cluster Accuracy
| Cluster | Accuracy | Size | Status |
|---------|----------|------|---------|
| 0 | 83.7% | 504 | ✓ Good |
| 1 | 99.6% | 461 | ✓ Excellent |
| 2 | 0.0% | 326 | ✗ Failed |
| 3 | 5.0% | 559 | ✗ Failed |
| 4 | 85.9% | 654 | ✓ Good |
| 5 | 0.0% | 415 | ✗ Failed |
| 6 | 0.0% | 229 | ✗ Failed |
| 7 | 98.2% | 811 | ✓ Excellent |
| 8 | 75.5% | 641 | ✓ Fair |
| 9 | 100.0% | 103 | ✓ Perfect |

## Memory Usage
- **Total samples in buffers**: 3,779
- **Memory usage**: 11.1 MB / 4,096 MB
- **Usage percentage**: 0.3%
- **Adaptations**: 1 event (at cluster 10)

## Key Findings

### Strengths ✓
1. **Memory efficiency**: Only 0.3% of budget used (11.1 MB / 4 GB)
2. **Strong recent performance**: 77.5% on recent clusters
3. **Excellent on some clusters**: 99.6%, 98.2%, 100% on clusters 1, 7, 9
4. **No OOM issues**: Well within memory constraints
5. **Auto-scaling worked**: 1 adaptation event triggered

### Issues ⚠️
1. **Catastrophic forgetting on middle clusters**: 4 clusters (2, 3, 5, 6) have 0-5% accuracy
2. **Overall accuracy below target**: 60.7% vs 80% target
3. **Moderate recency bias**: +9.3pp suggests some forgetting

### Analysis
The pipeline successfully processed 4,703 SwissProt sequences but shows **severe catastrophic forgetting** on certain clusters (especially clusters 2, 3, 5, 6). This suggests:

1. **DNABERT-2 not optimal for proteins**: The model was designed for DNA, not protein sequences
2. **Cluster interference**: Some clusters may have similar embeddings, causing confusion
3. **Buffer strategy needs tuning**: Despite 235 exemplars/cluster, some clusters were forgotten
4. **Training order effects**: Middle clusters suffered most (36.2% vs 68.2% early, 77.5% recent)

## Comparison: eDNA vs SwissProt

| Metric | eDNA (Real) | SwissProt |
|--------|-------------|-----------|
| Sequences | 1,000 | 4,703 |
| Clusters | 5 | 10 |
| Overall Acc | 71.5% | 60.7% |
| Early Acc | 59.9% | 68.2% |
| Middle Acc | 49.2% | 36.2% |
| Recent Acc | 91.5% | 77.5% |
| Recency Bias | +31.6pp | +9.3pp |
| Memory Used | 4.9 MB | 11.1 MB |
| Adaptations | 0 | 1 |

**Key differences**:
- SwissProt has **worse overall performance** (60.7% vs 71.5%)
- SwissProt has **more severe middle-cluster forgetting** (36.2% vs 49.2%)
- SwissProt is **4.7× larger dataset** with **2× more clusters**
- SwissProt triggered **1 adaptation**, eDNA triggered **0**
- Both show **catastrophic forgetting**, but different patterns

## Conclusions

1. ✓ **Pipeline scales**: Successfully handled 4,703 sequences with dynamic scaling
2. ✓ **Memory management works**: Only 0.3% of 4GB budget used
3. ⚠️ **Protein embeddings suboptimal**: DNABERT-2 designed for DNA, not proteins
4. ✗ **Forgetting still severe**: 60.7% overall, with 4 clusters nearly forgotten
5. ℹ️ **More work needed**: Consider protein-specific embedders (ESM, ProtTrans) for better results

## Next Steps

To improve SwissProt results:
1. **Use protein-specific embedder**: ESM-2, ProtBERT, or ProtTrans instead of DNABERT-2
2. **Increase exemplars**: Try 500-1000 exemplars/cluster for more stable memory
3. **Hierarchical clustering**: Group similar protein families together
4. **Curriculum learning**: Train on related clusters together, not in sequence
5. **Monitor cluster similarity**: Detect and handle overlapping clusters specially

## Files Generated
- `swissprot_simulation_output/embeddings/dnabert2_embeddings.npy`
- `swissprot_simulation_output/clustering/results.json`
- `swissprot_simulation_output/checkpoints/checkpoint_epoch10_*.pt`
- `swissprot_simulation_output/evaluation_results.json`
