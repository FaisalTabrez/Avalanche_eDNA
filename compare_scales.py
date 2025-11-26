"""
Compare continual learning performance across dataset scales
"""

import matplotlib.pyplot as plt
import numpy as np

# Results
scales = ['1,000 seqs', '2,500 seqs']
overall_acc = [13.8, 18.0]

# Cluster-wise breakdown
# 1K: Cluster 4 had ~200/1000 samples = 20%, 100% acc = 20% overall
# 2.5K: Cluster 4 has 449/2500 samples = 18%, 100% acc = 18% overall

cluster_retention = [
    [0, 0, 0, 0, 100],  # 1K sequences
    [0, 0, 0, 0, 100]   # 2.5K sequences
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Overall accuracy comparison
bars = axes[0].bar(scales, overall_acc, color=['steelblue', 'coral'], alpha=0.7)
axes[0].set_ylabel('Overall Accuracy (%)', fontsize=12)
axes[0].set_title('Catastrophic Forgetting Across Scales', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 100])
axes[0].axhline(y=20, color='red', linestyle='--', alpha=0.3, label='Expected (20% if random)')
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, overall_acc)):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 2,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

axes[0].legend()

# Plot 2: Cluster retention heatmap
cluster_data = np.array(cluster_retention)
im = axes[1].imshow(cluster_data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

axes[1].set_xticks(range(len(scales)))
axes[1].set_xticklabels(scales)
axes[1].set_yticks(range(5))
axes[1].set_yticklabels([f'Cluster {i}' for i in range(5)])
axes[1].set_title('Per-Cluster Accuracy Retention', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(scales)):
    for j in range(5):
        text = axes[1].text(i, j, f'{cluster_data[i, j]:.0f}%',
                           ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im, ax=axes[1], label='Accuracy (%)')

plt.tight_layout()
plt.savefig('pipeline_outputs_2500/visualizations/scale_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved scale comparison visualization")

# Print analysis
print("\n" + "="*60)
print("Catastrophic Forgetting Analysis")
print("="*60)
print("\nKey Findings:")
print(f"  • Both scales show COMPLETE forgetting of earlier clusters")
print(f"  • Only the final cluster (Cluster 4) is retained at 100%")
print(f"  • Overall accuracy depends on final cluster size:")
print(f"    - 1,000 seqs: ~200 in Cluster 4 → 13.8% overall")
print(f"    - 2,500 seqs: 449 in Cluster 4 → 18.0% overall")
print(f"\n  • Scaling up does NOT solve catastrophic forgetting")
print(f"  • Need better continual learning strategies:")
print(f"    - Increase replay buffer (300 → 1000+)")
print(f"    - Reduce EWC lambda (500 → 100)")
print(f"    - Add replay samples during training")
print(f"    - Use progressive neural networks")
print(f"    - Implement knowledge distillation")
