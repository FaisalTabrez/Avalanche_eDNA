"""
Compare passive replay (buffer only) vs active replay (mixed batches)
Run both strategies and visualize the difference in catastrophic forgetting
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(output_dir):
    """Load pipeline results from summary file."""
    summary_file = Path(output_dir) / 'pipeline_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None

def plot_comparison():
    """Create comparison visualization."""
    
    # Historical results (passive replay)
    passive_results = {
        'strategy': 'Passive Replay',
        'buffer_size': 300,
        'overall_accuracy': 18.0,
        'cluster_accuracies': [0.0, 0.0, 0.0, 0.0, 100.0]
    }
    
    # Active replay results (to be updated)
    active_file = Path('pipeline_outputs_2500_active/pipeline_summary.json')
    if active_file.exists():
        with open(active_file) as f:
            active_data = json.load(f)
            active_results = {
                'strategy': 'Active Replay',
                'buffer_size': 1000,
                'overall_accuracy': active_data['results']['overall_accuracy'],
                'cluster_accuracies': [
                    active_data['results']['cluster_results'][str(i)]['accuracy']
                    for i in range(5)
                ]
            }
    else:
        # Placeholder
        active_results = {
            'strategy': 'Active Replay',
            'buffer_size': 1000,
            'overall_accuracy': 0.0,
            'cluster_accuracies': [0.0] * 5
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Overall accuracy comparison
    strategies = ['Passive\nReplay', 'Active\nReplay']
    overall_accs = [passive_results['overall_accuracy'], active_results['overall_accuracy']]
    
    bars = axes[0, 0].bar(strategies, overall_accs, 
                          color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0, 100])
    axes[0, 0].axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Random (20%)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].legend()
    
    # Add value labels
    for bar, val in zip(bars, overall_accs):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 3,
                       f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    # Plot 2: Per-cluster accuracy heatmap
    cluster_data = np.array([
        passive_results['cluster_accuracies'],
        active_results['cluster_accuracies']
    ])
    
    im = axes[0, 1].imshow(cluster_data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(['Passive', 'Active'])
    axes[0, 1].set_yticks(range(5))
    axes[0, 1].set_yticklabels([f'Cluster {i}' for i in range(5)])
    axes[0, 1].set_title('Per-Cluster Accuracy Retention', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(5):
            text = axes[0, 1].text(i, j, f'{cluster_data[i, j]:.0f}%',
                                  ha="center", va="center", 
                                  color="white" if cluster_data[i, j] > 50 else "black",
                                  fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=axes[0, 1], label='Accuracy (%)')
    
    # Plot 3: Configuration comparison
    configs = [
        ['Buffer Size', passive_results['buffer_size'], active_results['buffer_size']],
        ['EWC Lambda', 500, 100],
        ['Replay Mode', 'Passive', 'Active']
    ]
    
    axes[1, 0].axis('tight')
    axes[1, 0].axis('off')
    table = axes[1, 0].table(
        cellText=configs,
        colLabels=['Parameter', 'Passive', 'Active'],
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 0].set_title('Strategy Configuration', fontsize=14, fontweight='bold', pad=20)
    
    # Plot 4: Retention pattern
    clusters = ['C0', 'C1', 'C2', 'C3', 'C4']
    x = np.arange(len(clusters))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, passive_results['cluster_accuracies'], 
                          width, label='Passive', color='#e74c3c', alpha=0.7)
    bars2 = axes[1, 1].bar(x + width/2, active_results['cluster_accuracies'],
                          width, label='Active', color='#2ecc71', alpha=0.7)
    
    axes[1, 1].set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Cluster-wise Retention Pattern', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(clusters)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].set_ylim([0, 110])
    
    plt.tight_layout()
    plt.savefig('replay_strategy_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved replay_strategy_comparison.png")
    
    # Print summary
    print("\n" + "="*70)
    print("REPLAY STRATEGY COMPARISON")
    print("="*70)
    print(f"\n{'Strategy':<20} {'Buffer':<10} {'Overall Acc':<15} {'Catastrophic Forgetting'}")
    print("-"*70)
    print(f"{'Passive Replay':<20} {passive_results['buffer_size']:<10} {passive_results['overall_accuracy']:>6.1f}%        {'SEVERE (only last cluster)'}")
    print(f"{'Active Replay':<20} {active_results['buffer_size']:<10} {active_results['overall_accuracy']:>6.1f}%        {'TBD'}")
    print("="*70)
    
    if active_results['overall_accuracy'] > 0:
        improvement = active_results['overall_accuracy'] - passive_results['overall_accuracy']
        print(f"\n🎯 Improvement: {improvement:+.1f} percentage points")
        
        # Count how many clusters retained
        passive_retained = sum(1 for acc in passive_results['cluster_accuracies'] if acc > 50)
        active_retained = sum(1 for acc in active_results['cluster_accuracies'] if acc > 50)
        print(f"📊 Clusters Retained (>50% acc):")
        print(f"   Passive: {passive_retained}/5 clusters")
        print(f"   Active:  {active_retained}/5 clusters")

if __name__ == "__main__":
    plot_comparison()
