"""
Generate visual summary of pipeline revision
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

fig = plt.figure(figsize=(16, 10))

# Title
fig.suptitle('eDNA Taxonomy Classification Pipeline - Complete Revision', 
             fontsize=18, fontweight='bold', y=0.98)

# Create 3x3 grid
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3, 
                      left=0.05, right=0.95, top=0.93, bottom=0.05)

# ============================================================================
# Panel 1: Problem Identification
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title('1. Problem: Catastrophic Forgetting', fontsize=12, fontweight='bold')
ax1.axis('off')

text1 = """
SIMULATION RESULTS:

Old Pipeline (Passive Replay):
├─ Buffer: 300 samples
├─ Strategy: Store but don't use
└─ Result: 18.0% accuracy ❌

Problem Identified:
• Only last cluster remembered
• All previous knowledge lost
• Unusable for sequential data
"""

ax1.text(0.05, 0.95, text1, va='top', fontsize=9, family='monospace',
         bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))

# ============================================================================
# Panel 2: Solution Discovery
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('2. Solution: Active Replay', fontsize=12, fontweight='bold')
ax2.axis('off')

text2 = """
NEW APPROACH:

Active Replay Strategy:
├─ Buffer: 1000 samples
├─ Strategy: Mix 50/50 in batches
├─ EWC: λ=100 (reduced)
└─ Result: 89.0% accuracy ✅

Key Insight:
• Must USE replay samples
• Every batch = 50% past + 50% now
• Model constantly reminded
"""

ax2.text(0.05, 0.95, text2, va='top', fontsize=9, family='monospace',
         bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

# ============================================================================
# Panel 3: Results Comparison
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title('3. Results: 71pp Improvement', fontsize=12, fontweight='bold')

strategies = ['Passive\nReplay', 'Active\nReplay']
accuracies = [18.0, 89.0]
colors = ['#e74c3c', '#27ae60']

bars = ax3.bar(strategies, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Overall Accuracy (%)', fontsize=10, fontweight='bold')
ax3.set_ylim([0, 100])
ax3.axhline(y=20, color='gray', linestyle='--', alpha=0.4, label='Random (20%)')
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, accuracies):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 3,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Add improvement arrow
ax3.annotate('', xy=(1, 89), xytext=(0, 18),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax3.text(0.5, 55, '+71.0pp', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# ============================================================================
# Panel 4: Architecture Comparison
# ============================================================================
ax4 = fig.add_subplot(gs[1, :])
ax4.set_title('4. Architecture: Old vs New', fontsize=12, fontweight='bold')
ax4.set_xlim([0, 10])
ax4.set_ylim([0, 6])
ax4.axis('off')

# Old architecture (top)
y_old = 4.5
ax4.text(0.5, y_old + 0.5, 'OLD PIPELINE', fontsize=10, fontweight='bold', color='red')

old_stages = ['Load\nSeqs', 'Generate\nEmbeddings', 'Cluster', 'Train\n(Passive)', 'Accuracy:\n18%']
old_colors = ['lightblue', 'lightblue', 'lightblue', '#ffcccc', '#ffcccc']
x_old = 1
for stage, color in zip(old_stages, old_colors):
    box = FancyBboxPatch((x_old, y_old-0.3), 1.5, 0.6, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color, linewidth=1.5)
    ax4.add_patch(box)
    ax4.text(x_old + 0.75, y_old, stage, ha='center', va='center', fontsize=8)
    if x_old < 8:
        ax4.arrow(x_old + 1.5, y_old, 0.4, 0, head_width=0.15, head_length=0.1, 
                 fc='black', ec='black')
    x_old += 1.9

# New architecture (bottom)
y_new = 2.0
ax4.text(0.5, y_new + 0.5, 'NEW PIPELINE', fontsize=10, fontweight='bold', color='green')

new_stages = ['Load\nSeqs', 'DNABERT-2\nEmbeddings', 'Cluster', 'Train\n(Active)', 'Accuracy:\n89%']
new_colors = ['lightblue', '#ccffcc', 'lightblue', '#ccffcc', '#ccffcc']
x_new = 1
for stage, color in zip(new_stages, new_colors):
    box = FancyBboxPatch((x_new, y_new-0.3), 1.5, 0.6,
                         boxstyle="round,pad=0.05",
                         edgecolor='black', facecolor=color, linewidth=2)
    ax4.add_patch(box)
    ax4.text(x_new + 0.75, y_new, stage, ha='center', va='center', fontsize=8, fontweight='bold')
    if x_new < 8:
        ax4.arrow(x_new + 1.5, y_new, 0.4, 0, head_width=0.15, head_length=0.1,
                 fc='green', ec='green', linewidth=2)
    x_new += 1.9

# Highlight key difference
highlight = Rectangle((6.5, 1.5), 2, 1.5, linewidth=3, edgecolor='green', 
                      facecolor='none', linestyle='--')
ax4.add_patch(highlight)
ax4.text(7.5, 0.8, 'KEY DIFFERENCE!', ha='center', fontsize=9, 
        color='green', fontweight='bold')

# ============================================================================
# Panel 5: Configuration Details
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_title('5. Configuration', fontsize=11, fontweight='bold')
ax5.axis('tight')
ax5.axis('off')

config_data = [
    ['Parameter', 'Old', 'New'],
    ['Buffer Size', '300', '1000'],
    ['EWC Lambda', '500', '100'],
    ['Replay Mode', 'Passive', 'Active'],
    ['Replay Ratio', 'N/A', '0.5'],
    ['Architecture', 'Shallow', 'Deep']
]

table = ax5.table(cellText=config_data, cellLoc='center', loc='center',
                 colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(3):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight new values
for i in range(1, 6):
    table[(i, 2)].set_facecolor('#ccffcc')
    table[(i, 2)].set_text_props(weight='bold')

# ============================================================================
# Panel 6: Per-Cluster Results
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_title('6. Cluster Retention', fontsize=11, fontweight='bold')

clusters = ['C0', 'C1', 'C2', 'C3', 'C4']
passive_acc = [0, 0, 0, 0, 100]
active_acc = [90.6, 75.2, 87.1, 96.9, 99.8]

x = np.arange(len(clusters))
width = 0.35

bars1 = ax6.bar(x - width/2, passive_acc, width, label='Passive',
               color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax6.bar(x + width/2, active_acc, width, label='Active',
               color='#27ae60', alpha=0.7, edgecolor='black')

ax6.set_ylabel('Accuracy (%)', fontsize=9, fontweight='bold')
ax6.set_xlabel('Cluster ID', fontsize=9, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(clusters)
ax6.legend(fontsize=8)
ax6.grid(axis='y', alpha=0.3)
ax6.set_ylim([0, 110])

# ============================================================================
# Panel 7: Implementation Status
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_title('7. Implementation Status', fontsize=11, fontweight='bold')
ax7.axis('off')

status_text = """
DELIVERABLES:

✅ Simulation Complete
   • 1K and 2.5K sequences tested
   • 89% accuracy validated

✅ Pipeline Implemented
   • run_taxonomy_pipeline_v2.py
   • Full active replay support

✅ Documentation Ready
   • Usage guide
   • Migration guide
   • Demo script

✅ Production Ready
   • Tested on CPU
   • Versioned models
   • Automated checkpoints

NEXT: Test on real data!
"""

ax7.text(0.05, 0.95, status_text, va='top', fontsize=8, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.9))

plt.savefig('pipeline_revision_visual_summary.png', dpi=150, bbox_inches='tight')
print("✓ Saved pipeline_revision_visual_summary.png")

# Also create a simple flowchart
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.axis('off')

fig2.suptitle('Active Replay Training Flow', fontsize=16, fontweight='bold')

# Flow stages
stages = [
    (5, 9, 'Start: Cluster 0'),
    (5, 8, 'Train on Cluster 0\n(No replay yet)'),
    (5, 7, 'Store samples\nin buffer'),
    (5, 6, 'Move to Cluster 1'),
    (5, 5, 'Mix 50% Cluster 1\n+ 50% from buffer'),
    (5, 4, 'Train on mixed batch'),
    (5, 3, 'Store more samples'),
    (5, 2, 'Repeat for all clusters'),
    (5, 1, 'Result: All clusters\nretained! (89%)')
]

for i, (x, y, text) in enumerate(stages):
    if i < len(stages) - 1:
        color = '#ccffcc' if i > 0 else 'lightblue'
    else:
        color = '#27ae60'
    
    box = FancyBboxPatch((x-1.5, y-0.3), 3, 0.6,
                         boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=10,
           fontweight='bold' if i == len(stages) - 1 else 'normal')
    
    # Arrow to next stage
    if i < len(stages) - 1:
        ax.arrow(x, y - 0.35, 0, -0.25, head_width=0.2, head_length=0.1,
                fc='black', ec='black', linewidth=2)

# Add annotation for key step
ax.annotate('KEY STEP:\nActive replay starts here!',
           xy=(5, 5), xytext=(7.5, 5.5),
           arrowprops=dict(arrowstyle='->', lw=2, color='red'),
           fontsize=10, color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.savefig('active_replay_flowchart.png', dpi=150, bbox_inches='tight')
print("✓ Saved active_replay_flowchart.png")

print("\n" + "="*70)
print("VISUAL SUMMARY COMPLETE")
print("="*70)
print("\nGenerated:")
print("  1. pipeline_revision_visual_summary.png - Complete revision overview")
print("  2. active_replay_flowchart.png - Training flow diagram")
print("\nThese visualizations summarize:")
print("  • Problem identification (18% accuracy)")
print("  • Solution discovery (active replay)")
print("  • Results (+71pp improvement)")
print("  • Architecture changes")
print("  • Configuration updates")
print("  • Implementation status")
print("="*70)
