"""
Final Visualization - Compare Balanced vs Worst Results
========================================================

Generate paper-ready plots comparing both objectives.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

# Load simplified results
results_dir = Path('./results/map_simple/cifar100_lt_if100')

with open(results_dir / 'rc_curve.json', 'r') as f:
    rc_data = json.load(f)

with open('./checkpoints/map_simple/cifar100_lt_if100/map_parameters.json', 'r') as f:
    params = json.load(f)

# Extract data
rej = np.array(rc_data['rejection_rates'])
err = np.array(rc_data['selective_errors'])
aurc = rc_data['aurc']
group_errors = rc_data['group_errors']

print("="*70)
print("📊 FINAL VISUALIZATION")
print("="*70)
print(f"\nData loaded:")
print(f"  Rejection range: [{rej.min():.3f}, {rej.max():.3f}]")
print(f"  Error range: [{err.min():.3f}, {err.max():.3f}]")
print(f"  AURC: {aurc:.4f}")
print(f"  Params: threshold={params['threshold']:.3f}, γ={params['gamma']:.3f}")

# Create 3-panel plot like your attached image
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Full range (0-1)
ax1 = axes[0]
ax1.plot(rej, err, 'b-', linewidth=2.5, label=f'Balanced (AURC={aurc:.4f})')
ax1.fill_between(rej, 0, err, alpha=0.2)
ax1.set_xlabel('Proportion of Rejections', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error', fontsize=12, fontweight='bold')
ax1.set_title('Error vs Rejection Rate (0-1)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1.0)
ax1.set_ylim(0, 1.0)

# Panel 2: Practical range (0-0.8)
ax2 = axes[1]
mask = rej <= 0.8
if mask.sum() > 0:
    ax2.plot(rej[mask], err[mask], 'b-', linewidth=2.5, label='Balanced')
    ax2.set_xlim(0, 0.8)
else:
    # All rejections > 0.8, show what we have
    ax2.plot(rej, err, 'b-', linewidth=2.5, label='Balanced')
    ax2.set_xlim(rej.min() - 0.02, 1.0)
    ax2.text(0.5, 0.5, 'Most samples\nrejected\n(> 93%)', 
            ha='center', va='center', transform=ax2.transAxes,
            fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax2.set_xlabel('Proportion of Rejections', fontsize=12, fontweight='bold')
ax2.set_ylabel('Error', fontsize=12, fontweight='bold')
ax2.set_title('Error vs Rejection Rate (0-0.8)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Panel 3: AURC comparison (simulated)
ax3 = axes[2]

# Full AURC
aurc_full = aurc

# Practical AURC (0.2-1.0 range like your image)
mask_prac = rej >= 0.2
if mask_prac.sum() > 1:
    rej_prac = rej[mask_prac]
    err_prac = err[mask_prac]
    # Normalize to [0.2, 1.0] range
    aurc_practical = np.trapz(err_prac, rej_prac) / 0.8
else:
    aurc_practical = aurc

objectives = ['Balanced']
x_pos = np.arange(len(objectives))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, [aurc_full], width,
               label='Full (0-1)', color='#2E86AB', alpha=0.8, edgecolor='black')
bars2 = ax3.bar(x_pos + width/2, [aurc_practical], width,
               label='Practical (0.2-1.0)', color='#2E86AB', alpha=0.5,
               edgecolor='black', hatch='///')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_ylabel('AURC', fontsize=12, fontweight='bold')
ax3.set_title('AURC Comparison (Full vs 0.2-1.0)', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(objectives)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, max(aurc_full, aurc_practical) * 1.2)

plt.tight_layout()

# Save
output_dir = Path('./outputs/visualizations/map_simple')
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / 'final_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: {save_path}")

plt.close()

# Additional plot: Group-wise errors
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Extract group errors
head_errors = np.array([ge[0] for ge in group_errors])
tail_errors = np.array([ge[1] for ge in group_errors])

ax.plot(rej, head_errors, 'b-', linewidth=2.5, marker='o', markersize=4,
        markevery=5, label='Head (Classes 0-49)', color='#2E86AB')
ax.plot(rej, tail_errors, 'r-', linewidth=2.5, marker='s', markersize=4,
        markevery=5, label='Tail (Classes 50-99)', color='#A23B72')
ax.plot(rej, err, 'k--', linewidth=2, label='Overall', alpha=0.7)

ax.set_xlabel('Rejection Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('Selective Error', fontsize=12, fontweight='bold')
ax.set_title('Group-wise RC Curves (Simplified MAP)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(rej.min() - 0.01, 1.0)
ax.set_ylim(0, 1.0)

# Add annotations
ax.annotate(f'Head: {head_errors[0]:.2%}\nTail: {tail_errors[0]:.2%}',
           xy=(rej[0], err[0]), xytext=(0.5, 0.3),
           fontsize=10, ha='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

plt.tight_layout()
save_path2 = output_dir / 'group_wise_curves.png'
plt.savefig(save_path2, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {save_path2}")

plt.close()

# Print summary table
print("\n" + "="*70)
print("📊 SUMMARY STATISTICS")
print("="*70)

print(f"\nParameters:")
print(f"  Confidence threshold: {params['threshold']:.3f}")
print(f"  Uncertainty penalty (γ): {params['gamma']:.3f}")
print(f"  Objective: {params['objective']}")

print(f"\nPerformance:")
print(f"  AURC (full): {aurc:.4f}")
print(f"  AURC (practical): {aurc_practical:.4f}")

print(f"\nKey Operating Points:")
# Find interesting points
for target in [rej.min(), (rej.min() + rej.max())/2, rej.max()]:
    idx = np.argmin(np.abs(rej - target))
    print(f"  Rej={rej[idx]:.1%}: Error={err[idx]:.4f}, "
          f"Head={head_errors[idx]:.4f}, Tail={tail_errors[idx]:.4f}")

print(f"\nGroup Performance (at min rejection):")
print(f"  Head error: {head_errors[0]:.2%}")
print(f"  Tail error: {tail_errors[0]:.2%}")
print(f"  Gap: {abs(head_errors[0] - tail_errors[0]):.2%}")

print(f"\n✅ Visualization complete!")
print(f"Output directory: {output_dir}")
