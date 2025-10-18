#!/usr/bin/env python3
"""
Visualization to explain reweighting concept.

This creates a visual comparison between:
1. Train distribution (long-tail)
2. Test distribution (balanced)  
3. Balanced vs Reweighted accuracy calculation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_reweighting_concept():
    """Create visualization explaining reweighting."""
    
    # Setup
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Reweighting in Long-Tail Learning: Visual Explanation', 
                 fontsize=16, fontweight='bold')
    
    # Example with 10 classes for clarity
    num_classes = 10
    class_names = [f'C{i}' for i in range(num_classes)]
    
    # Train distribution (long-tail, IF=100)
    train_counts = []
    for i in range(num_classes):
        count = int(500 * (100 ** (-i / (num_classes - 1))))
        train_counts.append(max(5, count))
    train_counts = np.array(train_counts)
    
    # Test distribution (balanced)
    test_counts = np.ones(num_classes) * 100
    
    # Simulate per-class accuracy (decreases from head to tail)
    per_class_acc = np.linspace(0.90, 0.60, num_classes)
    
    # Class weights for reweighting
    class_weights = train_counts / train_counts.sum()
    
    # ============ Plot 1: Train Distribution ============
    ax1 = axes[0, 0]
    bars1 = ax1.bar(class_names, train_counts, color='steelblue', alpha=0.7)
    ax1.set_title('(a) Training Distribution (Long-Tail)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Samples')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars1, train_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({class_weights[i]*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)
    
    ax1.text(0.95, 0.95, f'Imbalance Factor: {train_counts[0]/train_counts[-1]:.1f}',
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============ Plot 2: Test Distribution ============
    ax2 = axes[0, 1]
    bars2 = ax2.bar(class_names, test_counts, color='coral', alpha=0.7)
    ax2.set_title('(b) Test Distribution (Balanced)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Number of Samples')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, max(train_counts) * 1.2])
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n(10.0%)',
                ha='center', va='bottom', fontsize=8)
    
    ax2.text(0.95, 0.95, 'All classes equal',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # ============ Plot 3: Per-Class Accuracy ============
    ax3 = axes[1, 0]
    bars3 = ax3.bar(class_names, per_class_acc * 100, color='green', alpha=0.7)
    ax3.set_title('(c) Per-Class Accuracy on Test Set', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_ylim([0, 100])
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=per_class_acc.mean() * 100, color='blue', linestyle='--', 
                label=f'Balanced Avg: {per_class_acc.mean()*100:.1f}%')
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars3, per_class_acc)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.0f}%',
                ha='center', va='bottom', fontsize=8)
    
    ax3.legend()
    
    # ============ Plot 4: Metric Comparison ============
    ax4 = axes[1, 1]
    
    # Calculate metrics
    balanced_acc = per_class_acc.mean()
    reweighted_acc = (per_class_acc * class_weights).sum()
    
    # Bar chart
    metrics = ['Balanced\nAccuracy\n(WRONG)', 'Reweighted\nAccuracy\n(CORRECT)']
    values = [balanced_acc * 100, reweighted_acc * 100]
    colors_bar = ['lightcoral', 'lightgreen']
    
    bars4 = ax4.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_title('(d) Metric Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_ylim([0, 100])
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars4, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add formulas
    formula_balanced = r'$\frac{1}{C} \sum_i \mathrm{acc}_i$'
    formula_reweighted = r'$\sum_i w_i \cdot \mathrm{acc}_i$'
    
    ax4.text(0, -15, formula_balanced, ha='center', fontsize=10, 
             transform=ax4.transData)
    ax4.text(1, -15, formula_reweighted, ha='center', fontsize=10,
             transform=ax4.transData)
    
    # Add explanation
    explanation = (
        f"Reweighted is {reweighted_acc/balanced_acc:.2f}× higher\n"
        f"because head classes (C0-C2)\n"
        f"have both HIGH accuracy\n"
        f"AND HIGH train frequency"
    )
    ax4.text(0.98, 0.60, explanation,
             transform=ax4.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('demo_outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'reweighting_explanation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    
    plt.show()


def visualize_cifar100_comparison():
    """Visualize CIFAR-100-LT scenario."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('CIFAR-100-LT: Why Reweighting Matters', 
                 fontsize=14, fontweight='bold')
    
    # Create CIFAR-100-LT distribution
    num_classes = 100
    imb_factor = 100
    train_counts = []
    for i in range(num_classes):
        count = int(500 * (imb_factor ** (-i / (num_classes - 1))))
        train_counts.append(max(1, count))
    train_counts = np.array(train_counts)
    
    class_weights = train_counts / train_counts.sum()
    
    # Simulate accuracy (better on head)
    per_class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        # Head: 85%, Tail: 55%
        per_class_acc[i] = 0.85 - 0.30 * (i / 99)
        per_class_acc[i] += np.random.normal(0, 0.02)  # Add noise
        per_class_acc[i] = np.clip(per_class_acc[i], 0.4, 0.95)
    
    # ============ Plot 1: Distribution ============
    ax1 = axes[0]
    ax1.semilogy(range(num_classes), train_counts, 'b-', linewidth=2, label='Train (Long-tail)')
    ax1.axhline(y=75, color='r', linestyle='--', linewidth=2, label='Test (Balanced, 75/class)')
    ax1.set_title('(a) Class Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Class Index (sorted by frequency)')
    ax1.set_ylabel('Number of Samples (log scale)')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Add annotations
    ax1.annotate(f'Head: {train_counts[0]} samples', xy=(0, train_counts[0]), 
                xytext=(10, train_counts[0]*2),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10)
    ax1.annotate(f'Tail: {train_counts[-1]} samples', xy=(99, train_counts[-1]), 
                xytext=(80, train_counts[-1]*10),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10)
    
    # ============ Plot 2: Accuracy ============
    ax2 = axes[1]
    ax2.plot(range(num_classes), per_class_acc * 100, 'g-', linewidth=2)
    ax2.axhline(y=per_class_acc.mean() * 100, color='orange', linestyle='--', 
                linewidth=2, label=f'Balanced: {per_class_acc.mean()*100:.1f}%')
    ax2.set_title('(b) Per-Class Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Class Index')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim([40, 100])
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # Shade regions
    ax2.axvspan(0, 33, alpha=0.2, color='green', label='Head')
    ax2.axvspan(33, 66, alpha=0.2, color='yellow', label='Medium')
    ax2.axvspan(66, 100, alpha=0.2, color='red', label='Tail')
    
    # ============ Plot 3: Contribution to Accuracy ============
    ax3 = axes[2]
    
    # Contribution of each class
    contribution_balanced = per_class_acc / num_classes * 100
    contribution_reweighted = per_class_acc * class_weights * 100
    
    # Plot
    width = 0.8
    x = np.arange(num_classes)
    ax3.bar(x, contribution_reweighted, width, alpha=0.7, color='green', 
            label='Reweighted')
    ax3.bar(x, contribution_balanced, width, alpha=0.5, color='orange', 
            label='Balanced')
    
    ax3.set_title('(c) Contribution to Overall Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Class Index')
    ax3.set_ylabel('Contribution (%)')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add text
    balanced_acc = per_class_acc.mean()
    reweighted_acc = (per_class_acc * class_weights).sum()
    
    summary = (
        f"Balanced:   {balanced_acc*100:.2f}%\n"
        f"Reweighted: {reweighted_acc*100:.2f}%\n"
        f"Difference: +{(reweighted_acc - balanced_acc)*100:.2f}%"
    )
    ax3.text(0.98, 0.98, summary,
             transform=ax3.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
             fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('demo_outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'cifar100_reweighting.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("REWEIGHTING VISUALIZATION")
    print("=" * 60)
    
    print("\n📊 Creating visualizations...\n")
    
    # Create both visualizations
    print("[1/2] Simple 10-class example...")
    visualize_reweighting_concept()
    
    print("\n[2/2] CIFAR-100-LT example...")
    visualize_cifar100_comparison()
    
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print("\nVisualization files created in: demo_outputs/")
    print("  - reweighting_explanation.png")
    print("  - cifar100_reweighting.png")
