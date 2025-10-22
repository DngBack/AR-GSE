"""
Plot AURC curves comparing Balanced vs Worst objectives.
Creates 3-panel figure like the reference image.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_results(results_dir: Path, objective: str):
    """Load cost sweep results for an objective."""
    results_path = results_dir / f'cost_sweep_{objective}.json'
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Get best result (typically cost=0.0)
    best_result = data['results_per_cost'][0]
    
    return {
        'objective': objective,
        'rejection_rates': np.array(best_result['rc_curve']['rejection_rates']),
        'errors': np.array(best_result['rc_curve']['selective_errors']),
        'aurc_full': best_result['aurc']
    }


def compute_aurc_range(rejection_rates, errors, min_rejection, max_rejection):
    """Compute AURC for a specific rejection rate range."""
    mask = (rejection_rates >= min_rejection) & (rejection_rates <= max_rejection)
    
    if mask.sum() < 2:
        return 0.0
    
    rej_range = rejection_rates[mask]
    err_range = errors[mask]
    
    # Normalize by range width
    range_width = max_rejection - min_rejection
    if range_width > 0:
        aurc = np.trapz(err_range, rej_range) / range_width
    else:
        aurc = 0.0
    
    return aurc


def plot_comparison(balanced_data, worst_data, output_dir: Path):
    """
    Create 3-panel comparison plot:
    1. Error vs Rejection Rate (full 0-1)
    2. Error vs Rejection Rate (practical 0-0.8)
    3. AURC Comparison bars
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Colors
    color_balanced = 'green'
    color_worst = 'orange'
    
    # ========================================================================
    # Plot 1: Error vs Rejection Rate (Full 0-1)
    # ========================================================================
    ax1 = axes[0]
    
    ax1.plot(balanced_data['rejection_rates'], balanced_data['errors'], 
             'o-', linewidth=2, markersize=3, label=f"Balanced (AURC={balanced_data['aurc_full']:.4f})",
             color=color_balanced)
    
    ax1.plot(worst_data['rejection_rates'], worst_data['errors'],
             'o-', linewidth=2, markersize=3, label=f"Worst (AURC={worst_data['aurc_full']:.4f})",
             color=color_worst)
    
    ax1.set_xlabel('Proportion of Rejections', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('Error vs Rejection Rate (0-1)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    
    # ========================================================================
    # Plot 2: Error vs Rejection Rate (Practical 0-0.8)
    # ========================================================================
    ax2 = axes[1]
    
    # Filter for rejection <= 0.8
    mask_bal = balanced_data['rejection_rates'] <= 0.8
    mask_worst = worst_data['rejection_rates'] <= 0.8
    
    ax2.plot(balanced_data['rejection_rates'][mask_bal], 
             balanced_data['errors'][mask_bal],
             'o-', linewidth=2, markersize=3, label='Balanced', color=color_balanced)
    
    ax2.plot(worst_data['rejection_rates'][mask_worst],
             worst_data['errors'][mask_worst],
             'o-', linewidth=2, markersize=3, label='Worst', color=color_worst)
    
    ax2.set_xlabel('Proportion of Rejections', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('Error vs Rejection Rate (0-0.8)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.8])
    
    # ========================================================================
    # Plot 3: AURC Comparison (Full vs Practical 0.2-1.0)
    # ========================================================================
    ax3 = axes[2]
    
    # Compute AURC for practical range (0.2-1.0)
    aurc_bal_practical = compute_aurc_range(
        balanced_data['rejection_rates'], 
        balanced_data['errors'],
        0.2, 1.0
    )
    
    aurc_worst_practical = compute_aurc_range(
        worst_data['rejection_rates'],
        worst_data['errors'],
        0.2, 1.0
    )
    
    # Bar chart
    x = np.arange(2)  # Balanced, Worst
    width = 0.35
    
    aurcs_full = [balanced_data['aurc_full'], worst_data['aurc_full']]
    aurcs_practical = [aurc_bal_practical, aurc_worst_practical]
    
    bars_full = ax3.bar(x - width/2, aurcs_full, width, 
                        label='Full (0-1)', color=[color_balanced, color_worst], alpha=0.7)
    
    bars_practical = ax3.bar(x + width/2, aurcs_practical, width,
                             label='Practical (0.2-1.0)', 
                             color=[color_balanced, color_worst], alpha=0.4, hatch='///')
    
    ax3.set_ylabel('AURC', fontsize=12)
    ax3.set_title('AURC Comparison (Full vs 0.2-1.0)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Balanced', 'Worst'])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars_full:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars_practical:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Add percentage difference annotations
    for i, (full, practical) in enumerate(zip(aurcs_full, aurcs_practical)):
        if full > 0:
            pct_diff = ((practical - full) / full) * 100
            obj_name = ['Balanced', 'Worst'][i]
            ax3.text(i, max(full, practical) * 0.9,
                    f'{obj_name} full AURC {pct_diff:+.1f}% vs {obj_name}',
                    ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'aurc_comparison_balanced_vs_worst.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison plot to: {output_path}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("AURC SUMMARY")
    print("="*70)
    print(f"{'Objective':<12} {'Full (0-1)':<12} {'Practical (0.2-1.0)':<18} {'Diff %':<10}")
    print("-"*70)
    
    for i, obj in enumerate(['Balanced', 'Worst']):
        full = aurcs_full[i]
        practical = aurcs_practical[i]
        diff = ((practical - full) / full * 100) if full > 0 else 0
        print(f"{obj:<12} {full:<12.4f} {practical:<18.4f} {diff:+.2f}%")
    
    print("\nWorst vs Balanced:")
    print(f"  Full AURC: +{(aurcs_full[1] - aurcs_full[0]) / aurcs_full[0] * 100:.2f}%")
    print(f"  Practical AURC: +{(aurcs_practical[1] - aurcs_practical[0]) / aurcs_practical[0] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Plot AURC comparison between objectives')
    parser.add_argument('--results_dir', type=str, 
                       default='results_map/cifar100_lt_if100',
                       help='Directory containing cost sweep results')
    parser.add_argument('--output_dir', type=str,
                       default='results_map/cifar100_lt_if100',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("📊 PLOTTING AURC COMPARISON")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    try:
        balanced_data = load_results(results_dir, 'balanced')
        print(f"✅ Loaded balanced results (AURC={balanced_data['aurc_full']:.4f})")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run: python3 train_map_cost_sweep.py --objective balanced")
        return
    
    try:
        worst_data = load_results(results_dir, 'worst')
        print(f"✅ Loaded worst results (AURC={worst_data['aurc_full']:.4f})")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run: python3 train_map_cost_sweep.py --objective worst")
        return
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    plot_comparison(balanced_data, worst_data, output_dir)
    
    print("\n" + "="*70)
    print("🎉 DONE!")
    print("="*70)


if __name__ == '__main__':
    main()
