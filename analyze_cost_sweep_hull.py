"""
Analyze Cost Sweep Results and Compute AURC Hull
==================================================

This script analyzes the results from cost sweep training and computes
the AURC "hull" (lower convex envelope) which represents the best achievable
performance when the controller knows the rejection cost.

Usage:
    python analyze_cost_sweep_hull.py --objective balanced
    python analyze_cost_sweep_hull.py --objective worst
"""

import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def aurc_from_cost_sweep(rejection_rates, errors, costs):
    """
    Compute AURC "hull" from cost sweep (lower convex envelope).
    
    For each cost c, find point (r, e) that minimizes e + c·r.
    This gives the best achievable RC curve when controller knows c.
    """
    # For each cost, find optimal point
    idxs = [int(np.argmin(errors + c * rejection_rates)) for c in costs]
    
    # Keep unique points, sorted by rejection rate
    uniq = sorted(set(idxs), key=lambda j: rejection_rates[j])
    
    r_hull = np.array([rejection_rates[j] for j in uniq])
    e_hull = np.array([errors[j] for j in uniq])
    
    # Compute AURC
    aurc_hull = np.trapz(e_hull, r_hull)
    
    return aurc_hull, r_hull, e_hull


def analyze_cost_sweep(results_file: Path, output_dir: Path):
    """Analyze cost sweep results."""
    
    print("="*70)
    print("COST SWEEP HULL ANALYSIS")
    print("="*70)
    print(f"Results file: {results_file}")
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    objective = data['objective']
    results_per_cost = data['results_per_cost']
    
    print(f"\nObjective: {objective}")
    print(f"Number of costs: {len(results_per_cost)}")
    
    # Extract data for best cost (minimum AURC on test)
    best_idx = int(np.argmin([r['aurc_test'] for r in results_per_cost]))
    best_result = results_per_cost[best_idx]
    
    rejection_rates = np.array(best_result['rc_curve_test']['rejection_rates'])
    errors = np.array(best_result['rc_curve_test']['selective_errors'])
    costs = [r['cost'] for r in results_per_cost]
    
    # Compute AURC hull
    aurc_hull, r_hull, e_hull = aurc_from_cost_sweep(rejection_rates, errors, costs)
    
    print(f"\n✅ AURC (standard): {best_result['aurc_test']:.4f}")
    print(f"✅ AURC (hull): {aurc_hull:.4f}")
    print(f"   Improvement: {(best_result['aurc_test'] - aurc_hull) / best_result['aurc_test'] * 100:.2f}%")
    print(f"   Hull points: {len(r_hull)}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: RC curve with hull
    ax1 = axes[0]
    ax1.plot(rejection_rates, errors, 'o-', linewidth=1, markersize=3, 
             label=f'Standard (AURC={best_result["aurc_test"]:.4f})', 
             color='blue', alpha=0.5)
    ax1.plot(r_hull, e_hull, 's-', linewidth=2, markersize=6,
             label=f'Hull (AURC={aurc_hull:.4f})',
             color='red')
    
    ax1.set_xlabel('Proportion of Rejections', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title(f'{objective.capitalize()}: RC Curve vs Hull', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    
    # Plot 2: Cost vs AURC
    ax2 = axes[1]
    
    costs_arr = np.array([r['cost'] for r in results_per_cost])
    aurcs_val = np.array([r['aurc_val'] for r in results_per_cost])
    aurcs_test = np.array([r['aurc_test'] for r in results_per_cost])
    
    ax2.plot(costs_arr, aurcs_val, 'o-', label='VAL', color='blue')
    ax2.plot(costs_arr, aurcs_test, 's-', label='TEST', color='red')
    ax2.axhline(y=aurc_hull, color='green', linestyle='--', label=f'Hull={aurc_hull:.4f}')
    
    ax2.set_xlabel('Rejection Cost (c)', fontsize=12)
    ax2.set_ylabel('AURC', fontsize=12)
    ax2.set_title(f'{objective.capitalize()}: AURC vs Cost', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / f'cost_sweep_hull_{objective}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved plot to: {plot_path}")
    
    plt.close()
    
    # Save hull data
    hull_data = {
        'objective': objective,
        'aurc_standard': float(best_result['aurc_test']),
        'aurc_hull': float(aurc_hull),
        'improvement_pct': float((best_result['aurc_test'] - aurc_hull) / best_result['aurc_test'] * 100),
        'hull_points': len(r_hull),
        'hull_rejection_rates': r_hull.tolist(),
        'hull_errors': e_hull.tolist(),
    }
    
    hull_file = output_dir / f'hull_analysis_{objective}.json'
    with open(hull_file, 'w') as f:
        json.dump(hull_data, f, indent=2)
    
    print(f"✅ Saved hull data to: {hull_file}")
    
    return hull_data


def main():
    parser = argparse.ArgumentParser(description='Analyze Cost Sweep Hull')
    parser.add_argument('--objective', type=str, default='balanced',
                       choices=['balanced', 'worst'],
                       help='Optimization objective')
    
    args = parser.parse_args()
    
    results_dir = Path('./results/map_cost_sweep/cifar100_lt_if100')
    results_file = results_dir / f'cost_sweep_{args.objective}.json'
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print(f"   Please run train_map_cost_sweep.py first!")
        return
    
    hull_data = analyze_cost_sweep(results_file, results_dir)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
