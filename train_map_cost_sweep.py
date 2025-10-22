"""
MAP Plugin Training with Cost Sweep
====================================

Train MAP plugin with multiple rejection costs to analyze risk-coverage trade-off.

Usage:
    python train_map_cost_sweep.py --objective balanced
    python train_map_cost_sweep.py --objective worst --eg_outer
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from src.models.gating_network_map import GatingNetwork, compute_uncertainty_for_map
from src.models.map_selector_simple import (
    SimpleMAPSelector,
    SimpleMAPConfig,
    SimpleGridSearchOptimizer,
    RCCurveComputer,
    compute_selective_metrics
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits_fixed',
        'num_classes': 100,
        'num_groups': 2,
        'group_boundaries': [69],  # Head: 0-68 (69 classes), Tail: 69-99 (31 classes)
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/cifar100_lt_if100/',
    },
    'gating': {
        'checkpoint': './checkpoints/gating_map/cifar100_lt_if100/final_gating.pth',
    },
    'map': {
        # Grid ranges
        'threshold_grid': list(np.linspace(0.1, 0.9, 17)),
        'gamma_grid': [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0],
        
        # Cost sweep for RC curves
        'cost_sweep': [0.0, 0.1, 0.5, 0.75, 0.85, 0.91, 0.95, 0.97, 0.99],
        
        # EG-outer
        'eg_iterations': 10,
        'eg_xi': 0.1,
        
        # Uncertainty coefficients
        'uncertainty_coeff_a': 1.0,
        'uncertainty_coeff_b': 1.0,
        'uncertainty_coeff_d': 1.0,
    },
    'evaluation': {
        'threshold_grid': list(np.linspace(0.0, 1.0, 100)),  # For RC curve
        'use_reweighting': True,
    },
    'output': {
        'checkpoints_dir': './checkpoints/map_cost_sweep/',
        'results_dir': './results/map_cost_sweep/',
    },
    'seed': 42
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def mean_risk(r: np.ndarray, e: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Compute mean risk over rejection range [lo, hi].
    
    Args:
        r: rejection rates
        e: errors
        lo: lower bound
        hi: upper bound
    
    Returns:
        mean risk (integral / width)
    """
    mask = (r >= lo) & (r <= hi)
    if mask.sum() < 2:
        return 0.0
    r_m, e_m = r[mask], e[mask]
    integral = np.trapezoid(e_m, r_m)
    width = max(hi - lo, 1e-6)
    return integral / width


def aurc_from_cost_sweep(
    rejection_rates: np.ndarray,
    errors: np.ndarray,
    costs: List[float]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AURC "hull" from cost sweep (lower convex envelope).
    
    For each cost c, find point (r, e) that minimizes e + c·r.
    This gives the best achievable RC curve when controller knows c.
    
    Args:
        rejection_rates: [N] rejection rates from RC curve
        errors: [N] errors from RC curve
        costs: [K] list of rejection costs
    
    Returns:
        aurc_hull: area under hull
        r_hull: rejection rates on hull
        e_hull: errors on hull
    """
    # For each cost, find optimal point
    idxs = [int(np.argmin(errors + c * rejection_rates)) for c in costs]
    
    # Keep unique points, sorted by rejection rate
    uniq = sorted(set(idxs), key=lambda j: rejection_rates[j])
    
    r_hull = np.array([rejection_rates[j] for j in uniq])
    e_hull = np.array([errors[j] for j in uniq])
    
    # Compute AURC
    aurc_hull = np.trapezoid(e_hull, r_hull)
    
    return aurc_hull, r_hull, e_hull


# ============================================================================
# DATA LOADING (same as train_map_simple.py)
# ============================================================================

def load_expert_logits(expert_names, logits_dir, split_name, device='cpu'):
    """Load expert logits."""
    logits_list = []
    
    for expert_name in expert_names:
        logits_path = Path(logits_dir) / expert_name / f"{split_name}_logits.pt"
        
        if not logits_path.exists():
            raise FileNotFoundError(f"Logits not found: {logits_path}")
        
        logits_e = torch.load(logits_path, map_location=device).float()
        logits_list.append(logits_e)
    
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    return logits


def load_labels(splits_dir, split_name, device='cpu'):
    """Load labels from CIFAR-100 dataset."""
    import torchvision
    
    # Load indices
    indices_file = f"{split_name}_indices.json"
    with open(Path(splits_dir) / indices_file, 'r') as f:
        indices = json.load(f)
    
    # Determine if train or test split
    is_train = split_name in ['expert', 'gating', 'train', 'tunev']
    
    dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=is_train,
        download=False
    )
    
    # Extract labels for these indices
    labels = torch.tensor([dataset.targets[i] for i in indices], dtype=torch.long, device=device)
    return labels


def load_sample_weights(splits_dir, split_name, device='cpu'):
    """
    Load sample weights for reweighting evaluation metrics.
    
    Purpose: REBALANCE test/val distribution to match train distribution.
    
    Test/Val: Balanced (~10 samples per class)
    Train: Imbalanced (500 for head, 5 for tail with IF=100)
    
    Solution: Weight each test sample by its class frequency in train
    → w_c ∝ freq_train(c) = n_train(c) / N_train
    
    Result:
    - Head classes (0-68): HIGH weight (many samples in train)
    - Tail classes (69-99): LOW weight (few samples in train)
    
    This makes evaluation reflect performance on the ACTUAL train distribution,
    not the artificial balanced test distribution.
    """
    weights_path = Path(splits_dir) / 'class_weights.json'
    
    if not weights_path.exists():
        print(f"⚠️  Class weights not found: {weights_path}")
        return None
    
    with open(weights_path, 'r') as f:
        class_weights_list = json.load(f)
    
    # Convert to tensor
    class_weights = torch.tensor(
        class_weights_list,
        dtype=torch.float32,
        device=device
    )
    
    # Get labels to create per-sample weights
    labels = load_labels(splits_dir, split_name, device)
    sample_weights = class_weights[labels]
    
    return sample_weights


# ============================================================================
# COST SWEEP TRAINING
# ============================================================================

def train_with_cost_sweep(
    objective: str = 'balanced',
    use_eg_outer: bool = False,
    verbose: bool = True
):
    """
    Train MAP plugin with multiple rejection costs.
    
    For each cost c ∈ {0.0, 0.1, 0.5, 0.75, 0.85, 0.91, 0.95, 0.97, 0.99}:
        1. Grid search to find best (θ, γ) that minimize R(θ,γ;c) = error + c·ρ
        2. Compute RC curve with this (θ, γ)
        3. Save results
    """
    print("="*70)
    print("🚀 MAP COST SWEEP TRAINING")
    print("="*70)
    print(f"Objective: {objective}")
    print(f"EG-Outer: {use_eg_outer}")
    print(f"Device: {DEVICE}")
    print(f"Cost values: {CONFIG['map']['cost_sweep']}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # ========================================================================
    # 1. LOAD GATING
    # ========================================================================
    print("\n" + "="*70)
    print("1. LOADING GATING NETWORK")
    print("="*70)
    
    num_experts = len(CONFIG['experts']['names'])
    num_classes = CONFIG['dataset']['num_classes']
    
    gating = GatingNetwork(
        num_experts=num_experts,
        num_classes=num_classes,
        routing='dense'
    ).to(DEVICE)
    
    gating_checkpoint_path = Path(CONFIG['gating']['checkpoint'])
    checkpoint = torch.load(gating_checkpoint_path, map_location=DEVICE, weights_only=False)
    gating.load_state_dict(checkpoint['model_state_dict'])
    gating.eval()
    
    print(f"✅ Loaded gating from: {gating_checkpoint_path}")
    
    # ========================================================================
    # 2. LOAD DATA (val for optimization)
    # ========================================================================
    print("\n" + "="*70)
    print("2. LOADING DATA")
    print("="*70)
    
    # Use tunev for optimization (balanced set for selective training)
    optimization_split = 'tunev'
    expert_logits_val = load_expert_logits(
        CONFIG['experts']['names'],
        CONFIG['experts']['logits_dir'],
        optimization_split,
        DEVICE
    )
    
    labels_val = load_labels(CONFIG['dataset']['splits_dir'], optimization_split, DEVICE)
    
    sample_weights_val = None
    if CONFIG['evaluation']['use_reweighting']:
        sample_weights_val = load_sample_weights(
            CONFIG['dataset']['splits_dir'],
            optimization_split,
            DEVICE
        )
        print("✅ Reweighting enabled")
    
    print(f"✅ {optimization_split.capitalize()}: {expert_logits_val.shape[0]} samples")
    print(f"   Expert logits shape: {expert_logits_val.shape}")
    print(f"   Expert logits range: [{expert_logits_val.min():.3f}, {expert_logits_val.max():.3f}]")
    
    # Check for NaN/Inf in expert logits
    if torch.isnan(expert_logits_val).any():
        print(f"⚠️  WARNING: Expert logits contain NaN! Count: {torch.isnan(expert_logits_val).sum()}")
    if torch.isinf(expert_logits_val).any():
        print(f"⚠️  WARNING: Expert logits contain Inf! Count: {torch.isinf(expert_logits_val).sum()}")
    
    # ========================================================================
    # 3. COMPUTE MIXTURE POSTERIORS & UNCERTAINTY
    # ========================================================================
    print("\n" + "="*70)
    print("3. COMPUTING MIXTURE POSTERIORS")
    print("="*70)
    
    with torch.no_grad():
        expert_posteriors_val = F.softmax(expert_logits_val, dim=-1)
        
        # Get gating weights
        gating_output = gating(expert_posteriors_val)
        if isinstance(gating_output, tuple):
            gating_weights_val = gating_output[0]  # [B, E]
        else:
            gating_weights_val = gating_output
        
        # Check for NaN
        if torch.isnan(gating_weights_val).any():
            print("⚠️  WARNING: Gating produces NaN! Falling back to uniform weights")
            B, E = expert_logits_val.shape[0], expert_logits_val.shape[1]
            gating_weights_val = torch.ones(B, E, device=DEVICE) / E
        else:
            print("✅ Gating network working properly!")
        
        # Check weights sum to 1
        weights_sum = gating_weights_val.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), \
            f"Gating weights don't sum to 1: {weights_sum.min():.6f} - {weights_sum.max():.6f}"
        
        mixture_posterior_val = (gating_weights_val.unsqueeze(-1) * expert_posteriors_val).sum(dim=1)
        
        # Compute raw uncertainty
        uncertainty_val_raw = compute_uncertainty_for_map(
            posteriors=expert_posteriors_val,
            weights=gating_weights_val,
            mixture_posterior=mixture_posterior_val,
            coeffs={
                'a': CONFIG['map']['uncertainty_coeff_a'],
                'b': CONFIG['map']['uncertainty_coeff_b'],
                'd': CONFIG['map']['uncertainty_coeff_d']
            }
        )
        
        # Z-score normalization for stable γ
        U_mu = uncertainty_val_raw.mean()
        U_sigma = uncertainty_val_raw.std().clamp_min(1e-6)
        uncertainty_val = (uncertainty_val_raw - U_mu) / U_sigma
    
    print(f"✅ Mixture posteriors computed")
    print(f"   Gating weights range: [{gating_weights_val.min():.3f}, {gating_weights_val.max():.3f}]")
    print(f"   Gating weights sum: [{weights_sum.min():.6f}, {weights_sum.max():.6f}] (should be ~1.0)")
    print(f"   Mixture posterior range: [{mixture_posterior_val.min():.3f}, {mixture_posterior_val.max():.3f}]")
    print(f"   Mixture posterior sum: {mixture_posterior_val.sum(dim=-1).mean():.3f} (should be ~1.0)")
    print(f"   Uncertainty (raw) range: [{uncertainty_val_raw.min():.3f}, {uncertainty_val_raw.max():.3f}]")
    print(f"   Uncertainty (z-score) μ={U_mu:.3f}, σ={U_sigma:.3f}")
    print(f"   Uncertainty (normalized) range: [{uncertainty_val.min():.3f}, {uncertainty_val.max():.3f}]")
    
    # Check for NaNs
    if torch.isnan(uncertainty_val).any():
        print("⚠️  WARNING: Uncertainty contains NaN values!")
        print(f"   NaN count: {torch.isnan(uncertainty_val).sum()}/{len(uncertainty_val)}")
        # Replace NaN with 0 for now
        uncertainty_val = torch.nan_to_num(uncertainty_val, nan=0.0)
    
    # ========================================================================
    # 4. COST SWEEP
    # ========================================================================
    print("\n" + "="*70)
    print("4. COST SWEEP OPTIMIZATION")
    print("="*70)
    
    cost_sweep = CONFIG['map']['cost_sweep']
    results_per_cost = []
    
    for cost in cost_sweep:
        print(f"\n{'='*70}")
        print(f"Training with rejection_cost = {cost}")
        print(f"{'='*70}")
        
        # Create MAP config for this cost
        map_config = SimpleMAPConfig(
            num_classes=CONFIG['dataset']['num_classes'],
            num_groups=CONFIG['dataset']['num_groups'],
            group_boundaries=CONFIG['dataset']['group_boundaries'],
            threshold_grid=CONFIG['map']['threshold_grid'],
            gamma_grid=CONFIG['map']['gamma_grid'],
            objective=objective,
            rejection_cost=cost  # KEY: Set rejection cost
        )
        
        selector = SimpleMAPSelector(map_config).to(DEVICE)
        optimizer = SimpleGridSearchOptimizer(map_config)
        
        # Grid search
        best_result = optimizer.search(
            selector,
            mixture_posterior_val,
            uncertainty_val,
            labels_val,
            beta=None,  # TODO: Support EG-outer
            sample_weights=sample_weights_val,
            verbose=verbose
        )
        
        # Set best parameters
        selector.set_parameters(best_result.threshold, best_result.gamma)
        
        # ====================================================================
        # VAL: Compute RC curve with OPTIMAL gamma (from grid search)
        # ====================================================================
        rc_computer = RCCurveComputer(map_config)
        
        rc_data_val = rc_computer.compute_rc_curve(
            selector,
            mixture_posterior_val,
            uncertainty_val,
            labels_val,
            gamma=best_result.gamma,  # ← Use optimal γ* from grid search
            threshold_grid=np.linspace(0.0, 1.0, 200),  # Sweep θ from 0 to 1
            sample_weights=sample_weights_val
        )
        
        # ====================================================================
        # TEST: Evaluate on test set with same parameters
        # ====================================================================
        print(f"\n   Evaluating on TEST set...")
        
        # Load test data
        expert_logits_test = load_expert_logits(
            CONFIG['experts']['names'],
            CONFIG['experts']['logits_dir'],
            'test',
            DEVICE
        )
        labels_test = load_labels(CONFIG['dataset']['splits_dir'], 'test', DEVICE)
        
        # Compute test posteriors & uncertainty
        with torch.no_grad():
            expert_posteriors_test = F.softmax(expert_logits_test, dim=-1)
            
            gating_output_test = gating(expert_posteriors_test)
            if isinstance(gating_output_test, tuple):
                gating_weights_test = gating_output_test[0]
            else:
                gating_weights_test = gating_output_test
            
            mixture_posterior_test = (gating_weights_test.unsqueeze(-1) * expert_posteriors_test).sum(dim=1)
            
            # Use SAME U scaling as VAL (critical for deployment!)
            uncertainty_test_raw = compute_uncertainty_for_map(
                posteriors=expert_posteriors_test,
                weights=gating_weights_test,
                mixture_posterior=mixture_posterior_test,
                coeffs={
                    'a': CONFIG['map']['uncertainty_coeff_a'],
                    'b': CONFIG['map']['uncertainty_coeff_b'],
                    'd': CONFIG['map']['uncertainty_coeff_d']
                }
            )
            uncertainty_test = (uncertainty_test_raw - U_mu) / U_sigma
        
        # Sample weights for test
        sample_weights_test = None
        if CONFIG['evaluation']['use_reweighting']:
            sample_weights_test = load_sample_weights(
                CONFIG['dataset']['splits_dir'],
                'test',
                DEVICE
            )
        
        # RC curve on TEST with gamma* from VAL
        rc_data_test = rc_computer.compute_rc_curve(
            selector,
            mixture_posterior_test,
            uncertainty_test,
            labels_test,
            gamma=best_result.gamma,
            threshold_grid=np.linspace(0.0, 1.0, 200),
            sample_weights=sample_weights_test
        )
        
        print(f"   ✅ TEST: AURC = {rc_data_test['aurc']:.4f}")
        
        # ====================================================================
        # Compute objective value (for consistency with optimization)
        # ====================================================================
        if objective == 'balanced':
            objective_error = np.mean(best_result.group_errors)
        else:  # worst
            objective_error = np.max(best_result.group_errors)
        
        objective_value = objective_error + cost * (1.0 - best_result.coverage)
        
        # Store results (convert all to native Python types for JSON)
        result_dict = {
            'cost': float(cost),
            'threshold': float(best_result.threshold),
            'gamma': float(best_result.gamma),
            'selective_error': float(best_result.selective_error),
            'coverage': float(best_result.coverage),
            'group_errors': [float(x) for x in best_result.group_errors],
            'worst_group_error': float(best_result.worst_group_error),
            'objective_value': float(objective_value),
            # VAL metrics
            'aurc_val': float(rc_data_val['aurc']),
            'rc_curve_val': {
                'rejection_rates': [float(x) for x in rc_data_val['rejection_rates']],
                'selective_errors': [float(x) for x in rc_data_val['selective_errors']],
            },
            # TEST metrics (primary for reporting)
            'aurc_test': float(rc_data_test['aurc']),
            'rc_curve_test': {
                'rejection_rates': [float(x) for x in rc_data_test['rejection_rates']],
                'selective_errors': [float(x) for x in rc_data_test['selective_errors']],
            },
            # Uncertainty normalization params (for deployment)
            'uncertainty_mu': float(U_mu),
            'uncertainty_sigma': float(U_sigma),
        }
        
        results_per_cost.append(result_dict)
        
        print(f"\n✅ Cost={cost}:")
        print(f"   θ={best_result.threshold:.3f}, γ={best_result.gamma:.3f}")
        print(f"   Objective value: {objective_value:.4f} (error={objective_error:.4f} + cost×ρ={cost * (1-best_result.coverage):.4f})")
        print(f"   Selective error: {best_result.selective_error:.4f}")
        print(f"   Coverage: {best_result.coverage:.3f}")
        print(f"   AURC (VAL): {rc_data_val['aurc']:.4f}")
        print(f"   AURC (TEST): {rc_data_test['aurc']:.4f}")  # ← Primary metric
    
    # ========================================================================
    # 5. SAVE RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("5. SAVING RESULTS")
    print("="*70)
    
    checkpoint_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    results_dir = Path(CONFIG['output']['results_dir']) / CONFIG['dataset']['name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all results
    save_dict = {
        'objective': objective,
        'use_eg_outer': use_eg_outer,
        'cost_sweep': cost_sweep,
        'results_per_cost': results_per_cost,
        'config': CONFIG
    }
    
    output_path = results_dir / f'cost_sweep_{objective}.json'
    with open(output_path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"✅ Saved results to: {output_path}")
    
    # ========================================================================
    # 6. PLOT RC CURVES
    # ========================================================================
    print("\n" + "="*70)
    print("6. PLOTTING RC CURVES")
    print("="*70)
    
    plot_rc_curves(results_per_cost, results_dir, objective)
    
    return results_per_cost


def plot_rc_curves(results_per_cost: List[Dict], output_dir: Path, objective: str):
    """
    Plot RC curves with 3 panels:
    1. Error vs Rejection Rate (full range 0-1)
    2. Error vs Rejection Rate (practical range 0-0.8)
    3. AURC Comparison (Full vs Practical)
    
    Uses TEST data (not VAL) for all plots.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Select best result (minimum AURC on TEST)
    best_idx = int(np.argmin([r['aurc_test'] for r in results_per_cost]))
    best_result = results_per_cost[best_idx]
    
    # Extract TEST data
    rejection_rates = np.array(best_result['rc_curve_test']['rejection_rates'])
    errors = np.array(best_result['rc_curve_test']['selective_errors'])
    
    # ========================================================================
    # Plot 1: Error vs Rejection Rate (Full range 0-1)
    # ========================================================================
    ax1 = axes[0]
    ax1.plot(rejection_rates, errors, 'o-', linewidth=2, markersize=3, 
             label=f'{objective.capitalize()} (AURC={best_result["aurc_test"]:.4f})', 
             color='green')
    
    ax1.set_xlabel('Proportion of Rejections', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title(f'Error vs Rejection Rate (0-1)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, min(1.05, errors.max() * 1.1)])
    
    # ========================================================================
    # Plot 2: Error vs Rejection Rate (Practical range 0-0.8)
    # ========================================================================
    ax2 = axes[1]
    
    # Filter data for rejection rate <= 0.8
    mask = rejection_rates <= 0.8
    rejection_practical = rejection_rates[mask]
    errors_practical = errors[mask]
    
    ax2.plot(rejection_practical, errors_practical, 'o-', linewidth=2, markersize=3,
             label=f'{objective.capitalize()}', color='green')
    
    ax2.set_xlabel('Proportion of Rejections', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title(f'Error vs Rejection Rate (0-0.8)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.8])
    ax2.set_ylim([0, max(errors_practical) * 1.1 if len(errors_practical) > 0 else 1.0])
    
    # ========================================================================
    # Plot 3: Mean Risk Comparison (Full vs Practical 0.2-1.0)
    # ========================================================================
    ax3 = axes[2]
    
    def mean_risk(r, e, lo=0.0, hi=1.0):
        """Compute mean risk over rejection range [lo, hi]"""
        mask = (r >= lo) & (r <= hi)
        if mask.sum() < 2:
            return 0.0
        r_m, e_m = r[mask], e[mask]
        integral = np.trapezoid(e_m, r_m)
        width = max(hi - lo, 1e-6)
        return integral / width
    
    # Compute mean risks
    mean_risk_full = mean_risk(rejection_rates, errors, 0.0, 1.0)
    mean_risk_practical = mean_risk(rejection_rates, errors, 0.2, 1.0)
    
    # Bar chart
    x_pos = np.arange(1)
    width = 0.35
    
    bar1 = ax3.bar(x_pos - width/2, [mean_risk_full], width, label='Full (0-1)', 
                   color='green', alpha=0.7)
    bar2 = ax3.bar(x_pos + width/2, [mean_risk_practical], width, label='Practical (0.2-1.0)',
                   color='green', alpha=0.4, hatch='///')
    
    ax3.set_ylabel('Mean Risk', fontsize=12)
    ax3.set_title('Mean Risk Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([objective.capitalize()])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bar1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    for bar in bar2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Add percentage difference annotation
    if mean_risk_full > 0:
        pct_diff = ((mean_risk_practical - mean_risk_full) / mean_risk_full) * 100
        ax3.text(0, max(mean_risk_full, mean_risk_practical) * 0.95,
                f'{pct_diff:+.1f}% (Practical vs Full)',
                ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    plot_path = output_dir / f'aurc_curves_{objective}_test.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to: {plot_path}")
    
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MAP Plugin with Cost Sweep')
    parser.add_argument('--objective', type=str, default='balanced',
                       choices=['balanced', 'worst'],
                       help='Optimization objective')
    parser.add_argument('--eg_outer', action='store_true',
                       help='Use EG-outer for worst-group')
    parser.add_argument('--no_reweight', action='store_true',
                       help='Disable reweighting on test set')
    
    args = parser.parse_args()
    
    if args.no_reweight:
        CONFIG['evaluation']['use_reweighting'] = False
    
    results = train_with_cost_sweep(
        objective=args.objective,
        use_eg_outer=args.eg_outer,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("🎉 COST SWEEP COMPLETED!")
    print("="*70)
    print(f"\nSummary of {len(results)} configurations:")
    print(f"{'Cost':<8} {'θ':<8} {'γ':<8} {'Obj.Val':<10} {'Error':<10} {'Coverage':<10} {'AURC(VAL)':<12} {'AURC(TEST)':<12}")
    print("-" * 100)
    for r in results:
        # Compute objective-specific error
        if args.objective == 'balanced':
            obj_error = np.mean(r['group_errors'])
        else:  # worst
            obj_error = np.max(r['group_errors'])
        
        print(f"{r['cost']:<8.2f} {r['threshold']:<8.3f} {r['gamma']:<8.3f} "
              f"{r['objective_value']:<10.4f} {obj_error:<10.4f} {r['coverage']:<10.3f} "
              f"{r['aurc_val']:<12.4f} {r['aurc_test']:<12.4f}")


if __name__ == '__main__':
    main()
