"""
Comprehensive AURC Evaluation (Correct Methodology)

This script follows the proper AURC evaluation methodology:
  - Validation set: tuneV + val_lt (combined for threshold optimization)
  - Test set: test_lt (held-out for final evaluation)

The algorithm:
  1. For each rejection cost c and metric (standard/balanced/worst):
     - Find optimal threshold on validation set (tuneV + val_lt)
     - Apply that threshold to test set (test_lt) to measure coverage and risk
  2. Compute AURC by integrating risk over coverage

It loads optimal parameters (α*, μ*, and gating) from the plugin checkpoint,
computes mixture posteriors, constructs GSE margins as confidence scores, and
then sweeps rejection costs to produce risk–coverage curves.

Outputs:
  - aurc_detailed_results.csv: RC points for each metric
  - aurc_summary.json: AURC values for full and [0.2, 1.0] coverage ranges
  - aurc_curves.png: plots of RC curves and AURC comparison
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

# Custom modules
from src.models.argse import AR_GSE
from src.train.gse_balanced_plugin import compute_margin
from src.metrics.reweighted_metrics import ReweightedMetrics  # Import reweighted metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits_fixed',  # Updated to use fixed splits
        'num_classes': 100,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits_fixed',  # Updated to use recomputed logits
    },
    'aurc_eval': {
        'cost_values': np.linspace(0.0, 1.0, 81),  # 81 cost values from 0 to 1.0
        'metrics': ['standard', 'balanced', 'worst'],
    },
    'plugin_checkpoint': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
    'output_dir': './results_worst_eg_improved/cifar100_lt_if100',
    'seed': 42,
}

#############################################
# Data loading for comprehensive AURC only  #
#############################################

def load_aurc_splits_data():
    """
    Load all three splits for proper AURC evaluation:
      - tunev: tuning/validation split for threshold optimization
      - val: validation split for threshold optimization  
      - test: held-out test set for final evaluation
    
    Returns:
        Tuple of (tunev_data, val_data, test_data) where each is (logits, labels, indices)
    """
    logits_root = Path(CONFIG['experts']['logits_dir'])
    # Check if dataset subdirectory exists
    dataset_subdir = logits_root / CONFIG['dataset']['name']
    if dataset_subdir.exists():
        logits_root = dataset_subdir
    
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    num_experts = len(CONFIG['experts']['names'])
    num_classes = CONFIG['dataset']['num_classes']
    
    # Load splits indices
    print("📂 Loading AURC evaluation splits...")
    
    # tunev (from test set - balanced)
    with open(splits_dir / 'tunev_indices.json', 'r') as f:
        tunev_indices = json.load(f)
    
    # val (from test set - balanced)
    with open(splits_dir / 'val_indices.json', 'r') as f:
        val_indices = json.load(f)
    
    # test (from test set - balanced)
    with open(splits_dir / 'test_indices.json', 'r') as f:
        test_indices = json.load(f)
    
    print(f"✅ tunev: {len(tunev_indices)} samples (test set - balanced)")
    print(f"✅ val: {len(val_indices)} samples (test set - balanced)")
    print(f"✅ test: {len(test_indices)} samples (test set - balanced)")
    print(f"✅ Validation (tunev + val): {len(tunev_indices) + len(val_indices)} samples")
    
    # Load datasets - all splits come from test set now (balanced)
    cifar_test_full = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    
    # Helper function to load logits (supports both .npz and .pt)
    def load_expert_logits(split_name, indices):
        logits = torch.zeros(len(indices), num_experts, num_classes)
        for i, expert_name in enumerate(CONFIG['experts']['names']):
            # Try .npz first (new format), then .pt (old format)
            npz_path = logits_root / expert_name / f"{split_name}_logits.npz"
            pt_path = logits_root / expert_name / f"{split_name}_logits.pt"
            
            if npz_path.exists():
                data = np.load(npz_path)
                logits[:, i, :] = torch.from_numpy(data['logits'])
            elif pt_path.exists():
                logits[:, i, :] = torch.load(pt_path, map_location='cpu', weights_only=False)
            else:
                raise FileNotFoundError(f"Missing logits for {expert_name} split {split_name}: {npz_path} or {pt_path}")
        return logits
    
    # Load tunev data
    tunev_logits = load_expert_logits('tunev', tunev_indices)
    tunev_labels = torch.tensor(np.array(cifar_test_full.targets)[tunev_indices])
    
    # Load val data
    val_logits = load_expert_logits('val', val_indices)
    val_labels = torch.tensor(np.array(cifar_test_full.targets)[val_indices])
    
    # Load test data
    test_logits = load_expert_logits('test', test_indices)
    test_labels = torch.tensor(np.array(cifar_test_full.targets)[test_indices])
    
    return (tunev_logits, tunev_labels, tunev_indices), (val_logits, val_labels, val_indices), (test_logits, test_labels, test_indices)

def get_mixture_posteriors(model, logits):
    """Compute mixture posteriors η̃(x) from expert logits."""
    model.eval()
    with torch.no_grad():
        logits = logits.to(DEVICE)
        expert_posteriors = torch.softmax(logits, dim=-1)              # [B, E, C]
        gating_features = model.feature_builder(logits)
        gating_weights = torch.softmax(model.gating_net(gating_features), dim=1)  # [B, E]
        eta_mix = torch.einsum('be,bec->bc', gating_weights, expert_posteriors)   # [B, C]
    return eta_mix.cpu()

#############################################
# Core AURC utilities                        #
#############################################


def compute_group_risk_for_aurc(preds, labels, accepted_mask, class_to_group, K, 
                               class_weights=None, metric_type="balanced"):
    """
    Compute group-aware risk for AURC evaluation on accepted samples.
    
    Args:
        preds: [N] predictions
        labels: [N] true labels  
        accepted_mask: [N] boolean mask for accepted samples
        class_to_group: [C] class to group mapping
        K: number of groups
        class_weights: dict mapping class_id -> weight (for reweighting), or None
        metric_type: 'standard', 'balanced', or 'worst'
        
    Returns:
        risk: scalar risk value (error rate)
    """
    if accepted_mask.sum() == 0:
        return 1.0
    
    y = labels
    g = class_to_group[y]
    
    if metric_type == 'standard':
        correct = (preds[accepted_mask] == y[accepted_mask])
        
        # Apply reweighting if class_weights provided
        if class_weights is not None:
            weights = torch.tensor([class_weights[int(c)] for c in y[accepted_mask]], 
                                  dtype=torch.float32)
            weighted_correct = (correct.float() * weights).sum()
            total_weight = weights.sum()
            return 1.0 - (weighted_correct / total_weight).item()
        else:
            return 1.0 - correct.float().mean().item()
    
    group_errors = []
    for k in range(K):
        group_mask = (g == k)
        group_accepted = accepted_mask & group_mask
        if group_accepted.sum() == 0:
            group_errors.append(1.0)
        else:
            group_correct = (preds[group_accepted] == y[group_accepted])
            
            # Apply reweighting if class_weights provided
            if class_weights is not None:
                weights = torch.tensor([class_weights[int(c)] for c in y[group_accepted]], 
                                      dtype=torch.float32)
                weighted_correct = (group_correct.float() * weights).sum()
                total_weight = weights.sum()
                group_error = 1.0 - (weighted_correct / total_weight).item()
            else:
                group_error = 1.0 - group_correct.float().mean().item()
            
            group_errors.append(group_error)
    if metric_type == 'balanced':
        return float(np.mean(group_errors))
    elif metric_type == 'worst':
        return float(np.max(group_errors))
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

def find_optimal_threshold_for_cost(confidence_scores, preds, labels, class_to_group, K, 
                                   cost_c, class_weights=None, metric_type="balanced"):
    """
    Find optimal threshold that minimizes: risk + c * (1 - coverage)
    
    Args:
        confidence_scores: [N] confidence scores (GSE margins)
        preds: [N] predictions
        labels: [N] true labels
        class_to_group: [C] class to group mapping
        K: number of groups
        cost_c: rejection cost
        class_weights: dict mapping class_id -> weight (for reweighting), or None
        metric_type: risk metric type
        
    Returns:
        optimal_threshold: scalar threshold value
    """
    # Create candidate thresholds from unique confidence scores
    unique_scores = torch.unique(confidence_scores)
    thresholds = torch.cat([torch.tensor([confidence_scores.min().item() - 1.0]), 
                           unique_scores, 
                           torch.tensor([confidence_scores.max().item() + 1.0])])
    thresholds = torch.sort(thresholds, descending=True)[0]  # High to low
    
    best_cost = float('inf')
    best_threshold = 0.0
    
    for threshold in thresholds:
        accepted = confidence_scores >= threshold
        coverage = accepted.float().mean().item()
        risk = compute_group_risk_for_aurc(preds, labels, accepted, class_to_group, K, 
                                          class_weights, metric_type)
        
        # Objective: risk + c * rejection_rate
        objective = risk + cost_c * (1.0 - coverage)
        
        if objective < best_cost:
            best_cost = objective
            best_threshold = threshold.item()
    
    return best_threshold

def sweep_cost_values_aurc(confidence_scores_val, preds_val, labels_val, 
                          confidence_scores_test, preds_test, labels_test,
                          class_to_group, K, cost_values, class_weights=None, metric_type="balanced"):
    """
    Sweep cost values and return (cost, coverage, risk) points on test set.
    
    Args:
        confidence_scores_val: [N_val] validation confidence scores
        preds_val: [N_val] validation predictions
        labels_val: [N_val] validation labels
        confidence_scores_test: [N_test] test confidence scores
        preds_test: [N_test] test predictions
        labels_test: [N_test] test labels
        class_to_group: [C] class to group mapping
        K: number of groups
        cost_values: array of cost values to sweep
        class_weights: dict mapping class_id -> weight (for reweighting), or None
        metric_type: risk metric type
        
    Returns:
        rc_points: list of (cost, coverage, risk) tuples
    """
    rc_points = []
    
    print(f"🔄 Sweeping {len(cost_values)} cost values for {metric_type} metric...")
    
    for i, cost_c in enumerate(cost_values):
        # Find optimal threshold on validation
        optimal_threshold = find_optimal_threshold_for_cost(
            confidence_scores_val, preds_val, labels_val, class_to_group, K, cost_c, 
            class_weights, metric_type
        )
        
        # Apply to test set
        accepted_test = confidence_scores_test >= optimal_threshold
        coverage_test = accepted_test.float().mean().item()
        risk_test = compute_group_risk_for_aurc(preds_test, labels_test, accepted_test, 
                                               class_to_group, K, class_weights, metric_type)
        
        rc_points.append((cost_c, coverage_test, risk_test))
        
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(cost_values)} - Current: c={cost_c:.3f}, "
                  f"cov={coverage_test:.3f}, risk={risk_test:.3f}")
    
    return rc_points

def compute_aurc_from_points(rc_points, coverage_range='full'):
    """
    Compute AURC using trapezoidal integration.
    
    Args:
        rc_points: List of (cost, coverage, risk) tuples
        coverage_range: 'full' for [0, 1] or '0.2-1.0' for [0.2, 1.0]
        
    Returns:
        aurc: scalar AURC value
    """
    # Sort by coverage
    rc_points = sorted(rc_points, key=lambda x: x[1])
    
    coverages = [p[1] for p in rc_points]
    risks = [p[2] for p in rc_points]
    
    if coverage_range == '0.2-1.0':
        # Need to interpolate risk at coverage=0.2 BEFORE filtering
        # Find points around 0.2 for interpolation
        all_points = list(zip(coverages, risks))
        
        # Find the last point with coverage < 0.2 and first point with coverage >= 0.2
        points_below = [(c, r) for c, r in all_points if c < 0.2]
        points_above = [(c, r) for c, r in all_points if c >= 0.2]
        
        if not points_above:
            # No points >= 0.2, cannot compute
            return float('nan')
        
        # Interpolate risk at coverage = 0.2
        if points_below:
            # Have points on both sides of 0.2
            c_below, r_below = points_below[-1]  # Last point before 0.2
            c_above, r_above = points_above[0]   # First point after 0.2
            
            if c_above > c_below:
                # Linear interpolation
                risk_at_02 = r_below + (r_above - r_below) * (0.2 - c_below) / (c_above - c_below)
            else:
                # Should not happen, but use r_above as fallback
                risk_at_02 = r_above
        else:
            # No points below 0.2, use first point's risk
            risk_at_02 = points_above[0][1]
        
        # Build final curve starting from 0.2
        coverages = [0.2] + [c for c, r in points_above]
        risks = [risk_at_02] + [r for c, r in points_above]
        
        # Ensure endpoint at 1.0
        if coverages[-1] < 1.0:
            coverages = coverages + [1.0]
            risks = risks + [risks[-1]]
    else:
        # Full range [0, 1]
        # Ensure we have endpoints for proper integration
        if coverages[0] > 0.0:
            coverages = [0.0] + coverages
            # When coverage=0 (reject all), risk should be very high (no correct predictions)
            # Use the first available risk as approximation (conservative)
            risks = [risks[0]] + risks
        
        if coverages[-1] < 1.0:
            coverages = coverages + [1.0]
            risks = risks + [risks[-1]]  # Extend last risk to coverage=1
    
    # Trapezoidal integration
    aurc = np.trapz(risks, coverages)
    return aurc
#############################################
# Plotting                                    
#############################################

def plot_aurc_curves(all_rc_points, aurc_results, save_path):
    """Plot risk-coverage curves for different metrics."""
    plt.figure(figsize=(15, 5))
    
    colors = ['blue', 'red', 'green', 'orange']
    linestyles = ['-', '--', '-.', ':']
    
    # Full range plot
    plt.subplot(1, 3, 1)
    for i, (metric, rc_points) in enumerate(all_rc_points.items()):
        rc_points = sorted(rc_points, key=lambda x: x[1])
        coverages = [p[1] for p in rc_points]
        risks = [p[2] for p in rc_points]
        
        aurc = aurc_results[metric]
        plt.plot(coverages, risks, color=colors[i % len(colors)], 
                linestyle=linestyles[i % len(linestyles)], linewidth=2,
                label=f'{metric.title()} (AURC={aurc:.4f})')
    
    plt.xlabel('Coverage (Fraction Accepted)')
    plt.ylabel('Risk (Error on Accepted)')
    plt.title('Risk-Coverage Curves (Full Range)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, None)
    
    # Focused range plot (0.2-1.0)
    plt.subplot(1, 3, 2)
    for i, (metric, rc_points) in enumerate(all_rc_points.items()):
        rc_points = sorted(rc_points, key=lambda x: x[1])
        coverages = [p[1] for p in rc_points if p[1] >= 0.2]
        risks = [p[2] for p in rc_points if p[1] >= 0.2]
        
        plt.plot(coverages, risks, color=colors[i % len(colors)], 
                linestyle=linestyles[i % len(linestyles)], linewidth=2,
                label=f'{metric.title()}')
    
    plt.xlabel('Coverage (Fraction Accepted)')
    plt.ylabel('Risk (Error on Accepted)')
    plt.title('Risk-Coverage Curves (0.2-1.0)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0.2, 1.0)
    plt.ylim(0, None)
    
    # AURC comparison bar plot
    plt.subplot(1, 3, 3)
    metrics = list(aurc_results.keys())
    aurcs = list(aurc_results.values())
    
    bars = plt.bar(metrics, aurcs, color=colors[:len(metrics)], alpha=0.7)
    plt.ylabel('AURC Value')
    plt.title('AURC Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, aurc in zip(bars, aurcs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{aurc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved AURC plots to {save_path}")

def main():
    # Reproducibility
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    print("=== Comprehensive AURC Evaluation (Reweighted for Long-Tail) ===")

    # Output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class weights for reweighting
    class_weights_path = Path(CONFIG['dataset']['splits_dir']) / 'class_weights.json'
    if class_weights_path.exists():
        with open(class_weights_path, 'r') as f:
            class_weights_list = json.load(f)
        # Convert list to dict: class_id -> weight
        class_weights = {i: w for i, w in enumerate(class_weights_list)}
        print(f"✅ Loaded class weights from {class_weights_path}")
        print(f"   Sample weights: class 0={class_weights[0]:.4f}, class 99={class_weights[99]:.4f}")
    else:
        class_weights = None
        print("⚠️  No class_weights.json found - using uniform weighting")

    # 1) Load plugin checkpoint (α*, μ*, class_to_group, gating)
    plugin_ckpt_path = Path(CONFIG['plugin_checkpoint'])
    if not plugin_ckpt_path.exists():
        raise FileNotFoundError(f"Plugin checkpoint not found: {plugin_ckpt_path}")
    print(f"📂 Loading plugin checkpoint: {plugin_ckpt_path}")
    checkpoint = torch.load(plugin_ckpt_path, map_location=DEVICE, weights_only=False)
    alpha_star = checkpoint['alpha'].to(DEVICE)
    mu_star = checkpoint['mu'].to(DEVICE)
    class_to_group = checkpoint['class_to_group'].to(DEVICE)
    num_groups = checkpoint['num_groups']

    print("✅ Loaded optimal parameters:")
    print(f"   α* = {alpha_star.detach().cpu().tolist()}")
    print(f"   μ* = {mu_star.detach().cpu().tolist()}")

    # 2) Build AR-GSE and load gating
    num_experts = len(CONFIG['experts']['names'])
    with torch.no_grad():
        dummy_logits = torch.zeros(2, num_experts, CONFIG['dataset']['num_classes']).to(DEVICE)
        temp_model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, 1).to(DEVICE)
        gating_feature_dim = temp_model.feature_builder(dummy_logits).size(-1)
        del temp_model
    print(f"✅ Dynamic gating feature dim: {gating_feature_dim}")
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)

    # Load gating network weights with dimension compatibility check
    if 'gating_net_state_dict' in checkpoint:
        saved_state = checkpoint['gating_net_state_dict']
        current_state = model.gating_net.state_dict()
        
        compatible = True
        for key in saved_state.keys():
            if key in current_state and saved_state[key].shape != current_state[key].shape:
                print(f"⚠️  Dimension mismatch for {key}: saved {saved_state[key].shape} vs current {current_state[key].shape}")
                compatible = False
        
        if compatible:
            model.gating_net.load_state_dict(saved_state)
            print("✅ Gating network weights loaded successfully")
        else:
            print("❌ Gating checkpoint incompatible with enriched features. Using random weights.")
    else:
        print("⚠️ No gating network weights found in checkpoint")

    # Set optimal α*, μ*
    with torch.no_grad():
        model.alpha.copy_(alpha_star)
        model.mu.copy_(mu_star)
    print("✅ Model configured with optimal parameters and gating ready")

    # 3) Load AURC splits: tuneV + val_lt (validation), test_lt (test)
    print("\n" + "="*60)
    print("COMPREHENSIVE AURC EVALUATION")
    print("="*60)
    (tunev_logits, tunev_labels, _), (val_lt_logits, val_lt_labels, _), (test_logits, test_labels, _) = load_aurc_splits_data()
    
    print(f"📊 Validation set (tuneV + val_lt): {len(tunev_labels) + len(val_lt_labels)} samples")
    print(f"📊 Test set (test_lt): {len(test_labels)} samples")
    print("✅ Correct methodology: Optimize thresholds on (tuneV + val_lt), evaluate on test_lt")

    # 4) Compute mixture posteriors and GSE margins/predictions
    print("\n🔮 Computing mixture posteriors for all splits...")
    tunev_eta_mix = get_mixture_posteriors(model, tunev_logits)
    val_lt_eta_mix = get_mixture_posteriors(model, val_lt_logits)
    test_eta_mix = get_mixture_posteriors(model, test_logits)

    class_to_group_cpu = class_to_group.cpu()
    alpha_star_cpu = alpha_star.cpu()
    mu_star_cpu = mu_star.cpu()

    # Compute margins for all splits
    gse_margins_tunev = compute_margin(tunev_eta_mix, alpha_star_cpu, mu_star_cpu, 0.0, class_to_group_cpu)
    gse_margins_val_lt = compute_margin(val_lt_eta_mix, alpha_star_cpu, mu_star_cpu, 0.0, class_to_group_cpu)
    gse_margins_test = compute_margin(test_eta_mix, alpha_star_cpu, mu_star_cpu, 0.0, class_to_group_cpu)

    # Compute predictions for all splits
    preds_tunev = (alpha_star_cpu[class_to_group_cpu] * tunev_eta_mix).argmax(dim=1)
    preds_val_lt = (alpha_star_cpu[class_to_group_cpu] * val_lt_eta_mix).argmax(dim=1)
    preds_test = (alpha_star_cpu[class_to_group_cpu] * test_eta_mix).argmax(dim=1)
    
    # Combine tuneV + val_lt as validation set for threshold optimization
    gse_margins_val_combined = torch.cat([gse_margins_tunev, gse_margins_val_lt])
    preds_val_combined = torch.cat([preds_tunev, preds_val_lt])
    labels_val_combined = torch.cat([tunev_labels, val_lt_labels])
    
    print(f"✅ Combined validation set: {len(labels_val_combined)} samples")

    # 5) Sweep cost values and compute AURC
    cost_values = CONFIG['aurc_eval']['cost_values']
    metrics = CONFIG['aurc_eval']['metrics']
    print(f"\n🎯 Cost grid: {len(cost_values)} values from {cost_values[0]:.1f} to {cost_values[-1]:.1f}")

    aurc_results = {}
    all_rc_points = {}
    for metric in metrics:
        print(f"\n🔄 Processing {metric} metric {'(REWEIGHTED)' if class_weights else ''}...")
        print(f"   • Optimizing thresholds on validation (tunev + val): {len(labels_val_combined)} samples")
        print(f"   • Evaluating on test: {len(test_labels)} samples")
        
        rc_points = sweep_cost_values_aurc(
            gse_margins_val_combined, preds_val_combined, labels_val_combined,  # Validation
            gse_margins_test, preds_test, test_labels,                          # Test
            class_to_group_cpu, num_groups, cost_values, class_weights, metric  # Pass class_weights
        )
        aurc_full = compute_aurc_from_points(rc_points, coverage_range='full')
        aurc_02_10 = compute_aurc_from_points(rc_points, coverage_range='0.2-1.0')
        aurc_results[metric] = aurc_full
        aurc_results[f'{metric}_02_10'] = aurc_02_10
        all_rc_points[metric] = rc_points
        print(f"   ✅ {metric.upper()} AURC (0-1):     {aurc_full:.6f}")
        print(f"   ✅ {metric.upper()} AURC (0.2-1):   {aurc_02_10:.6f}")

    # 6) Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    aurc_rows = []
    for metric, rc_points in all_rc_points.items():
        for cost_c, coverage, risk in rc_points:
            aurc_rows.append({'metric': metric, 'cost': cost_c, 'coverage': coverage, 'risk': risk})
    pd.DataFrame(aurc_rows).to_csv(output_dir / 'aurc_detailed_results.csv', index=False)
    with open(output_dir / 'aurc_summary.json', 'w') as f:
        json.dump(aurc_results, f, indent=4)
    plot_aurc_curves(all_rc_points, aurc_results, output_dir / 'aurc_curves.png')

    # 7) Final summary
    print("\n" + "="*60)
    print("FINAL AURC RESULTS (REWEIGHTED FOR LONG-TAIL)" if class_weights else "FINAL AURC RESULTS")
    print("="*60)
    print("\n📊 AURC (Full Range 0-1):")
    for metric in metrics:
        print(f"   • {metric.upper():>12} AURC: {aurc_results[metric]:.6f}")
    print("\n📊 AURC (Practical Range 0.2-1):")
    for metric in metrics:
        print(f"   • {metric.upper():>12} AURC: {aurc_results.get(f'{metric}_02_10', float('nan')):.6f}")
    print("="*60)
    if class_weights:
        print("✅ Metrics reweighted by train class distribution (proper long-tail evaluation)")
    print("📝 Lower AURC is better (less area under risk-coverage curve)")
    print("🎯 Methodology: Optimize thresholds on (tunev + val), evaluate on test")

if __name__ == '__main__':
    main()