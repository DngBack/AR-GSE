"""
Simplified MAP Plugin Training Script
======================================

Uses confidence-based rejection instead of complex L2R margin.

Usage:
    python train_map_simple.py --objective balanced
    python train_map_simple.py --objective worst --eg_outer
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, Optional
import torchvision

from src.models.gating_network_map import GatingNetwork, compute_uncertainty_for_map
from src.models.map_selector_simple import (
    SimpleMAPSelector,
    SimpleMAPConfig,
    SimpleGridSearchOptimizer,
    RCCurveComputer
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
        'group_boundaries': [50],
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/cifar100_lt_if100/',
    },
    'gating': {
        'checkpoint': './checkpoints/gating_map/cifar100_lt_if100/best_gating.pth',
    },
    'map': {
        # Grid ranges (simplified)
        'threshold_grid': list(np.linspace(0.1, 0.9, 17)),  # Confidence thresholds
        'gamma_grid': [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0],  # Uncertainty penalties
        
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
        'checkpoints_dir': './checkpoints/map_simple/',
        'results_dir': './results/map_simple/',
    },
    'seed': 42
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# DATA LOADING (same as before)
# ============================================================================

def load_expert_logits(expert_names, logits_dir, split_name, device='cpu'):
    """Load expert logits."""
    logits_list = []
    
    for expert_name in expert_names:
        logits_path = Path(logits_dir) / expert_name / f"{split_name}_logits.pt"
        
        if not logits_path.exists():
            raise FileNotFoundError(f"Logits not found: {logits_path}")
        
        logits_e = torch.load(logits_path, map_location=device, weights_only=False).float()
        logits_list.append(logits_e)
    
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    return logits


def load_labels(splits_dir, split_name, device='cpu'):
    """Load labels."""
    indices_file = f"{split_name}_indices.json"
    indices_path = Path(splits_dir) / indices_file
    
    if not indices_path.exists():
        raise FileNotFoundError(f"Indices not found: {indices_path}")
    
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    
    if split_name in ['gating', 'expert', 'train', 'tunev']:
        cifar_train = True
    else:
        cifar_train = False
    
    dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=cifar_train,
        download=False
    )
    
    labels = torch.tensor([dataset.targets[i] for i in indices], device=device)
    return labels


def load_class_weights(splits_dir, device='cpu'):
    """Load class weights from training distribution."""
    weights_path = Path(splits_dir) / 'class_weights.json'
    
    if not weights_path.exists():
        print("⚠️  class_weights.json not found, using uniform weights")
        return torch.ones(100, device=device) / 100
    
    with open(weights_path, 'r') as f:
        weights_data = json.load(f)
    
    if isinstance(weights_data, list):
        weights = torch.tensor(weights_data, device=device, dtype=torch.float32)
    elif isinstance(weights_data, dict):
        weights = torch.tensor([weights_data[str(i)] for i in range(100)], 
                              device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unexpected format: {type(weights_data)}")
    
    return weights


def compute_sample_weights(labels, class_weights):
    """Convert class weights to per-sample weights."""
    return class_weights[labels]


def generate_mixture_posteriors(gating, expert_logits, device='cpu'):
    """Generate mixture posteriors và uncertainty."""
    gating.eval()
    
    posteriors = torch.softmax(expert_logits, dim=-1)
    
    with torch.no_grad():
        weights, aux = gating(posteriors)
    
    mixture = gating.get_mixture_posterior(posteriors, weights)
    
    uncertainty = compute_uncertainty_for_map(
        posteriors, weights, mixture,
        coeffs={
            'a': CONFIG['map']['uncertainty_coeff_a'],
            'b': CONFIG['map']['uncertainty_coeff_b'],
            'd': CONFIG['map']['uncertainty_coeff_d']
        }
    )
    
    return {
        'mixture_posteriors': mixture,
        'uncertainties': uncertainty,
        'gating_weights': weights,
        'expert_posteriors': posteriors
    }


# ============================================================================
# EG-OUTER OPTIMIZATION
# ============================================================================

def eg_outer_optimization(
    selector: SimpleMAPSelector,
    optimizer: SimpleGridSearchOptimizer,
    val_data: Dict,
    val_labels: torch.Tensor,
    val_weights: Optional[torch.Tensor],
    num_iterations: int = 10,
    xi: float = 0.1,
    verbose: bool = True
):
    """
    EG-outer for worst-group optimization.
    
    Returns:
        (best_result, best_beta)
    """
    num_groups = selector.config.num_groups
    
    # Initialize uniform beta
    beta = torch.ones(num_groups, device=DEVICE) / num_groups
    
    best_worst_error = float('inf')
    best_result = None
    best_beta = None
    
    if verbose:
        print("\n" + "="*70)
        print("EG-OUTER OPTIMIZATION (Worst-Group)")
        print("="*70)
    
    for iter_idx in range(num_iterations):
        if verbose:
            print(f"\nIteration {iter_idx + 1}/{num_iterations}:")
            print(f"  β = {beta.cpu().numpy()}")
        
        # Grid search with current beta
        result = optimizer.search(
            selector,
            val_data['mixture_posteriors'],
            val_data['uncertainties'],
            val_labels,
            beta=beta,
            sample_weights=val_weights,
            verbose=False
        )
        
        # Compute worst-group error
        worst_error = result.worst_group_error
        
        if verbose:
            print(f"  Group errors: {result.group_errors}")
            print(f"  Worst error: {worst_error:.4f}")
        
        # Update best
        if worst_error < best_worst_error:
            best_worst_error = worst_error
            best_result = result
            best_beta = beta.clone()
            
            if verbose:
                print(f"  → New best worst-error: {worst_error:.4f}")
        
        # Update beta (multiplicative weights)
        errors_tensor = torch.tensor(result.group_errors, device=DEVICE)
        beta = beta * torch.exp(xi * errors_tensor)
        beta = beta / beta.sum()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ EG-Outer completed!")
        print(f"   Best worst-group error: {best_worst_error:.4f}")
        print(f"   Final β: {best_beta.cpu().numpy()}")
    
    return best_result, best_beta


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_map_simple(
    objective: str = 'balanced',
    use_eg_outer: bool = False,
    verbose: bool = True
):
    """Main training function."""
    print("="*70)
    print("🚀 SIMPLIFIED MAP PLUGIN TRAINING")
    print("="*70)
    print(f"Objective: {objective}")
    print(f"EG-Outer: {use_eg_outer}")
    print(f"Device: {DEVICE}")
    
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
    # 2. LOAD DATA
    # ========================================================================
    print("\n" + "="*70)
    print("2. LOADING DATA & GENERATING MIXTURE POSTERIORS")
    print("="*70)
    
    expert_names = CONFIG['experts']['names']
    logits_dir = CONFIG['experts']['logits_dir']
    splits_dir = CONFIG['dataset']['splits_dir']
    
    # Val split (for optimization)
    print("\nValidation:")
    val_logits = load_expert_logits(expert_names, logits_dir, 'val', DEVICE)
    val_labels = load_labels(splits_dir, 'val', DEVICE)
    val_data = generate_mixture_posteriors(gating, val_logits, DEVICE)
    print(f"  ✓ {val_logits.shape[0]:,} samples")
    
    # Test split
    print("\nTest (balanced):")
    test_logits = load_expert_logits(expert_names, logits_dir, 'test', DEVICE)
    test_labels = load_labels(splits_dir, 'test', DEVICE)
    test_data = generate_mixture_posteriors(gating, test_logits, DEVICE)
    print(f"  ✓ {test_logits.shape[0]:,} samples")
    
    # Load class weights for reweighting
    if CONFIG['evaluation']['use_reweighting']:
        print("\nLoading class weights for reweighting...")
        class_weights = load_class_weights(splits_dir, DEVICE)
        
        val_weights = compute_sample_weights(val_labels, class_weights)
        test_weights = compute_sample_weights(test_labels, class_weights)
        
        print(f"  ✓ Class weights range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")
    else:
        val_weights = None
        test_weights = None
    
    # ========================================================================
    # 3. INITIALIZE SELECTOR
    # ========================================================================
    print("\n" + "="*70)
    print("3. INITIALIZING SIMPLIFIED MAP SELECTOR")
    print("="*70)
    
    map_config = SimpleMAPConfig(
        num_classes=CONFIG['dataset']['num_classes'],
        num_groups=CONFIG['dataset']['num_groups'],
        group_boundaries=CONFIG['dataset']['group_boundaries'],
        threshold_grid=CONFIG['map']['threshold_grid'],
        gamma_grid=CONFIG['map']['gamma_grid'],
        objective=objective
    )
    
    selector = SimpleMAPSelector(map_config).to(DEVICE)
    optimizer = SimpleGridSearchOptimizer(map_config)
    
    print(f"✅ Selector initialized")
    print(f"   Groups: {map_config.num_groups}")
    print(f"   Threshold grid: {len(map_config.threshold_grid)} points")
    print(f"   Gamma grid: {len(map_config.gamma_grid)} points")
    print(f"   Total combinations: {len(map_config.threshold_grid) * len(map_config.gamma_grid)}")
    
    # ========================================================================
    # 4. OPTIMIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("4. OPTIMIZATION")
    print("="*70)
    
    if use_eg_outer and objective == 'worst':
        # EG-outer
        best_result, best_beta = eg_outer_optimization(
            selector, optimizer,
            val_data, val_labels, val_weights,
            num_iterations=CONFIG['map']['eg_iterations'],
            xi=CONFIG['map']['eg_xi'],
            verbose=True
        )
    else:
        # Standard grid search
        print("\nGrid search...")
        best_result = optimizer.search(
            selector,
            val_data['mixture_posteriors'],
            val_data['uncertainties'],
            val_labels,
            beta=None,
            sample_weights=val_weights,
            verbose=True
        )
    
    # Set best parameters
    selector.set_parameters(
        threshold=best_result.threshold,
        gamma=best_result.gamma
    )
    
    print(f"\n✅ Optimization completed!")
    print(f"   Best threshold: {best_result.threshold:.3f}")
    print(f"   Best γ: {best_result.gamma:.3f}")
    print(f"   Selective error: {best_result.selective_error:.4f}")
    print(f"   Coverage: {best_result.coverage:.3f}")
    
    # ========================================================================
    # 5. EVALUATION ON TEST
    # ========================================================================
    print("\n" + "="*70)
    print("5. EVALUATION ON TEST SET")
    print("="*70)
    
    # RC Curve
    print("\nComputing RC curve...")
    rc_computer = RCCurveComputer(map_config)
    
    rc_data = rc_computer.compute_rc_curve(
        selector,
        test_data['mixture_posteriors'],
        test_data['uncertainties'],
        test_labels,
        gamma=best_result.gamma,
        threshold_grid=np.array(CONFIG['evaluation']['threshold_grid']),
        sample_weights=test_weights
    )
    
    print(f"✅ RC curve computed")
    print(f"   AURC: {rc_data['aurc']:.4f}")
    print(f"   Points: {len(rc_data['rejection_rates'])}")
    
    # Operating points
    print("\nOperating points:")
    for target_rej in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        idx = np.argmin(np.abs(rc_data['rejection_rates'] - target_rej))
        actual_rej = rc_data['rejection_rates'][idx]
        error = rc_data['selective_errors'][idx]
        group_errors = rc_data['group_errors_list'][idx]
        
        print(f"  Rej≈{target_rej:.1f}: error={error:.4f}, "
              f"head={group_errors[0]:.4f}, tail={group_errors[1]:.4f}")
    
    # ========================================================================
    # 6. SAVE RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("6. SAVING RESULTS")
    print("="*70)
    
    checkpoint_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    results_dir = Path(CONFIG['output']['results_dir']) / CONFIG['dataset']['name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    save_dict = {
        'threshold': best_result.threshold,
        'gamma': best_result.gamma,
        'objective': objective,
        'use_eg_outer': use_eg_outer,
        'config': CONFIG
    }
    
    with open(checkpoint_dir / 'map_parameters.json', 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"✅ Saved parameters to: {checkpoint_dir / 'map_parameters.json'}")
    
    # Save RC curve
    rc_save_dict = {
        'rejection_rates': rc_data['rejection_rates'].tolist(),
        'selective_errors': rc_data['selective_errors'].tolist(),
        'aurc': float(rc_data['aurc']),
        'threshold_grid': rc_data['threshold_grid'].tolist(),
        'group_errors': [g.tolist() for g in rc_data['group_errors_list']]
    }
    
    with open(results_dir / 'rc_curve.json', 'w') as f:
        json.dump(rc_save_dict, f, indent=2)
    
    print(f"✅ Saved RC curve to: {results_dir / 'rc_curve.json'}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print(f"AURC: {rc_data['aurc']:.4f}")
    print(f"Best configuration:")
    print(f"  threshold = {best_result.threshold:.3f}")
    print(f"  γ = {best_result.gamma:.3f}")
    
    return selector, best_result, rc_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Simplified MAP Plugin')
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
    
    selector, result, rc_data = train_map_simple(
        objective=args.objective,
        use_eg_outer=args.eg_outer,
        verbose=True
    )
    
    print("\n🎉 Done!")


if __name__ == '__main__':
    main()
