#!/usr/bin/env python3
"""
Comprehensive test script for improved EG-outer worst-group optimization.
This script incorporates all the optimizations:
- Anti-collapse β with floor and momentum
- Per-group thresholds on correct predictions only
- Blended alpha updates
- Adaptive lambda grid expansion
"""

import sys
sys.path.append('.')

from src.train.gse_balanced_plugin import main, CONFIG

if __name__ == '__main__':
    print("🚀 GSE Worst-Group EG-Outer with ALL Improvements")
    print("=" * 60)
    print("Configuration:")
    print(f"  Splits dir: {CONFIG['dataset']['splits_dir']}")
    print(f"  Logits dir: {CONFIG['experts']['logits_dir']}")
    print()
    print("Improvements applied:")
    print("✅ Anti-collapse β (floor=0.05, momentum=0.25)")
    print("✅ Reduced EG step size (xi=0.2)")
    print("✅ Error centering before EG update")
    print("✅ Early stopping with patience")
    print("✅ Per-group thresholds on correct predictions only")
    print("✅ More aggressive tail coverage (0.42 vs 0.58)")
    print("✅ Blended alpha updates (joint + conditional)")
    print("✅ Adaptive lambda grid expansion")
    print("✅ Reweighted metrics for balanced data")
    print("=" * 60)
    
    # Apply all improvements to CONFIG
    CONFIG['plugin_params'].update({
        'objective': 'worst',
        'use_eg_outer': True,
        'eg_outer_T': 40,           # More iterations for convergence
        'eg_outer_xi': 0.15,        # Reduced step size for stability  
        'use_conditional_alpha': True,  # Enable blended updates
        'M': 12,                    # Inner iterations (increased)
        'alpha_steps': 5,           # Alpha fixed-point steps (increased)
        'gamma': 0.20,              # EMA factor (reduced for stability)
    })
    
    # Update output directory for improved results
    CONFIG['output']['checkpoints_dir'] = './checkpoints/argse_worst_eg_improved/'
    
    print("\nConfiguration:")
    print(f"  EG outer iterations: {CONFIG['plugin_params']['eg_outer_T']}")
    print(f"  EG step size: {CONFIG['plugin_params']['eg_outer_xi']}")
    print(f"  Inner iterations: {CONFIG['plugin_params']['M']}")
    print(f"  Alpha method: {'blended' if CONFIG['plugin_params']['use_conditional_alpha'] else 'joint'}")
    print(f"  Output: {CONFIG['output']['checkpoints_dir']}")
    
    print("\nStarting training...")
    main()
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print("Next step: Run evaluation with:")
    print("  python -m src.train.eval_gse_plugin")
    print("(Update CONFIG['plugin_checkpoint'] to point to new checkpoint)")
    print("=" * 60)