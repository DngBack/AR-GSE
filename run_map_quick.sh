#!/bin/bash
# Quick MAP Pipeline (assumes experts and gating already trained)

set -e  # Exit on error

echo "=========================================="
echo "🚀 QUICK MAP PIPELINE"
echo "=========================================="
echo ""
echo "⚠️  Prerequisites:"
echo "  - Experts already trained (checkpoints/experts/)"
echo "  - Gating already trained (checkpoints/gating_map/)"
echo ""

# Step 1: Compute Expert Logits on all splits
echo "📊 STEP 1: Computing Expert Logits..."
echo "------------------------------------------"
PYTHONPATH=$PWD:$PYTHONPATH python3 recompute_logits.py
echo "✅ Logits computed!"
echo ""

# Step 2: Train MAP Plugin with Cost Sweep
echo "📊 STEP 2: Training MAP Plugin with Cost Sweep..."
echo "------------------------------------------"
echo "Sweeping costs: {0.0, 0.1, 0.5, 0.75, 0.85, 0.91, 0.95, 0.97, 0.99}"
PYTHONPATH=$PWD:$PYTHONPATH python3 train_map_cost_sweep.py --objective balanced
echo "✅ MAP Plugin trained with all costs!"
echo ""

echo "=========================================="
echo "🎉 PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - MAP checkpoint: checkpoints/map_simple/cifar100_lt_if100/"
echo "  - Evaluation results: results/map_simple/cifar100_lt_if100/"
echo ""
echo "📊 View results:"
echo "  cat results/map_simple/cifar100_lt_if100/optimization_log.json"
echo "  cat results/map_simple/cifar100_lt_if100/rc_curve.json"
