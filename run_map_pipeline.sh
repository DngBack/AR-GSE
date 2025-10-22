#!/bin/bash
# Complete MAP Pipeline with new splits (Expert 90% + Gating 10%)

set -e  # Exit on error

echo "=========================================="
echo "🚀 AR-GSE MAP PIPELINE"
echo "=========================================="
echo ""

# Step 1: Train Experts on Expert Split (90% of train)
echo "📊 STEP 1: Training Experts on Expert Split..."
echo "------------------------------------------"
PYTHONPATH=$PWD:$PYTHONPATH python3 train_experts.py
echo "✅ Experts trained!"
echo ""

# Step 2: Compute Expert Logits on all splits
echo "📊 STEP 2: Computing Expert Logits..."
echo "------------------------------------------"
PYTHONPATH=$PWD:$PYTHONPATH python3 recompute_logits.py
echo "✅ Logits computed!"
echo ""

# Step 3: Train Gating Network (Dense routing) on Gating Split (10%)
echo "📊 STEP 3: Training Gating Network..."
echo "------------------------------------------"
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/train_gating_map.py \
    --routing dense \
    --epochs 100 \
    --lr 0.001 \
    --batch-size 128
echo "✅ Gating trained!"
echo ""

# Step 4: Train & Evaluate MAP Plugin with Cost Sweep
echo "📊 STEP 4: Training MAP Plugin with Cost Sweep..."
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
echo "  - Expert checkpoints: checkpoints/experts/cifar100_lt_if100/"
echo "  - Gating checkpoint: checkpoints/gating_map/cifar100_lt_if100/"
echo "  - MAP checkpoints: checkpoints/map_cost_sweep/cifar100_lt_if100/"
echo "  - Results & plots: results/map_cost_sweep/cifar100_lt_if100/"
echo ""
echo "📊 View results:"
echo "  cat results/map_cost_sweep/cifar100_lt_if100/cost_sweep_balanced.json"
echo "  open results/map_cost_sweep/cifar100_lt_if100/rc_curves_balanced.png"
