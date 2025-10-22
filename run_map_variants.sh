#!/bin/bash
# Train MAP with different objectives

set -e  # Exit on error

echo "=========================================="
echo "🚀 MAP TRAINING - MULTIPLE OBJECTIVES"
echo "=========================================="
echo ""

# 1. Balanced objective
echo "📊 Training MAP with BALANCED objective..."
echo "------------------------------------------"
PYTHONPATH=$PWD:$PYTHONPATH python3 train_map_simple.py --objective balanced
echo "✅ Balanced MAP trained!"
echo ""

# 2. Worst-group objective (without EG-outer)
echo "📊 Training MAP with WORST-GROUP objective (no EG-outer)..."
echo "------------------------------------------"
PYTHONPATH=$PWD:$PYTHONPATH python3 train_map_simple.py --objective worst
echo "✅ Worst-group MAP trained!"
echo ""

# 3. Worst-group objective (with EG-outer)
echo "📊 Training MAP with WORST-GROUP objective (with EG-outer)..."
echo "------------------------------------------"
PYTHONPATH=$PWD:$PYTHONPATH python3 train_map_simple.py --objective worst --eg_outer
echo "✅ Worst-group MAP with EG-outer trained!"
echo ""

echo "=========================================="
echo "🎉 ALL VARIANTS TRAINED!"
echo "=========================================="
echo ""
echo "Compare results:"
echo "  - Balanced:           results/map_simple/cifar100_lt_if100/balanced/"
echo "  - Worst-group:        results/map_simple/cifar100_lt_if100/worst/"
echo "  - Worst-group + EG:   results/map_simple/cifar100_lt_if100/worst_eg/"
