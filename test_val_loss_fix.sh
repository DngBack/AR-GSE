#!/bin/bash

echo "=================================="
echo "QUICK TEST: Val Loss Fix"
echo "=================================="
echo ""

# Test với dense routing (should have low val loss now)
echo "1. Testing DENSE routing (load-balancing OFF)..."
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/train_gating_map.py \
    --routing dense \
    --epochs 10 \
    2>&1 | grep -E "(Val Loss|Val Acc|Load-balancing)"

echo ""
echo "2. Testing TOP-K routing (load-balancing ON)..."
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/train_gating_map.py \
    --routing top_k \
    --top_k 2 \
    --epochs 10 \
    2>&1 | grep -E "(Val Loss|Val Acc|Load-balancing)"

echo ""
echo "=================================="
echo "Expected results:"
echo "- Dense: Val Loss ≈ 0.01-0.05 (close to train NLL)"
echo "- Top-K: Val Loss can be higher due to LB penalty"
echo "=================================="
