#!/bin/bash
# Run MAP cost sweep for both objectives and create comparison plots

set -e

echo "========================================================================"
echo "🚀 RUNNING FULL AURC ANALYSIS"
echo "========================================================================"

# Activate environment
source ~/.bashrc
conda activate argse

# Set paths
export PYTHONPATH=$PWD:$PYTHONPATH
cd /home/duong.xuan.bach/AR-GSE

# Run balanced
echo ""
echo "========================================================================"
echo "1. Running BALANCED objective..."
echo "========================================================================"
python3 train_map_cost_sweep.py --objective balanced

# Run worst
echo ""
echo "========================================================================"
echo "2. Running WORST objective..."
echo "========================================================================"
python3 train_map_cost_sweep.py --objective worst

# Create comparison plot
echo ""
echo "========================================================================"
echo "3. Creating comparison plot..."
echo "========================================================================"
python3 plot_aurc_comparison.py

echo ""
echo "========================================================================"
echo "🎉 ALL DONE!"
echo "========================================================================"
echo ""
echo "Results saved to: results_map/cifar100_lt_if100/"
echo "  - aurc_curves_balanced.png"
echo "  - aurc_curves_worst.png"
echo "  - aurc_comparison_balanced_vs_worst.png"
