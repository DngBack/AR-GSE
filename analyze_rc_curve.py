"""
Analyze group errors diversity in RC curve data.
"""
import json
import numpy as np
from pathlib import Path

results_file = Path('results/map_cost_sweep/cifar100_lt_if100/cost_sweep_balanced.json')

if not results_file.exists():
    print(f"❌ File not found: {results_file}")
    exit(1)

with open(results_file, 'r') as f:
    data = json.load(f)

# Get cost=0.0 result
cost0 = data['results_per_cost'][0]
rc = cost0['rc_curve']

rejection_rates = np.array(rc['rejection_rates'])
errors = np.array(rc['selective_errors'])

print("="*70)
print("RC CURVE ANALYSIS")
print("="*70)

print(f"\nTotal points: {len(rejection_rates)}")
print(f"Rejection rate range: [{rejection_rates.min():.3f}, {rejection_rates.max():.3f}]")
print(f"Error range: [{errors.min():.3f}, {errors.max():.3f}]")

# Check for unique values
unique_rejections = len(np.unique(rejection_rates))
unique_errors = len(np.unique(errors))

print(f"\nUnique rejection rates: {unique_rejections}")
print(f"Unique errors: {unique_errors}")

# Sample points at different rejection rates
print("\nSample points:")
print(f"{'Rejection':<12} {'Error':<12}")
print("-"*25)

indices = [0, len(rejection_rates)//4, len(rejection_rates)//2, 3*len(rejection_rates)//4, -1]
for i in indices:
    print(f"{rejection_rates[i]:<12.4f} {errors[i]:<12.4f}")

print("\n" + "="*70)
print("PROBLEM DIAGNOSIS")
print("="*70)

if rejection_rates.min() > 0.1:
    print("❌ RC curve missing low rejection region (0.0-0.1)")
    print("   Reason: gamma or threshold range doesn't cover full spectrum")
    print("   Fix: Use gamma=0.0 and threshold_grid from 0.0 to 1.0")

if unique_errors < 50:
    print(f"❌ RC curve has too few unique error values ({unique_errors})")
    print("   Reason: Not enough variation in threshold sweep")
    print("   Fix: Increase threshold_grid resolution or adjust gamma")

print("\n" + "="*70)
