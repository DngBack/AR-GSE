import pandas as pd
import numpy as np

df = pd.read_csv('results_worst_eg_improved/cifar100_lt_if100/aurc_detailed_results.csv')
worst_df = df[df['metric'] == 'worst'].copy()

print("WORST METRIC ANALYSIS")
print("="*60)
print(f"\nTotal points: {len(worst_df)}")
print(f"Unique coverages: {worst_df['coverage'].nunique()}")
print(f"\nAll unique (coverage, risk) pairs:")
unique_pairs = worst_df[['coverage', 'risk']].drop_duplicates().sort_values('coverage')
print(unique_pairs.to_string(index=False))

print("\n" + "="*60)
print("AURC COMPUTATION FOR 0.2-1.0 RANGE:")
print("="*60)

# Simulate the computation
rc_points = list(zip(worst_df['cost'], worst_df['coverage'], worst_df['risk']))
rc_points = sorted(rc_points, key=lambda x: x[1])

coverages = [p[1] for p in rc_points]
risks = [p[2] for p in rc_points]

print(f"\nOriginal: {len(coverages)} points")
print(f"Coverage range: [{min(coverages):.4f}, {max(coverages):.4f}]")

# Filter for 0.2-1.0
filtered = [(c, r) for c, r in zip(coverages, risks) if c >= 0.2]
print(f"\nAfter filtering >= 0.2: {len(filtered)} points")

if filtered:
    coverages_filt, risks_filt = zip(*filtered)
    coverages_filt = list(coverages_filt)
    risks_filt = list(risks_filt)
    
    print(f"Filtered coverage range: [{min(coverages_filt):.4f}, {max(coverages_filt):.4f}]")
    print(f"Filtered risk range: [{min(risks_filt):.4f}, {max(risks_filt):.4f}]")
    
    # Add endpoint at 0.2 if needed
    if coverages_filt[0] > 0.2:
        print(f"\nAdding endpoint at coverage=0.2")
        print(f"  First point after filter: cov={coverages_filt[0]:.4f}, risk={risks_filt[0]:.4f}")
        
        # Check if we can interpolate
        if len(coverages_filt) >= 2:
            c0, r0 = coverages_filt[0], risks_filt[0]
            c1, r1 = coverages_filt[1], risks_filt[1]
            print(f"  Second point: cov={c1:.4f}, risk={r1:.4f}")
            
            if c1 > c0:
                risk_at_02 = r0 + (r1 - r0) * (0.2 - c0) / (c1 - c0)
                print(f"  Interpolated risk at 0.2: {risk_at_02:.4f}")
            else:
                risk_at_02 = r0
                print(f"  Using first risk (no interpolation): {risk_at_02:.4f}")
        else:
            risk_at_02 = risks_filt[0]
            print(f"  Only one point, using its risk: {risk_at_02:.4f}")
    
    # Compute AURC
    aurc = np.trapz(risks_filt, coverages_filt)
    print(f"\nAURC (without endpoints): {aurc:.6f}")
    
    # With proper endpoints
    if coverages_filt[0] > 0.2:
        if len(coverages_filt) >= 2:
            c0, r0 = coverages_filt[0], risks_filt[0]
            c1, r1 = coverages_filt[1], risks_filt[1]
            if c1 > c0:
                risk_at_02 = r0 + (r1 - r0) * (0.2 - c0) / (c1 - c0)
            else:
                risk_at_02 = r0
        else:
            risk_at_02 = risks_filt[0]
        coverages_final = [0.2] + coverages_filt
        risks_final = [risk_at_02] + risks_filt
    else:
        coverages_final = coverages_filt
        risks_final = risks_filt
    
    if coverages_final[-1] < 1.0:
        coverages_final = coverages_final + [1.0]
        risks_final = risks_final + [risks_final[-1]]
    
    aurc_final = np.trapz(risks_final, coverages_final)
    print(f"AURC (with endpoints 0.2 and 1.0): {aurc_final:.6f}")
    
    print(f"\nFinal points for integration:")
    for i, (c, r) in enumerate(zip(coverages_final, risks_final)):
        print(f"  {i+1}. coverage={c:.4f}, risk={r:.4f}")
