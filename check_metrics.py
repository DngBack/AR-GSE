import pandas as pd

df = pd.read_csv('results_worst_eg_improved/cifar100_lt_if100/aurc_detailed_results.csv')

for metric in ['standard', 'balanced', 'worst']:
    mdf = df[df['metric'] == metric]
    print(f"\n{metric.upper()} METRIC:")
    print(f"  Coverage range: [{mdf['coverage'].min():.4f}, {mdf['coverage'].max():.4f}]")
    print(f"  Unique coverage values: {len(mdf['coverage'].unique())}")
    print(f"  Points with coverage >= 0.2: {len(mdf[mdf['coverage'] >= 0.2])}")
    print(f"  Risk range: [{mdf['risk'].min():.4f}, {mdf['risk'].max():.4f}]")
    
    # Show coverage distribution
    cov_in_02_04 = len(mdf[(mdf['coverage'] >= 0.2) & (mdf['coverage'] <= 0.4)])
    cov_in_04_06 = len(mdf[(mdf['coverage'] >= 0.4) & (mdf['coverage'] <= 0.6)])
    cov_in_06_08 = len(mdf[(mdf['coverage'] >= 0.6) & (mdf['coverage'] <= 0.8)])
    cov_in_08_10 = len(mdf[(mdf['coverage'] >= 0.8) & (mdf['coverage'] <= 1.0)])
    
    print(f"  Coverage distribution:")
    print(f"    [0.2-0.4]: {cov_in_02_04} points")
    print(f"    [0.4-0.6]: {cov_in_04_06} points")
    print(f"    [0.6-0.8]: {cov_in_06_08} points")
    print(f"    [0.8-1.0]: {cov_in_08_10} points")
