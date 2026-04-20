"""EDA — run: python notebooks/02_eda.py"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('reports', exist_ok=True)
abt = pd.read_csv('data/processed/abt.csv')

sorted_rx = abt['total_rx_24m'].sort_values(ascending=False).reset_index(drop=True)
cum_rx = sorted_rx.cumsum() / sorted_rx.sum() * 100
pct_top_20 = cum_rx.iloc[int(len(cum_rx) * 0.2)]

print("=" * 50)
print("KEY INSIGHTS")
print("=" * 50)
print(f"1. Top 20% doctors drive {pct_top_20:.1f}% of Rx (Pareto)")
print(f"2. Cardiologists avg: {abt[abt['specialty']=='Cardiologist']['total_rx_24m'].mean():.0f} Rx")
print(f"   Pediatricians avg: {abt[abt['specialty']=='Pediatrician']['total_rx_24m'].mean():.0f} Rx")
print(f"3. Tier-1 avg: {abt[abt['city_tier']=='Tier-1']['total_rx_24m'].mean():.0f} Rx")
print(f"   Tier-3 avg: {abt[abt['city_tier']=='Tier-3']['total_rx_24m'].mean():.0f} Rx")
print(f"4. Bloomers: {abt['is_bloomer'].sum()} | Decliners: {abt['is_decliner'].sum()}")
print(f"5. Avg coverage: {abt['coverage_rate'].mean():.1%}")
print(f"6. Visit-Rx correlation: {abt[['total_visits_24m','total_rx_24m']].corr().iloc[0,1]:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

abt['total_rx_24m'].hist(bins=50, ax=axes[0, 0])
axes[0, 0].set_title('Rx Distribution')
axes[0, 0].set_xlabel('Total Rx 24m')

axes[0, 1].plot(range(len(cum_rx)), cum_rx)
axes[0, 1].axhline(80, color='red', linestyle='--', label='80%')
axes[0, 1].set_title('Pareto Curve')
axes[0, 1].set_xlabel('Doctor Rank')
axes[0, 1].legend()

abt.groupby('specialty')['total_rx_24m'].mean().sort_values().plot.barh(ax=axes[1, 0])
axes[1, 0].set_title('Avg Rx by Specialty')

abt.groupby('city_tier')['total_rx_24m'].mean().plot.bar(ax=axes[1, 1])
axes[1, 1].set_title('Avg Rx by City Tier')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('reports/eda_summary.png', dpi=100, bbox_inches='tight')
print(f"\nChart saved: reports/eda_summary.png")