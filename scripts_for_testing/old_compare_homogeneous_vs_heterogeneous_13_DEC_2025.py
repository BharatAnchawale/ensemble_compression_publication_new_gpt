"""
Compare Ensemble Performance: Homogeneous vs Heterogeneous Data
Purpose: Show when ensemble helps vs hurts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
canterbury = pd.read_csv('results/experiments/ensemble_improved_results_13_DEC_2025_comparison.csv')
heterogeneous = pd.read_csv('results/experiments/ensemble_heterogeneous_13_DEC_2025_comparison.csv')

# Add data type label
canterbury['data_type'] = 'Homogeneous (Canterbury)'
heterogeneous['data_type'] = 'Heterogeneous (Mixed)'

# Combine
all_data = pd.concat([canterbury, heterogeneous], ignore_index=True)

print("="*80)
print("ENSEMBLE PERFORMANCE: HOMOGENEOUS vs HETEROGENEOUS DATA")
print("="*80)

# Summary statistics
print("\n📊 HOMOGENEOUS DATA (Canterbury Corpus):")
print(f"  Average improvement: {canterbury['improvement_pct'].mean():+.2f}%")
print(f"  Win rate: {len(canterbury[canterbury['improvement_pct'] > 0])}/{len(canterbury)}")
print(f"  Best case: {canterbury['improvement_pct'].max():+.2f}%")
print(f"  Worst case: {canterbury['improvement_pct'].min():+.2f}%")

print("\n📊 HETEROGENEOUS DATA (Mixed Content):")
print(f"  Average improvement: {heterogeneous['improvement_pct'].mean():+.2f}%")
print(f"  Win rate: {len(heterogeneous[heterogeneous['improvement_pct'] > 0])}/{len(heterogeneous)}")
print(f"  Best case: {heterogeneous['improvement_pct'].max():+.2f}%")
print(f"  Worst case: {heterogeneous['improvement_pct'].min():+.2f}%")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_ind(canterbury['improvement_pct'], 
                                   heterogeneous['improvement_pct'])
print(f"\n📈 Statistical Comparison:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"  ✅ Statistically significant difference (p < 0.05)")
else:
    print(f"  ⚠️  Not statistically significant")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Ensemble Compression: Homogeneous vs Heterogeneous Data Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Box plot comparison
ax1 = axes[0, 0]
sns.boxplot(data=all_data, x='data_type', y='improvement_pct', ax=ax1, palette='Set2')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Break-even')
ax1.set_title('Improvement Distribution by Data Type', fontweight='bold', fontsize=12)
ax1.set_xlabel('Data Type', fontsize=11)
ax1.set_ylabel('Improvement over Best Baseline (%)', fontsize=11)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add mean markers
for i, dtype in enumerate(['Homogeneous (Canterbury)', 'Heterogeneous (Mixed)']):
    data_subset = all_data[all_data['data_type'] == dtype]
    mean_val = data_subset['improvement_pct'].mean()
    ax1.scatter(i, mean_val, color='red', s=200, marker='D', zorder=5, 
                edgecolors='black', linewidths=2)
    ax1.text(i, mean_val + 2, f'μ={mean_val:.1f}%', ha='center', 
             fontweight='bold', fontsize=10)

# 2. Individual file comparison
ax2 = axes[0, 1]
x_pos = np.arange(len(all_data))
colors = ['red' if x < 0 else 'green' for x in all_data['improvement_pct']]
bars = ax2.bar(x_pos, all_data['improvement_pct'], color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_title('Per-File Improvement: Red=Loss, Green=Gain', fontweight='bold', fontsize=12)
ax2.set_xlabel('File Index', fontsize=11)
ax2.set_ylabel('Improvement (%)', fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# Add vertical separator between data types
separator_idx = len(canterbury)
ax2.axvline(x=separator_idx - 0.5, color='blue', linestyle='--', linewidth=2, 
            label='Data Type Boundary')
ax2.legend()

# 3. File Size vs Improvement
ax3 = axes[1, 0]
for dtype in all_data['data_type'].unique():
    data_subset = all_data[all_data['data_type'] == dtype]
    ax3.scatter(data_subset['file_size_kb'], data_subset['improvement_pct'],
                label=dtype, s=100, alpha=0.7)

ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_title('File Size vs Improvement', fontweight='bold', fontsize=12)
ax3.set_xlabel('File Size (KB)', fontsize=11)
ax3.set_ylabel('Improvement (%)', fontsize=11)
ax3.legend(loc='best', frameon=True, shadow=True)
ax3.grid(alpha=0.3)
ax3.set_xscale('log')

# 4. Summary table
ax4 = axes[1, 1]
ax4.axis('off')

summary_data = []
for dtype in ['Homogeneous (Canterbury)', 'Heterogeneous (Mixed)']:
    data_subset = all_data[all_data['data_type'] == dtype]
    wins = len(data_subset[data_subset['improvement_pct'] > 0])
    ties = len(data_subset[data_subset['improvement_pct'] == 0])
    losses = len(data_subset[data_subset['improvement_pct'] < 0])
    
    summary_data.append([
        dtype.split('(')[0].strip(),
        len(data_subset),
        f"{data_subset['improvement_pct'].mean():.2f}%",
        f"{data_subset['improvement_pct'].std():.2f}%",
        f"{wins}/{len(data_subset)}",
        f"{data_subset['improvement_pct'].max():.2f}%",
        f"{data_subset['improvement_pct'].min():.2f}%"
    ])

table = ax4.table(
    cellText=summary_data,
    colLabels=['Data Type', 'Files', 'Mean\nImpr.', 'Std\nDev', 'Win\nRate', 'Best', 'Worst'],
    cellLoc='center',
    loc='center',
    colWidths=[0.15, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
for i in range(1, 3):
    for j in range(7):
        if i == 1:
            table[(i, j)].set_facecolor('#ffcccc')  # Red tint for homogeneous
        else:
            table[(i, j)].set_facecolor('#ccffcc')  # Green tint for heterogeneous
        table[(i, j)].set_alpha(0.5)

ax4.set_title('Performance Summary Statistics', fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig('results/figures/ensemble_homogeneous_vs_heterogeneous_13_DEC_2025.png', 
            dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved: results/figures/ensemble_homogeneous_vs_heterogeneous_13_DEC_2025.png")

# Print key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

diff = heterogeneous['improvement_pct'].mean() - canterbury['improvement_pct'].mean()
print(f"\n🎯 Ensemble performs {abs(diff):.2f}% {'better' if diff > 0 else 'worse'} on heterogeneous data")

hetero_wins = len(heterogeneous[heterogeneous['improvement_pct'] > 0])
hetero_total = len(heterogeneous)
print(f"\n✅ Heterogeneous data win rate: {hetero_wins}/{hetero_total} ({hetero_wins/hetero_total*100:.1f}%)")

canterbury_wins = len(canterbury[canterbury['improvement_pct'] > 0])
canterbury_total = len(canterbury)
print(f"❌ Homogeneous data win rate: {canterbury_wins}/{canterbury_total} ({canterbury_wins/canterbury_total*100:.1f}%)")

print("\n💡 CONCLUSION:")
if heterogeneous['improvement_pct'].mean() > 0:
    print("   Ensemble compression WORKS for heterogeneous data!")
    print("   Use ensemble for: mixed-content files, concatenated formats")
    print("   Use single algorithm for: uniform text files")
else:
    print("   Need to investigate further or adjust approach")

print("="*80)

plt.show()