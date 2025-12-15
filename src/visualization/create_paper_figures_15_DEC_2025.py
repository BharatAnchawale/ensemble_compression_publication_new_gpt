"""
Create All Publication Figures for Paper
Created: December 15, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

sys.path.insert(0, 'src')
from analysis.multi_objective_14_DEC_2025 import ParetoFrontier, PreferenceSelector # Added PreferenceSelector

# Load data
df = pd.read_csv('results/experiments/multi_objective_all_300plusfiles_complete_15_DEC_2025.csv')

# Add corpus labels
def get_corpus(filename):
    if any(x in filename for x in ['alice', 'asyoulik', 'cp.html', 'fields', 
                                   'grammar', 'kennedy', 'lcet', 'plrabn', 
                                   'ptt5', 'sum', 'xargs']):
        return 'Canterbury'
    else:
        return 'Heterogeneous'

df['corpus'] = df['filename'].apply(get_corpus)

# Set style
sns.set_style("whitegrid")
sns.set_palette("Set2")

os.makedirs('results/figures', exist_ok=True)

print("="*80)
print("CREATING PUBLICATION FIGURES")
print("="*80)

# FIGURE 1: Compression Ratio Comparison (Box Plots)
print("\n📊 Figure 1: Compression Ratio Box Plots")
fig1, ax = plt.subplots(figsize=(10, 7))

algorithms = sorted(df['algorithm'].unique())
box_data = [df[df['algorithm'] == algo]['compression_ratio'].values for algo in algorithms]

bp = ax.boxplot(box_data, labels=algorithms, patch_artist=True,
                showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', 
                                              markersize=8, markeredgecolor='black'))

# Color boxes
colors = sns.color_palette("Set2", len(algorithms))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add mean values on top
means = [np.mean(data) for data in box_data]
for i, mean in enumerate(means):
    ax.text(i+1, mean, f'{mean:.2f}x', ha='center', va='bottom', 
            fontweight='bold', fontsize=10)

ax.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
ax.set_ylabel('Compression Ratio', fontsize=13, fontweight='bold')
ax.set_title('Compression Ratio Distribution Across All Files\n(Red Diamond = Mean)', 
            fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/figures/figure1_compression_ratio_boxplots_300plus_files.png', dpi=300, bbox_inches='tight')
print("✅ Saved: figure1_compression_ratio_boxplots_300plus_files.png")

# --- START OF MODIFICATION ---

# Calculate Average Performance and Pareto Frontier (Required for Figure 2 and 4)
df_avg = df.groupby('algorithm').agg({
    'compression_ratio': 'mean',
    'speed_mbps': 'mean',
    'memory_mb': 'mean'
}).reset_index()

pareto_indices = ParetoFrontier.calculate_pareto_frontier(df_avg)
pareto_algos = df_avg.loc[pareto_indices, 'algorithm'].tolist()

# FIGURE 2: Pareto Frontier (Ratio vs Speed)
print("\n📊 Figure 2: Pareto Frontier (Ratio vs Speed) - REGENERATED")
fig2, ax = plt.subplots(figsize=(10, 8))

for i, row in df_avg.iterrows():
    # Highlight Pareto Optimal points in red
    color = 'red' if i in pareto_indices else 'blue'
    marker = 'D' if i in pareto_indices else 'o'
    size = 300 if i in pareto_indices else 150
    
    ax.scatter(row['speed_mbps'], row['compression_ratio'], 
              c=color, s=size, marker=marker, alpha=0.7,
              edgecolors='black', linewidths=2.5)
    ax.annotate(row['algorithm'], (row['speed_mbps'], row['compression_ratio']),
               xytext=(8, 8), textcoords='offset points', fontsize=12, fontweight='bold')

ax.set_xlabel('Compression Speed (MB/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Compression Ratio (higher = better)', fontsize=14, fontweight='bold')
ax.set_title('Multi-Objective Trade-off: Ratio vs Speed\n(Red = Pareto Optimal)', 
            fontsize=15, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')

# Manual legend
legend_elements = [
    plt.Line2D([0], [0], marker='D', color='w', label='Pareto Optimal',
               markerfacecolor='red', markersize=12, markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='o', color='w', label='Dominated',
               markerfacecolor='blue', markersize=10, markeredgecolor='black', markeredgewidth=2)
]
ax.legend(handles=legend_elements, loc='best', fontsize=12, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('results/figures/pareto_ratio_speed_14_DEC_2025_300plus_files.png', dpi=300, bbox_inches='tight')
print("✅ Saved: pareto_ratio_speed_14_DEC_2025_300plus_files.png (Updated with N=391 data)")

# --- END OF MODIFICATION ---


# FIGURE 3: Corpus-Specific Analysis
print("\n📊 Figure 3: Corpus-Specific Performance")
fig3, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Grouped bar chart by corpus
corpus_algo = df.groupby(['corpus', 'algorithm'])['compression_ratio'].mean().reset_index()

width = 0.35
x = np.arange(len(algorithms))

for i, corpus in enumerate(['Canterbury', 'Heterogeneous']):
    corpus_data = corpus_algo[corpus_algo['corpus'] == corpus]
    ratios = [corpus_data[corpus_data['algorithm'] == algo]['compression_ratio'].values[0] 
              if len(corpus_data[corpus_data['algorithm'] == algo]) > 0 else 0
              for algo in algorithms]
    
    offset = width * (i - 0.5)
    bars = axes[0].bar(x + offset, ratios, width, label=corpus, alpha=0.8)
    
    # Add value labels
    for bar, ratio in zip(bars, ratios):
        if ratio > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{ratio:.2f}', ha='center', va='bottom', fontsize=9)

axes[0].set_xlabel('Algorithm', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
axes[0].set_title('Performance by Corpus Type', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(algorithms)
axes[0].legend(loc='upper left', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# Right: Violin plot showing distribution
axes[1].violinplot([df[(df['corpus'] == 'Canterbury') & (df['algorithm'] == algo)]['compression_ratio'].values
                    for algo in algorithms],
                    positions=x - width/2,
                    widths=width,
                    showmeans=True,
                    showmedians=True)

axes[1].violinplot([df[(df['corpus'] == 'Heterogeneous') & (df['algorithm'] == algo)]['compression_ratio'].values
                    for algo in algorithms],
                    positions=x + width/2,
                    widths=width,
                    showmeans=True,
                    showmedians=True)

axes[1].set_xlabel('Algorithm', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
axes[1].set_title('Distribution Comparison', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(algorithms)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/figure3_corpus_analysis_300plus_files.png', dpi=300, bbox_inches='tight')
print("✅ Saved: figure3_corpus_analysis_300plus_files.png")

# FIGURE 4: Multi-Objective Preference Trade-offs
print("\n📊 Figure 4: Preference-Based Trade-offs")
fig4, axes = plt.subplots(2, 2, figsize=(14, 12))

# Top-left: Preference scores bar chart
preferences = ['fast', 'small', 'balanced', 'memory_efficient']
pref_colors = {'gzip': '#ff7f0e', 'bzip2': '#2ca02c', 'lzma': '#d62728', 
               'zstd': '#9467bd', 'brotli': '#8c564b'}

pref_winners = {}
for pref in preferences:
    best_algo, _, all_scores = PreferenceSelector.select_by_preference(df_avg, pref)
    pref_winners[pref] = (best_algo, all_scores)

# Create stacked/grouped visualization
x_pos = np.arange(len(preferences))
width = 0.15

for i, algo in enumerate(algorithms):
    scores = [pref_winners[pref][1][algo] for pref in preferences]
    axes[0, 0].bar(x_pos + i*width, scores, width, label=algo, 
                   color=pref_colors.get(algo, '#gray'), alpha=0.8)

axes[0, 0].set_xlabel('Preference Mode', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Preference Score', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Algorithm Scores by Preference', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x_pos + width * 2)
axes[0, 0].set_xticklabels([p.replace('_', '\n') for p in preferences], fontsize=10)
axes[0, 0].legend(loc='upper left', fontsize=9, ncol=2)
axes[0, 0].grid(axis='y', alpha=0.3)

# Top-right: Winner summary
pref_data = [[pref.replace('_', ' ').title(), pref_winners[pref][0]] 
             for pref in preferences]

axes[0, 1].axis('off')
table = axes[0, 1].table(cellText=pref_data,
                         colLabels=['Preference', 'Best Algorithm'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 3)

for i in range(2):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(pref_data) + 1):
    algo = pref_data[i-1][1]
    color = pref_colors.get(algo, '#gray')
    for j in range(2):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_alpha(0.3)

axes[0, 1].set_title('Preference Winners Summary', fontsize=13, fontweight='bold', pad=20)

# Bottom-left: Speed vs Ratio with preference regions
axes[1, 0].scatter(df_avg['speed_mbps'], df_avg['compression_ratio'], 
                   s=300, alpha=0.7, edgecolors='black', linewidths=2)

for _, row in df_avg.iterrows():
    axes[1, 0].annotate(row['algorithm'], 
                         (row['speed_mbps'], row['compression_ratio']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=11, fontweight='bold')

# Add preference region annotations
axes[1, 0].annotate('FAST\nPreference', xy=(14, 3.7), fontsize=12, 
                    ha='center', style='italic', 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
axes[1, 0].annotate('SMALL\nPreference', xy=(3.5, 5.0), fontsize=12, 
                    ha='center', style='italic',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

axes[1, 0].set_xlabel('Speed (MB/s)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Preference Regions in Objective Space', fontsize=13, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Bottom-right: Parallel coordinates
from pandas.plotting import parallel_coordinates

df_norm = PreferenceSelector.normalize_objectives(df_avg)
df_parallel = df_norm[['algorithm', 'compression_ratio_norm', 
                       'speed_mbps_norm', 'memory_mb_norm']].copy()
df_parallel.columns = ['algorithm', 'Ratio', 'Speed', 'Memory']

parallel_coordinates(df_parallel, 'algorithm', ax=axes[1, 1], 
                    color=list(pref_colors.values()), alpha=0.7, linewidth=2.5)

axes[1, 1].set_ylabel('Normalized Performance (0-1)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Multi-Objective Performance Profiles', fontsize=13, fontweight='bold')
axes[1, 1].legend(loc='best', fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/figure4_preference_tradeoffs_300plus_files.png', dpi=300, bbox_inches='tight')
print("✅ Saved: figure4_preference_tradeoffs_300plus_files.png")

print("\n" + "="*80)
print("✅ ALL PUBLICATION FIGURES COMPLETE!")
print("="*80)
print("\nGenerated figures:")
print("  1. figure1_compression_ratio_boxplots.png")
print("  2. pareto_ratio_speed_14_DEC_2025.png (Updated Figure 2)")
print("  3. figure3_corpus_analysis.png")
print("  4. figure4_preference_tradeoffs.png")

plt.show()