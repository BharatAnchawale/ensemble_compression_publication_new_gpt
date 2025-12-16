"""
Multi-Objective Visualization - FIXED VERSION
Create Pareto frontier plots and preference analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import sys
import os

sys.path.insert(0, 'src')
from analysis.multi_objective_14_DEC_2025 import ParetoFrontier, PreferenceSelector

# Load results
df = pd.read_csv('results/experiments/multi_objective_all_300plusfiles_complete_15_DEC_2025.csv')

def get_corpus(filename):
    fname = filename.lower()
    if any(x in fname for x in ['a.txt', 'aaa.txt', 'alphabet.txt', 'random.txt']):
        return 'Artificial'
    return 'Real'

df['corpus'] = df['filename'].apply(get_corpus)
df = df[df['corpus'] != 'Artificial'].copy()
print(f"✅ Filtered to {len(df)} rows (artificial excluded)")

print("="*80)
print("MULTI-OBJECTIVE ANALYSIS & VISUALIZATION")
print("="*80)
print(f"\nLoaded {len(df)} results")
print(f"Algorithms: {', '.join(df['algorithm'].unique())}")
print(f"Files: {df['filename'].nunique()}")

# Aggregate across all files (average performance)
df_avg = df.groupby('algorithm').agg({
    'compression_ratio': 'median',
    'speed_mbps': 'mean',
    'memory_mb': 'mean'
}).reset_index()

print("\n" + "="*80)
print("AVERAGE PERFORMANCE ACROSS ALL FILES")
print("="*80)
for _, row in df_avg.iterrows():
    print(f"{row['algorithm']:8s}: Ratio={row['compression_ratio']:.3f}x  "
          f"Speed={row['speed_mbps']:.2f} MB/s  Memory={row['memory_mb']:.1f} MB")

# Calculate Pareto frontier
pareto_indices = ParetoFrontier.calculate_pareto_frontier(df_avg)
pareto_algos = df_avg.loc[pareto_indices, 'algorithm'].tolist()

print("\n" + "="*80)
print("PARETO FRONTIER (Non-Dominated Solutions)")
print("="*80)
print(f"Algorithms on Pareto frontier: {', '.join(pareto_algos)}")
print(f"Pareto optimal solutions: {len(pareto_indices)}/{len(df_avg)}")

# Preference-based selection
print("\n" + "="*80)
print("PREFERENCE-BASED SELECTION")
print("="*80)

for pref in ['fast', 'small', 'balanced', 'memory_efficient']:
    best_algo, score, all_scores = PreferenceSelector.select_by_preference(df_avg, pref)
    print(f"\n{pref.upper()} preference:")
    print(f"  Best algorithm: {best_algo} (score: {score:.3f})")
    print(f"  Rankings:")
    for algo, s in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"    {algo:8s}: {s:.3f}")

# Create visualizations
os.makedirs('results/figures', exist_ok=True)

# Set style
sns.set_style("whitegrid")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Multi-Objective Compression Analysis: Ratio, Speed, and Memory', 
             fontsize=18, fontweight='bold', y=0.98)

# FIGURE 1: Ratio vs Speed (2D Pareto Frontier)
ax1 = plt.subplot(2, 3, 1)
colors = ['red' if i in pareto_indices else 'blue' for i in range(len(df_avg))]
sizes = [200 if i in pareto_indices else 100 for i in range(len(df_avg))]

for i, row in df_avg.iterrows():
    color = 'red' if i in pareto_indices else 'blue'
    marker = 'D' if i in pareto_indices else 'o'
    size = 200 if i in pareto_indices else 100
    label = 'Pareto Optimal' if i in pareto_indices and i == pareto_indices[0] else None
    if i not in pareto_indices and i == 0:
        label = 'Dominated'
    
    ax1.scatter(row['speed_mbps'], row['compression_ratio'], 
               c=color, s=size, marker=marker, alpha=0.7, 
               edgecolors='black', linewidths=2, label=label)
    ax1.annotate(row['algorithm'], (row['speed_mbps'], row['compression_ratio']),
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

ax1.set_xlabel('Speed (MB/s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Compression Ratio (higher = better)', fontsize=12, fontweight='bold')
ax1.set_title('Ratio vs Speed Trade-off', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(alpha=0.3)

# FIGURE 2: Ratio vs Memory (2D Pareto Frontier)
ax2 = plt.subplot(2, 3, 2)

for i, row in df_avg.iterrows():
    color = 'red' if i in pareto_indices else 'blue'
    marker = 'D' if i in pareto_indices else 'o'
    size = 200 if i in pareto_indices else 100
    
    ax2.scatter(row['memory_mb'], row['compression_ratio'], 
               c=color, s=size, marker=marker, alpha=0.7,
               edgecolors='black', linewidths=2)
    ax2.annotate(row['algorithm'], (row['memory_mb'], row['compression_ratio']),
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

ax2.set_xlabel('Memory Usage (MB, lower = better)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Compression Ratio (higher = better)', fontsize=12, fontweight='bold')
ax2.set_title('Ratio vs Memory Trade-off', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

# FIGURE 3: 3D Scatter Plot (All 3 Objectives)
ax3 = plt.subplot(2, 3, 3, projection='3d')

for i, row in df_avg.iterrows():
    color = 'red' if i in pareto_indices else 'blue'
    marker = 'D' if i in pareto_indices else 'o'
    size = 200 if i in pareto_indices else 100
    
    ax3.scatter(row['compression_ratio'], row['speed_mbps'], row['memory_mb'],
               c=color, s=size, marker=marker, alpha=0.7,
               edgecolors='black', linewidths=2)
    ax3.text(row['compression_ratio'], row['speed_mbps'], row['memory_mb'],
            row['algorithm'], fontsize=8, fontweight='bold')

ax3.set_xlabel('Compression Ratio', fontsize=10, fontweight='bold')
ax3.set_ylabel('Speed (MB/s)', fontsize=10, fontweight='bold')
ax3.set_zlabel('Memory (MB)', fontsize=10, fontweight='bold')
ax3.set_title('3D Multi-Objective Space', fontsize=13, fontweight='bold')

# FIGURE 4: Preference Comparison (Radar Chart) - FIXED
ax4 = plt.subplot(2, 3, 4, projection='polar')

# Create radar chart for normalized scores - FIXED
categories = ['Ratio', 'Speed', 'Memory']
num_vars = len(categories)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

df_norm = PreferenceSelector.normalize_objectives(df_avg)

for _, row in df_norm.iterrows():
    values = [
        row['compression_ratio_norm'], 
        row['speed_mbps_norm'], 
        row['memory_mb_norm']
    ]
    values += values[:1]  # Complete the circle
    
    ax4.plot(angles, values, 'o-', linewidth=2, 
            label=row['algorithm'], alpha=0.7, markersize=8)
    ax4.fill(angles, values, alpha=0.1)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax4.set_ylim(0, 1)
ax4.set_yticks([0.25, 0.5, 0.75, 1.0])
ax4.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
ax4.set_title('Normalized Performance Profiles', fontsize=13, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
ax4.grid(True)

# FIGURE 5: Preference Selection Bar Chart
ax5 = plt.subplot(2, 3, 5)

preferences = ['fast', 'small', 'balanced', 'memory_efficient']
pref_labels = []
pref_algos = []
pref_colors = []

color_map = {algo: plt.cm.Set3(i) for i, algo in enumerate(df_avg['algorithm'])}

for pref in preferences:
    best_algo, _, _ = PreferenceSelector.select_by_preference(df_avg, pref)
    pref_labels.append(pref.replace('_', '\n'))
    pref_algos.append(best_algo)
    pref_colors.append(color_map[best_algo])

bars = ax5.bar(range(len(pref_labels)), [1]*len(pref_labels), color=pref_colors, alpha=0.7,
              edgecolor='black', linewidth=2)
ax5.set_xticks(range(len(pref_labels)))
ax5.set_xticklabels(pref_labels, fontsize=10, fontweight='bold')
ax5.set_yticks([])
ax5.set_title('Best Algorithm by Preference', fontsize=13, fontweight='bold')

for i, (bar, algo) in enumerate(zip(bars, pref_algos)):
    ax5.text(bar.get_x() + bar.get_width()/2, 0.5, algo, 
            ha='center', va='center', fontsize=11, fontweight='bold')

# FIGURE 6: Summary Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create summary data
summary_data = []
for _, row in df_avg.iterrows():
    is_pareto = '✓' if row.name in pareto_indices else ''
    summary_data.append([
        row['algorithm'],
        f"{row['compression_ratio']:.2f}x",
        f"{row['speed_mbps']:.1f}",
        f"{row['memory_mb']:.1f}",
        is_pareto
    ])

table = ax6.table(
    cellText=summary_data,
    colLabels=['Algorithm', 'Ratio', 'Speed\n(MB/s)', 'Memory\n(MB)', 'Pareto'],
    cellLoc='center',
    loc='center',
    colWidths=[0.18, 0.15, 0.15, 0.15, 0.12]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
colors_table = sns.color_palette('Set3', len(summary_data))
for i in range(1, len(summary_data) + 1):
    for j in range(5):
        if i-1 in pareto_indices:
            table[(i, j)].set_facecolor('#ffcccc')  # Highlight Pareto
            table[(i, j)].set_alpha(0.6)
        else:
            table[(i, j)].set_facecolor(colors_table[i-1])
            table[(i, j)].set_alpha(0.3)

ax6.set_title('Performance Summary\n(Red = Pareto Optimal)', 
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('results/figures/multi_objective_analysis_14_DEC_2025_300plus_files_without_memory_post_tabel1n4_fix.png', 
           dpi=300, bbox_inches='tight')
print(f"\n✅ Main visualization saved: results/figures/multi_objective_analysis_14_DEC_2025_300plus_files_without_memory_post_tabel1n4_fix.png")

# Create individual publication-quality figures
# Individual Figure 1: Ratio vs Speed
fig1, ax = plt.subplots(figsize=(10, 8))
for i, row in df_avg.iterrows():
    color = 'red' if i in pareto_indices else 'blue'
    marker = 'D' if i in pareto_indices else 'o'
    size = 300 if i in pareto_indices else 150
    
    ax.scatter(row['speed_mbps'], row['compression_ratio'], 
              c=color, s=size, marker=marker, alpha=0.7,
              edgecolors='black', linewidths=2.5)
    ax.annotate(row['algorithm'], (row['speed_mbps'], row['compression_ratio']),
               xytext=(8, 8), textcoords='offset points', fontsize=12, fontweight='bold')

ax.set_xlabel('Compression Speed (MB/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Compression Ratio', fontsize=14, fontweight='bold')
ax.set_title('Multi-Objective Trade-off: Ratio vs Speed\n(Red = Pareto Optimal)', 
            fontsize=15, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')

# Manual legend
from matplotlib.patches import Patch
legend_elements = [
    plt.Line2D([0], [0], marker='D', color='w', label='Pareto Optimal',
               markerfacecolor='red', markersize=12, markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='o', color='w', label='Dominated',
               markerfacecolor='blue', markersize=10, markeredgecolor='black', markeredgewidth=2)
]
ax.legend(handles=legend_elements, loc='best', fontsize=12, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('results/figures/pareto_ratio_speed_14_DEC_2025_300plus_files_without_memory_post_tabel1n4_fix.png', dpi=300, bbox_inches='tight')
print(f"✅ Figure 1 saved: results/figures/pareto_ratio_speed_14_DEC_2025_300plus_files_without_memory_post_tabel1n4_fix.png")

# Individual Figure 2: Ratio vs Memory
fig2, ax = plt.subplots(figsize=(10, 8))
for i, row in df_avg.iterrows():
    color = 'red' if i in pareto_indices else 'blue'
    marker = 'D' if i in pareto_indices else 'o'
    size = 300 if i in pareto_indices else 150
    
    ax.scatter(row['memory_mb'], row['compression_ratio'], 
              c=color, s=size, marker=marker, alpha=0.7,
              edgecolors='black', linewidths=2.5)
    ax.annotate(row['algorithm'], (row['memory_mb'], row['compression_ratio']),
               xytext=(8, 8), textcoords='offset points', fontsize=12, fontweight='bold')

ax.set_xlabel('Memory Usage (MB, lower = better)', fontsize=14, fontweight='bold')
ax.set_ylabel('Compression Ratio', fontsize=14, fontweight='bold')
ax.set_title('Multi-Objective Trade-off: Ratio vs Memory\n(Red = Pareto Optimal)', 
            fontsize=15, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')
ax.legend(handles=legend_elements, loc='best', fontsize=12, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('results/figures/pareto_ratio_memory_14_DEC_2025_300plus_files_without_memory_post_tabel1n4_fix.png', dpi=300, bbox_inches='tight')
print(f"✅ Figure 2 saved: results/figures/pareto_ratio_memory_14_DEC_2025_300plus_files_without_memory_post_tabel1n4_fix.png")

# Individual Figure 3: 3D Plot
fig3 = plt.figure(figsize=(12, 10))
ax = fig3.add_subplot(111, projection='3d')

for i, row in df_avg.iterrows():
    color = 'red' if i in pareto_indices else 'blue'
    marker = 'D' if i in pareto_indices else 'o'
    size = 300 if i in pareto_indices else 150
    
    ax.scatter(row['compression_ratio'], row['speed_mbps'], row['memory_mb'],
              c=color, s=size, marker=marker, alpha=0.7,
              edgecolors='black', linewidths=2.5)
    ax.text(row['compression_ratio'], row['speed_mbps'], row['memory_mb'],
           '  ' + row['algorithm'], fontsize=10, fontweight='bold')

ax.set_xlabel('\nCompression Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('\nSpeed (MB/s)', fontsize=12, fontweight='bold')
ax.set_zlabel('\nMemory (MB)', fontsize=12, fontweight='bold')
ax.set_title('3D Multi-Objective Optimization Space\n(Red = Pareto Optimal)', 
            fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('results/figures/pareto_3d_space_14_DEC_2025_300plus_files_without_memory_post_tabel1n4_fix.png', dpi=300, bbox_inches='tight')
print(f"✅ Figure 3 saved: results/figures/pareto_3d_space_14_DEC_2025_300plus_files_without_memory_post_tabel1n4_fix.png")

print("\n" + "="*80)
print("✅ ALL VISUALIZATIONS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. multi_objective_analysis_14_DEC_2025.png (comprehensive 6-panel)")
print("  2. pareto_ratio_speed_14_DEC_2025.png (publication figure)")
print("  3. pareto_ratio_memory_14_DEC_2025.png (publication figure)")
print("  4. pareto_3d_space_14_DEC_2025.png (publication figure)")

plt.show()