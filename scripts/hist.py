import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the data
print("Loading data...")
data = []
# Change 'style_data.jsonl' to your actual filename
with open('/root/reliquary-miner/results/miner_analysis.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
df = pd.DataFrame(data)

# Ensure status is treated as a string/category
df['status'] = df['status'].astype(str)

# Set seaborn theme
sns.set_theme(style="whitegrid")

# ==========================================
# Plot 1: Overlapping Histograms (Layered)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Diff by Status
sns.histplot(data=df, x='diff', hue='status', ax=axes[0], 
             multiple='layer', alpha=0.5, palette='tab10', common_norm=False)
axes[0].set_title('Distribution of Diff by Status')
axes[0].set_xlabel('Diff')

# Perplexity by Status
sns.histplot(data=df, x='perplexity', hue='status', ax=axes[1], 
             multiple='layer', alpha=0.5, palette='tab10', common_norm=False)
axes[1].set_title('Distribution of Perplexity by Status')
axes[1].set_xlabel('Perplexity')

plt.tight_layout()
plt.savefig('1_histograms_overlapping_by_status.png', dpi=300)
# plt.show()

# ==========================================
# Plot 2: Faceted Histograms (Side-by-Side)
# ==========================================
# This creates a separate subplot for each unique 'status'

print("Generating faceted plots...")

# For Diff
g_diff = sns.displot(data=df, x='diff', col='status', hue='status', 
                     kind='hist', bins=30, height=4, aspect=1.2, 
                     palette='tab10', facet_kws={'sharey': False, 'sharex': False})
g_diff.fig.suptitle('Distribution of Diff by Status (Faceted)', y=1.05)
plt.savefig('2_histograms_faceted_diff_by_status.png', dpi=300, bbox_inches='tight')

# For Perplexity
g_perp = sns.displot(data=df, x='perplexity', col='status', hue='status', 
                     kind='hist', bins=30, height=4, aspect=1.2, 
                     palette='tab10', facet_kws={'sharey': False, 'sharex': False})
g_perp.fig.suptitle('Distribution of Perplexity by Status (Faceted)', y=1.05)
plt.savefig('3_histograms_faceted_perplexity_by_status.png', dpi=300, bbox_inches='tight')

# plt.show()
print("✅ Done! Check your folder for the saved PNGs.")

# ==========================================
# 1. Load and Process Data
# ==========================================
print("Loading data...")
data = []
# Change 'style_data.jsonl' to your actual filename
with open('/root/reliquary-miner/results/miner_analysis.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Extract scalars and aggregate vectors into a clean DataFrame
processed_data = []
for item in data:
    row = {
        'diff': item.get('diff'),
        'perplexity': item.get('perplexity'),
    }
    
    # Aggregate vectors (using mean, but you can use np.max or np.sum if preferred)
    if item.get('completion_length_vector'):
        row['avg_completion_length'] = np.mean(item['completion_length_vector'])
        
    if item.get('pstop_vector'):
        row['avg_pstop'] = np.mean(item['pstop_vector'])
        
    processed_data.append(row)

df = pd.DataFrame(processed_data).dropna()
print(f"Loaded {len(df)} valid records.")

sns.set_theme(style="whitegrid")

# ==========================================
# 2. Plot 2D Histograms (Bivariate Heatmaps)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('2D Histograms: Relationships between Metrics', fontsize=18, y=0.98)

# Define the pairs and their colors
plots = [
    {'x': 'avg_completion_length', 'y': 'diff', 'ax': axes[0, 0], 'cmap': 'Blues', 'title': 'Diff vs Completion Length'},
    {'x': 'avg_completion_length', 'y': 'perplexity', 'ax': axes[0, 1], 'cmap': 'Greens', 'title': 'Perplexity vs Completion Length'},
    {'x': 'avg_pstop', 'y': 'diff', 'ax': axes[1, 0], 'cmap': 'Oranges', 'title': 'Diff vs P-Stop'},
    {'x': 'avg_pstop', 'y': 'perplexity', 'ax': axes[1, 1], 'cmap': 'Purples', 'title': 'Perplexity vs P-Stop'},
]

for p in plots:
    # Create 2D histogram
    sns.histplot(
        data=df, x=p['x'], y=p['y'], ax=p['ax'], 
        bins=50, cmap=p['cmap'], cbar=True, 
        cbar_kws={'shrink': 0.8}
    )
    p['ax'].set_title(p['title'], fontsize=14)
    p['ax'].set_xlabel(p['x'].replace('_', ' ').title())
    p['ax'].set_ylabel(p['y'].replace('_', ' ').title())

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('1_2d_histograms_all_pairs.png', dpi=300, bbox_inches='tight')
# plt.show()

# ==========================================
# 3. Plot Joint Plots (Histograms on Margins)
# ==========================================
# Joint plots are often better for EDA because they show the marginal 
# 1D histograms on the top and right axes!

print("Generating Joint Plots...")
joint_pairs = [
    ('avg_completion_length', 'diff', 'Blues'),
    ('avg_completion_length', 'perplexity', 'Greens'),
    ('avg_pstop', 'diff', 'Oranges'),
    ('avg_pstop', 'perplexity', 'Purples')
]

for x_var, y_var, cmap in joint_pairs:
    # kind='hex' is usually cleaner than 'hist' for jointplots, 
    # but we use 'hist' to strictly match your request for histograms.
    g = sns.jointplot(
        data=df, x=x_var, y=y_var, 
        kind='hist', bins=40, cmap=cmap, 
        height=7, joint_kws={'edgecolor': 'white', 'linewidth': 0.5}
    )
    
    g.fig.suptitle(f'Joint Distribution: {y_var.title()} vs {x_var.replace("_", " ").title()}', y=1.03)
    
    # Save individual joint plots
    filename = f"2_jointplot_{y_var}_vs_{x_var}.png"
    g.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.close()

print("✅ Done! Check your folder for the saved images.")