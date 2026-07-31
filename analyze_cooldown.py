import json
import os
import matplotlib.pyplot as plt
from collections import Counter

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
JSONL_FILE = "/root/reliquary-miner/cooldown_problems.jsonl"
OUT_PNG = "cooldown_difficulty_distribution.png"

# ----------------------------------------------------------------------
# Read the JSONL file and extract difficulties
# ----------------------------------------------------------------------
def load_difficulties(file_path):
    difficulties = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            diff = data['difficulty']
            
            # If it's a list, extract the last value (or choose another method)
            if isinstance(diff, list):
                if not diff:          # empty list -> default to 0.0
                    diff = 0.0
                else:
                    diff = diff[-1]   # take the last element
            # If it's not a number (e.g., string), convert to float if possible
            try:
                diff = float(diff)
            except (TypeError, ValueError):
                diff = 0.0
            
            # Round to nearest 0.5
            diff = round(diff * 2) / 2.0
            # diff = round(diff, 1)
            difficulties.append(diff)
    return difficulties

difficulties = load_difficulties(JSONL_FILE)

if not difficulties:
    print("No difficulty data found. Exiting.")
    exit(1)

# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
print(f"Total problems with difficulty: {len(difficulties)}")
print(f"Min: {min(difficulties):.1f}")
print(f"Max: {max(difficulties):.1f}")
print(f"Mean: {sum(difficulties)/len(difficulties):.2f}")

# Count in range 7.0 – 9.5
lower, upper = 6.0, 7.5
in_range = [d for d in difficulties if lower <= d <= upper]
print(f"\nDifficulty in [{lower}, {upper}]: {len(in_range)} / {len(difficulties)} ({len(in_range)/len(difficulties)*100:.1f}%)")

# ----------------------------------------------------------------------
# Count per rounded difficulty (0.5 steps)
# ----------------------------------------------------------------------
diff_counts = Counter(difficulties)
sorted_diffs = sorted(diff_counts.items())

print("\nCounts per difficulty:")
for d, cnt in sorted_diffs:
    print(f"  {d:.1f}: {cnt}")

# ----------------------------------------------------------------------
# Bar chart
# ----------------------------------------------------------------------
difficulties_vals = [d for d, _ in sorted_diffs]
counts = [c for _, c in sorted_diffs]

plt.figure(figsize=(12, 6))
bars = plt.bar(difficulties_vals, counts, width=0.4, edgecolor='black', alpha=0.7, color='steelblue')

# Label bars with count
for bar, cnt in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             str(cnt), ha='center', va='bottom', fontsize=9)

# Threshold lines
plt.axvline(x=lower, color='red', linestyle='--', linewidth=2, label=f'Lower = {lower}')
plt.axvline(x=upper, color='red', linestyle='--', linewidth=2, label=f'Upper = {upper}')

plt.xlabel('Difficulty (rounded to 0.5)', fontsize=12)
plt.ylabel('Number of Problems', fontsize=12)
plt.title(f'Difficulty Distribution of Cooldown Problems\nTotal: {len(difficulties)}')
plt.xticks(difficulties_vals)
plt.grid(axis='y', alpha=0.3)
plt.legend()

# Annotate with in‑range count
plt.text(0.02, 0.98, f"In range: {len(in_range)} / {len(difficulties)} ({len(in_range)/len(difficulties)*100:.1f}%)",
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nPlot saved as: {OUT_PNG}")

plt.show()