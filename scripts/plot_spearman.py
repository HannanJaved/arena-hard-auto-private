import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot Spearman correlation across training iterations for multi-fidelity BO rung sizes.')
parser.add_argument('--csv_path', type=str, default='results/vs_base_llama3.1-8b/hard_prompt_leaderboard_all.csv', help='Path to the CSV file')
args = parser.parse_args()

# Load data
df = pd.read_csv(args.csv_path)

# Rename columns
df.rename(columns={'Model': 'model', 'Scores (%)': 'win_rate'}, inplace=True)

# Function to parse model name
def parse_model(model_name):
    if '-final' in model_name:
        base = model_name.replace('-final', '')
        step = 55000
    else:
        step_match = re.search(r'-step(\d+)', model_name)
        if step_match:
            base = model_name.replace(step_match.group(), '')
            step = int(step_match.group(1))
        else:
            base = model_name
            step = None
    rank_match = re.search(r'rank(\d+)', model_name)
    rank = int(rank_match.group(1)) if rank_match else None
    return base, step, rank

# Apply parsing
df[['base', 'step', 'rank']] = df['model'].apply(lambda x: pd.Series(parse_model(x)))

# Drop rows without step
df = df.dropna(subset=['step'])
df['step'] = df['step'].astype(int)

# Now, compute Spearman correlations per rank and step (used later)
step_correlations = {64: {}, 256: {}, 1024: {}}
for rank in [64, 256, 1024]:
    df_rank = df[df['rank'] == rank]
    if not df_rank.empty:
        final_win = df_rank[df_rank['step'] == 55000].set_index('base')['win_rate'].to_dict()
        for step in sorted(df_rank['step'].unique()):
            sub_df = df_rank[df_rank['step'] == step]
            if len(sub_df) > 5:  # Require at least 6 data points for correlation
                win_at_step = sub_df['win_rate'].tolist()
                final_wins = [final_win.get(row['base'], float('nan')) for _, row in sub_df.iterrows()]
                # drop pairs where final win is missing
                pairs = [(a, b) for a, b in zip(win_at_step, final_wins) if not (isinstance(b, float) and np.isnan(b))]
                if len(pairs) <= 5:
                    continue
                win_at_step_f, final_wins_f = zip(*pairs)
                if len(set(final_wins_f)) > 1:  # Ensure variation in final wins
                    corr, _ = spearmanr(win_at_step_f, final_wins_f)
                else:
                    continue
                step_correlations[rank][step] = corr

# Parse alpha and warmup from the base name so we can group combinations
def parse_alpha_and_warmup(base_name):
    # alpha examples: alpha1e5, alpha5e5, alpha1e6
    alpha_match = re.search(r'alpha(1e5|5e5|1e6)', base_name)
    alpha = alpha_match.group(1) if alpha_match else None
    # warmup examples: -001, -005, -010 (three digits)
    warmup_match = re.search(r'-(001|005|010)(?:$|[-_])', base_name)
    warmup = warmup_match.group(1) if warmup_match else None
    return alpha, warmup


df[['alpha', 'warmup']] = df['base'].apply(lambda x: pd.Series(parse_alpha_and_warmup(x)))

# If you want to plot averaged time series per (alpha, warmup) for each rank, do that
ranks = [64, 256, 1024]
alphas = ['1e5', '5e5', '1e6']
warmups = ['001', '005', '010']

# Prepare figure with 3 subplots (one per rank)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Define colors by learning rate (alpha) and linestyles/markers by warmup
color_map = {'1e5': '#1f77b4', '5e5': '#ff7f0e', '1e6': '#2ca02c'}
linestyle_map = {'001': '-', '005': '--', '010': ':'}
marker_map = {'001': 'o', '005': 's', '010': 'D'}

for ax, rank in zip(axes, ranks):
    df_rank = df[df['rank'] == rank]
    if df_rank.empty:
        ax.set_title(f'Rank {rank} (no data)')
        continue

    # determine steps to use (sorted)
    steps = sorted(df_rank['step'].unique())

    # For plotting a clean legend we collect representatives
    legend_entries = []
    plotted_any = False

    # For each alpha/warmup combo compute mean win_rate across matching bases at each step
    for alpha in alphas:
        for warmup in warmups:
            combo_label = f'alpha{alpha}-{warmup}'
            combo_df = df_rank[(df_rank['alpha'] == alpha) & (df_rank['warmup'] == warmup)]
            if combo_df.empty:
                # skip combos with no data for this rank
                continue

            mean_by_step = combo_df.groupby('step')['win_rate'].mean().reindex(steps)
            # Convert to numpy array so matplotlib draws gaps where there are NaNs
            y = mean_by_step.values.astype(float)
            line = ax.plot(steps, y, marker=marker_map.get(warmup, 'o'), linestyle=linestyle_map.get(warmup, '-'), 
                           color=color_map.get(alpha, None), label=combo_label)
            # store one handle per combo for legend
            legend_entries.append((line[0], combo_label))
            plotted_any = True

    if not plotted_any:
        ax.set_title(f'Rank {rank} (no matching LR/WR combos)')
        continue

    ax.set_xlabel('Training Iterations')
    ax.set_title(f'Rank {rank}')
    ax.grid(True)

axes[0].set_ylabel('Win Rate (%)')

# Build combined legend: show LR colors and WR linestyles/markers
# Create separate legend for combos (could be large) placed at top
handles = [h for h, l in legend_entries]
labels = [l for h, l in legend_entries]
fig.legend(handles, labels, loc='upper right', ncol=3)
fig.suptitle('Win Rate vs Training Iterations by Rank and LR/WR Combination')
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Save plot for averaged time series
save_path = os.path.join(os.path.dirname(args.csv_path), 'lr_wr_time_series_by_rank.png')
plt.savefig(save_path, bbox_inches='tight')
print(f"Plot saved to {save_path}")

# Show plot
plt.show()

# Plot
plt.figure(figsize=(10, 6))
for rank in [64, 256, 1024]:
    steps = list(step_correlations[rank].keys())
    corrs = list(step_correlations[rank].values())
    plt.plot(steps, corrs, marker='o', label=f'Rank {rank}')
plt.xlabel('Training Iterations')
plt.ylabel('Spearman Correlation with Final Performance')
plt.title('Spearman Correlation Across Training Iterations by Rank')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
save_path = os.path.join(os.path.dirname(args.csv_path), 'spearman_rung_correlations_by_rank.png')
plt.savefig(save_path, bbox_inches='tight')
print(f"Plot saved to {save_path}")

# Show plot
plt.show()
