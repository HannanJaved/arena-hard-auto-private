import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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
    # Extract base (without step)
    step_match = re.search(r'-step(\d+)', model_name)
    rank_match = re.search(r'rank(\d+)', model_name)
    if step_match:
        base = model_name.replace(step_match.group(), '')
        step = int(step_match.group(1))
    else:
        base = model_name
        step = None
    rank = int(rank_match.group(1)) if rank_match else None
    return base, step, rank

# Apply parsing
df[['base', 'step', 'rank']] = df['model'].apply(lambda x: pd.Series(parse_model(x)))

# Drop rows without step
df = df.dropna(subset=['step'])
df['step'] = df['step'].astype(int)

# Now, for each rank, compute correlations
step_correlations = {64: {}, 256: {}, 1024: {}}
for rank in [64, 256, 1024]:
    df_rank = df[df['rank'] == rank]
    if not df_rank.empty:
        final_win = df_rank.groupby('base')['win_rate'].max().to_dict()
        for step in sorted(df_rank['step'].unique()):
            sub_df = df_rank[df_rank['step'] == step]
            if len(sub_df) > 5:  # Require at least 6 data points for correlation
                win_at_step = sub_df['win_rate'].tolist()
                final_wins = [final_win[row['base']] for _, row in sub_df.iterrows()]
                if len(set(final_wins)) > 1:  # Ensure variation in final wins
                    corr, _ = spearmanr(win_at_step, final_wins)
                    step_correlations[rank][step] = corr

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
