import pandas as pd
import re
from scipy.stats import spearmanr
import argparse
import os
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot Spearman correlation between LR performance ranks and final ranks across iterations for tulu3 models.')
parser.add_argument('--csv_path', type=str, default='results/vs_tulu_llama3.1-8b/hard_prompt_leaderboard_all.csv', help='Path to the CSV file')
parser.add_argument('--save_csv', action='store_true', help='Save results to a .csv file')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results .csv file (default: same folder as CSV)')
args = parser.parse_args()

# Set default save path if not provided
if args.save_path is None and args.save_csv:
    args.save_path = os.path.join(os.path.dirname(args.csv_path), 'rank_correlation_results.csv')

# Read the CSV file
df = pd.read_csv(args.csv_path)

# Filter models starting with 'tulu3-'
df = df[df['Model'].str.startswith('tulu3-')]

# Function to parse rank, LR, and step from model name
def parse_model(model):
    # Extract rank
    rank_match = re.search(r'rank(\d+)', model)
    rank = int(rank_match.group(1)) if rank_match else None
    
    # Extract LR 
    LR_match = re.search(r'alpha(\w+)', model)
    if LR_match:
        LR_str = LR_match.group(1)
        # Convert to float, e.g., '5e5' -> 5e-5
        if 'e' in LR_str:
            parts = LR_str.split('e')
            LR = float(parts[0]) * (10 ** (-int(parts[1])))
        else:
            LR = float(LR_str)
    else:
        LR = None
    
    # Extract step
    step_match = re.search(r'step(\d+)', model)
    step = int(step_match.group(1)) if step_match else None
    
    return rank, LR, step

# Apply parsing
df['rank'], df['lr'], df['step'] = zip(*df['Model'].apply(parse_model))

# Drop rows with missing rank, lr, or step
df = df.dropna(subset=['rank', 'lr', 'step'])

# Convert Scores to float
df['Scores (%)'] = df['Scores (%)'].astype(float)

# Compute Spearman correlations per rank per step (ranking LRs by performance at each step vs final)
correlations = {64: {}, 256: {}, 1024: {}}
for rank in [64, 256, 1024]:
    df_rank = df[df['rank'] == rank]
    if df_rank.empty:
        continue
    final_step = df_rank['step'].max()
    # Get final ranks: sort LRs by win rate descending at final step
    final_df = df_rank[df_rank['step'] == final_step].sort_values('Scores (%)', ascending=False)
    final_df['final_rank'] = range(1, len(final_df) + 1)
    lr_to_final_rank = dict(zip(final_df['lr'], final_df['final_rank']))
    
    for step in sorted(df_rank['step'].unique()):
        if step == final_step:
            continue
        sub_df = df_rank[df_rank['step'] == step]
        if len(sub_df) < 2:
            continue  # Need at least 2 for correlation
        # Rank LRs by win rate descending at this step
        sub_df = sub_df.sort_values('Scores (%)', ascending=False).copy()
        sub_df['step_rank'] = range(1, len(sub_df) + 1)
        # Get corresponding final ranks
        final_ranks = [lr_to_final_rank[lr] for lr in sub_df['lr']]
        # Compute Spearman correlation between step ranks and final ranks
        corr, _ = spearmanr(sub_df['step_rank'], final_ranks)
        correlations[rank][step] = corr

# Plot the correlations
plt.figure(figsize=(10, 6))
for rank in [64, 256, 1024]:
    steps = list(correlations[rank].keys())
    corrs = list(correlations[rank].values())
    if steps:
        plt.plot(steps, corrs, marker='o', label=f'Rank {rank}')
plt.xlabel('Training Iterations')
plt.ylabel('Spearman Correlation between Step Ranks and Final Ranks')
plt.title('Spearman Correlation of LR Performance Ranks with Final Ranks Across Iterations by Rank')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
save_plot_path = os.path.join(os.path.dirname(args.csv_path), 'lr_rank_correlation_plot.png')
plt.savefig(save_plot_path, bbox_inches='tight')
print(f"Plot saved to {save_plot_path}")

# Show plot
plt.show()

# Optional: Save results if requested
if args.save_csv:
    results_df = pd.DataFrame([
        {'Rank': rank, 'Step': step, 'Spearman_Correlation': corr}
        for rank in [64, 256, 1024]
        for step, corr in correlations[rank].items()
    ])
    results_df.to_csv(args.save_path, index=False)
    print(f"Results saved to {args.save_path}")
