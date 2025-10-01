import pandas as pd
import re
from scipy.stats import spearmanr
import argparse
import os
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot Spearman correlation between LR and Win Rate across iterations for tulu3 models.')
parser.add_argument('--csv_path', type=str, default='results/vs_instruct_llama3.1-8b/hard_prompt_leaderboard_all.csv', help='Path to the CSV file')
parser.add_argument('--save_csv', action='store_true', help='Save results to a .csv file')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results .csv file (default: same folder as CSV)')
args = parser.parse_args()

# Set default save path if not provided
if args.save_path is None and args.save_csv:
    args.save_path = os.path.join(os.path.dirname(args.csv_path), 'correlation_results.csv')

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

# Compute Spearman correlations per rank per step
correlations = {64: {}, 256: {}, 1024: {}}
for rank in [64, 256, 1024]:
    df_rank = df[df['rank'] == rank]
    for step in sorted(df_rank['step'].unique()):
        sub_df = df_rank[df_rank['step'] == step]
        if len(sub_df) > 1:
            corr, p_value = spearmanr(sub_df['lr'], sub_df['Scores (%)'])
            correlations[rank][step] = corr

# Plot the correlations
plt.figure(figsize=(10, 6))
for rank in [64, 256, 1024]:
    steps = list(correlations[rank].keys())
    corrs = list(correlations[rank].values())
    plt.plot(steps, corrs, marker='o', label=f'Rank {rank}')
plt.xlabel('Training Iterations')
plt.ylabel('Spearman Correlation between LR and Win Rate')
plt.title('Spearman Correlation between LR and Win Rate Across Iterations by Rank')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
save_plot_path = os.path.join(os.path.dirname(args.csv_path), 'lr_winrate_correlation_plot.png')
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
