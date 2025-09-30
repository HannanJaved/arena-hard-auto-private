import pandas as pd
import re
from scipy.stats import spearmanr
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compute Spearman correlation between LR and Win Rate for tulu3 models.')
parser.add_argument('--csv_path', type=str, default='results/vs_tulu_llama3.1-8b/hard_prompt_leaderboard_all.csv', help='Path to the CSV file')
parser.add_argument('--save_txt', action='store_true', help='Save results to a .txt file')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results .txt file (default: same folder as CSV)')
args = parser.parse_args()

# Set default save path if not provided
if args.save_path is None and args.save_txt:
    args.save_path = os.path.join(os.path.dirname(args.csv_path), 'correlation_results.txt')

# Read the CSV file
df = pd.read_csv(args.csv_path)

# Filter models starting with 'tulu3-'
df = df[df['Model'].str.startswith('tulu3-')]

# Function to parse rank and LR from model name
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
    
    return rank, LR

# Apply parsing
df['rank'], df['lr'] = zip(*df['Model'].apply(parse_model))

# Drop rows with missing rank or lr
df = df.dropna(subset=['rank', 'lr'])

# Convert Scores to float
df['Scores (%)'] = df['Scores (%)'].astype(float)

# Group by rank and compute Spearman correlation
results = []
for rank in [64, 256, 1024]:
    sub_df = df[df['rank'] == rank]
    if len(sub_df) > 1:
        corr, p_value = spearmanr(sub_df['lr'], sub_df['Scores (%)'])
        result = f"Rank {rank}: Spearman correlation = {corr:.3f}, p-value = {p_value:.3f}"
        print(result)
        results.append(result)
    else:
        result = f"Rank {rank}: Not enough data points"
        print(result)
        results.append(result)

# Save results to a .txt file if requested
if args.save_txt:
    with open(args.save_path, 'w') as f:
        for result in results:
            f.write(result + '\n')
    print(f"Results saved to {args.save_path}")
