import pandas as pd
import re
from scipy.stats import spearmanr

# Read the CSV file
df = pd.read_csv('results/vs_base_llama3.1-8b/hard_prompt_leaderboard_all.csv')

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
for rank in [64, 256, 1024]:
    sub_df = df[df['rank'] == rank]
    if len(sub_df) > 1:
        corr, p_value = spearmanr(sub_df['lr'], sub_df['Scores (%)'])
        print(f"Rank {rank}: Spearman correlation = {corr:.3f}, p-value = {p_value:.3f}")
    else:
        print(f"Rank {rank}: Not enough data points")
