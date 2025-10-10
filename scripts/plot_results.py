import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_model(model):
    # Extract lalpha value, default to 64 if not found
    lalpha_match = re.search(r'lalpha(\d+)', model)
    lalpha = int(lalpha_match.group(1)) if lalpha_match else 64
    
    # Extract step value
    step_match = re.search(r'step(\d+)', model)
    if step_match:
        step = int(step_match.group(1))
    elif 'final' in model:
        step = 55000
    else:
        step = None  # Though from the data, this shouldn't happen
    
    return step, lalpha

# Read the CSV file
csv_path = 'arena-hard-auto/results/comparing_alphas_instruct/hard_prompt_leaderboard_all.csv'
df = pd.read_csv(csv_path)

# Parse model names to extract step and lalpha
df['step'], df['lalpha'] = zip(*df['Model'].apply(parse_model))

# Define colors for different lalpha values
colors = {64: 'blue', 128: 'red', 256: 'green', 512: 'orange'}

# Plot the data
plt.figure(figsize=(10, 6))
for lalpha, group in df.groupby('lalpha'):
    group = group.sort_values('step')  # Sort by step for proper line plotting
    plt.plot(group['step'], group['Scores (%)'], 
             label=f'Alpha={lalpha}', 
             color=colors.get(lalpha, 'black'),
             marker='o')

plt.xlabel('Training Iterations (Step)')
plt.ylabel('Scores (%)')
plt.title('Performance vs Instruct Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('performance_plot_vs_instruct.png', dpi=300, bbox_inches='tight')
print("Plot saved")

# Display the plot
plt.show()