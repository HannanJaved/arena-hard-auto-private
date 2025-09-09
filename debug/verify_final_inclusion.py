#!/usr/bin/env python3
"""
Verify that final checkpoints are now included at step 55000
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def parse_model_name(model_name):
    """Parse model name to extract hyperparameters"""
    parsed = {
        'model_name': model_name,
        'rank': None,
        'learning_rate': None,
        'warmup_ratio': None,
        'checkpoint_step': None,
        'is_final': False,
        'is_default': False
    }
    
    # Extract rank
    rank_match = re.search(r'rank(\d+)', model_name)
    if rank_match:
        parsed['rank'] = int(rank_match.group(1))
    
    # Check if it's a default configuration
    if 'default' in model_name:
        parsed['is_default'] = True
        # Extract step for default models
        if 'final' in model_name:
            parsed['is_final'] = True
            parsed['checkpoint_step'] = 55000  # Use 55000 for final
        else:
            step_match = re.search(r'step(\d+)', model_name)
            if step_match:
                parsed['checkpoint_step'] = int(step_match.group(1))
    else:
        # Parse alpha learning rate
        alpha_match = re.search(r'alpha([0-9e]+)', model_name)
        if alpha_match:
            lr_str = alpha_match.group(1)
            if 'e' in lr_str:
                parsed['learning_rate'] = float(lr_str.replace('e', 'e-'))
            else:
                parsed['learning_rate'] = float(lr_str)
        
        # Extract warmup ratio
        warmup_match = re.search(r'alpha[0-9e]+-([0-9]+)', model_name)
        if warmup_match:
            warmup_str = warmup_match.group(1)
            parsed['warmup_ratio'] = int(warmup_str) / 1000.0
        
        # Extract checkpoint step
        if 'final' in model_name:
            parsed['is_final'] = True
            parsed['checkpoint_step'] = 55000  # Use 55000 for final
        else:
            step_match = re.search(r'step(\d+)', model_name)
            if step_match:
                parsed['checkpoint_step'] = int(step_match.group(1))
    
    return parsed

def load_and_parse_data(csv_dir):
    """Load all CSV files and parse model names"""
    all_data = []
    csv_files = list(Path(csv_dir).glob("*leaderboard*.csv"))
    
    for csv_file in csv_files:
        if 'all.csv' in str(csv_file):
            continue
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            if row['Model'] == 'llama3.1-8b-instruct':
                continue
            parsed = parse_model_name(row['Model'])
            parsed['score'] = row['Scores (%)']
            all_data.append(parsed)
    
    return pd.DataFrame(all_data)

print("=== VERIFYING FINAL CHECKPOINTS INCLUSION ===")
df = load_and_parse_data('results')

print(f"Total models: {len(df)}")
print(f"Final models: {len(df[df['is_final']])}")
print(f"Non-final models: {len(df[~df['is_final']])}")

# Check final checkpoint steps
final_df = df[df['is_final']]
step_counts = final_df['checkpoint_step'].value_counts()
print(f"\nFinal checkpoint steps used:")
for step, count in step_counts.items():
    print(f"  Step {step}: {count} models")

# Check specific examples
print(f"\nSample final models:")
for _, row in final_df.head(5).iterrows():
    print(f"  {row['model_name']} -> Step: {row['checkpoint_step']}, Score: {row['score']:.1f}%")

# Check training curve data includes final checkpoints
print(f"\n=== TRAINING CURVE DATA ===")
print(f"All models in training curves: {len(df)}")

# Check steps available for one rank
rank = 256
rank_data = df[df['rank'] == rank]
steps = sorted(rank_data['checkpoint_step'].unique())
print(f"\nRank {rank} available steps: {steps}")

# Check if 55000 is included
if 55000 in steps:
    print(f"✅ SUCCESS: Step 55000 (final) is included in training curves!")
    final_count = len(rank_data[rank_data['checkpoint_step'] == 55000])
    print(f"   Number of final checkpoints at step 55000: {final_count}")
else:
    print(f"❌ ISSUE: Step 55000 (final) is missing from training curves")

# Show default vs non-default final models
print(f"\n=== FINAL MODELS BREAKDOWN ===")
final_default = final_df[final_df['is_default']]
final_nondefault = final_df[~final_df['is_default']]
print(f"Final default models: {len(final_default)}")
print(f"Final non-default models: {len(final_nondefault)}")

if len(final_default) > 0:
    print(f"\nSample final default models:")
    for _, row in final_default.head(3).iterrows():
        print(f"  {row['model_name']} -> Step: {row['checkpoint_step']}")

if len(final_nondefault) > 0:
    print(f"\nSample final non-default models:")
    for _, row in final_nondefault.head(3).iterrows():
        print(f"  {row['model_name']} -> Step: {row['checkpoint_step']}")
