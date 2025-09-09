#!/usr/bin/env python3
"""
Debug script to check learning rate parsing and filtering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def parse_model_name(model_name):
    """
    Parse model name to extract hyperparameters
    """
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
            parsed['checkpoint_step'] = 999999  # Use large number for final
        else:
            step_match = re.search(r'step(\d+)', model_name)
            if step_match:
                parsed['checkpoint_step'] = int(step_match.group(1))
    else:
        # Parse alpha learning rate
        alpha_match = re.search(r'alpha([0-9e]+)', model_name)
        if alpha_match:
            lr_str = alpha_match.group(1)
            # Convert notation like '1e5' to float
            if 'e' in lr_str:
                parsed['learning_rate'] = float(lr_str.replace('e', 'e-'))
            else:
                parsed['learning_rate'] = float(lr_str)
        
        # Extract warmup ratio (3 digits after alpha)
        warmup_match = re.search(r'alpha[0-9e]+-([0-9]+)', model_name)
        if warmup_match:
            warmup_str = warmup_match.group(1)
            # Convert 3-digit format to decimal (e.g., '010' -> 0.10)
            parsed['warmup_ratio'] = int(warmup_str) / 1000.0
        
        # Extract checkpoint step
        if 'final' in model_name:
            parsed['is_final'] = True
            parsed['checkpoint_step'] = 999999  # Use large number for final
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
            continue  # Skip the combined file for now
            
        df = pd.read_csv(csv_file)
        
        # Parse model names
        parsed_data = []
        for _, row in df.iterrows():
            # Skip the baseline model in individual files
            if row['Model'] == 'llama3.1-8b-instruct':
                continue
                
            parsed = parse_model_name(row['Model'])
            parsed['score'] = row['Scores (%)']
            parsed_data.append(parsed)
        
        all_data.extend(parsed_data)
    
    return pd.DataFrame(all_data)

# Load and analyze the data
print("=== DEBUGGING LEARNING RATE PARSING ===")
df = load_and_parse_data('results')

# Filter to non-final models for training curves
training_df = df[~df['is_final']].copy()
print(f"Training data models (non-final): {len(training_df)}")

# Check learning rates
print("\n=== LEARNING RATE ANALYSIS ===")
lr_counts = training_df['learning_rate'].value_counts().sort_index()
print("Learning rates found:")
for lr, count in lr_counts.items():
    if pd.notna(lr):
        print(f"  {lr:.6f} ({lr:.0e}): {count} models")

# Check for specific rank
rank = 256
print(f"\n=== RANK {rank} ANALYSIS ===")
rank_data = training_df[training_df['rank'] == rank]
print(f"Rank {rank} non-final models: {len(rank_data)}")

non_default = rank_data[~rank_data['is_default']]
print(f"Non-default models: {len(non_default)}")

print("\nGrouping by learning rate and warmup ratio:")
group_count = 0
for (lr, warmup), group in non_default.groupby(['learning_rate', 'warmup_ratio']):
    if pd.notna(lr) and pd.notna(warmup):
        print(f"  α={lr:.6f} ({lr:.0e}), w={warmup:.3f}: {len(group)} models")
        if len(group) > 1:
            print(f"    -> Would be plotted (group #{group_count + 1})")
            group_count += 1
        else:
            print(f"    -> SKIPPED (only 1 point)")
        
        # Show some sample model names
        for model_name in group['model_name'].head(2):
            print(f"      Sample: {model_name}")

print(f"\nTotal plottable groups: {group_count}")
print("Note: Only first 5 groups are plotted due to readability limit")
