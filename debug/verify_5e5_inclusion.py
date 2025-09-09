#!/usr/bin/env python3
"""
Verify that 5e-5 learning rate is now included in the plots
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
        if 'final' in model_name:
            parsed['is_final'] = True
            parsed['checkpoint_step'] = 999999
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
            parsed['checkpoint_step'] = 999999
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

# Simulate the new plotting logic
print("=== VERIFYING NEW PLOTTING LOGIC ===")
df = load_and_parse_data('results')
training_df = df[~df['is_final']].copy()

rank = 256
rank_data = training_df[training_df['rank'] == rank]
non_default = rank_data[~rank_data['is_default']]

print(f"Simulating plotting for Rank {rank}:")
print(f"Non-default models: {len(non_default)}")

config_count = 0
plotted_lrs = set()

print("\n--- FIRST PASS: One config per LR (prefer warmup=0.010) ---")
for (lr, warmup), group in non_default.groupby(['learning_rate', 'warmup_ratio']):
    if pd.notna(lr) and pd.notna(warmup) and len(group) > 1:
        if lr not in plotted_lrs and warmup == 0.010:
            print(f"  ✓ α={lr:.0e}, w={warmup:.3f} - {len(group)} points")
            plotted_lrs.add(lr)
            config_count += 1

print(f"\nAfter first pass: {config_count} configs, LRs plotted: {sorted(plotted_lrs)}")

print("\n--- SECOND PASS: Fill remaining LRs ---")
for (lr, warmup), group in non_default.groupby(['learning_rate', 'warmup_ratio']):
    if pd.notna(lr) and pd.notna(warmup) and len(group) > 1:
        if lr not in plotted_lrs:
            print(f"  ✓ α={lr:.0e}, w={warmup:.3f} - {len(group)} points")
            plotted_lrs.add(lr)
            config_count += 1
            if config_count >= 6:
                break

print(f"\nAfter second pass: {config_count} configs, LRs plotted: {sorted(plotted_lrs)}")

print("\n--- THIRD PASS: Add more configs if space allows ---")
remaining_slots = 8 - config_count
added_count = 0
for (lr, warmup), group in non_default.groupby(['learning_rate', 'warmup_ratio']):
    if pd.notna(lr) and pd.notna(warmup) and len(group) > 1 and added_count < remaining_slots:
        print(f"  ✓ α={lr:.0e}, w={warmup:.3f} - {len(group)} points (additional)")
        config_count += 1
        added_count += 1

print(f"\nFinal result: {config_count} total configs plotted")
print(f"Learning rates included: {sorted(plotted_lrs)}")

# Check specifically for 5e-5
if 5e-5 in plotted_lrs:
    print(f"\n🎉 SUCCESS: Learning rate 5e-5 (0.00005) IS NOW INCLUDED!")
else:
    print(f"\n❌ ISSUE: Learning rate 5e-5 (0.00005) is still missing")
