#!/usr/bin/env python3
"""
Debug script to check training curves data
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
    baseline_score = None
    
    # First, load the complete data to get baseline score
    all_csv = Path(csv_dir) / "hard_prompt_leaderboard_all.csv"
    if all_csv.exists():
        all_df = pd.read_csv(all_csv)
        baseline_row = all_df[all_df['Model'] == 'llama3.1-8b-instruct']
        if not baseline_row.empty:
            baseline_score = baseline_row.iloc[0]['Scores (%)']
            print(f"Baseline (llama3.1-8b-instruct) score: {baseline_score:.1f}%")
    
    csv_files = list(Path(csv_dir).glob("*leaderboard*.csv"))
    
    for csv_file in csv_files:
        if 'all.csv' in str(csv_file):
            continue  # Skip the combined file for now
            
        df = pd.read_csv(csv_file)
        print(f"\nProcessing {csv_file}")
        print(f"Found {len(df)} models in this file")
        
        # Extract rank category from filename
        if 'rank64' in str(csv_file):
            rank_category = 'rank64'
        elif 'rank256' in str(csv_file):
            rank_category = 'rank256'
        elif 'rank1024' in str(csv_file):
            rank_category = 'rank1024'
        elif 'default' in str(csv_file):
            rank_category = 'default'
        else:
            rank_category = 'unknown'
        
        # Parse model names
        parsed_data = []
        for _, row in df.iterrows():
            # Skip the baseline model in individual files
            if row['Model'] == 'llama3.1-8b-instruct':
                continue
                
            parsed = parse_model_name(row['Model'])
            parsed['score'] = row['Scores (%)']
            parsed['rank_category'] = rank_category
            
            # Debug info
            print(f"  Model: {row['Model']}")
            print(f"    Rank: {parsed['rank']}, Step: {parsed['checkpoint_step']}, Final: {parsed['is_final']}, Default: {parsed['is_default']}")
            
            parsed_data.append(parsed)
        
        all_data.extend(parsed_data)
    
    df = pd.DataFrame(all_data)
    df['baseline_score'] = baseline_score
    return df

# Load and analyze the data
print("=== DEBUGGING TRAINING CURVES DATA ===")
df = load_and_parse_data('results')

print(f"\n=== OVERALL STATS ===")
print(f"Total models: {len(df)}")
print(f"Models with rank: {len(df[df['rank'].notna()])}")
print(f"Models with checkpoint_step: {len(df[df['checkpoint_step'].notna()])}")
print(f"Final models: {len(df[df['is_final']])}")
print(f"Non-final models: {len(df[~df['is_final']])}")
print(f"Default models: {len(df[df['is_default']])}")
print(f"Non-default models: {len(df[~df['is_default']])}")

print(f"\n=== TRAINING DATA FOR CURVES ===")
training_df = df[~df['is_final']].copy()
print(f"Training data models (non-final): {len(training_df)}")

# Check each rank
for rank in sorted(df['rank'].unique()):
    if pd.notna(rank):
        rank_data = training_df[training_df['rank'] == rank]
        print(f"\nRank {int(rank)}:")
        print(f"  Total training models: {len(rank_data)}")
        print(f"  Default models: {len(rank_data[rank_data['is_default']])}")
        print(f"  Non-default models: {len(rank_data[~rank_data['is_default']])}")
        
        # Check steps available
        steps = sorted(rank_data['checkpoint_step'].dropna().unique())
        print(f"  Available steps: {steps}")
        
        # Show some example models
        print(f"  Sample models:")
        for _, row in rank_data.head(3).iterrows():
            print(f"    {row['model_name']} -> Step: {row['checkpoint_step']}, Score: {row['score']:.1f}%")
