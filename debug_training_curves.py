#!/usr/bin/env python3

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

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
        parsed['learning_rate'] = 2e-5  # Default LR = 2e-5
        parsed['warmup_ratio'] = 0.03   # Default WR = 0.03
        if 'final' in model_name:
            parsed['is_final'] = True
            parsed['checkpoint_step'] = 55000
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
            parsed['checkpoint_step'] = 55000
        else:
            step_match = re.search(r'step(\d+)', model_name)
            if step_match:
                parsed['checkpoint_step'] = int(step_match.group(1))
    
    return parsed

def load_and_parse_data(csv_dir):
    """Load all CSV files and parse model names"""
    all_data = []
    baseline_score = 50.0  # Arena Hard baseline
    
    csv_files = list(Path(csv_dir).glob("*leaderboard*.csv"))
    
    for csv_file in csv_files:
        if 'all.csv' in str(csv_file):
            continue
            
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            parsed = parse_model_name(row['Model'])
            parsed['score'] = row['Scores (%)']
            parsed['improvement_over_baseline'] = row['Scores (%)'] - baseline_score
            all_data.append(parsed)
    
    df = pd.DataFrame(all_data)
    df['baseline_score'] = baseline_score
    return df

def debug_training_curves(df, output_dir):
    """Debug version of plot_training_curves with detailed logging"""
    
    training_df = df.copy()
    baseline_score = df['baseline_score'].iloc[0] if not df.empty else 50.0
    
    print(f"=== DEBUGGING TRAINING CURVES ===")
    print(f"Training curves: {len(training_df)} models (including final checkpoints)")
    print(f"Baseline score: {baseline_score}")
    
    # Create plots for each rank
    ranks = training_df['rank'].unique()
    ranks = [r for r in ranks if pd.notna(r)]
    
    print(f"Ranks found: {ranks}")
    
    if len(ranks) == 0:
        print("❌ No rank data found for training curves")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    plot_count = 0
    for idx, rank in enumerate(sorted(ranks)):
        if idx >= 4:
            break
            
        ax = axes[idx]
        rank_data = training_df[training_df['rank'] == rank]
        
        print(f"\n--- Processing Rank {int(rank)} ---")
        print(f"  Total models: {len(rank_data)}")
        
        if rank_data.empty:
            print("  ❌ No data for this rank")
            continue
        
        # Check data quality
        valid_steps = rank_data['checkpoint_step'].notna()
        print(f"  Models with valid steps: {valid_steps.sum()}/{len(rank_data)}")
        
        if valid_steps.sum() == 0:
            print("  ❌ No valid checkpoint steps found")
            continue
            
        # Add baseline line
        ax.axhline(y=baseline_score, color='red', linestyle='--', 
                  linewidth=2, label=f'Baseline (even win rate): {baseline_score:.1f}%')
        print(f"  ✅ Added baseline line at {baseline_score}%")
        
        # Plot default configuration
        default_data = rank_data[rank_data['is_default'] & rank_data['checkpoint_step'].notna()]
        if not default_data.empty:
            default_sorted = default_data.sort_values('checkpoint_step')
            ax.plot(default_sorted['checkpoint_step'], default_sorted['score'], 
                   'o-', linewidth=3, markersize=8, label='Default', color='black')
            print(f"  ✅ Plotted default config: {len(default_data)} points")
            print(f"      Steps: {sorted(default_data['checkpoint_step'].values)}")
            print(f"      Scores: {default_data['score'].values}")
        else:
            print(f"  ❌ No default data with valid steps")
        
        # Plot non-default configurations
        non_default = rank_data[(~rank_data['is_default']) & rank_data['checkpoint_step'].notna()]
        
        if not non_default.empty:
            print(f"  Non-default models: {len(non_default)}")
            
            # Get unique learning rates
            unique_lrs = sorted([lr for lr in non_default['learning_rate'].unique() if pd.notna(lr)])
            print(f"  Unique learning rates: {unique_lrs}")
            
            # Define color families for each learning rate
            color_families = {
                1e-6: ['darkred', 'red', 'lightcoral'],
                1e-5: ['darkgreen', 'green', 'lightgreen'],  
                2e-5: ['navy', 'blue', 'lightblue'],
                5e-5: ['darkorange', 'orange', 'gold']
            }
            
            config_count = 0
            for lr in unique_lrs:
                lr_data = non_default[non_default['learning_rate'] == lr]
                colors = color_families.get(lr, ['gray', 'lightgray', 'silver'])
                
                warmup_ratios = sorted([w for w in lr_data['warmup_ratio'].unique() if pd.notna(w)])
                print(f"    LR {lr}: {len(lr_data)} models, warmup ratios: {warmup_ratios}")
                
                for i, warmup in enumerate(warmup_ratios[:3]):
                    warmup_data = lr_data[lr_data['warmup_ratio'] == warmup]
                    if len(warmup_data) > 1:
                        group_sorted = warmup_data.sort_values('checkpoint_step')
                        color = colors[i % len(colors)]
                        
                        label = f'LR={lr:.0e}, WR={warmup:.3f}'
                        
                        ax.plot(group_sorted['checkpoint_step'], group_sorted['score'], 
                               'o-', linewidth=2, markersize=4, label=label, 
                               color=color, alpha=0.8)
                        config_count += 1
                        print(f"      ✅ Plotted {label}: {len(group_sorted)} points")
            
            print(f"  ✅ Total non-default configs plotted: {config_count}")
        else:
            print(f"  ❌ No non-default configs with valid steps")
        
        ax.set_title(f'Rank {int(rank)} Training Curves')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Arena Hard Score (%)')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to show data clearly
        min_score = rank_data['score'].min()
        max_score = rank_data['score'].max()
        score_range = max_score - min_score
        ax.set_ylim(max(0, min_score - score_range * 0.1), 
                   min(100, max_score + score_range * 0.1))
        
        print(f"  Y-axis range: {ax.get_ylim()}")
        
        # Place legend outside plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plot_count += 1
        
        print(f"  ✅ Completed rank {int(rank)} subplot")
    
    # Remove empty subplots
    for idx in range(plot_count, 4):
        fig.delaxes(axes[idx])
        print(f"  Removed empty subplot {idx}")
    
    plt.tight_layout()
    output_file = f'{output_dir}/debug_training_curves_by_rank.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Debug training curves plot saved to {output_file}")
    print(f"   Total subplots created: {plot_count}")

def main():
    csv_dir = "results/vs_base_llama3.1-8b"
    output_dir = "analysis_output/vs_base_llama3.1-8b"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_and_parse_data(csv_dir)
    print(f"Loaded {len(df)} model results")
    
    debug_training_curves(df, output_dir)

if __name__ == "__main__":
    main()
