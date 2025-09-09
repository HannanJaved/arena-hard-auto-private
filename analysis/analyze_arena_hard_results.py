#!/usr/bin/env python3
"""
Arena Hard Results Analysis Script

Analyzes Arena Hard evaluation results comparing:
- Baseline: llama3.1-8b-instruct (instruction-tuned model)  
- Fine-tuned models: LoRA fine-tuning applied to base (non-instruct) model

Shows the effectiveness of LoRA fine-tuning in adapting base models for instruction-following tasks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
from pathlib import Path
import argparse

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_model_name(model_name):
    """
    Parse model name to extract hyperparameters
    Format: tulu3-8b-rank{RANK}-alpha{LR}-{WARMUP}-step{STEP}
    Or: tulu3-8b-rank{RANK}-default-step{STEP}
    Or: tulu3-8b-rank{RANK}-default-final
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
            parsed['checkpoint_step'] = 55000  # Use large number for final
        else:
            step_match = re.search(r'step(\d+)', model_name)
            if step_match:
                parsed['checkpoint_step'] = int(step_match.group(1))
    
    return parsed

def parse_confidence_interval(ci_str):
    """Parse confidence interval string like '(-1.8 / +1.7)' to get lower and upper bounds"""
    match = re.search(r'\(([+-]?\d+\.?\d*) / ([+-]?\d+\.?\d*)\)', ci_str)
    if match:
        lower = float(match.group(1))
        upper = float(match.group(2))
        return lower, upper
    return None, None

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
        
        df['rank_category'] = rank_category
        
        # Parse model names
        parsed_data = []
        for _, row in df.iterrows():
            # Skip the baseline model in individual files
            if row['Model'] == 'llama3.1-8b-instruct':
                continue
                
            parsed = parse_model_name(row['Model'])
            parsed['score'] = row['Scores (%)']
            parsed['absolute_score'] = row['Scores (%)']
            # Calculate improvement over baseline
            if baseline_score is not None:
                parsed['improvement_over_baseline'] = row['Scores (%)'] - baseline_score
            else:
                parsed['improvement_over_baseline'] = None
            ci_lower, ci_upper = parse_confidence_interval(row['CI (%)'])
            parsed['ci_lower'] = ci_lower
            parsed['ci_upper'] = ci_upper
            parsed['rank_category'] = rank_category
            parsed_data.append(parsed)
        
        all_data.extend(parsed_data)
    
    df = pd.DataFrame(all_data)
    df['baseline_score'] = baseline_score
    return df

def plot_training_curves(df, output_dir):
    """Plot performance vs training steps for different configurations"""
    
    # Include all checkpoints (including final) for training curves
    training_df = df.copy()
    baseline_score = df['baseline_score'].iloc[0] if not df.empty else 50.0
    
    print(f"Training curves: {len(training_df)} models (including final checkpoints)")
    
    # Create plots for each rank
    ranks = training_df['rank'].unique()
    ranks = [r for r in ranks if pd.notna(r)]
    
    if len(ranks) == 0:
        print("No rank data found for training curves")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    plot_count = 0
    for idx, rank in enumerate(sorted(ranks)):
        if idx >= 4:
            break
            
        ax = axes[idx]
        rank_data = training_df[training_df['rank'] == rank]
        
        print(f"  Rank {int(rank)}: {len(rank_data)} models")
        
        if rank_data.empty:
            continue
            
        # Add baseline line
        ax.axhline(y=baseline_score, color='red', linestyle='--', 
                  linewidth=2, label=f'Baseline (llama3.1-8b-instruct): {baseline_score:.1f}%')
        
        # Plot default configuration
        default_data = rank_data[rank_data['is_default']]
        if not default_data.empty:
            default_sorted = default_data.sort_values('checkpoint_step')
            ax.plot(default_sorted['checkpoint_step'], default_sorted['score'], 
                   'o-', linewidth=3, markersize=8, label='Default', color='black')
            print(f"    Default: {len(default_data)} points")
        
        # Plot different hyperparameter configurations
        non_default = rank_data[~rank_data['is_default']]
        
        # Group by learning rate and warmup ratio, but prioritize showing all learning rates
        # config_count = 0
        plotted_lrs = set()
        
        # First pass: plot one config per learning rate (with warmup=0.010 if available)
        # for (lr, warmup), group in non_default.groupby(['learning_rate', 'warmup_ratio']):
        #     if pd.notna(lr) and pd.notna(warmup) and len(group) > 1:
        #         if lr not in plotted_lrs and warmup == 0.010:  # Prefer warmup=0.010
        #             label = f'LR={lr:.0e}, WR={warmup:.3f}'
        #             group_sorted = group.sort_values('checkpoint_step')
        #             ax.plot(group_sorted['checkpoint_step'], group_sorted['score'], 
        #                    'o-', linewidth=2, markersize=4, label=label, alpha=0.8)
        #             plotted_lrs.add(lr)
        #             config_count += 1
        
        # # Second pass: fill remaining slots with other configs, avoiding duplicate LRs
        # for (lr, warmup), group in non_default.groupby(['learning_rate', 'warmup_ratio']):
        #     if pd.notna(lr) and pd.notna(warmup) and len(group) > 1:
        #         if lr not in plotted_lrs:  # Haven't plotted this LR yet
        #             label = f'LR={lr:.0e}, WR={warmup:.3f}'
        #             group_sorted = group.sort_values('checkpoint_step')
        #             ax.plot(group_sorted['checkpoint_step'], group_sorted['score'], 
        #                    'o-', linewidth=2, markersize=4, label=label, alpha=0.8)
        #             plotted_lrs.add(lr)
        #             config_count += 1
                    # if config_count >= 6:  # Increased limit slightly
                    #     break
        
        # Third pass: add a few more interesting configs if space allows
        for (lr, warmup), group in non_default.groupby(['learning_rate', 'warmup_ratio']):
            if pd.notna(lr) and pd.notna(warmup) and len(group) > 1:
                # if config_count < 13:
                label = f'LR={lr:.0e}, WR={warmup:.3f}'
                group_sorted = group.sort_values('checkpoint_step')
                ax.plot(group_sorted['checkpoint_step'], group_sorted['score'], 
                        'o-', linewidth=2, markersize=4, label=label, alpha=0.8)
                    # config_count += 1
                # else:
                #     break
        
        print(f"    Non-default configs plotted!  ")
        
        ax.set_title(f'Rank {int(rank)} Training Curves')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Arena Hard Score (%)')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to show data clearly
        min_score = rank_data['score'].min()
        max_score = rank_data['score'].max()
        score_range = max_score - min_score
        ax.set_ylim(max(0, min_score - score_range * 0.1), 
                   min(baseline_score + 5, max_score + score_range * 0.1))
        
        # Place legend outside plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plot_count += 1
    
    # Remove empty subplots
    for idx in range(plot_count, 4):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves_by_rank.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves plot saved with {plot_count} subplots")

def plot_hyperparameter_heatmap(df, output_dir):
    """Create heatmaps showing performance across hyperparameters"""
    
    # Filter to non-default configurations and final checkpoints
    hparam_df = df[(~df['is_default']) & (df['is_final'])].copy()
    
    if hparam_df.empty:
        print("No hyperparameter data found for final checkpoints")
        return
    
    ranks = hparam_df['rank'].unique()
    ranks = [r for r in ranks if pd.notna(r)]
    
    fig, axes = plt.subplots(1, len(ranks), figsize=(6*len(ranks), 5))
    if len(ranks) == 1:
        axes = [axes]
    
    for idx, rank in enumerate(sorted(ranks)):
        rank_data = hparam_df[hparam_df['rank'] == rank]
        
        if rank_data.empty:
            continue
        
        # Create pivot table for heatmap
        pivot_data = rank_data.pivot_table(
            values='score', 
            index='learning_rate', 
            columns='warmup_ratio', 
            aggfunc='mean'
        )
        
        # Create heatmap
        im = axes[idx].imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto')
        
        # Set labels
        axes[idx].set_title(f'Rank {int(rank)} Final Performance')
        axes[idx].set_xlabel('Warmup Ratio')
        axes[idx].set_ylabel('Learning Rate')
        
        # Set tick labels
        axes[idx].set_xticks(range(len(pivot_data.columns)))
        axes[idx].set_xticklabels([f'{x:.3f}' for x in pivot_data.columns])
        axes[idx].set_yticks(range(len(pivot_data.index)))
        axes[idx].set_yticklabels([f'{x:.0e}' for x in pivot_data.index])
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], label='Arena Hard Score (%)')
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                if pd.notna(pivot_data.values[i, j]):
                    axes[idx].text(j, i, f'{pivot_data.values[i, j]:.1f}', 
                                 ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_rank_comparison(df, output_dir):
    """Compare performance across different ranks"""
    
    # Get final performance for each configuration
    final_df = df[df['is_final']].copy()
    
    plt.figure(figsize=(12, 8))
    
    # Create box plots for each rank
    ranks = sorted([r for r in final_df['rank'].unique() if pd.notna(r)])
    
    rank_data = []
    rank_labels = []
    
    for rank in ranks:
        rank_scores = final_df[final_df['rank'] == rank]['score'].values
        if len(rank_scores) > 0:
            rank_data.append(rank_scores)
            rank_labels.append(f'Rank {int(rank)}')
    
    bp = plt.boxplot(rank_data, tick_labels=rank_labels, patch_artist=True)
    
    # Color the boxes
    colors = sns.color_palette("husl", len(rank_data))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Performance Distribution Across Ranks (Final Checkpoints)')
    plt.xlabel('Rank')
    plt.ylabel('Arena Hard Score (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rank_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_improvement_over_baseline(df, output_dir):
    """Plot improvement over baseline for all models"""
    
    baseline_score = df['baseline_score'].iloc[0] if not df.empty else 50.0
    
    # Filter to final checkpoints
    final_df = df[df['is_final']].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Improvement by rank
    ax1 = axes[0]
    ranks = sorted([r for r in final_df['rank'].unique() if pd.notna(r)])
    
    # Prepare data for box plot
    improvements_by_rank = []
    rank_labels = []
    
    for rank in ranks:
        rank_data = final_df[final_df['rank'] == rank]
        improvements_by_rank.append(rank_data['improvement_over_baseline'].values)
        rank_labels.append(f'Rank {int(rank)}')
    
    # Create box plot
    bp = ax1.boxplot(improvements_by_rank, tick_labels=rank_labels, patch_artist=True)
    
    # Color the boxes
    colors = sns.color_palette("Set2", len(bp['boxes']))
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.7)
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Baseline')
    ax1.set_ylabel('Improvement over Baseline (percentage points)')
    ax1.set_title('Improvement over llama3.1-8b-instruct by Rank')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best vs Default configurations
    ax2 = axes[1]
    
    # Get default and best tuned for each rank
    rank_comparison = []
    for rank in ranks:
        rank_data = final_df[final_df['rank'] == rank]
        
        # Default configuration
        default_models = rank_data[rank_data['is_default']]
        if not default_models.empty:
            default_score = default_models['improvement_over_baseline'].max()
        else:
            default_score = 0
        
        # Best tuned configuration
        best_tuned = rank_data[~rank_data['is_default']]
        if not best_tuned.empty:
            best_score = best_tuned['improvement_over_baseline'].max()
        else:
            best_score = 0
        
        rank_comparison.append({
            'rank': rank,
            'default': default_score,
            'best_tuned': best_score
        })
    
    comp_df = pd.DataFrame(rank_comparison)
    
    x = np.arange(len(ranks))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, comp_df['default'], width, 
                   label='Default Config', alpha=0.8, color='lightcoral')
    bars2 = ax2.bar(x + width/2, comp_df['best_tuned'], width,
                   label='Best Tuned Config', alpha=0.8, color='lightblue')
    
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Improvement over Baseline (percentage points)')
    ax2.set_title('Default vs Best Tuned Configurations')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Rank {int(r)}' for r in ranks])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_over_baseline.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_rate_effect(df, output_dir):
    """Analyze effect of learning rate across ranks"""
    
    # Filter to final checkpoints and non-default
    lr_df = df[(df['is_final']) & (~df['is_default'])].copy()
    
    if lr_df.empty:
        print("No learning rate data found")
        return
    
    plt.figure(figsize=(12, 6))
    
    ranks = sorted([r for r in lr_df['rank'].unique() if pd.notna(r)])
    
    for rank in ranks:
        rank_data = lr_df[lr_df['rank'] == rank]
        
        # Group by learning rate
        lr_performance = rank_data.groupby('learning_rate')['score'].agg(['mean', 'std']).reset_index()
        
        plt.errorbar(lr_performance['learning_rate'], lr_performance['mean'], 
                    yerr=lr_performance['std'], marker='o', linewidth=2, 
                    markersize=6, label=f'Rank {int(rank)}', capsize=5)
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Arena Hard Score (%)')
    plt.title('Learning Rate Effect on Final Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_rate_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(df, output_dir):
    """Create summary statistics table"""
    
    baseline_score = df['baseline_score'].iloc[0] if not df.empty else 50.0
    
    # Best overall performance
    best_overall = df.loc[df['score'].idxmax()]
    
    # Best per rank
    best_per_rank = df.groupby('rank').apply(lambda x: x.loc[x['score'].idxmax()]).reset_index(drop=True)
    
    # Default vs best tuned comparison
    default_final = df[(df['is_default']) & (df['is_final'])]
    tuned_final = df[(~df['is_default']) & (df['is_final'])]
    
    summary_stats = {
        'Baseline (llama3.1-8b-instruct)': {
            'Score': f"{baseline_score:.1f}%",
            'Improvement': "0.0 (baseline)",
            'Rank': "N/A",
            'Learning Rate': "N/A",
            'Warmup Ratio': "N/A"
        },
        'Overall Best': {
            'Model': best_overall['model_name'],
            'Score': f"{best_overall['score']:.1f}%",
            'Improvement': f"+{best_overall['improvement_over_baseline']:.1f}pp",
            'Rank': best_overall['rank'],
            'Learning Rate': best_overall['learning_rate'],
            'Warmup Ratio': best_overall['warmup_ratio']
        }
    }
    
    for _, row in best_per_rank.iterrows():
        rank = row['rank']
        if pd.notna(rank):
            summary_stats[f'Best Rank {int(rank)}'] = {
                'Model': row['model_name'],
                'Score': f"{row['score']:.1f}%",
                'Improvement': f"+{row['improvement_over_baseline']:.1f}pp",
                'Learning Rate': row['learning_rate'],
                'Warmup Ratio': row['warmup_ratio']
            }
    
    # Save to CSV
    summary_df = pd.DataFrame(summary_stats).T
    summary_df.to_csv(f'{output_dir}/performance_summary.csv')
    
    print("\n=== PERFORMANCE SUMMARY vs llama3.1-8b-instruct ===")
    print(summary_df.to_string())
    
    if not default_final.empty and not tuned_final.empty:
        print(f"\nBaseline (llama3.1-8b-instruct): {baseline_score:.1f}%")
        print(f"Default Tuned Performance (avg): {default_final['score'].mean():.1f}% (+{default_final['improvement_over_baseline'].mean():.1f}pp)")
        print(f"Best Tuned Performance (avg): {tuned_final['score'].mean():.1f}% (+{tuned_final['improvement_over_baseline'].mean():.1f}pp)")
        print(f"Best vs Default improvement: {tuned_final['score'].mean() - default_final['score'].mean():.1f} percentage points")
    
    # Add improvement analysis
    print(f"\n=== IMPROVEMENT ANALYSIS ===")
    print(f"Best overall improvement: +{best_overall['improvement_over_baseline']:.1f}pp ({best_overall['model_name']})")
    
    # Rank-wise improvements
    for rank in sorted(df['rank'].unique()):
        if pd.notna(rank):
            rank_data = df[df['rank'] == rank]
            best_rank = rank_data.loc[rank_data['score'].idxmax()]
            print(f"Best Rank {int(rank)} improvement: +{best_rank['improvement_over_baseline']:.1f}pp ({best_rank['model_name']})")

def main():
    parser = argparse.ArgumentParser(description='Analyze Arena Hard results')
    parser.add_argument('--csv-dir', default='results', help='Directory containing CSV files')
    parser.add_argument('--output-dir', default='analysis_output', help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading and parsing data...")
    df = load_and_parse_data(args.csv_dir)
    
    print(f"Loaded {len(df)} model results")
    
    # Generate all plots and analysis
    print("Generating training curves...")
    plot_training_curves(df, args.output_dir)
    
    print("Generating improvement analysis...")
    plot_improvement_over_baseline(df, args.output_dir)
    
    print("Generating hyperparameter heatmap...")
    plot_hyperparameter_heatmap(df, args.output_dir)
    
    print("Generating rank comparison...")
    plot_rank_comparison(df, args.output_dir)
    
    print("Generating learning rate analysis...")
    plot_learning_rate_effect(df, args.output_dir)
    
    print("Creating summary table...")
    create_summary_table(df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
