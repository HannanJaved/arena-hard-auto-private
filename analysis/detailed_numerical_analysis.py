#!/usr/bin/env python3
"""
Detailed numerical analysis of Arena Hard results
"""

import pandas as pd
import numpy as np
from pathlib import Path

def detailed_analysis():
    # Load the all results CSV
    csv_path = Path("results/hard_prompt_leaderboard_all.csv")
    
    if not csv_path.exists():
        print("CSV file not found. Please run show_result.py first.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Get baseline
    baseline = df[df['Model'] == 'llama3.1-8b-instruct']['Scores (%)'].iloc[0]
    
    # Filter out baseline for analysis
    tuned_df = df[df['Model'] != 'llama3.1-8b-instruct'].copy()
    
    print("="*80)
    print("DETAILED ARENA HARD ANALYSIS")
    print("="*80)
    print(f"Baseline (llama3.1-8b-instruct): {baseline:.1f}%")
    print(f"Total fine-tuned models analyzed: {len(tuned_df)}")
    print()
    
    # Basic statistics
    print("PERFORMANCE STATISTICS:")
    print("-" * 30)
    print(f"Best fine-tuned score: {tuned_df['Scores (%)'].max():.1f}%")
    print(f"Worst fine-tuned score: {tuned_df['Scores (%)'].min():.1f}%")
    print(f"Average fine-tuned score: {tuned_df['Scores (%)'].mean():.1f}%")
    print(f"Median fine-tuned score: {tuned_df['Scores (%)'].median():.1f}%")
    print(f"Standard deviation: {tuned_df['Scores (%)'].std():.1f}%")
    print()
    
    # Performance degradation analysis
    tuned_df['degradation'] = baseline - tuned_df['Scores (%)']
    
    print("DEGRADATION FROM BASELINE:")
    print("-" * 30)
    print(f"Minimum degradation: {tuned_df['degradation'].min():.1f}pp")
    print(f"Maximum degradation: {tuned_df['degradation'].max():.1f}pp")
    print(f"Average degradation: {tuned_df['degradation'].mean():.1f}pp")
    print(f"Median degradation: {tuned_df['degradation'].median():.1f}pp")
    print()
    
    # Models that perform "best" (least bad)
    top_10 = tuned_df.nlargest(10, 'Scores (%)')
    print("TOP 10 PERFORMING FINE-TUNED MODELS:")
    print("-" * 50)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        degradation = baseline - row['Scores (%)']
        print(f"{i:2d}. {row['Model'][:50]:<50} {row['Scores (%)']:5.1f}% (-{degradation:.1f}pp)")
    print()
    
    # Rank analysis
    rank_analysis = {}
    for rank in ['rank64', 'rank256', 'rank1024', 'default']:
        rank_models = tuned_df[tuned_df['Model'].str.contains(rank, na=False)]
        if len(rank_models) > 0:
            rank_analysis[rank] = {
                'count': len(rank_models),
                'best': rank_models['Scores (%)'].max(),
                'worst': rank_models['Scores (%)'].min(),
                'average': rank_models['Scores (%)'].mean(),
                'best_model': rank_models.loc[rank_models['Scores (%)'].idxmax(), 'Model']
            }
    
    print("RANK-WISE ANALYSIS:")
    print("-" * 50)
    for rank, stats in rank_analysis.items():
        degradation = baseline - stats['best']
        print(f"{rank.upper():8s}: {stats['count']:3d} models | Best: {stats['best']:5.1f}% (-{degradation:.1f}pp) | Avg: {stats['average']:5.1f}%")
        print(f"         Best model: {stats['best_model']}")
        print()
    
    # Learning rate analysis (for non-default models)
    lr_models = tuned_df[tuned_df['Model'].str.contains('alpha', na=False)]
    
    print("LEARNING RATE PATTERNS:")
    print("-" * 30)
    lr_patterns = {}
    for _, row in lr_models.iterrows():
        model_name = row['Model']
        if 'alpha1e5' in model_name:
            lr = '1e-5'
        elif 'alpha5e5' in model_name:
            lr = '5e-5'
        elif 'alpha1e6' in model_name:
            lr = '1e-6'
        else:
            continue
            
        if lr not in lr_patterns:
            lr_patterns[lr] = []
        lr_patterns[lr].append(row['Scores (%)'])
    
    for lr, scores in lr_patterns.items():
        print(f"Learning Rate {lr:4s}: {len(scores):3d} models | Best: {max(scores):5.1f}% | Avg: {np.mean(scores):5.1f}%")
    print()
    
    # Training step analysis
    print("TRAINING STEP ANALYSIS:")
    print("-" * 30)
    step_performance = {}
    for _, row in tuned_df.iterrows():
        model_name = row['Model']
        if 'step' in model_name:
            try:
                step = model_name.split('-step')[-1]
                step_num = int(step)
                if step_num not in step_performance:
                    step_performance[step_num] = []
                step_performance[step_num].append(row['Scores (%)'])
            except:
                continue
    
    sorted_steps = sorted(step_performance.keys())
    for step in sorted_steps:
        scores = step_performance[step]
        if len(scores) >= 3:  # Only show steps with multiple models
            print(f"Step {step:5d}: {len(scores):3d} models | Best: {max(scores):5.1f}% | Avg: {np.mean(scores):5.1f}%")
    
    print()
    print("="*80)
    print("CONCLUSION: All fine-tuned models show significant degradation from baseline.")
    print("This suggests fundamental issues with the fine-tuning approach or data.")
    print("="*80)

if __name__ == "__main__":
    detailed_analysis()
