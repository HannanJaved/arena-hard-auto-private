#!/usr/bin/env python3
"""
Arena Hard Results Text-based Analysis
Provides detailed insights into training patterns and hyperparameter effects
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
import argparse

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
        # Set default hyperparameters
        parsed['learning_rate'] = 2e-5  # Default LR = 2e-5
        parsed['warmup_ratio'] = 0.03   # Default WR = 0.03
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
        
        # Parse model names
        for _, row in df.iterrows():
            parsed = parse_model_name(row['Model'])
            parsed['score'] = row['Scores (%)']
            all_data.append(parsed)
    
    return pd.DataFrame(all_data)

def analyze_training_patterns(df):
    """Analyze training patterns and identify optimal stopping points"""
    print("\n" + "="*60)
    print("TRAINING PATTERN ANALYSIS")
    print("="*60)
    
    # Group by configuration and analyze training progression
    for rank in sorted(df['rank'].unique()):
        if pd.isna(rank):
            continue
            
        print(f"\n--- RANK {int(rank)} ANALYSIS ---")
        rank_data = df[df['rank'] == rank]
        
        # Default configuration analysis
        default_data = rank_data[rank_data['is_default'] & ~rank_data['is_final']].sort_values('checkpoint_step')
        if not default_data.empty:
            print("\nDefault Configuration Training Curve:")
            best_step = default_data.loc[default_data['score'].idxmax(), 'checkpoint_step']
            best_score = default_data['score'].max()
            print(f"  Best step: {best_step:,} (Score: {best_score:.1f}%)")
            
            # Check for overfitting
            final_score = default_data.iloc[-1]['score']
            if final_score < best_score - 1.0:  # 1% drop threshold
                print(f"  ⚠️  Potential overfitting detected (final: {final_score:.1f}%, best: {best_score:.1f}%)")
        
        # Hyperparameter configurations
        tuned_data = rank_data[~rank_data['is_default']]
        if not tuned_data.empty:
            print("\nHyperparameter Tuning Results:")
            
            # Group by LR and warmup
            for (lr, warmup), group in tuned_data.groupby(['learning_rate', 'warmup_ratio']):
                if pd.notna(lr) and pd.notna(warmup):
                    group_sorted = group[~group['is_final']].sort_values('checkpoint_step')
                    if not group_sorted.empty:
                        best_step = group_sorted.loc[group_sorted['score'].idxmax(), 'checkpoint_step']
                        best_score = group_sorted['score'].max()
                        print(f"  LR={lr:.0e}, Warmup={warmup:.3f}: Best at step {best_step:,} ({best_score:.1f}%)")

def analyze_hyperparameter_effects(df):
    """Analyze the effect of different hyperparameters"""
    print("\n" + "="*60)
    print("HYPERPARAMETER EFFECTS ANALYSIS")
    print("="*60)
    
    # Focus on final/converged models
    final_data = df[df['is_final'] | (df['checkpoint_step'] >= 36000)]
    
    for rank in sorted(df['rank'].unique()):
        if pd.isna(rank):
            continue
            
        print(f"\n--- RANK {int(rank)} HYPERPARAMETER EFFECTS ---")
        rank_data = final_data[final_data['rank'] == rank]
        
        # Default baseline
        default_score = rank_data[rank_data['is_default']]['score'].mean()
        print(f"Default baseline: {default_score:.1f}%")
        
        # Learning rate effects
        tuned_data = rank_data[~rank_data['is_default']]
        if not tuned_data.empty:
            print("\nLearning Rate Effects:")
            for lr in sorted(tuned_data['learning_rate'].unique()):
                if pd.notna(lr):
                    lr_scores = tuned_data[tuned_data['learning_rate'] == lr]['score']
                    mean_score = lr_scores.mean()
                    improvement = mean_score - default_score
                    print(f"  LR {lr:.0e}: {mean_score:.1f}% (Δ{improvement:+.1f}%)")
            
            print("\nWarmup Ratio Effects:")
            for warmup in sorted(tuned_data['warmup_ratio'].unique()):
                if pd.notna(warmup):
                    warmup_scores = tuned_data[tuned_data['warmup_ratio'] == warmup]['score']
                    mean_score = warmup_scores.mean()
                    improvement = mean_score - default_score
                    print(f"  Warmup {warmup:.3f}: {mean_score:.1f}% (Δ{improvement:+.1f}%)")

def analyze_rank_scaling(df):
    """Analyze how performance scales with rank"""
    print("\n" + "="*60)
    print("RANK SCALING ANALYSIS")
    print("="*60)
    
    final_data = df[df['is_final'] | (df['checkpoint_step'] >= 36000)]
    
    print("\nBest Performance by Rank:")
    for rank in sorted(df['rank'].unique()):
        if pd.isna(rank):
            continue
            
        rank_data = final_data[final_data['rank'] == rank]
        
        # Best overall for this rank
        best_overall = rank_data.loc[rank_data['score'].idxmax()]
        print(f"  Rank {int(rank):4d}: {best_overall['score']:.1f}% - {best_overall['model_name']}")
        
        # Best default for this rank
        default_data = rank_data[rank_data['is_default']]
        if not default_data.empty:
            best_default = default_data.loc[default_data['score'].idxmax()]
            print(f"            (Default: {best_default['score']:.1f}%)")

def analyze_convergence_patterns(df):
    """Analyze when models converge and identify optimal training length"""
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    
    # Look for configurations with multiple checkpoints
    configs = []
    for rank in df['rank'].unique():
        if pd.isna(rank):
            continue
        rank_data = df[df['rank'] == rank]
        
        # Default configs
        default_data = rank_data[rank_data['is_default'] & ~rank_data['is_final']]
        if len(default_data) > 2:
            configs.append(('default', rank, default_data))
        
        # Tuned configs
        for (lr, warmup), group in rank_data.groupby(['learning_rate', 'warmup_ratio']):
            if pd.notna(lr) and pd.notna(warmup):
                group_data = group[~group['is_final']]
                if len(group_data) > 2:
                    configs.append((f'LR={lr:.0e}_W={warmup:.3f}', rank, group_data))
    
    print("\nTraining Length Recommendations:")
    for config_name, rank, data in configs[:5]:  # Show top 5
        sorted_data = data.sort_values('checkpoint_step')
        
        # Find optimal stopping point (highest score)
        best_idx = sorted_data['score'].idxmax()
        best_step = sorted_data.loc[best_idx, 'checkpoint_step']
        best_score = sorted_data.loc[best_idx, 'score']
        
        # Check if continuing training helped
        later_data = sorted_data[sorted_data['checkpoint_step'] > best_step]
        if not later_data.empty:
            final_score = sorted_data.iloc[-1]['score']
            if final_score < best_score - 0.5:
                status = "⚠️  Early stopping recommended"
            else:
                status = "✅ Continued training beneficial"
        else:
            status = "📊 Optimal point reached"
        
        print(f"  Rank {int(rank)} {config_name}: Optimal at step {best_step:,} ({best_score:.1f}%) {status}")

def generate_recommendations(df):
    """Generate actionable recommendations based on analysis"""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    final_data = df[df['is_final'] | (df['checkpoint_step'] >= 36000)]
    
    # Best configuration overall
    best_model = final_data.loc[final_data['score'].idxmax()]
    print(f"\n🏆 BEST OVERALL CONFIGURATION:")
    print(f"   Model: {best_model['model_name']}")
    print(f"   Score: {best_model['score']:.1f}%")
    if not best_model['is_default']:
        print(f"   LR: {best_model['learning_rate']:.0e}")
        print(f"   Warmup: {best_model['warmup_ratio']:.3f}")
    
    # Best per rank
    print(f"\n📊 BEST CONFIGURATION PER RANK:")
    for rank in sorted(df['rank'].unique()):
        if pd.isna(rank):
            continue
        rank_data = final_data[final_data['rank'] == rank]
        best_rank = rank_data.loc[rank_data['score'].idxmax()]
        
        config_type = "Default" if best_rank['is_default'] else "Tuned"
        print(f"   Rank {int(rank):4d}: {best_rank['score']:.1f}% ({config_type})")
        
        if not best_rank['is_default']:
            print(f"              LR={best_rank['learning_rate']:.0e}, Warmup={best_rank['warmup_ratio']:.3f}")
    
    # Learning rate recommendations
    tuned_data = final_data[~final_data['is_default']]
    if not tuned_data.empty:
        lr_performance = tuned_data.groupby('learning_rate')['score'].mean().sort_values(ascending=False)
        print(f"\n🎯 LEARNING RATE RANKING:")
        for lr, score in lr_performance.head(3).items():
            print(f"   {lr:.0e}: {score:.1f}% avg")
        
        warmup_performance = tuned_data.groupby('warmup_ratio')['score'].mean().sort_values(ascending=False)
        print(f"\n🎯 WARMUP RATIO RANKING:")
        for warmup, score in warmup_performance.head(3).items():
            print(f"   {warmup:.3f}: {score:.1f}% avg")

def main():
    parser = argparse.ArgumentParser(description='Analyze Arena Hard results (text-based)')
    parser.add_argument('--csv-dir', default='results', help='Directory containing CSV files')
    args = parser.parse_args()
    
    print("Loading and parsing Arena Hard results...")
    df = load_and_parse_data(args.csv_dir)
    print(f"Loaded {len(df)} model results")
    
    # Run all analyses
    analyze_training_patterns(df)
    analyze_hyperparameter_effects(df)
    analyze_rank_scaling(df)
    analyze_convergence_patterns(df)
    generate_recommendations(df)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
