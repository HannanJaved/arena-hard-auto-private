#!/usr/bin/env python3
"""
Memory-optimized Arena Hard results processor
Processes judgment files in batches to avoid OOM issues
"""

import pandas as pd
import argparse
import os
import torch
import gc
from glob import glob
from tqdm import tqdm
import tempfile
import pickle

from utils.judge_utils import JUDGE_SETTINGS
from utils.math_utils import one_hot_encode, to_winrate_probabilities, bootstrap_pairwise_model


def load_judgments_batch(judge_names, benchmark, batch_size=50, weight=3):
    """Load judgments in batches to avoid memory issues"""
    print(f"Loading judgments in batches of {batch_size}...")
    
    all_files = []
    for judge_name in judge_names:
        files = glob(os.path.join(
            "data",
            benchmark, 
            "model_judgment", 
            judge_name, 
            "*.jsonl"
        ))
        all_files.extend(files)
    
    print(f"Found {len(all_files)} judgment files to process")
    
    # Process files in batches
    processed_batches = []
    
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size} ({len(batch_files)} files)")
        
        # Load batch
        batch_dfs = []
        for f in tqdm(batch_files, desc="Loading files"):
            try:
                df = pd.read_json(f, lines=True)
                if not df.empty:
                    batch_dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
                continue
        
        if not batch_dfs:
            continue
            
        # Concatenate batch
        batch_data = pd.concat(batch_dfs, ignore_index=True)
        
        # Clean up individual DataFrames
        del batch_dfs
        gc.collect()
        
        # Process batch data
        processed_batch = process_batch_data(batch_data, weight)
        processed_batches.append(processed_batch)
        
        # Clean up batch data
        del batch_data
        gc.collect()
        
        print(f"Batch {i//batch_size + 1} processed: {len(processed_batch)} battles")
    
    # Combine all processed batches
    if processed_batches:
        print("Combining all batches...")
        final_battles = pd.concat(processed_batches, ignore_index=True)
        
        # Clean up batch data
        del processed_batches
        gc.collect()
        
        print(f"Total battles loaded: {len(final_battles)}")
        return final_battles
    else:
        print("No valid judgment data found!")
        return pd.DataFrame()


def process_batch_data(data, weight=3):
    """Process a batch of judgment data"""
    
    # Filter out null judgments
    null_indices = data.games.map(lambda x: x[0] is None or x[1] is None or x[0]['score'] is None or x[1]['score'] is None)
    _data = data[~null_indices].reset_index(drop=True)
    
    if len(data) - len(_data) > 0:
        print(f"  Filtered out {len(data) - len(_data)} null judgments from batch")
    
    if _data.empty:
        return pd.DataFrame()
    
    # Map label to score
    label_to_score = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }

    _data['scores'] = _data.games.map(
        lambda x: label_to_score[x[1]['score']] + [1 - s for s in label_to_score[x[0]['score']]]
    )
    
    # Explode scores to create individual battles
    battles = _data[['uid', 'model', 'category', 'scores']].explode('scores').reset_index(drop=True)
    
    return battles


def load_judgments_memory_mapped(judge_names, benchmark, weight=3):
    """Alternative approach using temporary files for very large datasets"""
    print("Using memory-mapped approach for large dataset...")
    
    # Create temporary file for storing processed data
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "battles.pkl")
    
    try:
        all_files = []
        for judge_name in judge_names:
            files = glob(os.path.join(
                "data",
                benchmark, 
                "model_judgment", 
                judge_name, 
                "*.jsonl"
            ))
            all_files.extend(files)
        
        print(f"Found {len(all_files)} judgment files")
        
        # Process one file at a time and append to temporary storage
        all_battles = []
        
        for i, f in enumerate(tqdm(all_files, desc="Processing files")):
            try:
                df = pd.read_json(f, lines=True)
                if df.empty:
                    continue
                    
                # Process this file
                batch_battles = process_batch_data(df, weight)
                if not batch_battles.empty:
                    all_battles.append(batch_battles)
                
                # Every 50 files, save intermediate results and clear memory
                if (i + 1) % 50 == 0:
                    if all_battles:
                        combined = pd.concat(all_battles, ignore_index=True)
                        
                        # Save to temporary file
                        if os.path.exists(temp_file):
                            existing = pd.read_pickle(temp_file)
                            combined = pd.concat([existing, combined], ignore_index=True)
                        
                        combined.to_pickle(temp_file)
                        
                        # Clear memory
                        del all_battles, combined
                        all_battles = []
                        gc.collect()
                        
                        print(f"  Saved intermediate results after {i+1} files")
                
            except Exception as e:
                print(f"Warning: Failed to process {f}: {e}")
                continue
        
        # Save any remaining battles
        if all_battles:
            combined = pd.concat(all_battles, ignore_index=True)
            
            if os.path.exists(temp_file):
                existing = pd.read_pickle(temp_file)
                combined = pd.concat([existing, combined], ignore_index=True)
            
            combined.to_pickle(temp_file)
            del all_battles, combined
            gc.collect()
        
        # Load final result
        if os.path.exists(temp_file):
            final_battles = pd.read_pickle(temp_file)
            print(f"Total battles loaded: {len(final_battles)}")
            return final_battles
        else:
            print("No valid battles found!")
            return pd.DataFrame()
            
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file):
            os.remove(temp_file)
        os.rmdir(temp_dir)


def load_judgments(judge_names, benchmark, weight=3, memory_efficient=True, batch_size=30):
    """
    Load judgments with memory optimization
    
    Args:
        judge_names: List of judge names
        benchmark: Benchmark name
        weight: Weight for strong preferences
        memory_efficient: Use memory-efficient loading
        batch_size: Number of files to process at once
    """
    if memory_efficient:
        if batch_size > 0:
            return load_judgments_batch(judge_names, benchmark, batch_size, weight)
        else:
            return load_judgments_memory_mapped(judge_names, benchmark, weight)
    else:
        # Original implementation (for small datasets)
        dfs = []
        for judge_name in judge_names:
            print(f"Loading {judge_name} judgments...")
            files = glob(os.path.join(
                "data",
                benchmark, 
                "model_judgment", 
                judge_name, 
                "*.jsonl"
            ))
            print(f"Found {len(files)} files")
            
            if len(files) > 100:
                print(f"WARNING: {len(files)} files detected. Consider using --memory-efficient flag.")
            
            dfs.extend([
                pd.read_json(f, lines=True) for f in tqdm(files)
            ])
        
        data = pd.concat(dfs).reset_index(drop=True)
        return process_batch_data(data, weight)


def get_model_style_metadata(benchmark):
    model_metadata = {}
    for file in glob(os.path.join("data", benchmark, "model_answer", "*.jsonl")):
        df = pd.read_json(file, lines=True)
        model_metadata[df.iloc[0]['model']] = df.set_index('uid')['metadata'].to_dict()
        
    return model_metadata


def format_confidence_interval(mean_scores, lower_scores, upper_scores, baseline=None):
    leaderboard = pd.merge(
        mean_scores, 
        lower_scores, 
        on="model"
    ).merge(
        upper_scores, 
        on="model"
    )
    
    leaderboard["Scores (%)"] = leaderboard["scores"].map(lambda x: round(x * 100, 1))
    
    leaderboard["CI (%)"] = leaderboard.apply(
        lambda row: f"(-{round((row['scores'] - row['lower']) * 100, 1)} / +{round((row['upper'] - row['scores']) * 100, 1)})", 
        axis=1
    )
    
    _leaderboard = leaderboard.rename(
        columns={"model": "Model"}
    ).drop(
        columns=["lower", "upper", "scores"]
    )
    
    if baseline:
        _leaderboard = pd.concat(
            [_leaderboard, pd.DataFrame({"Model": baseline, "Scores (%)": 50.0, "CI (%)": "(-0.0 / +0.0)"}, index=[0])]
        )
    
    return _leaderboard.sort_values(by="Scores (%)", ascending=False).reset_index(drop=True)


def print_leaderboard(battles, category):
    baseline = JUDGE_SETTINGS[category]["baseline"]
    
    _battles = battles.drop(columns=['category'])[['model', 'scores']]
    
    # remove model path
    _battles['model'] = _battles['model'].map(lambda x: x.split('/')[-1])
    
    print(f"Computing bootstrap confidence intervals for {len(_battles['model'].unique())} models...")
    
    bootstraps = pd.concat([
        _battles.groupby("model").sample(frac=1.0, replace=True).groupby("model").mean()
        for _ in tqdm(range(100), desc="Bootstrap sampling")
    ])
    
    bootstraps["scores"] = bootstraps["scores"].astype(float)
    
    mean_scores = bootstraps.groupby("model").mean().reset_index()
    lower_scores = bootstraps.groupby("model").quantile(0.05).reset_index().rename(columns={"scores": "lower"})
    upper_scores = bootstraps.groupby("model").quantile(0.95).reset_index().rename(columns={"scores": "upper"})
    
    _leaderboard = format_confidence_interval(mean_scores, lower_scores, upper_scores, baseline)
    
    print(f"##### Category: {category} #####")
    print(_leaderboard.to_string())
        

def print_leaderboard_with_style_features(battles, benchmark, category, control_features):        
    style_metadata = get_model_style_metadata(benchmark)
    
    model_features = battles.apply(lambda row: 
        style_metadata[row['model']][row['uid']], 
        axis=1
    ).tolist()
    baseline_features = battles.apply(
        lambda row: style_metadata[JUDGE_SETTINGS[row['category']]["baseline"]][row['uid']], 
        axis=1
    ).tolist()
    
    # remove model path
    battles['model'] = battles['model'].map(lambda x: x.split('/')[-1])
    
    model_feature_tensor = torch.tensor([
        [v if isinstance(v, int) else sum(v.values()) for k, v in metadata.items()]
        for metadata in model_features
    ], dtype=torch.float32)

    baseline_feature_tensor = torch.tensor([
        [v if isinstance(v, int) else sum(v.values()) for k, v in metadata.items()]
        for metadata in baseline_features
    ], dtype=torch.float32)
    
    final_feature_tensor = torch.zeros_like(model_feature_tensor)
    final_feature_tensor[:, 0] = (
        model_feature_tensor[:, 0] - baseline_feature_tensor[:, 0]
    ) / (
        model_feature_tensor[:, 0] + baseline_feature_tensor[:, 0]
    )
    
    model_md_density = model_feature_tensor[:, 1:] / (model_feature_tensor[:, :1] + 1)
    baseline_md_density = baseline_feature_tensor[:, 1:] / (baseline_feature_tensor[:, :1] + 1)
    
    assert not model_md_density.isnan().any()
    assert not baseline_md_density.isnan().any()
    
    final_feature_tensor[:, 1:] = (
        model_md_density - baseline_md_density
    ) / (
        model_md_density + baseline_md_density + 1
    )
    
    assert not final_feature_tensor.isnan().any()
    
    normalized_feature_tensor = (
        final_feature_tensor - torch.mean(final_feature_tensor, axis=0)
    ) / torch.std(
        final_feature_tensor, axis=0
    )
    
    assert not normalized_feature_tensor.isnan().any()
    
    outcomes = torch.tensor(battles.scores.tolist())
    
    assert not outcomes.isnan().any()
    
    model_features, unique_models = one_hot_encode(
        battles.model.tolist(), 
        baseline=JUDGE_SETTINGS[category]["baseline"]
    )
    all_features = torch.cat([model_features, normalized_feature_tensor], dim=1)
    
    assert not all_features.isnan().any()
    
    if "length" in control_features and "markdown" in control_features:
        num_features = 4
    elif "length" in control_features:
        all_features = all_features[:, :1]
        num_features = 1
    elif "markdown" in control_features:
        all_features = all_features[:, 1:]
        num_features = 3
    else:
        assert False, "Invalid control features"
        
    coefs, _ = bootstrap_pairwise_model(all_features, outcomes, loss_type="bt")
    
    _coefs = coefs[:, :-num_features]
    
    table = pd.DataFrame(
        columns=unique_models, 
        data=to_winrate_probabilities(
            _coefs, 
            unique_models,
            baseline_model=JUDGE_SETTINGS[category]["baseline"]
        ).tolist()
    )
    
    _leaderboard = format_confidence_interval(
        table.quantile(0.5).to_frame("scores").reset_index().rename(columns={"index": "model"}), 
        table.quantile(0.05).to_frame("lower").reset_index().rename(columns={"index": "model"}), 
        table.quantile(0.95).to_frame("upper").reset_index().rename(columns={"index": "model"}), 
    )

    print(f"##### Category: {category} #####")
    print(_leaderboard.to_string())
    print(f"Feature Coefs: {torch.quantile(coefs[:, -num_features:], 0.5, axis=0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", type=str, default="arena-hard-v2.0")
    parser.add_argument("--judge-names", "-j", nargs="+", default=["neuralmagic-llama3.1-70b-instruct-fp8"])
    parser.add_argument("--control-features", "-f", nargs="+", default=[])
    parser.add_argument("--category", "-c", nargs="+", default=['hard_prompt'])
    parser.add_argument("--memory-efficient", action="store_true", default=True, 
                       help="Use memory-efficient loading (recommended for large datasets)")
    parser.add_argument("--batch-size", type=int, default=30,
                       help="Number of files to process at once (0 for single-file processing)")
    parser.add_argument("--no-memory-efficient", action="store_true", 
                       help="Disable memory-efficient loading (original behavior)")
    args = parser.parse_args()
    
    # Handle memory efficiency flags
    if args.no_memory_efficient:
        args.memory_efficient = False
    
    print(f"Loading judgments from {args.judge_names} for {args.benchmark}")
    print(f"Memory efficient: {args.memory_efficient}, Batch size: {args.batch_size}")
    
    battles = load_judgments(
        args.judge_names, 
        args.benchmark, 
        memory_efficient=args.memory_efficient,
        batch_size=args.batch_size
    )
    
    if battles.empty:
        print("ERROR: No battles data loaded. Check if judgment files exist.")
        exit(1)
    
    for category in args.category:
        if category not in battles.category.unique():
            print(f"WARNING: Category '{category}' not found. Available categories: {list(battles.category.unique())}")
            continue
            
        category_battles = battles[battles.category == category].reset_index(drop=True)
        
        if category_battles.empty:
            print(f"WARNING: No battles found for category '{category}'")
            continue
        
        print(f"Processing {len(category_battles)} battles for category '{category}'")
        
        if args.control_features:
            print(f"INFO: Control features: {args.control_features}")
            
            print_leaderboard_with_style_features(
                category_battles, 
                args.benchmark, 
                category,
                args.control_features
            )
                
        else:
            print_leaderboard(category_battles, category)
