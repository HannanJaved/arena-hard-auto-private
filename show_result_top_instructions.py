#!/usr/bin/env python3
"""
Arena Hard results processor for top X instructions
Processes only the first X unique instructions (UIDs) found in the first file,
and ensures all other files are filtered to match those instructions.
"""

import pandas as pd
import argparse
import os
from glob import glob
from tqdm import tqdm
import gc

from utils.judge_utils import JUDGE_SETTINGS
from utils.math_utils import one_hot_encode, to_winrate_probabilities, bootstrap_pairwise_model

def get_top_instruction_uids(files, num_instructions):
    """Get the first num_instructions unique instruction UIDs from the first file."""
    if not files:
        return []
    df = pd.read_json(files[0], lines=True)
    uids = df['uid'].tolist()
    # Only keep unique, preserve order
    seen = set()
    top_uids = []
    for uid in uids:
        if uid not in seen:
            seen.add(uid)
            top_uids.append(uid)
        if len(top_uids) >= num_instructions:
            break
    return top_uids

def load_judgments_top_instructions(judge_names, benchmark, baseline, num_instructions, batch_size=30, weight=3):
    """
    Load judgments, but only for the top num_instructions UIDs found in the first file.
    """
    print(f"Loading judgments for top {num_instructions} instructions...")
    all_files = []
    for judge_name in judge_names:
        files = glob(os.path.join(
            "data",
            benchmark,
            "model_judgment",
            judge_name,
            baseline,
            "*.jsonl"
        ))
        all_files.extend(files)
    print(f"Found {len(all_files)} judgment files to process")
    if not all_files:
        print("No judgment files found!")
        return pd.DataFrame()
    # Get top instruction UIDs from the first file
    top_uids = get_top_instruction_uids(all_files, num_instructions)
    print(f"Top {len(top_uids)} instruction UIDs selected.")
    # Process files in batches
    processed_batches = []
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size} ({len(batch_files)} files)")
        batch_dfs = []
        for f in tqdm(batch_files, desc="Loading files"):
            try:
                df = pd.read_json(f, lines=True)
                # Only keep rows with UID in top_uids
                df = df[df['uid'].isin(top_uids)]
                if not df.empty:
                    batch_dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
                continue
        if not batch_dfs:
            continue
        batch_data = pd.concat(batch_dfs, ignore_index=True)
        del batch_dfs
        gc.collect()
        processed_batch = process_batch_data(batch_data, weight)
        processed_batches.append(processed_batch)
        del batch_data
        gc.collect()
        print(f"Batch {i//batch_size + 1} processed: {len(processed_batch)} battles")
    if processed_batches:
        print("Combining all batches...")
        final_battles = pd.concat(processed_batches, ignore_index=True)
        del processed_batches
        gc.collect()
        print(f"Total battles loaded: {len(final_battles)}")
        return final_battles
    else:
        print("No valid judgment data found!")
        return pd.DataFrame()

def process_batch_data(data, weight=3):
    # ...existing code...
    null_indices = data.games.map(lambda x: x[0] is None or x[1] is None or x[0]['score'] is None or x[1]['score'] is None)
    _data = data[~null_indices].reset_index(drop=True)
    if len(data) - len(_data) > 0:
        print(f"  Filtered out {len(data) - len(_data)} null judgments from batch")
    if _data.empty:
        return pd.DataFrame()
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
    def safe_score_mapping(x):
        try:
            score1 = x[0]['score']
            score2 = x[1]['score']
            if score1 not in label_to_score:
                print(f"WARNING: Invalid score '{score1}' found, skipping this judgment")
                return None
            if score2 not in label_to_score:
                print(f"WARNING: Invalid score '{score2}' found, skipping this judgment")
                return None
            return label_to_score[score2] + [1 - s for s in label_to_score[score1]]
        except (KeyError, TypeError) as e:
            print(f"WARNING: Error processing scores: {e}, skipping this judgment")
            return None
    _data['scores'] = _data.games.map(safe_score_mapping)
    initial_count = len(_data)
    _data = _data[_data['scores'].notna()].reset_index(drop=True)
    filtered_count = initial_count - len(_data)
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} judgments with invalid scores from batch")
    battles = _data[['uid', 'model', 'category', 'scores']].explode('scores').reset_index(drop=True)
    return battles

def format_confidence_interval(mean_scores, lower_scores, upper_scores, baseline=None):
    # ...existing code...
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

def print_leaderboard(battles, category, output_csv=False, output_dir=None):
    baseline = JUDGE_SETTINGS[category]["baseline"]
    _battles = battles.drop(columns=['category'])[['model', 'scores']]
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
    if output_csv and output_dir:
        overall_filename = os.path.join(output_dir, f"{category}_leaderboard_top{len(battles['uid'].unique())}.csv")
        _leaderboard.to_csv(overall_filename, index=False)
        print(f"Saved overall leaderboard to: {overall_filename}")
    return _leaderboard

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", type=str, default="arena-hard-v2.0")
    parser.add_argument("--judge-names", "-j", nargs="+", default=["neuralmagic-llama3.1-70b-instruct-fp8"])
    parser.add_argument("--baseline", "-m", type=str, default=None)
    parser.add_argument("--category", "-c", nargs="+", default=['hard_prompt'])
    parser.add_argument("--num-instructions", "-n", type=int, default=200,
                       help="Number of top instructions to process")
    parser.add_argument("--batch-size", type=int, default=30,
                       help="Number of files to process at once")
    parser.add_argument("--output-csv", action="store_true",
                       help="Output results to CSV files")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save CSV files (default: results)")
    args = parser.parse_args()
    if args.output_csv:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"CSV output enabled. Results will be saved to: {args.output_dir}")
    print(f"Loading judgments from {args.judge_names} for {args.benchmark}")
    print(f"Processing only top {args.num_instructions} instructions")
    # combine baseline path as "compared_with_" + "baseline"
    baseline = "compared_with_" + args.baseline

    battles = load_judgments_top_instructions(
        args.judge_names,
        args.benchmark,
        baseline,
        args.num_instructions,
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
        print_leaderboard(
            category_battles,
            category,
            output_csv=args.output_csv,
            output_dir=args.output_dir
        )
