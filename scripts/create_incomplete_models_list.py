#!/usr/bin/env python3
"""
Create Incomplete Models List
Generates a simple list of models that have incomplete answers (< 750 lines).
This list can be used with automate_arena_hard_generation.py to re-generate
incomplete answers.
"""

import os
import glob
from pathlib import Path

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
MODEL_ANSWER_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto/data/arena-hard-v2.0/model_answer"
CONFIG_FILE = f"{WORKSPACE_ROOT}/arena_hard_models_to_test.txt"
OUTPUT_FILE = f"{WORKSPACE_ROOT}/incomplete_models_list.txt"
MIN_LINES = 750  # Expected minimum lines for complete answers

def load_models_from_config():
    """Load model names from the configuration file, ignoring comments and empty lines."""
    models = []
    try:
        with open(CONFIG_FILE, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    models.append(line)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {CONFIG_FILE}")
        return []
    return models

def count_lines_in_file(file_path):
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except:
        return 0

def scan_for_incomplete_models():
    """Scan for models that have incomplete answers (less than MIN_LINES)."""
    if not os.path.exists(MODEL_ANSWER_DIR):
        print(f"ERROR: Model answer directory not found: {MODEL_ANSWER_DIR}")
        return []
    
    # Load models from configuration
    config_models = set(load_models_from_config())
    if not config_models:
        return []
    
    print(f"📋 Checking {len(config_models)} models from configuration...")
    
    # Find all .jsonl files
    jsonl_files = glob.glob(f"{MODEL_ANSWER_DIR}/*.jsonl")
    print(f"📁 Found {len(jsonl_files)} .jsonl files to scan...")
    
    incomplete_models = []
    checked = 0
    
    for file_path in jsonl_files:
        model_name = os.path.basename(file_path).replace('.jsonl', '')
        
        # Only check models that are in our configuration
        if model_name in config_models:
            checked += 1
            if checked % 10 == 0:
                print(f"   Checked {checked}/{len(config_models)} models...")
            
            line_count = count_lines_in_file(file_path)
            if line_count < MIN_LINES:
                incomplete_models.append((model_name, line_count))
    
    print(f"✅ Scanned {checked} configured models")
    return sorted(incomplete_models, key=lambda x: x[1])  # Sort by line count

def main():
    print("🔍 Scanning for Incomplete Model Answers...")
    print("=" * 50)
    
    incomplete_models = scan_for_incomplete_models()
    
    if not incomplete_models:
        print("✅ No incomplete models found! All models have 750+ lines.")
        # Create empty file to indicate completion
        with open(OUTPUT_FILE, 'w') as f:
            f.write("")
        print(f"📄 Empty list saved to: {OUTPUT_FILE}")
        return
    
    print(f"⚠️  Found {len(incomplete_models)} incomplete models:")
    print(f"   (Models with less than {MIN_LINES} lines)")
    print()
    
    # Display incomplete models with line counts
    for i, (model, lines) in enumerate(incomplete_models[:10], 1):
        print(f"  {i:2d}. {model} ({lines} lines)")
    
    if len(incomplete_models) > 10:
        print(f"   ... and {len(incomplete_models) - 10} more")
    
    # Save to file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            for model, lines in incomplete_models:
                f.write(f"{model}\n")
        
        print(f"\n📄 Incomplete models list saved to: {OUTPUT_FILE}")
        print(f"📊 Total incomplete models: {len(incomplete_models)}")
        
        # Show statistics
        if incomplete_models:
            min_lines = min(lines for _, lines in incomplete_models)
            max_lines = max(lines for _, lines in incomplete_models)
            avg_lines = sum(lines for _, lines in incomplete_models) / len(incomplete_models)
            
            print(f"\n📈 Line Count Statistics:")
            print(f"   Min: {min_lines} lines")
            print(f"   Max: {max_lines} lines") 
            print(f"   Avg: {avg_lines:.1f} lines")
            print(f"   Target: {MIN_LINES} lines")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Re-generate incomplete models:")
        print(f"      python3 automate_arena_hard_generation.py --missing-models-file {os.path.basename(OUTPUT_FILE)} --submit")
        print(f"   2. Monitor progress:")
        print(f"      python3 monitor_arena_hard_jobs.py")
        print(f"   3. Re-check completion:")
        print(f"      python3 create_incomplete_models_list.py")
        
    except Exception as e:
        print(f"❌ Failed to write output file: {e}")

if __name__ == "__main__":
    main()
