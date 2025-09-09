#!/usr/bin/env python3
"""
Create Missing Judgments List
Creates a plain text list of models that need judgments (missing or incomplete).
"""

import os
import glob
from pathlib import Path

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
JUDGMENT_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto/data/arena-hard-v2.0/model_judgment/neuralmagic-llama3.1-70b-instruct-fp8"
CONFIG_FILE = f"{WORKSPACE_ROOT}/arena_hard_models_to_test.txt"
MISSING_JUDGMENTS_FILE = f"{WORKSPACE_ROOT}/missing_judgments_list.txt"
INCOMPLETE_JUDGMENTS_FILE = f"{WORKSPACE_ROOT}/incomplete_judgments_list.txt"

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

def scan_judgment_directory():
    """Scan the judgment directory for .jsonl files."""
    if not os.path.exists(JUDGMENT_DIR):
        print(f"ERROR: Judgment directory not found: {JUDGMENT_DIR}")
        return {}
    
    # Find all .jsonl files
    jsonl_files = glob.glob(f"{JUDGMENT_DIR}/*.jsonl")
    
    # Extract model names and line counts
    judgment_status = {}
    for file_path in jsonl_files:
        model_name = os.path.basename(file_path).replace('.jsonl', '')
        line_count = count_lines_in_file(file_path)
        judgment_status[model_name] = line_count
    
    return judgment_status

def main():
    print("🔍 Scanning for missing and incomplete judgments...")
    
    # Load models from configuration
    config_models = load_models_from_config()
    if not config_models:
        return
    
    print(f"📋 Found {len(config_models)} models in configuration file")
    
    # Scan judgment directory
    judgment_status = scan_judgment_directory()
    if judgment_status is None:
        return
    
    # Find missing and incomplete judgments
    missing_judgments = []
    incomplete_judgments = []
    
    for model in config_models:
        if model not in judgment_status:
            # No judgment file exists
            missing_judgments.append(model)
        else:
            line_count = judgment_status[model]
            # Arena Hard v2.0 has 750 questions, so we expect ~750 judgments
            if line_count < 735:  # Allow some tolerance
                incomplete_judgments.append((model, line_count))
    
    # Write missing judgments list
    if missing_judgments:
        with open(MISSING_JUDGMENTS_FILE, 'w') as f:
            for model in missing_judgments:
                f.write(f"{model}\n")
        print(f"📄 Created missing judgments list: {MISSING_JUDGMENTS_FILE}")
        print(f"   Contains {len(missing_judgments)} models without judgment files")
    else:
        # Create empty file
        with open(MISSING_JUDGMENTS_FILE, 'w') as f:
            pass
        print(f"✅ No missing judgments found - created empty file: {MISSING_JUDGMENTS_FILE}")
    
    # Write incomplete judgments list
    if incomplete_judgments:
        with open(INCOMPLETE_JUDGMENTS_FILE, 'w') as f:
            for model, line_count in incomplete_judgments:
                f.write(f"{model}\n")
        print(f"📄 Created incomplete judgments list: {INCOMPLETE_JUDGMENTS_FILE}")
        print(f"   Contains {len(incomplete_judgments)} models with incomplete judgments")
        print(f"   Range: {min(lc for _, lc in incomplete_judgments)} - {max(lc for _, lc in incomplete_judgments)} lines (target: 750)")
    else:
        # Create empty file  
        with open(INCOMPLETE_JUDGMENTS_FILE, 'w') as f:
            pass
        print(f"✅ No incomplete judgments found - created empty file: {INCOMPLETE_JUDGMENTS_FILE}")
    
    # Summary
    total_need_judgment = len(missing_judgments) + len(incomplete_judgments)
    complete_judgments = len(config_models) - total_need_judgment
    
    print(f"\n📊 JUDGMENT STATUS SUMMARY:")
    print(f"  ✅ Complete judgments: {complete_judgments}/{len(config_models)} ({complete_judgments/len(config_models)*100:.1f}%)")
    print(f"  ❌ Missing judgments: {len(missing_judgments)}")
    print(f"  ⚠️  Incomplete judgments: {len(incomplete_judgments)}")
    print(f"  🎯 Total needing judgment: {total_need_judgment}")
    
    if total_need_judgment > 0:
        print(f"\n💡 NEXT STEPS:")
        print(f"  1. Generate judgments for missing models:")
        print(f"     python3 automate_arena_hard_judgment.py --missing-models-file missing_judgments_list.txt --submit")
        print(f"  2. Generate judgments for incomplete models:")
        print(f"     python3 automate_arena_hard_judgment.py --missing-models-file incomplete_judgments_list.txt --submit")
        print(f"  3. Or use the enhanced workflow:")
        print(f"     python3 missing_judgments_workflow.py --step all")
    else:
        print(f"\n🎉 ALL JUDGMENTS COMPLETE!")
        print(f"     All {len(config_models)} models have complete judgments.")

if __name__ == "__main__":
    main()
