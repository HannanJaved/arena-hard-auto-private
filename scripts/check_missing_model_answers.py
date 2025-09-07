#!/usr/bin/env python3
"""
Arena Hard Model Answer Status Checker
Scans the model_answer directory and compares with arena_hard_models_to_test.txt
to identify which models are missing answers.
"""

import os
import glob
from pathlib import Path
from datetime import datetime

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
MODEL_ANSWER_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto/data/arena-hard-v2.0/model_answer"
CONFIG_FILE = f"{WORKSPACE_ROOT}/arena_hard_models_to_test.txt"
OUTPUT_FILE = f"{WORKSPACE_ROOT}/missing_model_answers_report.txt"

def load_models_from_config():
    """Load model names from the configuration file, ignoring comments and empty lines."""
    models = []
    try:
        with open(CONFIG_FILE, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    models.append((line, line_num))
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {CONFIG_FILE}")
        return []
    return models

def scan_model_answer_directory():
    """Scan the model_answer directory for .jsonl files."""
    if not os.path.exists(MODEL_ANSWER_DIR):
        print(f"ERROR: Model answer directory not found: {MODEL_ANSWER_DIR}")
        return [], []
    
    # Find all .jsonl files
    jsonl_files = glob.glob(f"{MODEL_ANSWER_DIR}/*.jsonl")
    
    # Extract model names (filename without .jsonl extension)
    model_files = []
    for file_path in jsonl_files:
        model_name = os.path.basename(file_path).replace('.jsonl', '')
        file_size = os.path.getsize(file_path)
        file_mtime = os.path.getmtime(file_path)
        model_files.append((model_name, file_path, file_size, file_mtime))
    
    return sorted(model_files), sorted(jsonl_files)

def count_lines_in_file(file_path):
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except:
        return 0

def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def format_timestamp(timestamp):
    """Format timestamp to readable date."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def categorize_models_by_pattern(models):
    """Categorize models by their pattern (rank, alpha, etc.)."""
    categories = {
        'rank64': {'alpha1e5': {'001': [], '005': [], '010': []}, 
                   'alpha1e6': {'001': [], '005': [], '010': []},
                   'alpha5e5': {'001': [], '005': [], '010': []},
                   'default': []},
        'rank256': {'alpha1e5': {'001': [], '005': [], '010': []}, 
                    'alpha1e6': {'001': [], '005': [], '010': []},
                    'alpha5e5': {'001': [], '005': [], '010': []},
                    'default': []},
        'rank1024': {'alpha1e5': {'001': [], '005': [], '010': []}, 
                     'alpha1e6': {'001': [], '005': [], '010': []},
                     'alpha5e5': {'001': [], '005': [], '010': []},
                     'default': []},
        'other': []
    }
    
    for model, line_num in models:
        # Parse model name
        if 'rank64' in model:
            rank = 'rank64'
        elif 'rank256' in model:
            rank = 'rank256'
        elif 'rank1024' in model:
            rank = 'rank1024'
        else:
            categories['other'].append((model, line_num))
            continue
        
        if 'default' in model:
            categories[rank]['default'].append((model, line_num))
        elif 'alpha1e5' in model:
            if '001' in model:
                categories[rank]['alpha1e5']['001'].append((model, line_num))
            elif '005' in model:
                categories[rank]['alpha1e5']['005'].append((model, line_num))
            elif '010' in model:
                categories[rank]['alpha1e5']['010'].append((model, line_num))
        elif 'alpha1e6' in model:
            if '001' in model:
                categories[rank]['alpha1e6']['001'].append((model, line_num))
            elif '005' in model:
                categories[rank]['alpha1e6']['005'].append((model, line_num))
            elif '010' in model:
                categories[rank]['alpha1e6']['010'].append((model, line_num))
        elif 'alpha5e5' in model:
            if '001' in model:
                categories[rank]['alpha5e5']['001'].append((model, line_num))
            elif '005' in model:
                categories[rank]['alpha5e5']['005'].append((model, line_num))
            elif '010' in model:
                categories[rank]['alpha5e5']['010'].append((model, line_num))
        else:
            categories['other'].append((model, line_num))
    
    return categories

def generate_report():
    """Generate a comprehensive report of missing model answers."""
    print("🔍 Arena Hard Model Answer Status Checker")
    print("=" * 60)
    
    # Load models from configuration
    config_models = load_models_from_config()
    if not config_models:
        return
    
    config_model_names = {model for model, _ in config_models}
    print(f"📋 Found {len(config_models)} models in configuration file")
    
    # Scan model answer directory
    model_files, all_files = scan_model_answer_directory()
    if model_files is None:
        return
    
    available_model_names = {model_name for model_name, _, _, _ in model_files}
    print(f"📁 Found {len(model_files)} .jsonl files in model_answer directory")
    
    # Find missing and extra models
    missing_models = config_model_names - available_model_names
    extra_models = available_model_names - config_model_names
    present_models = config_model_names & available_model_names
    
    print(f"✅ Models with answers: {len(present_models)}")
    print(f"❌ Missing models: {len(missing_models)}")
    print(f"➕ Extra models (not in config): {len(extra_models)}")
    
    # Create detailed report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ARENA HARD MODEL ANSWER STATUS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Configuration file: {CONFIG_FILE}")
    report_lines.append(f"Model answer directory: {MODEL_ANSWER_DIR}")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total models in config: {len(config_models)}")
    report_lines.append(f"Models with answers: {len(present_models)}")
    report_lines.append(f"Missing models: {len(missing_models)}")
    report_lines.append(f"Extra models: {len(extra_models)}")
    report_lines.append(f"Coverage: {len(present_models)/len(config_models)*100:.1f}%")
    report_lines.append("")
    
    # Missing models section
    if missing_models:
        report_lines.append("MISSING MODELS (No answer files found)")
        report_lines.append("-" * 50)
        
        # Get missing models with line numbers
        missing_with_lines = [(model, line_num) for model, line_num in config_models if model in missing_models]
        
        # Categorize missing models
        missing_categories = categorize_models_by_pattern(missing_with_lines)
        
        for rank in ['rank64', 'rank256', 'rank1024']:
            if any(missing_categories[rank]['alpha1e5'].values()) or \
               any(missing_categories[rank]['alpha1e6'].values()) or \
               any(missing_categories[rank]['alpha5e5'].values()) or \
               missing_categories[rank]['default']:
                
                report_lines.append(f"\n{rank.upper()}:")
                
                # Alpha configurations
                for alpha in ['alpha1e5', 'alpha1e6', 'alpha5e5']:
                    for wr in ['001', '005', '010']:
                        if missing_categories[rank][alpha][wr]:
                            report_lines.append(f"  {alpha}-{wr}: {len(missing_categories[rank][alpha][wr])} missing")
                            for model, line_num in missing_categories[rank][alpha][wr]:
                                report_lines.append(f"    Line {line_num:3d}: {model}")
                
                # Default configuration
                if missing_categories[rank]['default']:
                    report_lines.append(f"  default: {len(missing_categories[rank]['default'])} missing")
                    for model, line_num in missing_categories[rank]['default']:
                        report_lines.append(f"    Line {line_num:3d}: {model}")
        
        # Other models
        if missing_categories['other']:
            report_lines.append(f"\nOTHER:")
            for model, line_num in missing_categories['other']:
                report_lines.append(f"  Line {line_num:3d}: {model}")
        
        report_lines.append("")
    
    # Present models section (summary)
    if present_models:
        report_lines.append("MODELS WITH ANSWERS (Summary)")
        report_lines.append("-" * 40)
        
        # Get present models with details
        present_with_details = []
        for model_name, file_path, file_size, file_mtime in model_files:
            if model_name in present_models:
                line_count = count_lines_in_file(file_path)
                present_with_details.append((model_name, file_size, file_mtime, line_count))
        
        # Categorize present models
        present_model_lines = [(model, 0) for model, _, _, _ in present_with_details]  # Line numbers not needed for summary
        present_categories = categorize_models_by_pattern(present_model_lines)
        
        for rank in ['rank64', 'rank256', 'rank1024']:
            rank_total = 0
            for alpha in ['alpha1e5', 'alpha1e6', 'alpha5e5']:
                for wr in ['001', '005', '010']:
                    rank_total += len(present_categories[rank][alpha][wr])
            rank_total += len(present_categories[rank]['default'])
            
            if rank_total > 0:
                report_lines.append(f"{rank.upper()}: {rank_total} models")
        
        if present_categories['other']:
            report_lines.append(f"OTHER: {len(present_categories['other'])} models")
        
        # File size statistics
        if present_with_details:
            total_size = sum(size for _, size, _, _ in present_with_details)
            avg_lines = sum(lines for _, _, _, lines in present_with_details) / len(present_with_details)
            report_lines.append(f"\nTotal size: {format_file_size(total_size)}")
            report_lines.append(f"Average lines per file: {avg_lines:.0f}")
        
        report_lines.append("")
    
    # Extra models section
    if extra_models:
        report_lines.append("EXTRA MODELS (Not in configuration)")
        report_lines.append("-" * 45)
        for model_name, file_path, file_size, file_mtime in model_files:
            if model_name in extra_models:
                line_count = count_lines_in_file(file_path)
                report_lines.append(f"{model_name} ({format_file_size(file_size)}, {line_count} lines)")
        report_lines.append("")
    
    # Detailed file information
    if present_models:
        report_lines.append("DETAILED FILE INFORMATION")
        report_lines.append("-" * 30)
        report_lines.append("Model Name".ljust(50) + " | Size     | Lines | Last Modified")
        report_lines.append("-" * 90)
        
        for model_name, file_path, file_size, file_mtime in sorted(model_files):
            if model_name in present_models:
                line_count = count_lines_in_file(file_path)
                size_str = format_file_size(file_size)
                time_str = format_timestamp(file_mtime)
                report_lines.append(f"{model_name.ljust(49)} | {size_str.rjust(8)} | {str(line_count).rjust(5)} | {time_str}")
    
    # Write report to file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\n📄 Detailed report saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Failed to write report file: {e}")
        return
    
    # Display summary on console
    if missing_models:
        print(f"\n❌ MISSING MODELS ({len(missing_models)}):")
        missing_list = sorted(list(missing_models))
        for i, model in enumerate(missing_list[:10]):  # Show first 10
            print(f"  {i+1:2d}. {model}")
        if len(missing_list) > 10:
            print(f"  ... and {len(missing_list) - 10} more (see report file for full list)")
    
    if extra_models:
        print(f"\n➕ EXTRA MODELS ({len(extra_models)}):")
        extra_list = sorted(list(extra_models))
        for i, model in enumerate(extra_list[:5]):  # Show first 5
            print(f"  {i+1}. {model}")
        if len(extra_list) > 5:
            print(f"  ... and {len(extra_list) - 5} more")
    
    print(f"\n📊 COMPLETION STATUS:")
    print(f"  ✅ Complete: {len(present_models)}/{len(config_models)} ({len(present_models)/len(config_models)*100:.1f}%)")
    print(f"  ❌ Missing: {len(missing_models)}")
    
    # Suggestions
    if missing_models:
        print(f"\n💡 NEXT STEPS:")
        print(f"  1. Review missing models in: {OUTPUT_FILE}")
        print(f"  2. Generate answers for missing models:")
        print(f"     python3 automate_arena_hard_generation.py --submit")
        print(f"  3. Monitor progress:")
        print(f"     python3 monitor_arena_hard_jobs.py")

def main():
    generate_report()

if __name__ == "__main__":
    main()
