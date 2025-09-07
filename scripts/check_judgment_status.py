#!/usr/bin/env python3
"""
Arena Hard Judgment Status Checker
Scans the model_judgment directory to check which models have been judged
and compares with the configuration file to identify missing judgments.
"""

import os
import glob
from pathlib import Path
from datetime import datetime

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
JUDGMENT_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto/data/arena-hard-v2.0/model_judgment/neuralmagic-llama3.1-70b-instruct-fp8"
CONFIG_FILE = f"{WORKSPACE_ROOT}/arena_hard_models_to_test.txt"
OUTPUT_FILE = f"{WORKSPACE_ROOT}/judgment_status_report.txt"

def load_models_from_config():
    """Load model names from the configuration file, ignoring comments and empty lines."""
    models = []
    try:
        with open(CONFIG_FILE, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    models.append((line, line_num))
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

def scan_judgment_directory():
    """Scan the judgment directory for .jsonl files."""
    if not os.path.exists(JUDGMENT_DIR):
        print(f"ERROR: Judgment directory not found: {JUDGMENT_DIR}")
        return []
    
    # Find all .jsonl files
    jsonl_files = glob.glob(f"{JUDGMENT_DIR}/*.jsonl")
    
    # Extract model names and file info
    judgment_files = []
    for file_path in jsonl_files:
        model_name = os.path.basename(file_path).replace('.jsonl', '')
        file_size = os.path.getsize(file_path)
        file_mtime = os.path.getmtime(file_path)
        line_count = count_lines_in_file(file_path)
        judgment_files.append((model_name, file_path, file_size, file_mtime, line_count))
    
    return sorted(judgment_files)

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
    """Generate a comprehensive report of judgment status."""
    print("🔍 Arena Hard Judgment Status Checker")
    print("=" * 60)
    
    # Load models from configuration
    config_models = load_models_from_config()
    if not config_models:
        return
    
    config_model_names = {model for model, _ in config_models}
    print(f"📋 Found {len(config_models)} models in configuration file")
    
    # Scan judgment directory
    judgment_files = scan_judgment_directory()
    if judgment_files is None:
        return
    
    judged_model_names = {model_name for model_name, _, _, _, _ in judgment_files}
    print(f"⚖️  Found {len(judgment_files)} judgment files in directory")
    
    # Find missing and extra judgments
    missing_judgments = config_model_names - judged_model_names
    extra_judgments = judged_model_names - config_model_names
    completed_judgments = config_model_names & judged_model_names
    
    print(f"✅ Models judged: {len(completed_judgments)}")
    print(f"❌ Missing judgments: {len(missing_judgments)}")
    print(f"➕ Extra judgments (not in config): {len(extra_judgments)}")
    
    # Check for incomplete judgments (expected ~750 judgments per model)
    incomplete_judgments = []
    complete_judgments = []
    
    for model_name, file_path, file_size, file_mtime, line_count in judgment_files:
        if model_name in completed_judgments:
            # Arena Hard v2.0 has 750 questions, so we expect ~750 judgments
            if line_count < 735:  # Allow some tolerance
                incomplete_judgments.append((model_name, line_count, file_size, file_mtime))
            else:
                complete_judgments.append((model_name, line_count, file_size, file_mtime))
    
    print(f"✅ Complete judgments (700+ lines): {len(complete_judgments)}")
    print(f"⚠️  Incomplete judgments (< 735 lines): {len(incomplete_judgments)}")

    # Create detailed report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ARENA HARD JUDGMENT STATUS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Configuration file: {CONFIG_FILE}")
    report_lines.append(f"Judgment directory: {JUDGMENT_DIR}")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total models in config: {len(config_models)}")
    report_lines.append(f"Models with judgments: {len(completed_judgments)}")
    report_lines.append(f"  - Complete (700+ judgments): {len(complete_judgments)}")
    report_lines.append(f"  - Incomplete (< 700 judgments): {len(incomplete_judgments)}")
    report_lines.append(f"Missing judgments: {len(missing_judgments)}")
    report_lines.append(f"Extra judgments: {len(extra_judgments)}")
    report_lines.append(f"Judgment coverage: {len(completed_judgments)/len(config_models)*100:.1f}%")
    report_lines.append(f"Complete coverage: {len(complete_judgments)/len(config_models)*100:.1f}%")
    report_lines.append("")
    
    # Missing judgments section
    if missing_judgments:
        report_lines.append("MISSING JUDGMENTS (No judgment files found)")
        report_lines.append("-" * 50)
        
        # Get missing models with line numbers
        missing_with_lines = [(model, line_num) for model, line_num in config_models if model in missing_judgments]
        
        # Categorize missing models
        missing_categories = categorize_models_by_pattern(missing_with_lines)
        
        for rank in ['rank64', 'rank256', 'rank1024']:
            if any(missing_categories[rank]['alpha1e5'].values()) or \
               any(missing_categories[rank]['alpha1e6'].values()) or \
               any(missing_categories[rank]['alpha5e5'].values()) or \
               missing_categories[rank]['default']:
                
                report_lines.append(f"\n{rank.upper()}:")
                
                # Alpha variants
                for alpha in ['alpha1e5', 'alpha1e6', 'alpha5e5']:
                    for weight in ['001', '005', '010']:
                        if missing_categories[rank][alpha][weight]:
                            report_lines.append(f"  {alpha}-{weight}: {len(missing_categories[rank][alpha][weight])} missing")
                            for model, line_num in missing_categories[rank][alpha][weight]:
                                report_lines.append(f"    Line {line_num:3d}: {model}")
                
                # Default models
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
    
    # Incomplete judgments section
    if incomplete_judgments:
        report_lines.append("INCOMPLETE JUDGMENTS (Less than 700 judgments)")
        report_lines.append("-" * 50)
        
        for model, line_count, file_size, file_mtime in sorted(incomplete_judgments, key=lambda x: x[1]):
            report_lines.append(f"  {model}: {line_count} judgments ({format_file_size(file_size)})")
        
        report_lines.append("")
    
    # Complete judgments section (summary)
    if complete_judgments:
        report_lines.append("COMPLETE JUDGMENTS (Summary)")
        report_lines.append("-" * 40)
        
        # File size statistics
        if complete_judgments:
            total_size = sum(size for _, _, size, _ in complete_judgments)
            avg_judgments = sum(lines for _, lines, _, _ in complete_judgments) / len(complete_judgments)
            min_judgments = min(lines for _, lines, _, _ in complete_judgments)
            max_judgments = max(lines for _, lines, _, _ in complete_judgments)
            
            report_lines.append(f"Total complete judgments: {len(complete_judgments)}")
            report_lines.append(f"Total size: {format_file_size(total_size)}")
            report_lines.append(f"Average judgments per model: {avg_judgments:.0f}")
            report_lines.append(f"Judgment count range: {min_judgments} - {max_judgments}")
        
        report_lines.append("")
    
    # Extra judgments section
    if extra_judgments:
        report_lines.append("EXTRA JUDGMENTS (Not in configuration)")
        report_lines.append("-" * 45)
        for model_name, file_path, file_size, file_mtime, line_count in judgment_files:
            if model_name in extra_judgments:
                report_lines.append(f"{model_name} ({format_file_size(file_size)}, {line_count} judgments)")
        report_lines.append("")
    
    # Detailed file information for complete judgments
    if complete_judgments:
        report_lines.append("DETAILED JUDGMENT INFORMATION")
        report_lines.append("-" * 35)
        report_lines.append("Model Name".ljust(50) + " | Size     | Judgments | Last Modified")
        report_lines.append("-" * 95)
        
        for model_name, file_path, file_size, file_mtime, line_count in sorted(judgment_files):
            if model_name in completed_judgments and line_count >= 700:
                report_lines.append(f"{model_name:<50} | {format_file_size(file_size):>8} | {line_count:>9} | {format_timestamp(file_mtime)}")
    
    # Write report to file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\n📄 Detailed report saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Failed to write report file: {e}")
        return
    
    # Display summary on console
    if missing_judgments:
        print(f"\n❌ MISSING JUDGMENTS ({len(missing_judgments)}):")
        missing_list = sorted(list(missing_judgments))
        for i, model in enumerate(missing_list[:10]):  # Show first 10
            print(f"  {i+1:2d}. {model}")
        if len(missing_list) > 10:
            print(f"  ... and {len(missing_list) - 10} more (see report file for full list)")
    
    if incomplete_judgments:
        print(f"\n⚠️  INCOMPLETE JUDGMENTS ({len(incomplete_judgments)}):")
        incomplete_list = sorted(incomplete_judgments, key=lambda x: x[1])  # Sort by judgment count
        for i, (model, lines, _, _) in enumerate(incomplete_list[:10]):  # Show first 10
            print(f"  {i+1:2d}. {model} ({lines} judgments)")
        if len(incomplete_list) > 10:
            print(f"  ... and {len(incomplete_list) - 10} more (see report file for full list)")
    
    if extra_judgments:
        print(f"\n➕ EXTRA JUDGMENTS ({len(extra_judgments)}):")
        extra_list = sorted(list(extra_judgments))
        for i, model in enumerate(extra_list[:5]):  # Show first 5
            print(f"  {i+1}. {model}")
        if len(extra_list) > 5:
            print(f"  ... and {len(extra_list) - 5} more")
    
    print(f"\n📊 JUDGMENT STATUS:")
    print(f"  ✅ Complete: {len(complete_judgments)}/{len(config_models)} ({len(complete_judgments)/len(config_models)*100:.1f}%)")
    print(f"  ⚠️  Incomplete: {len(incomplete_judgments)}")
    print(f"  ❌ Missing: {len(missing_judgments)}")
    
    # Suggestions
    if missing_judgments or incomplete_judgments:
        print(f"\n💡 NEXT STEPS:")
        print(f"  1. Review missing/incomplete judgments in: {OUTPUT_FILE}")
        if missing_judgments:
            print(f"  2. Generate judgments for missing models:")
            print(f"     python3 automate_arena_hard_judgment.py --submit --batch-size 10")
        if incomplete_judgments:
            print(f"  3. Re-generate judgments for incomplete models:")
            print(f"     # Create a list of incomplete models and re-run judgment")
        print(f"  4. Monitor progress:")
        print(f"     squeue -u $USER")
        print(f"     python3 monitor_arena_hard_judgments.py")
    else:
        print(f"\n🎉 ALL JUDGMENTS COMPLETE!")
        print(f"     All {len(config_models)} models have been successfully judged.")
        print(f"     Ready for results analysis!")

def main():
    generate_report()

if __name__ == "__main__":
    main()
