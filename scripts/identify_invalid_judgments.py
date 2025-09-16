#!/usr/bin/env python3
"""
Invalid Judgments Detection and Regeneration Script
Identifies judgment files with invalid score labels and creates lists for regeneration.
"""

import json
import os
import glob
import argparse
from pathlib import Path
from collections import defaultdict

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
JUDGMENT_BASE_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto/data/arena-hard-v2.0/model_judgment"
JUDGE_MODEL = "neuralmagic-llama3.1-70b-instruct-fp8"
BASELINE_DIRS = ["compared_with_instruct", "compared_with_base", "compared_with_tulu_finetuned"]

# Valid judgment scores based on the label_to_score mapping
VALID_SCORES = {
    "A>B", "A>>B", "A=B", "A<<B", "A<B",
    "B>A", "B>>A", "B=A", "B<<A", "B<A"
}

def count_lines_in_file(file_path):
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except:
        return 0

def analyze_judgment_file(file_path):
    """Analyze a judgment file for invalid scores and missing judgments."""
    results = {
        'total_lines': 0,
        'valid_judgments': 0,
        'invalid_judgments': 0,
        'missing_judgments': 0,
        'invalid_details': [],
        'file_corrupted': False
    }
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                results['total_lines'] += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    games = data.get('games', [])
                    
                    if not games or len(games) != 2:
                        results['missing_judgments'] += 1
                        results['invalid_details'].append(f"Line {line_num}: Missing or incomplete games")
                        continue
                    
                    line_valid = True
                    for game_idx, game in enumerate(games):
                        score = game.get('score')
                        if score not in VALID_SCORES:
                            results['invalid_judgments'] += 1
                            results['invalid_details'].append(
                                f"Line {line_num}, Game {game_idx + 1}: Invalid score '{score}'"
                            )
                            line_valid = False
                    
                    if line_valid:
                        results['valid_judgments'] += 1
                        
                except json.JSONDecodeError as e:
                    results['invalid_judgments'] += 1
                    results['invalid_details'].append(f"Line {line_num}: JSON decode error - {e}")
                except Exception as e:
                    results['invalid_judgments'] += 1
                    results['invalid_details'].append(f"Line {line_num}: Error - {e}")
                    
    except Exception as e:
        results['file_corrupted'] = True
        results['invalid_details'].append(f"File error: {e}")
    
    return results

def scan_judgment_files():
    """Scan all judgment files for invalid judgments."""
    print("🔍 Scanning judgment files for invalid scores...")
    
    all_results = {}
    
    for baseline_dir in BASELINE_DIRS:
        judgment_dir = os.path.join(JUDGMENT_BASE_DIR, JUDGE_MODEL, baseline_dir)
        
        if not os.path.exists(judgment_dir):
            print(f"⚠️  Judgment directory not found: {judgment_dir}")
            continue
        
        print(f"\n📂 Scanning {baseline_dir}...")
        
        # Find all .jsonl files
        jsonl_files = glob.glob(os.path.join(judgment_dir, "*.jsonl"))
        print(f"   Found {len(jsonl_files)} judgment files")
        
        baseline_results = {}
        
        for file_path in sorted(jsonl_files):
            model_name = os.path.basename(file_path).replace('.jsonl', '')
            results = analyze_judgment_file(file_path)
            baseline_results[model_name] = {
                'file_path': file_path,
                'results': results
            }
        
        all_results[baseline_dir] = baseline_results
    
    return all_results

def generate_invalid_models_lists(all_results):
    """Generate lists of models with invalid judgments for regeneration."""
    
    invalid_models_by_baseline = {}
    
    for baseline_dir, models in all_results.items():
        invalid_models = []
        
        for model_name, model_data in models.items():
            results = model_data['results']
            
            # Consider a model invalid if it has any invalid judgments or is corrupted
            if results['invalid_judgments'] > 0 or results['file_corrupted']:
                invalid_models.append(model_name)
        
        invalid_models_by_baseline[baseline_dir] = invalid_models
    
    # Write separate lists for each baseline
    for baseline_dir, invalid_models in invalid_models_by_baseline.items():
        if invalid_models:
            list_filename = f"{WORKSPACE_ROOT}/invalid_judgments_{baseline_dir.replace('compared_with_', '')}.txt"
            with open(list_filename, 'w') as f:
                for model in sorted(invalid_models):
                    f.write(f"{model}\n")
            print(f"📄 Created invalid judgments list: {list_filename}")
            print(f"   Contains {len(invalid_models)} models with invalid judgments")
        else:
            # Create empty file
            list_filename = f"{WORKSPACE_ROOT}/invalid_judgments_{baseline_dir.replace('compared_with_', '')}.txt"
            with open(list_filename, 'w') as f:
                pass
            print(f"✅ No invalid judgments found for {baseline_dir} - created empty file: {list_filename}")
    
    return invalid_models_by_baseline

def print_detailed_report(all_results):
    """Print a detailed report of invalid judgments."""
    
    print(f"\n📊 INVALID JUDGMENTS DETAILED REPORT:")
    print("=" * 80)
    
    total_files = 0
    total_invalid_files = 0
    total_invalid_judgments = 0
    
    for baseline_dir, models in all_results.items():
        print(f"\n🎯 BASELINE: {baseline_dir}")
        print("-" * 60)
        
        baseline_invalid_files = 0
        baseline_invalid_judgments = 0
        
        for model_name, model_data in models.items():
            results = model_data['results']
            total_files += 1
            
            has_issues = results['invalid_judgments'] > 0 or results['file_corrupted']
            
            if has_issues:
                total_invalid_files += 1
                baseline_invalid_files += 1
                total_invalid_judgments += results['invalid_judgments']
                baseline_invalid_judgments += results['invalid_judgments']
                
                status = "❌ CORRUPTED" if results['file_corrupted'] else "⚠️  INVALID"
                
                print(f"{status} {model_name}")
                print(f"    Total lines: {results['total_lines']}")
                print(f"    Valid judgments: {results['valid_judgments']}")
                print(f"    Invalid judgments: {results['invalid_judgments']}")
                print(f"    Missing judgments: {results['missing_judgments']}")
                
                # Show first few invalid details
                if results['invalid_details']:
                    print(f"    Sample issues:")
                    for detail in results['invalid_details'][:3]:  # Show first 3 issues
                        print(f"      • {detail}")
                    if len(results['invalid_details']) > 3:
                        print(f"      • ... and {len(results['invalid_details']) - 3} more issues")
                print()
        
        print(f"BASELINE SUMMARY - {baseline_dir}:")
        print(f"  Files with issues: {baseline_invalid_files}/{len(models)}")
        print(f"  Total invalid judgments: {baseline_invalid_judgments}")
    
    print(f"\n🔍 OVERALL SUMMARY:")
    print(f"  Total files scanned: {total_files}")
    print(f"  Files with invalid judgments: {total_invalid_files}")
    print(f"  Total invalid judgments: {total_invalid_judgments}")
    print(f"  Success rate: {((total_files - total_invalid_files) / total_files * 100):.1f}%")

def print_valid_scores_reference():
    """Print the reference of valid scores."""
    print(f"\n📋 VALID JUDGMENT SCORES REFERENCE:")
    print("Valid scores that should appear in judgment files:")
    for score in sorted(VALID_SCORES):
        print(f"  • {score}")
    print(f"\nTotal valid scores: {len(VALID_SCORES)}")
    print("Any other score values are considered invalid and need regeneration.")

def backup_invalid_files(all_results):
    """Create backups of files with invalid judgments before regeneration."""
    backup_dir = f"{WORKSPACE_ROOT}/invalid_judgments_backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    backed_up_files = []
    
    for baseline_dir, models in all_results.items():
        for model_name, model_data in models.items():
            results = model_data['results']
            
            if results['invalid_judgments'] > 0 or results['file_corrupted']:
                source_file = model_data['file_path']
                backup_subdir = os.path.join(backup_dir, baseline_dir)
                os.makedirs(backup_subdir, exist_ok=True)
                
                backup_file = os.path.join(backup_subdir, f"{model_name}.jsonl")
                
                try:
                    import shutil
                    shutil.copy2(source_file, backup_file)
                    backed_up_files.append(backup_file)
                except Exception as e:
                    print(f"⚠️  Failed to backup {source_file}: {e}")
    
    if backed_up_files:
        print(f"\n💾 BACKUP CREATED:")
        print(f"  Backed up {len(backed_up_files)} invalid judgment files to:")
        print(f"  {backup_dir}")
    
    return backed_up_files

def remove_invalid_judgments(file_path):
    """Remove lines with invalid judgments from a judgment file."""
    temp_file = f"{file_path}.tmp"
    removed_lines = 0
    kept_lines = 0
    
    try:
        with open(file_path, 'r') as infile, open(temp_file, 'w') as outfile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    games = data.get('games', [])
                    
                    # Check if this line has valid judgments
                    line_valid = True
                    if not games or len(games) != 2:
                        line_valid = False
                    else:
                        for game in games:
                            score = game.get('score')
                            if score not in VALID_SCORES:
                                line_valid = False
                                break
                    
                    if line_valid:
                        outfile.write(line + '\n')
                        kept_lines += 1
                    else:
                        removed_lines += 1
                        
                except (json.JSONDecodeError, Exception):
                    # Remove corrupted lines
                    removed_lines += 1
        
        # Replace original file with cleaned version
        if removed_lines > 0:
            import shutil
            shutil.move(temp_file, file_path)
            return removed_lines, kept_lines
        else:
            # No changes needed, remove temp file
            os.remove(temp_file)
            return 0, kept_lines
            
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

def clean_invalid_judgments(all_results):
    """Remove invalid judgments from files, making them appear as incomplete."""
    print(f"\n🧹 CLEANING INVALID JUDGMENTS FROM FILES...")
    
    cleaned_files = []
    total_removed = 0
    
    for baseline_dir, models in all_results.items():
        for model_name, model_data in models.items():
            results = model_data['results']
            
            if results['invalid_judgments'] > 0 or results['file_corrupted']:
                file_path = model_data['file_path']
                
                try:
                    removed, kept = remove_invalid_judgments(file_path)
                    if removed > 0:
                        cleaned_files.append((model_name, file_path, removed, kept))
                        total_removed += removed
                        print(f"  ✅ {model_name}: Removed {removed} invalid judgments, kept {kept} valid ones")
                    else:
                        print(f"  ⚠️  {model_name}: No invalid judgments found to remove")
                        
                except Exception as e:
                    print(f"  ❌ {model_name}: Failed to clean file - {e}")
    
    if cleaned_files:
        print(f"\n🧹 CLEANING SUMMARY:")
        print(f"  Files cleaned: {len(cleaned_files)}")
        print(f"  Total invalid judgments removed: {total_removed}")
        print(f"  These files now appear incomplete and will be detected by the missing judgments workflow")
    else:
        print(f"  No files needed cleaning")
    
    return cleaned_files

def main():
    parser = argparse.ArgumentParser(description='Identify and help regenerate invalid Arena Hard judgments')
    parser.add_argument('--baseline', type=str, choices=['instruct', 'base', 'tulu_finetuned'], 
                       help='Only check specific baseline (default: check all)')
    parser.add_argument('--clean', action='store_true', 
                       help='Remove invalid judgments from files (makes them appear incomplete)')
    parser.add_argument('--regenerate', action='store_true', 
                       help='Automatically regenerate invalid judgments using existing workflow')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Create backup of invalid files before cleaning (default: True)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    print("🔍 Invalid Judgments Detection and Analysis")
    print("=" * 50)
    
    # Print valid scores reference
    print_valid_scores_reference()
    
    # Scan all judgment files (or specific baseline)
    global BASELINE_DIRS
    original_baseline_dirs = BASELINE_DIRS
    
    if args.baseline:
        baseline_dirs = [f"compared_with_{args.baseline}"]
        print(f"\n🎯 Checking only baseline: {args.baseline}")
        BASELINE_DIRS = baseline_dirs
    else:
        baseline_dirs = BASELINE_DIRS
        print(f"\n🌐 Checking all baselines")
    
    all_results = scan_judgment_files()
    
    # Restore original baseline dirs
    BASELINE_DIRS = original_baseline_dirs
    
    if not any(all_results.values()):
        print("❌ No judgment files found to analyze")
        return
    
    # Generate lists of models with invalid judgments
    invalid_models_by_baseline = generate_invalid_models_lists(all_results)
    
    # Print detailed report
    print_detailed_report(all_results)
    
    # Summary and next steps
    total_invalid_models = sum(len(models) for models in invalid_models_by_baseline.values())
    
    if total_invalid_models > 0:
        print(f"\n💡 NEXT STEPS:")
        print(f"  Total models needing regeneration: {total_invalid_models}")
        
        if args.clean or args.regenerate:
            if not args.baseline:
                print("❌ ERROR: --clean and --regenerate require --baseline to be specified")
                return
            
            baseline_dir = f"compared_with_{args.baseline}"
            if baseline_dir in invalid_models_by_baseline and invalid_models_by_baseline[baseline_dir]:
                invalid_models = invalid_models_by_baseline[baseline_dir]
                
                if args.backup and not args.dry_run:
                    backed_up_files = backup_invalid_files(all_results)
                
                if args.clean:
                    print(f"\n🧹 CLEANING INVALID JUDGMENTS for {args.baseline} baseline...")
                    print(f"   Models to clean: {len(invalid_models)}")
                    
                    if args.dry_run:
                        print("   DRY RUN - would remove invalid judgments from:")
                        for model in invalid_models:
                            print(f"     • {model}")
                    else:
                        cleaned_files = clean_invalid_judgments(all_results)
                        print(f"\n✅ Files have been cleaned and now appear incomplete.")
                        print(f"   Use existing missing judgments workflow to regenerate:")
                        print(f"   python3 create_missing_judgments_list.py")
                        print(f"   python3 missing_judgments_workflow.py --step all")
                
                if args.regenerate:
                    if args.clean and not args.dry_run:
                        print(f"\n🚀 REGENERATING CLEANED JUDGMENTS using existing workflow...")
                        print(f"   Running missing judgments detection first...")
                        
                        # Run the missing judgments workflow
                        try:
                            import subprocess
                            
                            # First create the missing judgments list
                            cmd1 = ["python3", f"{WORKSPACE_ROOT}/create_missing_judgments_list.py"]
                            print(f"   Running: {' '.join(cmd1)}")
                            result1 = subprocess.run(cmd1, cwd=WORKSPACE_ROOT, capture_output=True, text=True)
                            
                            if result1.returncode == 0:
                                print("   ✅ Missing judgments list updated!")
                                
                                # Then run the missing judgments workflow
                                cmd2 = ["python3", f"{WORKSPACE_ROOT}/missing_judgments_workflow.py", "--step", "all"]
                                print(f"   Running: {' '.join(cmd2)}")
                                result2 = subprocess.run(cmd2, cwd=WORKSPACE_ROOT, capture_output=True, text=True)
                                
                                if result2.returncode == 0:
                                    print("   ✅ Regeneration workflow started successfully!")
                                else:
                                    print(f"   ❌ Regeneration workflow failed: {result2.stderr}")
                            else:
                                print(f"   ❌ Missing judgments detection failed: {result1.stderr}")
                                
                        except Exception as e:
                            print(f"   ❌ Failed to run regeneration workflow: {e}")
                    else:
                        print(f"   ⚠️  --regenerate requires --clean to be used together")
            else:
                print(f"   ✅ No invalid judgments found for {args.baseline} baseline")
        else:
            print(f"  1. Review the invalid judgment files above")
            print(f"  2. Clean invalid judgments (removes them so they appear incomplete):")
            
            for baseline_dir, invalid_models in invalid_models_by_baseline.items():
                if invalid_models:
                    baseline_name = baseline_dir.replace('compared_with_', '')
                    print(f"     python3 identify_invalid_judgments.py --baseline {baseline_name} --clean")
            
            print(f"  3. Use existing missing judgments workflow to regenerate:")
            print(f"     python3 create_missing_judgments_list.py")
            print(f"     python3 missing_judgments_workflow.py --step all")
            
            print(f"  4. Or do both steps automatically:")
            for baseline_dir, invalid_models in invalid_models_by_baseline.items():
                if invalid_models:
                    baseline_name = baseline_dir.replace('compared_with_', '')
                    print(f"     python3 identify_invalid_judgments.py --baseline {baseline_name} --clean --regenerate")
            
            print(f"  5. Re-run this script to verify all issues are resolved")
    else:
        print(f"\n🎉 ALL JUDGMENTS ARE VALID!")
        print(f"     No invalid judgment scores detected.")

if __name__ == "__main__":
    main()
