#!/usr/bin/env python3
"""
Missing Judgments Workflow Manager
Manages the entire process from detection to judgment generation for missing/incomplete judgments.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
MISSING_JUDGMENTS_FILE = f"{WORKSPACE_ROOT}/missing_judgments_list.txt"
INCOMPLETE_JUDGMENTS_FILE = f"{WORKSPACE_ROOT}/incomplete_judgments_list.txt"

def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_missing_judgments():
    """Check for missing and incomplete judgments."""
    print("=" * 60)
    print("STEP 1: CHECKING FOR MISSING JUDGMENTS")
    print("=" * 60)
    
    return run_command(
        f"python3 {WORKSPACE_ROOT}/create_missing_judgments_list.py",
        "Scanning for missing and incomplete judgments"
    )

def check_judgment_status():
    """Run detailed judgment status check."""
    print("=" * 60) 
    print("STEP 2: DETAILED JUDGMENT STATUS")
    print("=" * 60)
    
    return run_command(
        f"python3 {WORKSPACE_ROOT}/check_judgment_status.py",
        "Generating detailed judgment status report"
    )

def count_models_in_file(file_path):
    """Count the number of models in a file."""
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, 'r') as f:
            return sum(1 for line in f if line.strip())
    except:
        return 0

def generate_judgments_for_missing(dry_run=False):
    """Generate judgments for missing models."""
    missing_count = count_models_in_file(MISSING_JUDGMENTS_FILE)
    
    if missing_count == 0:
        print("✅ No missing judgments found - skipping generation")
        return True
    
    print("=" * 60)
    print(f"STEP 3A: GENERATING JUDGMENTS FOR {missing_count} MISSING MODELS")
    print("=" * 60)
    
    command = f"python3 {WORKSPACE_ROOT}/automate_arena_hard_judgment.py --missing-models-file {MISSING_JUDGMENTS_FILE}"
    if not dry_run:
        command += " --submit"
    else:
        command += " --dry-run"
    
    return run_command(command, f"Generating judgments for {missing_count} missing models")

def generate_judgments_for_incomplete(dry_run=False):
    """Generate judgments for incomplete models.""" 
    incomplete_count = count_models_in_file(INCOMPLETE_JUDGMENTS_FILE)
    
    if incomplete_count == 0:
        print("✅ No incomplete judgments found - skipping generation")
        return True
    
    print("=" * 60)
    print(f"STEP 3B: GENERATING JUDGMENTS FOR {incomplete_count} INCOMPLETE MODELS") 
    print("=" * 60)
    
    command = f"python3 {WORKSPACE_ROOT}/automate_arena_hard_judgment.py --missing-models-file {INCOMPLETE_JUDGMENTS_FILE}"
    if not dry_run:
        command += " --submit"
    else:
        command += " --dry-run"
    
    return run_command(command, f"Generating judgments for {incomplete_count} incomplete models")

def main():
    parser = argparse.ArgumentParser(description='Workflow manager for missing/incomplete judgments')
    parser.add_argument('--step', choices=['check', 'incomplete', 'missing', 'generate', 'all'],
                       default='all',
                       help='Which step to run (default: all)')
    parser.add_argument('--detailed-report', action='store_true',
                       help='Include detailed judgment status report')
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate scripts but do not submit jobs')
    
    args = parser.parse_args()
    
    print("🎯 Missing Judgments Workflow Manager")
    print(f"📁 Workspace: {WORKSPACE_ROOT}")
    print()
    
    success = True
    
    # Step 1: Check for missing judgments
    if args.step in ['check', 'all']:
        if not check_missing_judgments():
            success = False
            if args.step != 'all':
                sys.exit(1)
    
    # Step 2: Detailed report (optional)
    if args.detailed_report and args.step in ['check', 'all']:
        if not check_judgment_status():
            success = False
    
    # Step 3: Generate judgments
    if args.step in ['missing', 'generate', 'all']:
        if not generate_judgments_for_missing(args.dry_run):
            success = False
            
    if args.step in ['incomplete', 'generate', 'all']:
        if not generate_judgments_for_incomplete(args.dry_run):
            success = False
    
    # Final summary
    print("=" * 60)
    print("WORKFLOW SUMMARY")
    print("=" * 60)
    
    if success:
        print("✅ Workflow completed successfully!")
        
        # Show counts
        missing_count = count_models_in_file(MISSING_JUDGMENTS_FILE)
        incomplete_count = count_models_in_file(INCOMPLETE_JUDGMENTS_FILE)
        total_pending = missing_count + incomplete_count
        
        if total_pending > 0:
            print(f"📊 Judgment generation status:")
            print(f"   - Missing models: {missing_count}")
            print(f"   - Incomplete models: {incomplete_count}")
            print(f"   - Total jobs submitted: {total_pending}")
            
            if not args.dry_run:
                print(f"\n💡 Monitor progress with:")
                print(f"   squeue -u $USER")
                print(f"   python3 {WORKSPACE_ROOT}/monitor_arena_hard_judgments.py")
            else:
                print(f"\n💡 This was a dry run - no jobs were submitted")
                print(f"   Run without --dry-run to submit jobs")
        else:
            print(f"🎉 All judgments are complete!")
            
    else:
        print("❌ Workflow completed with errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
