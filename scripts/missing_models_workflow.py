#!/usr/bin/env python3
"""
Arena Hard Missing Models Workflow
Complete workflow for identifying and generating answers for missing models.
"""

import subprocess
import argparse
import os

WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"

def run_command(cmd, description, check_success=True):
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, cwd=WORKSPACE_ROOT)
        print("✅ Success")
        if result.stdout.strip():
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        if check_success:
            print(f"❌ Failed: {e}")
            if e.stderr:
                print(f"Error: {e.stderr}")
            return False
        else:
            print(f"⚠️  Command failed (continuing): {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Arena Hard Missing Models Workflow')
    parser.add_argument('--step', choices=['check', 'generate', 'all'], required=True,
                       help='Which step to run: check=identify missing, generate=create answers, all=both')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--detailed-report', action='store_true', help='Generate detailed report')
    
    args = parser.parse_args()
    
    print("=== Arena Hard Missing Models Workflow ===")
    print(f"Working directory: {WORKSPACE_ROOT}")
    print(f"Step: {args.step}")
    
    if args.step == 'check' or args.step == 'all':
        print("\n" + "="*60)
        print("🔍 STEP 1: CHECKING FOR MISSING MODELS")
        print("="*60)
        
        if args.detailed_report:
            # Generate detailed report
            if not run_command("python3 check_missing_model_answers.py", 
                             "Generating detailed missing models report"):
                return
        
        # Generate simple missing models list
        if not run_command("python3 create_missing_models_list.py", 
                         "Creating missing models list"):
            return
        
        # Check if we have missing models
        missing_file = f"{WORKSPACE_ROOT}/missing_models_list.txt"
        if os.path.exists(missing_file):
            with open(missing_file, 'r') as f:
                missing_models = [line.strip() for line in f if line.strip()]
            
            if missing_models:
                print(f"\n📊 Found {len(missing_models)} missing models")
                print("📄 Lists saved to:")
                print(f"   - {missing_file}")
                if args.detailed_report:
                    print(f"   - {WORKSPACE_ROOT}/missing_model_answers_report.txt")
                
                if args.step == 'all':
                    response = input(f"\n❓ Proceed to generate answers for {len(missing_models)} missing models? (y/N): ")
                    if response.lower() != 'y':
                        print("Stopping here. Run with --step generate when ready.")
                        return
            else:
                print("\n🎉 No missing models found! All models have answers.")
                return
        else:
            print("❌ Failed to create missing models list")
            return
    
    if args.step == 'generate' or args.step == 'all':
        print("\n" + "="*60)
        print("🚀 STEP 2: GENERATING MISSING ANSWERS")
        print("="*60)
        
        missing_file = f"{WORKSPACE_ROOT}/missing_models_list.txt"
        
        # Check if missing models file exists
        if not os.path.exists(missing_file):
            print(f"❌ Missing models file not found: {missing_file}")
            print("Run with --step check first to identify missing models")
            return
        
        # Count missing models
        with open(missing_file, 'r') as f:
            missing_models = [line.strip() for line in f if line.strip()]
        
        if not missing_models:
            print("🎉 No missing models to process!")
            return
        
        print(f"📋 Processing {len(missing_models)} missing models")
        
        # Generate answers for missing models
        cmd = f"python3 automate_arena_hard_generation.py --missing-models-file {missing_file}"
        if args.dry_run:
            cmd += " --dry-run"
        else:
            cmd += " --submit"
        
        if not run_command(cmd, f"Generating answers for {len(missing_models)} missing models"):
            return
        
        if not args.dry_run:
            print("\n📋 Monitor answer generation with:")
            print("   python3 monitor_arena_hard_jobs.py")
            print("   squeue -u $USER")
            
            # Ask if user wants to monitor
            monitor = input("\n❓ Start monitoring now? (y/N): ")
            if monitor.lower() == 'y':
                run_command("python3 monitor_arena_hard_jobs.py", "Monitoring answer generation", check_success=False)
    
    print("\n" + "="*60)
    print("🎉 WORKFLOW COMPLETE")
    print("="*60)
    
    print("\n📋 Useful commands:")
    print("   # Check status again")
    print("   python3 check_missing_model_answers.py")
    print("   python3 create_missing_models_list.py")
    print("")
    print("   # Monitor answer generation")
    print("   python3 monitor_arena_hard_jobs.py")
    print("   squeue -u $USER")
    print("")
    print("   # After answers are complete, generate judgments")
    print("   python3 automate_arena_hard_judgment.py --submit")

if __name__ == "__main__":
    main()
