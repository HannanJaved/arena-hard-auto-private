#!/usr/bin/env python3
"""
Arena Hard Missing Models Workflow
Complete workflow manager for missing and incomplete model handling.
"""

import argparse
import subprocess
import os

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"

def run_missing_check(detailed_report=False):
    """Run the missing models checker."""
    try:
        cmd = ["python3", "check_missing_model_answers.py"]
        if detailed_report:
            print(f"🔍 Running detailed missing models check: {' '.join(cmd)}")
        else:
            print(f"🔍 Running missing models check: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_ROOT)
        
        if result.returncode == 0:
            print("✅ Missing models check completed")
            print(result.stdout)
        else:
            print("❌ Missing models check failed")
            print(result.stderr)
        
        # Also create simple missing models list
        cmd = ["python3", "create_missing_models_list.py"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_ROOT)
        if result.returncode == 0:
            print("✅ Missing models list created")
        
        return True
    except Exception as e:
        print(f"❌ Error running missing models check: {e}")
        return False

def run_incomplete_check(dry_run=False):
    """Run the incomplete models checker."""
    try:
        cmd = ["python3", "create_incomplete_models_list.py"]
        if dry_run:
            print(f"🔍 Would run: {' '.join(cmd)}")
            return True
        
        print(f"🔍 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_ROOT)
        
        if result.returncode == 0:
            print("✅ Incomplete models check completed")
            print(result.stdout)
            return True
        else:
            print("❌ Incomplete models check failed")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running incomplete models check: {e}")
        return False

def run_answer_generation(dry_run=False, missing_only=False, incomplete_only=False):
    """Run the answer generation automation."""
    try:
        if incomplete_only:
            models_file = "incomplete_models_list.txt"
            print(f"🔧 Generating answers for incomplete models from: {models_file}")
        else:
            models_file = "missing_models_list.txt"
            print(f"🔧 Generating answers for missing models from: {models_file}")
        
        # Check if the models file exists and has content
        models_file_path = os.path.join(WORKSPACE_ROOT, models_file)
        if not os.path.exists(models_file_path):
            print(f"⚠️  Models file not found: {models_file}")
            print(f"   Run the appropriate check step first")
            return False
        
        with open(models_file_path, 'r') as f:
            models = [line.strip() for line in f if line.strip()]
        
        if not models:
            print(f"✅ No models to process in {models_file}")
            return True
        
        print(f"📋 Found {len(models)} models to process")
        
        cmd = ["python3", "automate_arena_hard_generation.py", 
               "--missing-models-file", models_file]
        
        if not dry_run:
            cmd.append("--submit")
        
        if dry_run:
            print(f"🔍 Would run: {' '.join(cmd)}")
            return True
        
        print(f"🚀 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=WORKSPACE_ROOT)
        
        if result.returncode == 0:
            print("✅ Answer generation completed")
            return True
        else:
            print("❌ Answer generation failed")
            return False
    except Exception as e:
        print(f"❌ Error running answer generation: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Arena Hard Missing Models Workflow")
    parser.add_argument("--step", choices=["check", "generate", "incomplete", "all"], required=True,
                      help="Which step to run: check=identify missing, generate=create answers, incomplete=handle incomplete models, all=missing+incomplete")
    parser.add_argument("--dry-run", action="store_true", 
                      help="Show what would be done without executing")
    parser.add_argument("--detailed-report", action="store_true",
                      help="Generate detailed report")
    
    args = parser.parse_args()
    
    print("🚀 Arena Hard Missing Models Workflow")
    print("=" * 50)
    
    if args.step in ["check", "all"]:
        print("\n📊 STEP 1: Checking for missing models...")
        run_missing_check(args.detailed_report)
    
    if args.step in ["incomplete", "all"]:
        print("\n⚠️  STEP 2: Checking for incomplete models...")
        run_incomplete_check(args.dry_run)
    
    if args.step in ["generate", "all"]:
        print("\n🔧 STEP 3: Generating answers for missing models...")
        run_answer_generation(args.dry_run, missing_only=True)
        
        if args.step == "all":
            print("\n🔧 STEP 4: Generating answers for incomplete models...")
            run_answer_generation(args.dry_run, incomplete_only=True)
    
    if args.step == "incomplete":
        print("\n🔧 STEP 2: Generating answers for incomplete models...")
        run_answer_generation(args.dry_run, incomplete_only=True)
    
    print(f"\n✅ Workflow completed!")
    print(f"💡 Monitor progress with: python3 monitor_arena_hard_jobs.py")

if __name__ == "__main__":
    main()
