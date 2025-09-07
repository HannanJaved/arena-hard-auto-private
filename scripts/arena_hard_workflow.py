#!/usr/bin/env python3
"""
Arena Hard Complete Workflow Manager
Manages the entire Arena Hard evaluation pipeline: answer generation → judgment → results
"""

import argparse
import subprocess
import os

WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Arena Hard Complete Workflow Manager')
    parser.add_argument('--step', choices=['answers', 'judgments', 'results', 'all'], 
                       required=True, help='Which step to run')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for parallel jobs')
    
    args = parser.parse_args()
    
    os.chdir(WORKSPACE_ROOT)
    
    print("=== Arena Hard Complete Workflow Manager ===")
    print(f"Working directory: {WORKSPACE_ROOT}")
    print(f"Step: {args.step}")
    print(f"Dry run: {args.dry_run}")
    
    if args.step == 'answers' or args.step == 'all':
        print("\\n" + "="*60)
        print("📝 STEP 1: GENERATING ANSWERS")
        print("="*60)
        
        # Check current status
        if not run_command("python3 monitor_arena_hard_jobs.py", "Checking current answer generation status"):
            print("⚠️  Monitor script failed, continuing anyway...")
        
        # Generate answers
        cmd = f"python3 automate_arena_hard_generation.py"
        if args.dry_run:
            cmd += " --dry-run"
        else:
            cmd += " --submit"
            
        if not run_command(cmd, "Generating answers for all models"):
            print("❌ Answer generation failed!")
            return
        
        if not args.dry_run:
            print("\\n📋 Monitor answer generation with:")
            print("   python3 monitor_arena_hard_jobs.py")
            print("   squeue -u $USER")
            
            if args.step == 'all':
                input("\\n⏸️  Press Enter after answer generation is complete to continue with judgments...")
    
    if args.step == 'judgments' or args.step == 'all':
        print("\\n" + "="*60)
        print("⚖️  STEP 2: GENERATING JUDGMENTS")
        print("="*60)
        
        # Validate models are ready for judgment
        if not run_command("python3 automate_arena_hard_judgment.py --validate-only", 
                          "Validating models for judgment"):
            print("❌ Validation failed! Generate answers first.")
            return
        
        # Check current judgment status
        if not run_command("python3 monitor_arena_hard_judgments.py", "Checking current judgment status"):
            print("⚠️  Monitor script failed, continuing anyway...")
        
        # Generate judgments
        cmd = f"python3 automate_arena_hard_judgment.py --batch-size {args.batch_size}"
        if args.dry_run:
            cmd += " --dry-run"
        else:
            cmd += " --submit"
            
        if not run_command(cmd, f"Generating judgments (batch size: {args.batch_size})"):
            print("❌ Judgment generation failed!")
            return
        
        if not args.dry_run:
            print("\\n📋 Monitor judgment generation with:")
            print("   python3 monitor_arena_hard_judgments.py")
            print("   squeue -u $USER")
            
            if args.step == 'all':
                input("\\n⏸️  Press Enter after judgment generation is complete to continue with results...")
    
    if args.step == 'results' or args.step == 'all':
        print("\\n" + "="*60)
        print("📊 STEP 3: ANALYZING RESULTS")
        print("="*60)
        
        # Check judgment status
        if not run_command("python3 monitor_arena_hard_judgments.py", "Checking judgment completion"):
            print("⚠️  Monitor script failed, continuing anyway...")
        
        # Show results
        arena_hard_dir = f"{WORKSPACE_ROOT}/arena-hard-auto"
        if os.path.exists(f"{arena_hard_dir}/show_result.py"):
            if not run_command(f"cd {arena_hard_dir} && python3 show_result.py", 
                              "Generating final results"):
                print("⚠️  Results generation failed, check manually")
        else:
            print("⚠️  show_result.py not found, check results manually")
        
        # Display result directories
        print("\\n📁 Result directories:")
        result_dirs = [
            f"{arena_hard_dir}/data/arena-hard-v2.0/model_answer",
            f"{arena_hard_dir}/data/arena-hard-v2.0/model_judgment",
        ]
        
        for result_dir in result_dirs:
            if os.path.exists(result_dir):
                print(f"   ✅ {result_dir}")
            else:
                print(f"   ❌ {result_dir} (not found)")
    
    print("\\n" + "="*60)
    print("🎉 WORKFLOW COMPLETE")
    print("="*60)
    
    if not args.dry_run:
        print("\\n📋 Useful commands:")
        print("   # Monitor jobs")
        print("   squeue -u $USER")
        print("   python3 monitor_arena_hard_jobs.py")
        print("   python3 monitor_arena_hard_judgments.py")
        print("")
        print("   # Cancel all jobs if needed")
        print("   scancel -u $USER")
        print("")
        print("   # Show final results")
        print("   cd arena-hard-auto && python3 show_result.py")

if __name__ == "__main__":
    main()
