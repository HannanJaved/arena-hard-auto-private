#!/usr/bin/env python3
"""
Monitor Arena Hard job progress and results
"""

import os
import subprocess
import glob
from pathlib import Path

WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
LOGS_DIR = f"{WORKSPACE_ROOT}/logs/arena-hard"
RESULTS_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto/data/arena-hard-v2.0/model_answer"

def get_job_status():
    """Get current SLURM job status for the user."""
    try:
        result = subprocess.run(['squeue', '-u', os.getenv('USER', ''), '-h'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []

def find_log_files():
    """Find all arena hard log files."""
    log_files = []
    for root, dirs, files in os.walk(LOGS_DIR):
        for file in files:
            if file.endswith(('.out', '.err')):
                log_files.append(os.path.join(root, file))
    return sorted(log_files)

def find_result_files():
    """Find generated answer files."""
    result_files = glob.glob(f"{RESULTS_DIR}/tulu3-8b-rank64-*.jsonl")
    return sorted(result_files)

def get_file_size(file_path):
    """Get file size in a human readable format."""
    if not os.path.exists(file_path):
        return "N/A"
    
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def count_lines(file_path):
    """Count lines in a file."""
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except:
        return 0

def main():
    print("=== Arena Hard Job Monitor ===\n")
    
    # Check running jobs
    jobs = get_job_status()
    if jobs and jobs[0]:  # Check if there are actual jobs (not empty list with empty string)
        print(f"🏃 Running Jobs ({len(jobs)}):")
        for job in jobs:
            if job.strip():  # Only print non-empty job lines
                print(f"  {job}")
    else:
        print("🏃 Running Jobs: None")
    
    print("\n" + "="*50)
    
    # Check log files
    log_files = find_log_files()
    print(f"\n📄 Log Files ({len(log_files)}):")
    
    recent_logs = sorted(log_files, key=lambda x: os.path.getmtime(x), reverse=True)[:10]
    for log_file in recent_logs:
        rel_path = os.path.relpath(log_file, WORKSPACE_ROOT)
        size = get_file_size(log_file)
        mtime = os.path.getmtime(log_file)
        import datetime
        mod_time = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {rel_path} ({size}, {mod_time})")
    
    if len(log_files) > 10:
        print(f"  ... and {len(log_files) - 10} more")
    
    print("\n" + "="*50)
    
    # Check result files
    result_files = find_result_files()
    print(f"\n📊 Result Files ({len(result_files)}):")
    
    for result_file in result_files:
        rel_path = os.path.relpath(result_file, RESULTS_DIR)
        size = get_file_size(result_file)
        lines = count_lines(result_file)
        print(f"  {rel_path} ({size}, {lines} answers)")
    
    if result_files:
        print(f"\n📈 Total Answers Generated: {sum(count_lines(f) for f in result_files)}")
    
    print("\n" + "="*50)
    print("\nCommands:")
    print("  squeue -u $USER                    # Check your jobs")
    print("  scancel -u $USER                   # Cancel all your jobs")
    print("  tail -f <log_file>                 # Follow log file")
    print(f"  ls {RESULTS_DIR}/                  # List result files")

if __name__ == "__main__":
    main()
