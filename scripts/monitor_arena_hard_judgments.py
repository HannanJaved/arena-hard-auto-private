#!/usr/bin/env python3
"""
Monitor Arena Hard judgment progress and results
"""

import os
import subprocess
import glob
import json
from pathlib import Path

WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
LOGS_DIR = f"{WORKSPACE_ROOT}/logs/arena-hard"
JUDGMENT_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto/data/arena-hard-v2.0/model_judgment"
ANSWER_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto/data/arena-hard-v2.0/model_answer"

# Judge configuration
JUDGE_MODEL = "neuralmagic-llama3.1-70b-instruct-fp8"
BASELINE_MODEL = "llama3.1-8b-instruct"

def get_job_status():
    """Get current SLURM job status for the user."""
    try:
        result = subprocess.run(['squeue', '-u', os.getenv('USER', ''), '-h'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []

def find_judgment_log_files():
    """Find all arena hard judgment log files."""
    log_files = []
    judgment_log_dir = f"{LOGS_DIR}/judgment_batches"
    if os.path.exists(judgment_log_dir):
        for root, dirs, files in os.walk(judgment_log_dir):
            for file in files:
                if file.endswith(('.out', '.err')):
                    log_files.append(os.path.join(root, file))
    return sorted(log_files, key=lambda x: os.path.getmtime(x), reverse=True)

def find_judgment_files():
    """Find generated judgment files."""
    judgment_files = []
    judge_dir = f"{JUDGMENT_DIR}/{JUDGE_MODEL}"
    if os.path.exists(judge_dir):
        judgment_files = glob.glob(f"{judge_dir}/tulu3-8b-rank*.jsonl")
    return sorted(judgment_files)

def find_answer_files():
    """Find answer files to check what's available for judgment."""
    answer_files = glob.glob(f"{ANSWER_DIR}/tulu3-8b-rank*.jsonl")
    return sorted(answer_files)

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

def count_judgments(file_path):
    """Count judgments in a judgment file (each line should have 2 games)."""
    if not os.path.exists(file_path):
        return 0, 0
    
    try:
        total_lines = 0
        total_games = 0
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    total_lines += 1
                    try:
                        data = json.loads(line)
                        if 'games' in data:
                            total_games += len(data['games'])
                    except:
                        pass
        return total_lines, total_games
    except:
        return 0, 0

def check_baseline_exists():
    """Check if baseline model answers exist."""
    baseline_file = f"{ANSWER_DIR}/{BASELINE_MODEL}.jsonl"
    return os.path.exists(baseline_file)

def load_models_from_file():
    """Load models from the test configuration file."""
    config_file = f"{WORKSPACE_ROOT}/arena_hard_models_to_test.txt"
    models = []
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    models.append(line)
    except FileNotFoundError:
        pass
    return models

def main():
    print("=== Arena Hard Judgment Monitor ===\n")
    
    # Check running jobs
    jobs = get_job_status()
    judgment_jobs = [job for job in jobs if job and 'judge' in job.lower()]
    
    if judgment_jobs:
        print(f"🏃 Running Judgment Jobs ({len(judgment_jobs)}):")
        for job in judgment_jobs:
            if job.strip():
                print(f"  {job}")
    else:
        print("🏃 Running Judgment Jobs: None")
    
    print("\n" + "="*60)
    
    # Check baseline
    if check_baseline_exists():
        baseline_answers = count_lines(f"{ANSWER_DIR}/{BASELINE_MODEL}.jsonl")
        print(f"✅ Baseline Model ({BASELINE_MODEL}): {baseline_answers} answers")
    else:
        print(f"❌ Baseline Model ({BASELINE_MODEL}): NOT FOUND")
        print("   You need baseline answers before judging!")
    
    print("\n" + "="*60)
    
    # Check available answers vs judgments
    answer_files = find_answer_files()
    judgment_files = find_judgment_files()
    models_in_config = set(load_models_from_file())
    
    print(f"\n📊 Judgment Progress:")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Total models in config: {len(models_in_config)}")
    print(f"Models with answers: {len(answer_files)}")
    print(f"Models with judgments: {len(judgment_files)}")
    
    # Create mapping of models to their status
    model_status = {}
    
    # Check answers
    for answer_file in answer_files:
        model_name = os.path.basename(answer_file).replace('.jsonl', '')
        if model_name.startswith('tulu3-8b-rank'):
            answers = count_lines(answer_file)
            model_status[model_name] = {'answers': answers, 'judgments': (0, 0)}
    
    # Check judgments
    for judgment_file in judgment_files:
        model_name = os.path.basename(judgment_file).replace('.jsonl', '')
        if model_name in model_status:
            questions, games = count_judgments(judgment_file)
            model_status[model_name]['judgments'] = (questions, games)
    
    # Display detailed status
    print(f"\n📈 Detailed Model Status:")
    print("Model Name".ljust(50) + " | Answers | Judgments (Q/G)")
    print("-" * 80)
    
    models_with_answers = 0
    models_with_judgments = 0
    models_ready_for_judgment = 0
    
    for model in sorted(model_status.keys()):
        status = model_status[model]
        answers = status['answers']
        questions, games = status['judgments']
        
        if answers > 0:
            models_with_answers += 1
            if questions == 0:
                models_ready_for_judgment += 1
            else:
                models_with_judgments += 1
        
        status_icon = "✅" if questions > 0 else ("🟡" if answers > 0 else "❌")
        print(f"{status_icon} {model.ljust(47)} | {str(answers).rjust(7)} | {str(questions).rjust(3)}/{str(games).rjust(3)}")
    
    print("\n" + "="*60)
    
    # Summary statistics
    print(f"\n📊 Summary:")
    print(f"  Models with answers: {models_with_answers}")
    print(f"  Models with judgments: {models_with_judgments}")
    print(f"  Models ready for judgment: {models_ready_for_judgment}")
    print(f"  Models in config but no answers: {len(models_in_config) - models_with_answers}")
    
    if models_ready_for_judgment > 0:
        print(f"\n💡 {models_ready_for_judgment} models are ready for judgment!")
    
    # Check recent log files
    log_files = find_judgment_log_files()
    if log_files:
        print(f"\n📄 Recent Judgment Logs ({min(5, len(log_files))}):")
        for log_file in log_files[:5]:
            rel_path = os.path.relpath(log_file, WORKSPACE_ROOT)
            size = get_file_size(log_file)
            mtime = os.path.getmtime(log_file)
            import datetime
            mod_time = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {rel_path} ({size}, {mod_time})")
    
    print("\n" + "="*60)
    print("\nCommands:")
    print("  python automate_arena_hard_judgment.py --validate --all  # Validate all models")
    print("  python automate_arena_hard_judgment.py --submit --all    # Submit judgment jobs")
    print("  squeue -u $USER                                          # Check your jobs")
    print("  scancel -u $USER                                         # Cancel all jobs")
    print(f"  ls {JUDGMENT_DIR}/{JUDGE_MODEL}/                         # List judgment files")

if __name__ == "__main__":
    main()
