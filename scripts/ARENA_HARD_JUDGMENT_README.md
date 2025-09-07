# Arena Hard Judgment Automation

This automation system generates Arena Hard judgments for multiple models in parallel using SLURM job scheduling. It compares each model against the baseline using a judge model.

## Configuration

- **Judge Model**: `neuralmagic-llama3.1-70b-instruct-fp8`
- **Baseline Model**: `llama3.1-8b-instruct`
- **Models to Judge**: Listed in `arena_hard_models_to_test.txt`

## Files Created

1. **`automate_arena_hard_judgment.py`** - Main judgment automation script
2. **`submit_arena_hard_judgment.sh`** - Simple bash wrapper for common operations
3. **`monitor_arena_hard_judgments.py`** - Monitor judgment progress and results

## Prerequisites

Before running judgment automation, ensure you have:

1. **Generated answers** for all models you want to judge
2. **Generated answers** for the baseline model (`llama3.1-8b-instruct`)
3. **Judge model** available in your API config

## Usage Options

### Option 1: Using the Configuration File (Recommended)

The script automatically reads from `arena_hard_models_to_test.txt`:

```bash
# Validate models (check if answers exist)
python3 automate_arena_hard_judgment.py --validate-only

# Dry run (generate scripts without submitting)
python3 automate_arena_hard_judgment.py --dry-run

# Submit judgment jobs
python3 automate_arena_hard_judgment.py --submit
```

### Option 2: Using the Bash Wrapper

```bash
# Judge all models from configuration file
./submit_arena_hard_judgment.sh --all

# Dry run
./submit_arena_hard_judgment.sh --all --dry-run

# Validate models only
./submit_arena_hard_judgment.sh --all --validate

# Judge specific models
./submit_arena_hard_judgment.sh --models tulu3-8b-rank64-alpha1e5-001-step48000 tulu3-8b-rank64-alpha1e5-001-final

# Control batch size (models per job)
./submit_arena_hard_judgment.sh --all --batch-size 5
```

### Option 3: Command Line Arguments

```bash
# Judge all tulu3 models from API config
python3 automate_arena_hard_judgment.py --all --submit

# Judge specific models
python3 automate_arena_hard_judgment.py --models tulu3-8b-rank64-alpha1e5-001-step48000 --submit

# Control batch size (default is 10 models per job)
python3 automate_arena_hard_judgment.py --all --batch-size 5 --submit
```

## How It Works

The judgment automation:

1. **Validates Models**: Checks that all models have generated answers
2. **Creates Batches**: Groups models into batches for parallel processing
3. **Generates Configs**: Creates individual Arena Hard config files for each batch
4. **Creates SLURM Scripts**: Generates job scripts that:
   - Start the judge model server (vLLM)
   - Run `gen_judgment.py` for the batch of models
   - Compare each model against the baseline
   - Generate 2 rounds of judgment per question (A vs B, B vs A)
   - Clean up resources

## Directory Structure Created

```
/data/horse/ws/hama901h-BFTranslation/
├── generated_judgment_scripts/          # SLURM job scripts for judgment
│   ├── run_arena_hard_judgment_batch_1.sh
│   └── run_arena_hard_judgment_batch_2.sh
├── generated_judgment_configs/          # Arena Hard config files
│   ├── arena_hard_judgment_batch_1.yaml
│   └── arena_hard_judgment_batch_2.yaml
├── logs/arena-hard/judgment_batches/    # Judgment job logs
│   ├── judge-batch-1_12345_20250907_143000.out
│   └── judge-batch-1_12345_20250907_143000.err
└── arena-hard-auto/data/arena-hard-v2.0/model_judgment/
    └── neuralmagic-llama3.1-70b-instruct-fp8/    # Generated judgments
        ├── tulu3-8b-rank64-alpha1e5-001-step48000.jsonl
        └── tulu3-8b-rank64-alpha1e5-001-final.jsonl
```

## Monitoring Progress

### Check Job Status
```bash
# Monitor judgment progress
python3 monitor_arena_hard_judgments.py

# Check SLURM jobs
squeue -u $USER

# Cancel all jobs if needed
scancel -u $USER
```

### Sample Monitoring Output
```
=== Arena Hard Judgment Monitor ===

🏃 Running Judgment Jobs (2):
  12345  capella judge-batch-1  user    R  15:30  1  c1

✅ Baseline Model (llama3.1-8b-instruct): 500 answers

📊 Judgment Progress:
Judge Model: neuralmagic-llama3.1-70b-instruct-fp8
Total models in config: 117
Models with answers: 45
Models with judgments: 12

📈 Detailed Model Status:
Model Name                                         | Answers | Judgments (Q/G)
--------------------------------------------------------------------------------
✅ tulu3-8b-rank64-alpha1e5-001-step48000          |     500 | 500/1000
🟡 tulu3-8b-rank64-alpha1e5-001-final             |     500 |   0/0
❌ tulu3-8b-rank64-alpha1e5-005-step6000           |       0 |   0/0
```

## Judgment File Format

Each judgment file (`model.jsonl`) contains:
- One line per question
- Each line has 2 "games" (A vs B, B vs A comparisons)
- Judgments in format like `[[A]]`, `[[B]]`, `[[A>B]]`, etc.

Example:
```json
{
  "question_id": "arena-hard-001",
  "games": [
    {"judgment": "[[A>B]]", ...},
    {"judgment": "[[B>A]]", ...}
  ]
}
```

## Features

- **Batch Processing**: Groups models to efficiently use GPU resources
- **Validation**: Checks model availability and answer files before processing
- **Error Handling**: Robust error checking and resource cleanup
- **Organized Logging**: Structured logs for each batch
- **Progress Monitoring**: Detailed status tracking
- **Flexible Batching**: Configurable batch sizes for optimal resource usage

## Customization

### Modify Batch Size
```bash
# Smaller batches for faster turnaround
python3 automate_arena_hard_judgment.py --batch-size 5 --submit

# Larger batches for efficiency
python3 automate_arena_hard_judgment.py --batch-size 20 --submit
```

### Change Judge Configuration
Edit the script constants:
```python
JUDGE_MODEL = "neuralmagic-llama3.1-70b-instruct-fp8"
BASELINE_MODEL = "llama3.1-8b-instruct"
JUDGE_PATH = "/path/to/your/judge/model"
```

### Modify Job Resources
Edit SLURM parameters in `create_judgment_slurm_script()`:
```bash
#SBATCH --mem=64G                # Memory allocation
#SBATCH --time=02:00:00          # Time limit (2 hours)
#SBATCH --cpus-per-task=8        # CPU cores
```

## Workflow Example

1. **Generate answers for all models**:
   ```bash
   python3 automate_arena_hard_generation.py --submit
   ```

2. **Validate models are ready for judgment**:
   ```bash
   python3 automate_arena_hard_judgment.py --validate-only
   ```

3. **Submit judgment jobs**:
   ```bash
   python3 automate_arena_hard_judgment.py --submit --batch-size 10
   ```

4. **Monitor progress**:
   ```bash
   python3 monitor_arena_hard_judgments.py
   ```

5. **Analyze results** (after completion):
   ```bash
   cd arena-hard-auto
   python show_result.py
   ```

## Troubleshooting

1. **Missing baseline answers**: Generate answers for `llama3.1-8b-instruct` first
2. **Judge server fails**: Check GPU availability and CUDA module loading
3. **Models not found**: Ensure model names match exactly with `api_config.yaml`
4. **Out of memory**: Reduce batch size or increase memory allocation
5. **Timeout errors**: Increase time limit in SLURM configuration

## Best Practices

1. **Start small**: Test with a few models before running large batches
2. **Validate first**: Always run `--validate-only` before submitting jobs
3. **Monitor resources**: Watch GPU memory usage and adjust batch sizes
4. **Check baseline**: Ensure baseline model answers are complete
5. **Organize results**: Use consistent naming for easy analysis
