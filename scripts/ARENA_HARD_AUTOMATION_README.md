# Arena Hard Complete Automation

This automation system provides end-to-end Arena Hard evaluation: answer generation → judgment → results analysis.

## Quick Start

```bash
# Complete workflow (dry run first)
python3 arena_hard_workflow.py --step all --dry-run

# Complete workflow (submit jobs)
python3 arena_hard_workflow.py --step all

# Individual steps
python3 arena_hard_workflow.py --step answers
python3 arena_hard_workflow.py --step judgments  
python3 arena_hard_workflow.py --step results
```

## Files Overview

### Answer Generation
1. **`automate_arena_hard_generation.py`** - Main answer generation script
2. **`submit_arena_hard_batch.sh`** - Bash wrapper for answer generation
3. **`monitor_arena_hard_jobs.py`** - Monitor answer generation progress

### Judgment Generation  
4. **`automate_arena_hard_judgment.py`** - Main judgment automation script
5. **`submit_arena_hard_judgment.sh`** - Bash wrapper for judgment generation
6. **`monitor_arena_hard_judgments.py`** - Monitor judgment progress

### Workflow Management
7. **`arena_hard_workflow.py`** - Complete workflow manager
8. **`arena_hard_models_to_test.txt`** - Configuration file listing models to test

## Configuration

- **Judge Model**: `neuralmagic-llama3.1-70b-instruct-fp8`
- **Baseline Model**: `llama3.1-8b-instruct`
- **Models to Test**: Listed in `arena_hard_models_to_test.txt` (270 models)

## How It Works

The automation system:
1. Reads your `api_config.yaml` to get available models
2. Creates individual SLURM job scripts for each model
3. Creates individual `gen_answer_config.yaml` files for each model
4. Organizes logs by model configuration (Rank/Alpha/Step)
5. Submits jobs to run in parallel

## Usage Options

### Option 1: Using the Configuration File (Recommended)

1. Edit `arena_hard_models_to_test.txt` to list the models you want to test:
   ```
   tulu3-8b-rank64-alpha1e5-001-step48000
   tulu3-8b-rank64-alpha1e5-001-final
   tulu3-8b-rank64-alpha1e5-005-step48000
   ```

2. Run the automation:
   ```bash
   # Dry run (generate scripts without submitting)
   python3 automate_arena_hard_generation.py --dry-run
   
   # Submit jobs
   python3 automate_arena_hard_generation.py --submit
   ```

### Option 2: Using the Bash Wrapper

```bash
# Process models from configuration file
./submit_arena_hard_batch.sh --all

# Dry run (generate scripts without submitting)
./submit_arena_hard_batch.sh --all --dry-run

# Process specific models
./submit_arena_hard_batch.sh --models tulu3-8b-rank64-alpha1e5-001-step48000 tulu3-8b-rank64-alpha1e5-001-final

# List available models
./submit_arena_hard_batch.sh --list
```

### Option 3: Command Line Arguments

```bash
# Process all tulu3 models from API config
python3 automate_arena_hard_generation.py --all --submit

# Process specific models
python3 automate_arena_hard_generation.py --models tulu3-8b-rank64-alpha1e5-001-step48000 tulu3-8b-rank64-alpha1e5-001-final --submit

# Use custom model list file
python3 automate_arena_hard_generation.py --models-file my_custom_models.txt --submit
```

## Directory Structure Created

```
/data/horse/ws/hama901h-BFTranslation/
├── generated_scripts/          # Individual SLURM job scripts
│   ├── run_arena_hard_tulu3-8b-rank64-alpha1e5-001-step48000.sh
│   └── run_arena_hard_tulu3-8b-rank64-alpha1e5-001-final.sh
├── generated_configs/          # Individual gen_answer config files
│   ├── gen_answer_config_tulu3-8b-rank64-alpha1e5-001-step48000.yaml
│   └── gen_answer_config_tulu3-8b-rank64-alpha1e5-001-final.yaml
└── logs/arena-hard/            # Organized log files
    ├── Rank64/
    │   └── alpha_1e5_001/
    │       ├── step_48000.out
    │       ├── step_48000.err
    │       ├── step_48000_vllm_model_server.log
    │       ├── final.out
    │       ├── final.err
    │       └── final_vllm_model_server.log
    └── misc/                   # For models that don't match expected naming
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel specific job
scancel <job_id>

# View job output in real-time
tail -f /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/Rank64/alpha_1e5_001/step_48000.out
```

## Features

- **Parallel Processing**: Each model runs as a separate SLURM job
- **Organized Logging**: Logs are organized by model configuration
- **Error Handling**: Each job includes error checking and cleanup
- **Flexible Model Selection**: Choose models via file, command line, or process all
- **Dry Run Mode**: Generate scripts without submitting for review
- **Resource Management**: Each job gets dedicated GPU and memory allocation

## Customization

### Modify Job Resources

Edit the SLURM parameters in `automate_arena_hard_generation.py`:
```python
#SBATCH --mem=32G                # Memory allocation
#SBATCH --time=01:00:00          # Time limit
#SBATCH --cpus-per-task=4        # CPU cores
```

### Change Model Server Wait Time

Modify the sleep duration in the generated scripts (currently 15 minutes):
```bash
echo "Waiting 15 mins for server to load..."
sleep 900  # Change this value
```

### Add Different Model Families

Modify the `extract_tulu_models()` function to include other model patterns:
```python
def extract_tulu_models(api_config):
    """Extract models matching specific patterns."""
    models = {}
    for model_name, config in api_config.items():
        if model_name.startswith('tulu3-8b-rank64') or model_name.startswith('other-pattern'):
            models[model_name] = config
    return models
```

## Troubleshooting

1. **Jobs fail to start**: Check SLURM configuration and partition availability
2. **Model server fails**: Check CUDA module loading and GPU availability  
3. **Permission errors**: Ensure scripts are executable (`chmod +x`)
4. **Missing models**: Verify model names match exactly with `api_config.yaml`

## Best Practices

1. Start with a dry run to review generated scripts
2. Test with 1-2 models before running large batches
3. Monitor disk space for logs and outputs
4. Use the configuration file for reproducible experiments
5. Keep model naming consistent with your API config
