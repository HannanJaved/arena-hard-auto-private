# Arena Hard Missing Models Management

This set of tools helps you identify and manage missing model answers in your Arena Hard evaluation pipeline.

## Tools Overview

### 1. Missing Models Detection

#### `check_missing_model_answers.py` - Comprehensive Analysis
Generates a detailed report comparing your configuration file with available answer files.

**Features:**
- Categorizes missing models by rank, alpha, and weight ratio
- Shows file sizes and line counts for existing answers
- **Identifies incomplete models (< 750 lines)**
- **Separates complete vs incomplete model statistics**
- Identifies extra models not in configuration
- Provides completion percentage and statistics

**Usage:**
```bash
python3 check_missing_model_answers.py
```

**Output:**
- `missing_model_answers_report.txt` - Detailed report with categories and statistics

#### `create_missing_models_list.py` - Simple List Generator
Creates a plain text list of missing model names for easy processing.

**Usage:**
```bash
python3 create_missing_models_list.py
```

**Output:**
- `missing_models_list.txt` - Simple list of missing model names (one per line)

#### `create_incomplete_models_list.py` - Incomplete Models List Generator
**NEW**: Creates a plain text list of models that have incomplete answers (< 750 lines).

**Features:**
- Scans only models in your configuration
- Shows progress during scanning
- Provides statistics on line counts
- Identifies models that need re-generation

**Usage:**
```bash
python3 create_incomplete_models_list.py
```

**Output:**
- `incomplete_models_list.txt` - Simple list of incomplete model names (one per line)

### 2. Missing Models Processing

#### `missing_models_workflow.py` - Complete Workflow Manager
**ENHANCED**: Now handles both missing and incomplete models.

**Usage:**
```bash
# Check for missing models only
python3 missing_models_workflow.py --step check

# Check for incomplete models only  
python3 missing_models_workflow.py --step incomplete

# Generate answers for missing models only  
python3 missing_models_workflow.py --step generate

# Complete workflow: check missing + incomplete + generate for both
python3 missing_models_workflow.py --step all

# Include detailed report
python3 missing_models_workflow.py --step check --detailed-report

# Dry run (no job submission)
python3 missing_models_workflow.py --step all --dry-run
```

#### `missing_models_workflow.py` - Complete Workflow Manager
Manages the entire process from detection to answer generation.

**Usage:**
```bash
# Check for missing models only
python3 missing_models_workflow.py --step check

# Generate answers for missing models only  
python3 missing_models_workflow.py --step generate

# Complete workflow: check + generate
python3 missing_models_workflow.py --step all

# Include detailed report
python3 missing_models_workflow.py --step check --detailed-report

# Dry run (no job submission)
python3 missing_models_workflow.py --step all --dry-run
```

#### Enhanced Answer Generation
The main automation script now supports missing models files:

```bash
# Generate answers for missing models specifically
python3 automate_arena_hard_generation.py --missing-models-file missing_models_list.txt --submit

# Generate answers for incomplete models specifically
python3 automate_arena_hard_generation.py --missing-models-file incomplete_models_list.txt --submit

# Also supports the original methods
python3 automate_arena_hard_generation.py --models-file arena_hard_models_to_test.txt --submit
python3 automate_arena_hard_generation.py --all --submit
```

## Example Workflow

### Step 1: Identify Missing Models
```bash
# Generate detailed analysis
python3 check_missing_model_answers.py

# Create simple list for processing
python3 create_missing_models_list.py

# NEW: Identify incomplete models
python3 create_incomplete_models_list.py
```

### Step 2: Review Results
```bash
# View detailed report
cat missing_model_answers_report.txt

# Count missing models
wc -l missing_models_list.txt

# Count incomplete models
wc -l incomplete_models_list.txt
```

### Step 3: Generate Missing Answers
```bash
# Option A: Use workflow manager (handles both missing and incomplete)
python3 missing_models_workflow.py --step all

# Option B: Use automation directly for missing models
python3 automate_arena_hard_generation.py --missing-models-file missing_models_list.txt --submit

# Option C: Use automation directly for incomplete models
python3 automate_arena_hard_generation.py --missing-models-file incomplete_models_list.txt --submit
```

### Step 4: Monitor Progress
```bash
# Monitor answer generation
python3 monitor_arena_hard_jobs.py

# Check SLURM jobs
squeue -u $USER
```

### Step 5: Verify Completion
```bash
# Re-check for missing models
python3 create_missing_models_list.py

# Re-check for incomplete models
python3 create_incomplete_models_list.py

# Should show 0 missing and 0 incomplete if complete
```

## Current Status

Based on your latest scan:

### Missing Models: 2 models
- `tulu3-8b-rank1024-alpha1e6-010-step48000`
- `tulu3-8b-rank64-alpha1e5-005-step24000`

### Incomplete Models: 124 models
- **Range**: 28 to 727 lines (target: 750 lines)
- **Average**: 388.3 lines per file
- **Most incomplete**: `tulu3-8b-rank1024-alpha1e5-005-step24000` (28 lines)

### Complete Models: 144 models
- Models with full 750 lines generated

### Total Coverage: 268/270 models (99.3% with answers)
- **Complete coverage**: 144/270 models (53.3%)
- **Need re-generation**: 124 models

## Sample Report Output

```
ARENA HARD MODEL ANSWER STATUS REPORT
================================================================================
Generated: 2025-09-07 13:32:32

SUMMARY
----------------------------------------
Total models in config: 270
Models with answers: 87
Missing models: 183
Coverage: 32.2%

MISSING MODELS (No answer files found)
--------------------------------------------------

RANK64:
  alpha1e5-005: 1 missing
    Line  24: tulu3-8b-rank64-alpha1e5-005-step24000
  alpha1e6-005: 1 missing  
    Line  56: tulu3-8b-rank64-alpha1e6-005-step24000

RANK256:
  alpha1e5-001: 9 missing
    Line 120: tulu3-8b-rank256-alpha1e5-001-step6000
    Line 121: tulu3-8b-rank256-alpha1e5-001-step12000
    ...
```

## File Structure

```
/data/horse/ws/hama901h-BFTranslation/
├── arena_hard_models_to_test.txt           # Configuration (270 models)
├── missing_models_list.txt                 # Simple missing models list
├── missing_model_answers_report.txt        # Detailed report
├── check_missing_model_answers.py          # Comprehensive analyzer  
├── create_missing_models_list.py           # Simple list generator
├── missing_models_workflow.py              # Workflow manager
├── automate_arena_hard_generation.py       # Enhanced with missing models support
└── arena-hard-auto/data/arena-hard-v2.0/model_answer/    # Answer files directory
```

## Integration with Existing Tools

The missing models tools integrate seamlessly with your existing automation:

1. **Answer Generation**: Enhanced to support missing models files
2. **Judgment Generation**: Works with whatever answers are available
3. **Monitoring**: Shows progress for both complete and missing models
4. **Workflow Management**: Coordinates the entire pipeline

## Best Practices

1. **Regular Checks**: Run missing models checks periodically to track progress
2. **Batch Processing**: Use missing models lists to process only what's needed
3. **Validation**: Always verify completion after answer generation
4. **Documentation**: Keep reports for tracking and debugging
5. **Resource Management**: Generate answers for missing models in smaller batches if needed

## Troubleshooting

### Common Issues

1. **"Configuration file not found"**
   - Ensure `arena_hard_models_to_test.txt` exists
   - Check file path and permissions

2. **"Model answer directory not found"**
   - Verify Arena Hard auto directory structure
   - Check if answer generation has been run

3. **"Model not found in API config"**
   - Check `api_config.yaml` for model definitions
   - Ensure model names match exactly

4. **Zero missing models but low coverage**
   - Check for extra models not in configuration
   - Verify configuration file completeness

### Commands for Quick Fixes

```bash
# Check if answer directory exists
ls -la arena-hard-auto/data/arena-hard-v2.0/model_answer/

# Count actual answer files
ls arena-hard-auto/data/arena-hard-v2.0/model_answer/*.jsonl | wc -l

# Check configuration file
head -20 arena_hard_models_to_test.txt

# Verify model exists in API config
grep "tulu3-8b-rank64-alpha1e5-001-step48000" arena-hard-auto/config/api_config.yaml
```
