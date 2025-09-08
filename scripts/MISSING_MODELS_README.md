# Arena Hard Missing Models & Judgments Management

This set of tools helps you identify and manage missing model answers and judgments in your Arena Hard evaluation pipeline.

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

### 2. Missing Judgments Detection

#### `check_judgment_status.py` - Comprehensive Judgment Analysis
Generates a detailed report comparing your configuration file with available judgment files.

**Features:**
- Shows which models have been judged vs missing judgments
- **Identifies incomplete judgments (< 750 lines)**
- **Separates complete vs incomplete judgment statistics**
- Categorizes missing judgments by rank, alpha, and weight ratio
- Shows file sizes and line counts for existing judgments
- Provides completion percentage and statistics

**Usage:**
```bash
python3 check_judgment_status.py
```

**Output:**
- `judgment_status_report.txt` - Detailed report with categories and statistics

#### `create_missing_judgments_list.py` - Simple Judgment List Generator
Creates plain text lists of models that need judgments (missing or incomplete).

**Features:**
- Scans judgment directory for existing judgment files
- Identifies models without judgment files (missing)
- Identifies models with incomplete judgments (< 750 lines)
- Shows progress and statistics during scanning

**Usage:**
```bash
python3 create_missing_judgments_list.py
```

**Output:**
- `missing_judgments_list.txt` - Simple list of models without judgment files (one per line)
- `incomplete_judgments_list.txt` - Simple list of models with incomplete judgments (one per line)

### 3. Missing Models Processing

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

### 4. Missing Judgments Processing

#### `missing_judgments_workflow.py` - Complete Judgment Workflow Manager
**NEW**: Manages the entire process from judgment detection to judgment generation.

**Features:**
- Detects both missing and incomplete judgments
- Processes missing and incomplete judgments separately  
- Supports batch processing for efficient resource usage
- Provides detailed progress reporting and statistics

**Usage:**
```bash
# Check for missing judgments only
python3 missing_judgments_workflow.py --step check

# Check for incomplete judgments only  
python3 missing_judgments_workflow.py --step incomplete

# Generate judgments for missing models only  
python3 missing_judgments_workflow.py --step missing

# Generate judgments for all models needing judgments
python3 missing_judgments_workflow.py --step generate

# Complete workflow: check missing + incomplete + generate for both
python3 missing_judgments_workflow.py --step all

# Include detailed report
python3 missing_judgments_workflow.py --step check --detailed-report

# Dry run (no job submission)
python3 missing_judgments_workflow.py --step all --dry-run
```

#### Enhanced Judgment Generation
The main judgment automation script now supports missing judgments files:

```bash
# Generate judgments for missing models specifically
python3 automate_arena_hard_judgment.py --missing-models-file missing_judgments_list.txt --submit

# Generate judgments for incomplete models specifically
python3 automate_arena_hard_judgment.py --missing-models-file incomplete_judgments_list.txt --submit

# Also supports the original methods
python3 automate_arena_hard_generation.py --models-file arena_hard_models_to_test.txt --submit
python3 automate_arena_hard_generation.py --all --submit
```

## Example Workflows

### Complete Pipeline Workflow

#### Step 1: Identify Missing Models & Judgments
```bash
# Generate detailed analysis for answers
python3 check_missing_model_answers.py

# Create simple lists for processing
python3 create_missing_models_list.py
python3 create_incomplete_models_list.py

# Generate detailed analysis for judgments
python3 check_judgment_status.py

# Create simple lists for judgment processing
python3 create_missing_judgments_list.py
```

#### Step 2: Review Results
```bash
# View detailed reports
cat missing_model_answers_report.txt
cat judgment_status_report.txt

# Count missing items
wc -l missing_models_list.txt
wc -l incomplete_models_list.txt
wc -l missing_judgments_list.txt
wc -l incomplete_judgments_list.txt
```

#### Step 3: Generate Missing Answers (if needed)
```bash
# Option A: Use workflow manager (handles both missing and incomplete)
python3 missing_models_workflow.py --step all

# Option B: Use automation directly for missing models
python3 automate_arena_hard_generation.py --missing-models-file missing_models_list.txt --submit

# Option C: Use automation directly for incomplete models
python3 automate_arena_hard_generation.py --missing-models-file incomplete_models_list.txt --submit
```

#### Step 4: Generate Missing Judgments
```bash
# Option A: Use workflow manager (recommended - handles both missing and incomplete)
python3 missing_judgments_workflow.py --step all

# Option B: Use automation directly for missing judgments
python3 automate_arena_hard_judgment.py --missing-models-file missing_judgments_list.txt --submit

# Option C: Use automation directly for incomplete judgments
python3 automate_arena_hard_judgment.py --missing-models-file incomplete_judgments_list.txt --submit
```

#### Step 5: Monitor Progress
```bash
# Monitor answer and judgment generation
python3 monitor_arena_hard_jobs.py
python3 monitor_arena_hard_judgments.py

# Check SLURM jobs
squeue -u $USER
```

#### Step 6: Verify Completion
```bash
# Re-check for missing models
python3 create_missing_models_list.py
python3 create_incomplete_models_list.py

# Re-check for missing judgments  
python3 create_missing_judgments_list.py

# Should show 0 missing and 0 incomplete for both if complete
```

### Quick Answer-Only Workflow

#### Step 1: Identify Missing Models
```bash
# Generate detailed analysis
python3 check_missing_model_answers.py

# Create simple list for processing
python3 create_missing_models_list.py

# NEW: Identify incomplete models
python3 create_incomplete_models_list.py
```

#### Step 2: Review Results
```bash
# View detailed report
cat missing_model_answers_report.txt

# Count missing models
wc -l missing_models_list.txt

# Count incomplete models
wc -l incomplete_models_list.txt
```

#### Step 3: Generate Missing Answers
```bash
# Option A: Use workflow manager (handles both missing and incomplete)
python3 missing_models_workflow.py --step all

# Option B: Use automation directly for missing models
python3 automate_arena_hard_generation.py --missing-models-file missing_models_list.txt --submit

# Option C: Use automation directly for incomplete models
python3 automate_arena_hard_generation.py --missing-models-file incomplete_models_list.txt --submit
```

### Quick Judgment-Only Workflow

#### Step 1: Identify Missing Judgments
```bash
# Generate detailed analysis
python3 check_judgment_status.py

# Create simple lists for processing  
python3 create_missing_judgments_list.py
```

#### Step 2: Review Results
```bash
# View detailed report
cat judgment_status_report.txt

# Count missing judgments
wc -l missing_judgments_list.txt
wc -l incomplete_judgments_list.txt
```

#### Step 3: Generate Missing Judgments
```bash
# Option A: Use workflow manager (recommended)
python3 missing_judgments_workflow.py --step all

# Option B: Use automation directly
python3 automate_arena_hard_judgment.py --missing-models-file missing_judgments_list.txt --submit
python3 automate_arena_hard_judgment.py --missing-models-file incomplete_judgments_list.txt --submit
```

## Current Status

Based on your latest scan:

### Model Answers Status
#### Missing Models: 2 models
- `tulu3-8b-rank1024-alpha1e6-010-step48000`
- `tulu3-8b-rank64-alpha1e5-005-step24000`

#### Incomplete Models: 124 models
- **Range**: 28 to 727 lines (target: 750 lines)
- **Average**: 388.3 lines per file
- **Most incomplete**: `tulu3-8b-rank1024-alpha1e5-005-step24000` (28 lines)

#### Complete Models: 144 models
- Models with full 750 lines generated

#### Total Coverage: 268/270 models (99.3% with answers)
- **Complete coverage**: 144/270 models (53.3%)
- **Need re-generation**: 124 models

### Judgment Status
To check current judgment status, run:
```bash
python3 check_judgment_status.py
python3 create_missing_judgments_list.py
```

This will generate detailed reports showing:
- Models without judgment files (missing judgments)
- Models with incomplete judgments (< 750 lines)
- Complete judgment statistics
- Categorized breakdown by model patterns

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
├── incomplete_models_list.txt              # Simple incomplete models list
├── missing_model_answers_report.txt        # Detailed answers report
├── missing_judgments_list.txt              # Simple missing judgments list  
├── incomplete_judgments_list.txt           # Simple incomplete judgments list
├── judgment_status_report.txt              # Detailed judgments report
├── check_missing_model_answers.py          # Comprehensive answers analyzer  
├── create_missing_models_list.py           # Simple answers list generator
├── create_incomplete_models_list.py        # Incomplete answers list generator
├── check_judgment_status.py               # Comprehensive judgments analyzer
├── create_missing_judgments_list.py       # Simple judgments list generator
├── missing_models_workflow.py             # Answers workflow manager
├── missing_judgments_workflow.py          # Judgments workflow manager
├── automate_arena_hard_generation.py      # Enhanced with missing models support
├── automate_arena_hard_judgment.py        # Enhanced with missing judgments support
├── arena-hard-auto/data/arena-hard-v2.0/model_answer/    # Answer files directory
└── arena-hard-auto/data/arena-hard-v2.0/model_judgment/  # Judgment files directory
```

## Integration with Existing Tools

The missing models and judgments tools integrate seamlessly with your existing automation:

1. **Answer Generation**: Enhanced to support missing models files
2. **Judgment Generation**: Enhanced to support missing judgments files  
3. **Monitoring**: Shows progress for both complete and missing models/judgments
4. **Workflow Management**: Coordinates the entire pipeline from answers to judgments
5. **Status Tracking**: Comprehensive reporting for both answers and judgments

## Best Practices

1. **Regular Checks**: Run missing models and judgments checks periodically to track progress
2. **Batch Processing**: Use missing models/judgments lists to process only what's needed
3. **Validation**: Always verify completion after answer/judgment generation
4. **Documentation**: Keep reports for tracking and debugging
5. **Resource Management**: Generate answers/judgments for missing models in smaller batches if needed
6. **Sequential Processing**: Generate answers first, then judgments (judgments require existing answers)
7. **Progress Monitoring**: Use workflow managers for automated progress tracking

## Troubleshooting

### Common Issues

#### Answer Generation Issues

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

#### Judgment Generation Issues

5. **"Judgment directory not found"**
   - Verify Arena Hard auto directory structure: `arena-hard-auto/data/arena-hard-v2.0/model_judgment/neuralmagic-llama3.1-70b-instruct-fp8/`
   - Check if any judgment generation has been run

6. **"No judgment files found"**
   - Ensure answers exist before generating judgments (judgments require existing model answers)
   - Check that `baseline_model` (llama3.1-8b-instruct) has answers
   - Verify judge model is properly configured

7. **"Incomplete judgments (< 750 lines)"**
   - Job may have been interrupted or timed out
   - Check SLURM logs for errors in judgment generation
   - Re-run judgment generation for affected models

8. **"Judge server startup failed"**
   - Check GPU availability and memory requirements
   - Verify judge model path exists
   - Check vLLM server logs for detailed error messages

### Commands for Quick Fixes

#### Answer-Related Fixes
```bash
# Check if answer directory exists
ls -la arena-hard-auto/data/arena-hard-v2.0/model_answer/

# Count actual answer files
ls arena-hard-auto/data/arena-hard-v2.0/model_answer/*.jsonl | wc -l

# Check configuration file
head -20 arena_hard_models_to_test.txt

# Verify model exists in API config
grep "tulu3-8b-rank64-alpha1e5-001-step48000" arena-hard-auto/config/api_config.yaml

# Check specific model answer file
ls -la arena-hard-auto/data/arena-hard-v2.0/model_answer/tulu3-8b-rank64-alpha1e5-001-step48000.jsonl
wc -l arena-hard-auto/data/arena-hard-v2.0/model_answer/tulu3-8b-rank64-alpha1e5-001-step48000.jsonl
```

#### Judgment-Related Fixes
```bash
# Check if judgment directory exists
ls -la arena-hard-auto/data/arena-hard-v2.0/model_judgment/neuralmagic-llama3.1-70b-instruct-fp8/

# Count actual judgment files
ls arena-hard-auto/data/arena-hard-v2.0/model_judgment/neuralmagic-llama3.1-70b-instruct-fp8/*.jsonl | wc -l

# Check specific model judgment file
ls -la arena-hard-auto/data/arena-hard-v2.0/model_judgment/neuralmagic-llama3.1-70b-instruct-fp8/tulu3-8b-rank64-alpha1e5-001-step48000.jsonl
wc -l arena-hard-auto/data/arena-hard-v2.0/model_judgment/neuralmagic-llama3.1-70b-instruct-fp8/tulu3-8b-rank64-alpha1e5-001-step48000.jsonl

# Check baseline model has answers (required for judgments)
ls -la arena-hard-auto/data/arena-hard-v2.0/model_answer/llama3.1-8b-instruct.jsonl

# Check judge model path
ls -la /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Meta-Llama-3.1-70B-Instruct-FP8/

# Check recent SLURM logs for judgment errors
ls -la logs/arena-hard/judgment_batches/ | tail -10

# Monitor running judgment jobs
squeue -u $USER | grep judge
```
