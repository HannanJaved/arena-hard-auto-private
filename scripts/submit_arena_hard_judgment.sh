#!/bin/bash
# Quick batch submission script for Arena Hard judgment generation

WORKSPACE_ROOT="/data/horse/ws/hama901h-BFTranslation"
SCRIPT_PATH="$WORKSPACE_ROOT/automate_arena_hard_judgment.py"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all                      Judge all tulu3 models"
    echo "  --models MODEL...          Judge specific models"
    echo "  --batch-size N             Number of models per judgment job (default: 10)"
    echo "  --dry-run                  Generate scripts without submitting"
    echo "  --validate                 Only validate models without generating scripts"
    echo "  --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all                                           # Judge all models from config file"
    echo "  $0 --models tulu3-8b-rank64-alpha1e5-001-step48000  # Judge specific model"
    echo "  $0 --dry-run --all                                # Generate scripts without submitting"
    echo "  $0 --validate --all                               # Only validate models"
    echo "  $0 --all --batch-size 5                           # Judge all models, 5 per job"
}

# Parse command line arguments
BATCH_SIZE=""
DRY_RUN=""
MODELS=""
VALIDATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            usage
            exit 0
            ;;
        --all)
            ACTION="--all"
            shift
            ;;
        --models)
            ACTION="--models"
            shift
            # Collect all model names until next option or end
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS="$MODELS $1"
                shift
            done
            ;;
        --batch-size)
            BATCH_SIZE="--batch-size $2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --validate)
            VALIDATE="--validate-only"
            shift
            ;;
        *)
            echo "Error: Unknown option '$1'"
            usage
            exit 1
            ;;
    esac
done

# Check if action is set
if [[ -z "$ACTION" ]]; then
    echo "Error: Must specify --all or --models"
    usage
    exit 1
fi

# Build command
CMD="python3 $SCRIPT_PATH $ACTION"

if [[ -n "$MODELS" ]]; then
    CMD="$CMD$MODELS"
fi

if [[ -n "$BATCH_SIZE" ]]; then
    CMD="$CMD $BATCH_SIZE"
fi

if [[ -n "$VALIDATE" ]]; then
    CMD="$CMD $VALIDATE"
elif [[ -n "$DRY_RUN" ]]; then
    CMD="$CMD $DRY_RUN"
else
    CMD="$CMD --submit"
fi

echo "Executing: $CMD"
eval $CMD
