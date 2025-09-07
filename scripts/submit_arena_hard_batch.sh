#!/bin/bash
# Quick batch submission script for Arena Hard answer generation

WORKSPACE_ROOT="/data/horse/ws/hama901h-BFTranslation"
SCRIPT_PATH="$WORKSPACE_ROOT/automate_arena_hard_generation.py"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all              Process all tulu3 models"
    echo "  --models MODEL...  Process specific models"
    echo "  --dry-run          Generate scripts without submitting"
    echo "  --list             List available models"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all                                    # Process all models"
    echo "  $0 --models tulu3-8b-rank64-alpha1e5-001-step48000  # Process specific model"
    echo "  $0 --dry-run --all                         # Generate scripts without submitting"
}

# Function to list available models
list_models() {
    echo "Available tulu3 models:"
    python3 -c "
import yaml
with open('$WORKSPACE_ROOT/arena-hard-auto/config/api_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
for model_name in sorted(config.keys()):
    if model_name.startswith('tulu3-8b-'):
        print(f'  {model_name}')
"
}

# Parse command line arguments
case "$1" in
    --help|-h)
        usage
        exit 0
        ;;
    --list)
        list_models
        exit 0
        ;;
    --all)
        if [[ "$2" == "--dry-run" ]]; then
            python3 "$SCRIPT_PATH" --dry-run
        else
            python3 "$SCRIPT_PATH" --submit
        fi
        ;;
    --models)
        shift
        if [[ "$*" == *"--dry-run"* ]]; then
            # Remove --dry-run from model list
            models=($(echo "$@" | sed 's/--dry-run//g'))
            python3 "$SCRIPT_PATH" --models "${models[@]}" --dry-run
        else
            python3 "$SCRIPT_PATH" --models "$@" --submit
        fi
        ;;
    --dry-run)
        if [[ "$2" == "--all" ]]; then
            python3 "$SCRIPT_PATH" --dry-run
        else
            echo "Error: --dry-run must be used with --all or --models"
            usage
            exit 1
        fi
        ;;
    "")
        echo "Error: No arguments provided"
        usage
        exit 1
        ;;
    *)
        echo "Error: Unknown option '$1'"
        usage
        exit 1
        ;;
esac
