#!/usr/bin/env python3
"""
Simple Missing Models List Generator
Creates a plain text file with just the list of missing model names.
"""

import os
import glob

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
MODEL_ANSWER_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto/data/arena-hard-v2.0/model_answer"
CONFIG_FILE = f"{WORKSPACE_ROOT}/arena_hard_models_to_test.txt"
MISSING_MODELS_FILE = f"{WORKSPACE_ROOT}/missing_models_list.txt"

def load_models_from_config():
    """Load model names from the configuration file."""
    models = set()
    try:
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    models.add(line)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {CONFIG_FILE}")
        return set()
    return models

def get_available_models():
    """Get model names that have answer files."""
    if not os.path.exists(MODEL_ANSWER_DIR):
        print(f"ERROR: Model answer directory not found: {MODEL_ANSWER_DIR}")
        return set()
    
    jsonl_files = glob.glob(f"{MODEL_ANSWER_DIR}/*.jsonl")
    available_models = set()
    
    for file_path in jsonl_files:
        model_name = os.path.basename(file_path).replace('.jsonl', '')
        available_models.add(model_name)
    
    return available_models

def main():
    # Load models from config and scan directory
    config_models = load_models_from_config()
    available_models = get_available_models()
    
    if not config_models:
        return
    
    # Find missing models
    missing_models = config_models - available_models
    
    # Write missing models to file
    try:
        with open(MISSING_MODELS_FILE, 'w') as f:
            for model in sorted(missing_models):
                f.write(f"{model}\n")
        
        print(f"📋 Total models in config: {len(config_models)}")
        print(f"✅ Models with answers: {len(available_models & config_models)}")
        print(f"❌ Missing models: {len(missing_models)}")
        print(f"📄 Missing models list saved to: {MISSING_MODELS_FILE}")
        
        if missing_models:
            print(f"\n💡 To generate answers for missing models:")
            print(f"   python3 automate_arena_hard_generation.py --models-file {MISSING_MODELS_FILE} --submit")
        
    except Exception as e:
        print(f"❌ Failed to write missing models file: {e}")

if __name__ == "__main__":
    main()
