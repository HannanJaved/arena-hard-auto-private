#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def copy_files_to_step_dirs(parent_path):
    """
    Copy specified files from parent directory to all step_xxxx subdirectories.
    
    Args:
        parent_path: Path to the parent directory containing step_xxxx subdirs
    """
    # Files to copy
    files_to_copy = [
        'chat_template.jinja',
        'config.json',
        'generation_config.json',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'tokenizer.json'
    ]
    
    parent_path = Path(parent_path)
    
    # Find all step_xxxx directories
    step_dirs = [d for d in parent_path.iterdir() 
                 if d.is_dir() and d.name.startswith('step_')]
    
    if not step_dirs:
        print(f"No step_xxxx directories found in {parent_path}")
        return
    
    print(f"Found {len(step_dirs)} step directories")
    
    # Copy files to each step directory
    for step_dir in sorted(step_dirs):
        print(f"\nProcessing {step_dir.name}...")
        
        for filename in files_to_copy:
            source_file = parent_path / filename
            dest_file = step_dir / filename
            
            if source_file.exists():
                try:
                    shutil.copy2(source_file, dest_file)
                    print(f"  ✓ Copied {filename}")
                except Exception as e:
                    print(f"  ✗ Error copying {filename}: {e}")
            else:
                print(f"  ⚠ {filename} not found in parent directory")
    
    print("\nDone!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python copy_files_for_intermediate_checkpoints.py <parent_directory>")
        sys.exit(1)
    
    copy_files_to_step_dirs(sys.argv[1])