import os
import glob

def check_jsonl_lines(directory, min_lines=750):
    pattern = os.path.join(directory, "*.jsonl")
    files = glob.glob(pattern)
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        if n_lines < min_lines:
            print(f"{os.path.basename(file)}: {n_lines} lines")

if __name__ == "__main__":
    check_jsonl_lines("/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/data/arena-hard-v2.0/model_judgment/Qwen3-Next-80B-A3B-Instruct-FP8")