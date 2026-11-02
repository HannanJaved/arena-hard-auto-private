#!/usr/bin/env python3
"""Append api_config.yaml entries (and a models list) for the fp32-master reruns of the Qwen3-0.6B / 1.7B Dolci DPO grids.

Checkpoints come from finetuning/generate_fp32_master_sweeps.py:
  0.6B  fp32 DDP         key infix "fp32",  dirs ...-AO-FP32-alpha (LR2e-6/beta0.01: ...-AO-FP32, from capella)
  1.7B  bf16 cfg, ZeRO-3 key infix "zero3", dirs ...-AO-ZeRO3-alpha
Naming follows the existing grid: unsuffixed key = final export, -stepN = trainer checkpoint (200..2000).
Idempotent: keys already in api_config.yaml are skipped, so it can be rerun when more runs finish.
The models list only contains models whose checkpoint already has config.json.

Usage: python add_fp32_rerun_models.py [--write]   (default: dry run)
"""
import argparse
import re
import shutil
import time
from pathlib import Path

import yaml

WS = Path("/data/horse/ws/hama901h-BFTranslation")
CFG = WS / "arena-hard-auto/config/api_config.yaml"
OUT_LIST = WS / "arena-hard-auto/scripts/models_qwen3_{size}_fp32_rerun.txt"   # one list per size: different --baseline
SFT = WS / "checkpoints/Qwen"
STEPS = range(200, 2001, 200)
FIRST_PORT = 9011          # existing entries use <= 9010

SIZES = {
    "0.6b": {"root": SFT / "Qwen3-0.6B-Base/SFT-sweep/Qwen3-0.6B-SFT-LR3e-5-Assistant-Only/olmo3-dpo",
             "name": "Qwen3-0.6B-SFT-LR3e-5-AO", "infix": "fp32", "dir_suffix": "FP32-alpha",
             "lrs": ["1e-6", "2e-6", "4e-6", "8e-6"]},
    "1.7b": {"root": SFT / "Qwen3-1.7B-Base/SFT-sweep/Qwen3-1.7B-SFT-LR3e-5-Assistant-Only/olmo3-dpo",
             "name": "Qwen3-1.7B-SFT-LR3e-5-AO", "infix": "zero3", "dir_suffix": "ZeRO3-alpha",
             "lrs": ["1e-6", "2e-6", "4e-6", "8e-6", "1e-5"]},
}
BETAS = ["0.01", "0.02", "0.04"]
GRID = {("0.6b", "1e-6"): BETAS, ("0.6b", "2e-6"): BETAS, ("0.6b", "4e-6"): BETAS, ("0.6b", "8e-6"): ["0.01"],
        ("1.7b", "1e-6"): BETAS, ("1.7b", "2e-6"): BETAS, ("1.7b", "4e-6"): BETAS, ("1.7b", "8e-6"): ["0.01"],
        ("1.7b", "1e-5"): ["0.01"]}

ENTRY = """{key}:
  model: {model}
  endpoints:
  - api_base: http://localhost:{port}/v1
    api_key: '-'
  api_type: openai
  parallel: 32
  max_tokens: 4096
  temperature: 0.0

"""


def entries():
    """Yield (key, checkpoint_dir) in list order."""
    for (size, lr), betas in GRID.items():
        s = SIZES[size]
        for beta in betas:
            d = s["root"] / f"DPO-LR{lr}_Beta{beta}-{s['name']}-{s['dir_suffix']}"
            if size == "0.6b" and lr == "2e-6" and beta == "0.01":   # capella run: full 11-checkpoint trajectory
                d = s["root"] / f"DPO-LR{lr}_Beta{beta}-{s['name']}-FP32"
            base = f"dpo-qwen3-{size}-sft-lr3e5-ao-{s['infix']}-lr{lr.replace('-', '')}-beta{beta}"
            yield base, d
            for n in STEPS:
                yield f"{base}-step{n}", d / f"checkpoint-{n}"
    # replicate of 0.6B LR2e-6/beta0.01 (alpha, final only) to measure run-to-run variance
    s = SIZES["0.6b"]
    yield ("dpo-qwen3-0.6b-sft-lr3e5-ao-fp32r2-lr2e6-beta0.01",
           s["root"] / f"DPO-LR2e-6_Beta0.01-{s['name']}-FP32-alpha")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    existing = yaml.safe_load(CFG.read_text()) or {}
    port = max(FIRST_PORT, max(int(p) for p in re.findall(r"localhost:(\d+)/", CFG.read_text())) + 1)

    text, ready, missing, skipped = "\n", [], [], 0
    for key, path in entries():
        if key in existing:
            skipped += 1
            continue
        text += ENTRY.format(key=key, model=path, port=port)
        port += 1
        (ready if (path / "config.json").exists() else missing).append(key)
    print(f"{len(ready) + len(missing)} new entries ({len(ready)} checkpoints ready, {len(missing)} not finished), "
          f"{skipped} already in api_config.yaml")
    for k in missing:
        print("  not ready yet:", k)
    if not args.write:
        print("dry run; pass --write to append")
        return
    if text.strip():
        shutil.copy(CFG, f"{CFG}.bak.{int(time.time())}")
        with open(CFG, "a") as fh:
            fh.write(text)
    all_ready = [k for k, p in entries() if (p / "config.json").exists()]
    for size, s in SIZES.items():
        keys = [k for k in all_ready if k.startswith(f"dpo-qwen3-{size}-")]
        out = Path(str(OUT_LIST).format(size=size))
        out.write_text(f"# fp32-master reruns of Qwen3-{size.upper()} (key infix {s['infix']}). "
                       "Unsuffixed keys are the final export, -stepN keys are trainer checkpoints.\n"
                       + "\n".join(keys) + "\n")
        print(f"wrote {len(keys)} models to {out}")
    print(f"appended to {CFG}")

if __name__ == "__main__":
    main()
