# Static LM-eval tasks

Covers the 7 non-LLM-judge benchmarks run via
[EleutherAI's `lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness):
`arc_challenge`, `gpqa`, `gsm8k`, `hellaswag`, `ifeval`, `piqa`, `truthfulqa`.
Each has its own submit script — `submit_{task}_from_list.py` — that are all
structurally identical (same flags, same SLURM template), just with a
different `--tasks` value passed to `lm_eval`. This doc covers all seven at
once; anything said about one applies to the others unless noted.

Unlike the LLM-judge evals (Arena-Hard, AlpacaEval, MT-Bench, ELO), these
don't need a vLLM server or a judge model — `lm_eval --model hf` loads the
checkpoint directly with Hugging Face `transformers` and runs the whole
benchmark within a single SLURM job.

## Prerequisites

1. **`lm-evaluation-harness` checked out** as a sibling directory
   (`/data/horse/ws/hama901h-BFTranslation/lm-evaluation-harness`) with
   `lm_eval` importable/runnable from it.
2. **The `venv-lm-eval` virtualenv** with `lm_eval` and its dependencies
   installed.
3. **A model entry in `api_config.yaml`** — same as every other script in
   this directory, models are resolved by key against
   [`arena-hard-auto/config/api_config.yaml`](../config/api_config.yaml),
   reading the `model` field as the HF-loadable checkpoint path.
4. **A models file** — one model key per line, `#`-comments allowed.
5. **`module load CUDA`** must work in your shell — every generated SLURM
   script runs this ([Lmod](https://lmod.readthedocs.io/) environment
   modules) before invoking `lm_eval`. See the main
   [`README.md`](README.md#prerequisites) for more on this and on adapting
   it if your cluster doesn't use Lmod.

## Hardcoded paths

Each `submit_{task}_from_list.py` defines its own copy of these constants
near the top of the file — there's no shared config, so if you're running
from a different workspace, edit all 7 files:

| Constant | Default | Purpose |
|---|---|---|
| `DEFAULT_API_CONFIG` | `arena-hard-auto/config/api_config.yaml` | where model keys resolve to checkpoint paths |
| `DEFAULT_VENV_ACTIVATE` | `venv-lm-eval/bin/activate` | venv sourced inside the SLURM job |
| `DEFAULT_LM_EVAL_DIR` | `lm-evaluation-harness` | the harness is `cd`'d into before running `lm_eval` |
| `DEFAULT_LOG_DIR` | `logs/LM-eval` | SLURM stdout/stderr; `submit_evals.py --skip-completed` scans this for `END TIME:` to detect completed runs |
| `DEFAULT_OUTPUT_DIR` | `evaluation_results/{task}` | where `lm_eval --output_path` writes results |
| `DEFAULT_HF_HOME` / `DEFAULT_HF_DATASETS_CACHE` | `.cache` | HF model/dataset cache, exported inside the job |
| `DEFAULT_PYTHONPATH` | `venv-lm-eval/lib/python3.11/site-packages` | forced onto `PYTHONPATH` inside the job (note the hardcoded `python3.11` — update if your venv uses a different Python version) |

All of these can also be overridden per-invocation via CLI flags
(`--api-config`, `--venv-activate`, `--lm-eval-dir`, `--log-dir`,
`--output-dir`, `--hf-home`, `--hf-datasets-cache`, `--pythonpath`) without
editing source — unlike the LLM-judge scripts, you don't have to hand-edit
the file just to relocate the workspace, as long as you pass all seven flags.

## Usage

Submission behavior differs slightly from the rest of the scripts in this
directory: there's no `--submit` flag — jobs submit by default, and
`--dry-run` is what suppresses submission (prints the generated sbatch script
instead).

```bash
cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts

# Preview the generated SLURM script without submitting
python submit_ifeval_from_list.py --models-file models.txt --dry-run

# Submit for real
python submit_ifeval_from_list.py --models-file models.txt

# Same pattern for the other six tasks
python submit_arc_challenge_from_list.py --models-file models.txt
python submit_gpqa_from_list.py          --models-file models.txt
python submit_gsm8k_from_list.py         --models-file models.txt
python submit_hellaswag_from_list.py     --models-file models.txt
python submit_piqa_from_list.py          --models-file models.txt
python submit_truthfulqa_from_list.py    --models-file models.txt
```

Key flags (same across all 7 scripts):
- `--models-file` (required) — text file of model keys.
- `--batch-size` (default 16) — per-GPU batch size passed to `lm_eval`.
- `--dtype` (default `bfloat16`).
- `--gres` (default `gpu:1`) — pass `gpu:N` with N>1 to switch to the
  multi-GPU SLURM template, which wraps the run in `accelerate launch` across
  `srun`.
- `--partition`, `--time`, `--cpus-per-task`, `--mem`, `--exclusive` — SLURM
  resource overrides (defaults: `capella`, `04:00:00`, 4 CPUs, `16G`).
- `--job-name-prefix` (default e.g. `ifeval_`) — SLURM job name prefix; this
  is also embedded in log filenames (`{prefix}{model}_%j.out`), which is what
  `submit_evals.py`'s `--skip-completed` pattern-matches against.

One job is submitted per model (no batching across models like the judge
scripts do). Each job runs `lm_eval --model hf --model_args
pretrained={model_path},dtype=... --tasks {task} --output_path {output_dir}`
directly against the checkpoint — no server, no judge.

## Monitoring & output

- SLURM logs: `logs/LM-eval/{prefix}{model}_<jobid>.out` /
  `.err`. A completed run's `.out` file ends with `END TIME: <date>` — this
  is the exact string `submit_evals.py --skip-completed` greps for to decide
  a model/task pair is already done.
- Results: `evaluation_results/{task}/` (per-task `lm_eval` output — nested
  by model/config per the harness's own conventions).
- `squeue -u $USER` to watch jobs, `scancel -u $USER` (or a specific job ID)
  to cancel.

## Notes

- `ifeval` uses `--gen_kwargs max_new_tokens=1280`; check each script's
  `SBATCH_BODY` template if you need to tune generation-eval tasks
  differently — `arc_challenge`/`gpqa`/`gsm8k`/`hellaswag`/`piqa`/`truthfulqa`
  are mostly likelihood/multiple-choice tasks and don't need that flag, but
  the template structure is otherwise identical across all seven.
- These scripts don't check for existing results before submitting — running
  twice for the same model will submit a duplicate job. Use
  `submit_evals.py --skip-completed` (which wraps all 7) if you want
  automatic skip-if-done behavior, or check `logs/LM-eval/` yourself first.
- `submit_evals.py` invokes these with `--dry-run` when *not* submitting
  (opposite of its own `--dry-run`/`--submit` convention) — that's because
  these scripts submit by default and `submit_evals.py` has to explicitly
  suppress that when the user hasn't asked to submit anything yet.
