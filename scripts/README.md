# scripts/ — Evaluation Automation

This directory automates running full evaluation suites (Arena-Hard, AlpacaEval,
MT-Bench, ELO estimation, and static LM-eval tasks) over lists of locally-hosted
model checkpoints, by generating and submitting SLURM jobs on the cluster.

Everything here is workspace-specific: paths are hardcoded to
`/data/horse/ws/hama901h-BFTranslation`, models are looked up by key in
[`arena-hard-auto/config/api_config.yaml`](../config/api_config.yaml), and jobs
are submitted with `sbatch`. This README covers the top-level orchestrator
(`submit_evals.py`) and the individual automation scripts it wraps. For
narrower, older docs on specific pieces see the other `*_README.md` files in
this directory (`ARENA_HARD_AUTOMATION_README.md`,
`ARENA_HARD_JUDGMENT_README.md`, `ALPACA_EVAL_AUTOMATION_README.md`,
`README_ifeval_submit.md`, `README_ifbench_submit.md`, `MISSING_MODELS_README.md`).

Two of the wrapped scripts (`automate_mtbench.py`, `automate_elo_estimation.py`)
live in the sibling `JudgeArena`/`OpenJury` repos rather than here. Reference
copies of those two scripts, plus every file underneath them that carries
local modifications on top of upstream JudgeArena/OpenJury, are vendored
under [`vendor/`](vendor/) — see the dedicated sections below for what's
vendored and why it's a snapshot rather than a runnable copy.

## Hardcoded paths — what to change if you clone this elsewhere

All of these scripts hardcode `/data/horse/ws/hama901h-BFTranslation` (this
workspace's absolute root) as `WORKSPACE` / `WORKSPACE_ROOT` at the top of the
file. If you're running from a different location — a different cluster user,
a different workspace mount, or someone else's clone — update the constant in
**every** file below (there's no single shared config; each script defines its
own copy):

| File | Constant | Line |
|---|---|---|
| `submit_evals.py` | `WORKSPACE` | [`submit_evals.py:26`](submit_evals.py#L26) |
| `automate_arena_hard_generation_olmo3.py` | `WORKSPACE_ROOT` | [`automate_arena_hard_generation_olmo3.py:15`](automate_arena_hard_generation_olmo3.py#L15) |
| `automate_arena_hard_judgment_olmo3.py` | `WORKSPACE_ROOT` | [`automate_arena_hard_judgment_olmo3.py:17`](automate_arena_hard_judgment_olmo3.py#L17) |
| `vendor/automate_mtbench.py` | `WORKSPACE_ROOT` | [`vendor/automate_mtbench.py:28`](vendor/automate_mtbench.py#L28) |
| `vendor/automate_elo_estimation.py` | `WORKSPACE_ROOT` | [`vendor/automate_elo_estimation.py:19`](vendor/automate_elo_estimation.py#L19) |

A few paths are hardcoded *beyond* the `WORKSPACE_ROOT` substitution and need
separate attention:

- **`automate_arena_hard_judgment_olmo3.py`**:
  - `JUDGE_PATH` ([line 26](automate_arena_hard_judgment_olmo3.py#L26)) — a
    literal absolute path to a default judge checkpoint
    (`checkpoints/meta-llama/Meta-Llama-3.1-70B-Instruct-FP8`), used only if
    `--judge-model` isn't resolvable via `api_config.yaml`.
  - Activates `{WORKSPACE_ROOT}/ah-eval/bin/activate` ([line 259](automate_arena_hard_judgment_olmo3.py#L259))
    — note this is a **different venv** (`ah-eval`) than the one
    `submit_evals.py` otherwise uses for Arena-Hard (`arena-hard-auto/venv`).
    Make sure `ah-eval` exists or change this line to point at your Arena-Hard venv.
  - Chat template `{WORKSPACE_ROOT}/checkpoints/olmo3_chat_template.jinja`
    ([line 292](automate_arena_hard_judgment_olmo3.py#L292)) — swap for
    `checkpoints/meta-llama/tulu_template.j2` if judging Tulu models (see the
    comment at the top of the file).
- **`automate_arena_hard_generation_olmo3.py`**: same chat template path,
  hardcoded at [line 198](automate_arena_hard_generation_olmo3.py#L198).
- **`vendor/automate_mtbench.py`**: `DEFAULT_VLLM_EXEC` points at
  `{WORKSPACE_ROOT}/arena-hard-auto/venv/bin/python` ([line 57](vendor/automate_mtbench.py#L57))
  for the judge server, separately from `DEFAULT_PYTHON_EXEC`
  (`venv-openjury`) used for the JudgeArena driver — both need to exist.
- All five scripts also assume sibling directories at the workspace root:
  `arena-hard-auto/`, `JudgeArena/` (or its vendored copy — see below),
  `OpenJury/` (or its vendored copy), `logs/`, and the four `venv*`
  directories listed in Prerequisites. Moving any of these relative to each
  other breaks the defaults.

None of these are read from environment variables or a shared config file —
they're plain Python string constants, so "changing the workspace path" means
editing source, not exporting a variable.

## Prerequisites

1. **A model entry in `api_config.yaml`.** Every script resolves models by key
   against [`arena-hard-auto/config/api_config.yaml`](../config/api_config.yaml).
   Add your checkpoint there before running anything, e.g.:
   ```yaml
   my-model-name:
     model: /data/horse/ws/hama901h-BFTranslation/checkpoints/my-model
     endpoints: null
     api_type: openai
     parallel: 32
   ```
2. **A models file** — a plain text file with one model key per line (lines
   starting with `#` are ignored). This is the `--models-file` argument
   accepted by every script below.
3. **The right virtualenv.** Each evaluation family runs in its own venv
   (see table below) because of conflicting dependency versions (vLLM,
   alpaca-eval, lm-eval, etc). Scripts are normally invoked with that venv's
   `python`, not whatever `python` is on your `$PATH`.
4. **SLURM access** on the `capella` partition (or pass `--partition` to
   override where supported).

| Venv | Path | Used for |
|---|---|---|
| Arena-Hard | `arena-hard-auto/venv/bin/python` | Arena-Hard generation + judgment |
| AlpacaEval | `venv-alpacaeval/bin/python` | AlpacaEval generation + judgment |
| OpenJury | `venv-openjury/bin/python` | MT-Bench (JudgeArena) + ELO estimation (OpenJury) |
| LM-eval | `venv-lm-eval/bin/python` | arc_challenge, gpqa, gsm8k, hellaswag, ifeval, piqa, truthfulqa |

All scripts share a common pattern:
- Called with no submission flag → **generate** SLURM scripts/configs only, print what *would* run.
- `--dry-run` → generate scripts/configs, print the `sbatch` commands, don't submit.
- `--submit` → generate and actually `sbatch` the jobs.

Monitor jobs with `squeue -u $USER`; cancel with `scancel -u $USER` (or
`scancel <job_id>` for a single job).

---

## submit_evals.py — run everything for a model list

The top-level orchestrator. Given one models file and a baseline, it runs (in
order) Arena-Hard generation → Arena-Hard judgment → AlpacaEval generation →
AlpacaEval judgment → MT-Bench → ELO estimation → the 7 static LM-eval tasks,
calling each dedicated script under the correct venv.

It shells out to scripts in **three repos**: this directory
(`arena-hard-auto/scripts/`), `JudgeArena/scripts/automate_mtbench.py`, and
`OpenJury/scripts/automate_elo_estimation.py` — all assumed to be siblings
under the workspace root.

```bash
cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts

# Preview what would run for every task, without submitting anything
python submit_evals.py --models-file models.txt --baseline llama3.1-8b-instruct

# Generate scripts/configs but don't submit (sbatch commands are printed)
python submit_evals.py --models-file models.txt --baseline llama3.1-8b-instruct --dry-run

# Actually submit all jobs
python submit_evals.py --models-file models.txt --baseline llama3.1-8b-instruct --submit

# Skip any task a model has already completed (checks output files/logs on disk)
python submit_evals.py --models-file models.txt --baseline llama3.1-8b-instruct --submit --skip-completed
```

Key flags:
- `--baseline` (required) — baseline model key used for MT-Bench and Arena-Hard
  pairwise judgment.
- `--judge-model` (default `Qwen3-Next-80B-A3B-Instruct-FP8`) — judge used for
  Arena-Hard, AlpacaEval, MT-Bench, and ELO.
- `--skip-completed` — before submitting, checks each task's output location
  (`model_answer/{model}.jsonl`, `alpaca_eval_outputs/{model}/model_outputs.json`,
  `evaluation_results/judgearena-mtbench/**/results-*.json`,
  `evaluation_results/openjury-elo/**/summary.json`,
  `logs/LM-eval/{task}_{model}_*.out` containing `END TIME:`) and only submits
  models that are still pending, per task. Also drops any model whose
  checkpoint path in `api_config.yaml` doesn't exist on disk.
- `--rerun` — force MT-Bench to rerun for all models instead of skipping
  existing results.
- When submitting (`--submit`), Arena-Hard and AlpacaEval judgment jobs are
  chained via SLURM `--dependency afterok:<job_id>` on their own generation
  jobs, so judgment won't start until generation finishes.

Run this when you want the full evaluation sweep for a batch of checkpoints
without hand-invoking each script. To run just one piece (e.g. only
Arena-Hard, or only re-judging), call that script directly — see below.

---

## automate_arena_hard_generation_olmo3.py — Arena-Hard answer generation

Generates model answers for the Arena-Hard benchmark. For each model, it
writes a `gen_answer_config_{model}.yaml` and a SLURM script that: starts a
vLLM OpenAI-compatible server for the model on GPU 0, waits for its `/health`
endpoint, runs [`gen_answer.py`](../gen_answer.py) against it, then tears the
server down.

```bash
cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts
source ../venv/bin/activate   # arena-hard-auto/venv

# Generate configs/scripts only (prints sbatch commands to run manually)
python automate_arena_hard_generation_olmo3.py --models-file models.txt

# Generate + submit
python automate_arena_hard_generation_olmo3.py --models-file models.txt --submit

# Process every tulu3-* model with an "alpha" variant found in api_config.yaml
python automate_arena_hard_generation_olmo3.py --all --submit

# Pick a different benchmark variant
python automate_arena_hard_generation_olmo3.py --models-file models.txt \
    --bench-name arena-hard-v2.0 --submit
```

Notes:
- `--models` accepts either explicit model keys or a single `.txt` file path.
- `--missing-models-file` targets only models missing answers (typically the
  output of `check_missing_model_answers.py` / `create_missing_models_list.py`).
- Uses `checkpoints/olmo3_chat_template.jinja` as the chat template and
  `--partition capella --gres=gpu:1 --time 04:00:00` by default; edit the
  script's `create_slurm_script()` to change SLURM resources.
- Output lands in `data/arena-hard-v2.0/model_answer/{model}.jsonl`. This is
  what `submit_evals.py`'s `--skip-completed` checks for.
- There's also a non-olmo3 variant (`automate_arena_hard_generation.py`) and a
  no-vLLM variant (`automate_arena_hard_generation_no_vllm.py`) for
  API-hosted models — use `_olmo3` for local checkpoints using the olmo3 chat
  template (this is what `submit_evals.py` calls).

---

## Arena-Hard judgment — automate_arena_hard_judgment_olmo3.py + judgement.py (gen_judgment.py)

Judging is two layers: `automate_arena_hard_judgment_olmo3.py` generates and
submits SLURM batches; each batch job starts a vLLM judge server and calls
[`gen_judgment.py`](../gen_judgment.py) (the actual pairwise LLM-judge logic)
against it.

### automate_arena_hard_judgment_olmo3.py

```bash
cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts

# Validate: models exist in api_config and have generated answers
python automate_arena_hard_judgment_olmo3.py --models-file models.txt \
    --baseline instruct --validate-only

# Dry run
python automate_arena_hard_judgment_olmo3.py --models-file models.txt \
    --baseline instruct --dry-run

# Submit, batching N models per judge-server job
python automate_arena_hard_judgment_olmo3.py --models-file models.txt \
    --baseline instruct --judge-model Qwen3-Next-80B-A3B-Instruct-FP8 \
    --batch-size 5 --submit
```

Key flags:
- `--baseline` — one of the keys in `BASELINE_CONFIGS` inside the script
  (`instruct`, `base`, `tulu_finetuned`, `tulu_sft`, `tulu_dpo`), each mapping
  to a concrete model key (e.g. `instruct` → `llama3.1-8b-instruct`). Update
  `BASELINE_CONFIGS` in the script if you need a new baseline alias.
- `--judge-model` (default `neuralmagic-llama3.1-70b-instruct-fp8` in this
  script; `submit_evals.py` overrides it to `Qwen3-Next-80B-A3B-Instruct-FP8`).
- `--batch-size` — how many models share one judge-server SLURM job
  (default 1; larger batches amortize the judge server startup cost).
- `--dependency afterok:<job_id>[:<job_id>...]` — wait for upstream
  generation jobs before starting (used internally by `submit_evals.py`).
- Judgments land in
  `data/arena-hard-v2.0/model_judgment/{judge_model}/compared_with_{baseline}/{model}.jsonl`.

### gen_judgment.py ("judgement.py")

Called by the SLURM job generated above — not usually invoked by hand, but you
can run it directly against an already-running judge server:

```bash
cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto
python gen_judgment.py \
    --setting-file generated_judgment_configs/arena_hard_judgment_batch_1.yaml \
    --endpoint-file config/api_config.yaml
```

- `--setting-file` (default `config/arena-hard-v2.0.yaml`) defines
  `judge_model`, `baseline`, `temperature`, `max_tokens`, `bench_name`,
  regex patterns for parsing `[[A>B]]`-style verdicts, the judge prompt
  template, and `model_list` (the models to judge in this run).
- `--endpoint-file` (default `config/api_config.yaml`) provides the judge
  model's endpoint/API settings.
- For each model in `model_list` it runs the pairwise judge twice per question
  (A-vs-B and B-vs-A, "round 1"/"round 2") against the baseline, and appends
  results to `data/{bench_name}/model_judgment/{judge_model}/{model}.jsonl`.
  Already-judged question IDs are skipped on rerun (idempotent/resumable).
- Requires a live OpenAI-compatible endpoint for `judge_model` (per
  `api_config.yaml`) — the automation script above handles spinning up vLLM
  for you; running `gen_judgment.py` standalone means you must already have a
  server up at that endpoint.

See [`ARENA_HARD_JUDGMENT_README.md`](ARENA_HARD_JUDGMENT_README.md) for
details on invalid-judgment detection/cleanup (`identify_invalid_judgments.py`)
and progress monitoring (`monitor_arena_hard_judgments.py`).

---

## automate_mtbench.py — MT-Bench / JudgeArena

Canonically lives in `JudgeArena/scripts/automate_mtbench.py` (sibling repo)
— `submit_evals.py` calls it there, under `venv-openjury`. Runs each model
against a baseline ("model_A" vs "model_B") on MT-Bench, judged by an LLM.

A copy is vendored at [`vendor/automate_mtbench.py`](vendor/automate_mtbench.py)
for reference, along with the two `judgearena` modules it/`generate_and_evaluate`
depend on that carry local modifications beyond upstream JudgeArena
(`vendor/judgearena/utils.py`, `vendor/judgearena/mt_bench/mt_bench_utils.py`).
**The vendored copy is not wired up to run** — `automate_mtbench.py` still
imports the full `judgearena` package (`judgearena.generate_and_evaluate`,
etc.) at runtime, so you need an actual clone of `JudgeArena` as a sibling
directory (with those two files' local changes applied) for anything to
execute; see [Hardcoded paths](#hardcoded-paths--what-to-change-if-you-clone-this-elsewhere).
The vendored files are there so the modified logic is visible without
checking out JudgeArena separately — treat them as a snapshot, not a working
copy, and keep them in sync by hand if JudgeArena changes.

```bash
cd /data/horse/ws/hama901h-BFTranslation
source venv-openjury/bin/activate

# Pair every model in models.txt against a fixed baseline
python JudgeArena/scripts/automate_mtbench.py \
    --models-file models.txt --baseline-model llama3.1-8b-instruct \
    --judge Qwen3-Next-80B-A3B-Instruct-FP8 --submit

# Or supply explicit pairs instead of models-file + baseline
python JudgeArena/scripts/automate_mtbench.py \
    --pairs-file pairs.txt --submit   # each line: model_a, model_b

# Skip pairs that already have results; rerun everything with --rerun-all
python JudgeArena/scripts/automate_mtbench.py \
    --models-file models.txt --baseline-model llama3.1-8b-instruct \
    --skip-existing --submit
```

Notes:
- Each job starts a vLLM server for the **judge**, then runs
  `judgearena.generate_and_evaluate` in-process for model_A/model_B (in-process
  vLLM providers, prefixed `VLLM/`, to sidestep CUDA-graph conflicts with a
  second server).
- Uses 2 GPUs and 128G mem by default (`--num-gpus`, `--mem`, `--partition`,
  `--time` to override); judge server and model generation run on separate
  `CUDA_VISIBLE_DEVICES` within the same job.
- Results land under `evaluation_results/judgearena-mtbench/{dataset}-{model_a}-{model_b}-{judge}-{swap_mode}/`,
  which is what `_is_mtbench_completed()` in `submit_evals.py` scans
  (matching on the `model_A` field inside `results-*.json`, since long names
  get hashed in the directory name).

---

## automate_elo_estimation.py — OpenJury ELO estimation

Canonically lives in `OpenJury/scripts/automate_elo_estimation.py` (sibling
repo) — `submit_evals.py` calls it there, under `venv-openjury`. Estimates an
Elo rating for each model by sampling battles from a reference arena
(LMArena or ComparIA) and judging them with an LLM.

A copy is vendored at [`vendor/automate_elo_estimation.py`](vendor/automate_elo_estimation.py),
along with the three `openjury` modules it depends on that carry local
modifications beyond upstream OpenJury (`vendor/openjury/estimate_elo_ratings.py`,
`vendor/openjury/evaluate.py`, `vendor/openjury/generate.py`). Same caveat as
MT-Bench above: **this is a reference snapshot, not a runnable copy** —
`automate_elo_estimation.py` calls `OPENJURY_DIR/openjury/estimate_elo_ratings.py`
as a subprocess against the real `OpenJury/` clone, so that sibling directory
must exist (with those three files' local changes applied) for jobs to run.

```bash
cd /data/horse/ws/hama901h-BFTranslation
source venv-openjury/bin/activate

python OpenJury/scripts/automate_elo_estimation.py \
    --models-file models.txt --judge Qwen3-Next-80B-A3B-Instruct-FP8 \
    --arena LMArena --n_instructions 500 --submit
```

Notes:
- `--arena` is `LMArena` (default) or `ComparIA`.
- `--n_instructions` (default 500) controls how many battles are sampled per
  model; `--swap_mode both` (default) runs both answer orderings to correct
  for position bias.
- Model and judge both load in-process via vLLM (`VLLM/{path}` provider
  strings) inside the same SLURM job — no separate server process, so
  `--num-gpus 1` is usually enough unless the judge is very large.
- Results land in `evaluation_results/openjury-elo/**/summary.json`, matched
  by `model` + `arena` fields — what `submit_evals.py`'s `--skip-completed`
  checks for.
- Like the other scripts, existing results trigger an interactive
  rerun/skip/quit prompt unless you pass `--skip-existing` or `--rerun-all`.

---

## Typical end-to-end workflow

```bash
cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts

# 1. Add your model(s) to config/api_config.yaml, then list their keys
cat > /tmp/models.txt <<EOF
my-model-checkpoint-1
my-model-checkpoint-2
EOF

# 2. Dry-run the full sweep to sanity-check what will happen
python submit_evals.py --models-file /tmp/models.txt \
    --baseline llama3.1-8b-instruct --dry-run

# 3. Submit everything, skipping anything already completed
python submit_evals.py --models-file /tmp/models.txt \
    --baseline llama3.1-8b-instruct --submit --skip-completed

# 4. Monitor
squeue -u $USER

# 5. Once done, aggregate Arena-Hard results
cd ..
python show_result.py
```
