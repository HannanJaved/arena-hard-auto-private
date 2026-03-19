# OpenJury ELO submission helper

This helper submits Slurm jobs for `openjury/estimate_elo_ratings.py` using model
names listed in a text file and resolved from `arena-hard-auto/config/api_config.yaml`.
It uses the VLLM provider by prefixing model paths with `VLLM/`.

## Usage

- Prepare a text file with one model name per line (matching keys in `api_config.yaml`) or pass
  model names directly.
- Pick a judge model name that also exists in `api_config.yaml`.
- Run the helper to submit jobs or use `--dry-run` to print the generated scripts.

### Example

```bash
python /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts/submit_elo_from_list.py \
  --models /data/horse/ws/hama901h-BFTranslation/random_search_dpo.txt \
  --judge-model llama3.1-8b-instruct \
  --dry-run

### Single model

```bash
python /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts/submit_elo_from_list.py \
  --models llama3.1-8b-instruct \
  --judge-model llama3.1-8b-instruct \
  --dry-run
```
```

## Notes

- Defaults assume the OpenJury repo lives at `OpenJury/` and the venv is
  `venv-openjury`. Use `--openjury-dir` and `--venv-activate` to override.
- Use `--engine-kwargs` to pass vLLM settings, e.g.
  `'{"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9}'`.
- Missing models in `api_config.yaml` are listed at the end of the run.
