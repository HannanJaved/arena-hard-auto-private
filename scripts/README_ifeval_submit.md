# IFEval submission helper

This helper submits IFEval Slurm jobs for a list of model names stored in
`arena-hard-auto/config/api_config.yaml`.

## Usage

- Prepare a text file with one model name per line (matching keys in `api_config.yaml`).
- Run the helper to submit jobs or use `--dry-run` to print the generated Slurm scripts.

### Example

```bash
python /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts/submit_ifeval_from_list.py \
  --models-txt /data/horse/ws/hama901h-BFTranslation/random_search_dpo.txt \
  --dry-run
```

## Notes

- The defaults mirror `lm-eval-scripts/ifeval.sh` (paths, Slurm settings, environment).
- Use flags like `--partition`, `--time`, or `--batch-size` to customize each run.
- The helper skips models that do not have a `model:` entry in `api_config.yaml`.
