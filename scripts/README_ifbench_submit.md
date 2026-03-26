# IFBench submission helper

This helper submits IFBench Slurm jobs for a list of model names stored in a text
file. Each model name maps to a response JSONL file on disk, and the helper can
optionally generate those responses via the IFBench `generate_responses.py` script
using endpoints listed in `arena-hard-auto/config/api_config.yaml`.

## Usage

- Prepare a text file with one model name per line.
- If you already have responses, store them using the default template:
  `{response_dir}/{model_name}{response_suffix}`.
- If you do not have responses, the helper will generate them automatically
  using `api_config.yaml` and launch a vLLM server inside the Slurm job
  (unless `--no-generate-responses` or `--no-start-vllm` is set).
- Run the helper to submit jobs or use `--dry-run` to print the generated Slurm scripts.

### Example

```bash
python /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts/submit_ifbench_from_list.py \
  --models-file /data/horse/ws/hama901h-BFTranslation/random_search_dpo.txt \
  --response-dir /data/horse/ws/hama901h-BFTranslation/IFBench/outputs \
  --api-model-field name \
  --dry-run
```

## Notes

- The defaults assume IFBench was cloned to `/data/horse/ws/hama901h-BFTranslation/IFBench`.
- Use `--response-template` if your response filenames are not `{model_name}.jsonl`.
- Use `--api-model-field` or `--api-model-override` if your OpenAI server expects a
  different model identifier than the config key.
- Use `--no-start-vllm` if you already have a running OpenAI-compatible server.
- vLLM startup uses the model path from `api_config.yaml` and the port from the
  configured endpoint (or 8000 if missing).
- Results are written to one subdirectory per model under `evaluation_results/ifbench`.
