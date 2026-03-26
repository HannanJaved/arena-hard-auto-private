# AlpacaEval automation

This workflow adds AlpacaEval generation + scoring alongside the Arena Hard scripts. It uses VLLM OpenAI-compatible servers for **both** generation and judgment, mirroring the Arena Hard flow, and computes **both** raw and length-controlled win rates from the same annotations.

## What gets created

- Generation job scripts under `generated_alpaca_eval_scripts/`
- Judgment job scripts under `generated_alpaca_eval_judgment_scripts/`
- Outputs under `alpaca_eval_outputs/<model_name>/`
  - `model_outputs.json`
  - `annotations.json`
  - `leaderboard_length_controlled.csv`
  - `leaderboard_raw.csv`

## Run for a single model

Use `run_alpaca_eval.py` for generating model outputs and `run_alpaca_eval_judgment.py` to score them.

```bash
python /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts/run_alpaca_eval.py \
  --model-name tulu3-8b-rank64-alpha1e5-001-step48000 \
  --openai-base-url http://localhost:8000/v1 \
  --requires-chatml

python /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts/run_alpaca_eval_judgment.py \
  --model-name tulu3-8b-rank64-alpha1e5-001-step48000 \
  --annotators-config /data/horse/ws/hama901h-BFTranslation/generated_alpaca_eval_judgment_configs/alpaca_eval_judge_Qwen3-Next-80B-A3B-Instruct-FP8.yaml \
  --openai-base-url http://localhost:8001/v1 \
  --save-raw \
  --save-length-controlled
```

## Batch generation with SLURM

Generate scripts for multiple models (using the same `api_config.yaml` as Arena Hard):

```bash
python /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts/automate_alpaca_eval.py \
  --models /data/horse/ws/hama901h-BFTranslation/tulu3.txt \
  --requires-chatml \
  --dry-run
```

Submit jobs after verifying the scripts:

```bash
python /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts/automate_alpaca_eval.py \
  --models /data/horse/ws/hama901h-BFTranslation/tulu3.txt \
  --requires-chatml \
  --submit

## Judgment jobs

Use `automate_alpaca_eval_judgment.py` to run Qwen3-Next (default) as the judge:

```bash
python /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/scripts/automate_alpaca_eval_judgment.py \
  --models /data/horse/ws/hama901h-BFTranslation/tulu3.txt \
  --save-raw \
  --save-length-controlled \
  --dry-run
```
```

## Notes

- The default prompt template for generation is `{instruction}`; set `--chat-template` in the generation automation script if your VLLM server needs a specific chat template.
- The default judge prompt template is AlpacaEval’s classifier prompt `alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt`.
- Use `--max-instances` in generation for a quick smoke test.
