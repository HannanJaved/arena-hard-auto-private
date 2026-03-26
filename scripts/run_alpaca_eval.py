#!/usr/bin/env python3
"""
Generate AlpacaEval model outputs using a local VLLM OpenAI-compatible server.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from alpaca_eval import utils
from alpaca_eval.decoders.openai import openai_completions

WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
DEFAULT_OUTPUT_DIR = f"{WORKSPACE_ROOT}/alpaca_eval_outputs"
DEFAULT_DATASET_REPO = "tatsu-lab/alpaca_eval"
DEFAULT_DATASET_FILE = "alpaca_eval_gpt4_baseline.json"
DEFAULT_PROMPT_TEMPLATE = "{instruction}"


def load_dataset(dataset_file: str, repo_id: str) -> pd.DataFrame:
    path = Path(dataset_file)
    if not path.exists():
        path = Path(
            hf_hub_download(repo_id=repo_id, filename=dataset_file, repo_type="dataset")
        )
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame.from_records(data)


def build_openai_config(base_url: str, output_dir: Path) -> Path:
    config_path = output_dir / "openai_local_config.yaml"
    config_path.write_text(
        "default:\n"
        "  - api_key: 'EMPTY'\n"
        f"    base_url: '{base_url}'\n"
    )
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AlpacaEval outputs using a VLLM server")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET_FILE)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--prompt-template-base", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-config-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-procs", type=int, default=None)
    parser.add_argument("--requires-chatml", action="store_true")
    parser.add_argument("--chatml-wrap", action="store_true", default=True)
    parser.add_argument("--max-instances", type=int, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.openai_config_path:
        openai_config_path = Path(args.openai_config_path)
    elif args.openai_base_url:
        openai_config_path = build_openai_config(args.openai_base_url, output_dir)
    else:
        openai_config_path = None

    df_dataset = load_dataset(args.dataset_file, args.dataset_repo)
    if args.max_instances is not None:
        df_dataset = df_dataset.iloc[: args.max_instances]

    template = utils.read_or_return(args.prompt_template, relative_to=args.prompt_template_base)
    prompts, _ = utils.make_prompts(df_dataset, template=template, batch_size=1)
    if args.requires_chatml and args.chatml_wrap:
        wrapped_prompts = []
        for prompt in prompts:
            stripped = prompt.strip()
            if stripped.startswith("<|im_start|>") and stripped.endswith("<|im_end|>"):
                wrapped_prompts.append(stripped)
            else:
                wrapped_prompts.append(f"<|im_start|>user\n{stripped}\n<|im_end|>")
        prompts = wrapped_prompts

    decoding_kwargs = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_procs": args.num_procs,
        "batch_size": args.batch_size,
    }
    decoding_kwargs["requires_chatml"] = args.requires_chatml
    if openai_config_path is not None:
        decoding_kwargs["client_config_path"] = str(openai_config_path)

    served_model_name = args.served_model_name or args.model_name
    completions = openai_completions(
        prompts=prompts,
        model_name=served_model_name,
        **decoding_kwargs,
    )["completions"]

    outputs = df_dataset.drop(columns=["output", "generator"], errors="ignore").copy()
    outputs["output"] = completions
    outputs["generator"] = args.model_name

    outputs_path = output_dir / "model_outputs.json"
    outputs.to_json(outputs_path, orient="records", indent=2)
    print(f"Saved model outputs to {outputs_path}")


if __name__ == "__main__":
    main()
