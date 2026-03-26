#!/usr/bin/env python3
"""
Run AlpacaEval judgment for a single model output file using a local VLLM judge.
Generates annotations and computes raw + length-controlled win rates.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from alpaca_eval import annotators, metrics, utils

WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
DEFAULT_OUTPUT_DIR = f"{WORKSPACE_ROOT}/alpaca_eval_outputs"
DEFAULT_DATASET_REPO = "tatsu-lab/alpaca_eval"
DEFAULT_DATASET_FILE = "alpaca_eval_gpt4_baseline.json"


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
    config_path = output_dir / "openai_local_judge_config.yaml"
    config_path.write_text(
        "default:\n"
        "  - api_key: 'EMPTY'\n"
        f"    base_url: '{base_url}'\n"
    )
    return config_path


def write_leaderboard(output_dir: Path, filename: str, model_name: str, metrics_dict: dict) -> None:
    df = pd.DataFrame.from_dict({model_name: metrics_dict}, orient="index")
    df.to_csv(output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlpacaEval judgment for a model output file")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-outputs", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET_FILE)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--annotators-config", required=True)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-config-path", default=None)
    parser.add_argument("--annotation-chunksize", type=int, default=128)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-length-controlled", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_outputs_path = Path(args.model_outputs) if args.model_outputs else output_dir / "model_outputs.json"
    if not model_outputs_path.exists():
        raise FileNotFoundError(f"Model outputs not found: {model_outputs_path}")

    if args.openai_config_path:
        openai_config_path = Path(args.openai_config_path)
    elif args.openai_base_url:
        openai_config_path = build_openai_config(args.openai_base_url, output_dir)
    else:
        openai_config_path = None

    reference_outputs = load_dataset(args.dataset_file, args.dataset_repo)
    model_outputs = utils.load_or_convert_to_dataframe(model_outputs_path)

    annotators_config = Path(args.annotators_config)
    if not annotators_config.exists():
        raise FileNotFoundError(f"Annotators config not found: {annotators_config}")

    annotator = annotators.PairwiseAnnotator(
        annotators_config=str(annotators_config),
    )

    decoding_kwargs = {"chunksize": args.annotation_chunksize}
    if openai_config_path is not None:
        decoding_kwargs["client_config_path"] = str(openai_config_path)

    annotations = annotator.annotate_head2head(
        outputs_1=reference_outputs,
        outputs_2=model_outputs,
        **decoding_kwargs,
    )

    annotations_df = utils.convert_to_dataframe(annotations)
    annotations_path = output_dir / "annotations.json"
    annotations_df.to_json(annotations_path, orient="records", indent=2)

    avg_length = int(model_outputs["output"].str.len().mean())

    save_raw = args.save_raw or not args.save_length_controlled
    save_length_controlled = args.save_length_controlled or not args.save_raw

    if save_length_controlled:
        lc_metrics = metrics.get_length_controlled_winrate(
            annotations_df,
            save_weights_dir=output_dir / "weights",
        )
        lc_metrics["avg_length"] = avg_length
        write_leaderboard(output_dir, "leaderboard_length_controlled.csv", args.model_name, lc_metrics)

    if save_raw:
        raw_metrics = metrics.get_winrate(annotations_df)
        raw_metrics["avg_length"] = avg_length
        write_leaderboard(output_dir, "leaderboard_raw.csv", args.model_name, raw_metrics)

    print(f"Saved annotations to {annotations_path}")


if __name__ == "__main__":
    main()
