import json
import yaml
import argparse
import os
import concurrent.futures

from tqdm import tqdm

from utils.completion import (
    load_questions,
    registered_api_completion,
    load_questions,
    load_model_answers,
    get_endpoint,
    make_config,
)

from utils.judge_utils import JUDGE_SETTINGS


def _judgment_baseline_matches(path, baseline_name):
    """Return True if the first jsonl record's baseline matches baseline_name."""
    try:
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
        if not first.strip():
            return True
        return json.loads(first).get("baseline") == baseline_name
    except (OSError, json.JSONDecodeError):
        return False


def resolve_judgment_output_file(judge_dir, model, baseline_name):
    """Prefer an existing judgment jsonl (ICLR/compared_with_* first); else write under ICLR.

    Incomplete files under ICLR/compared_with_{baseline}/{model}.jsonl are reused so
    gen_judgment appends in place instead of creating a sibling at the judge root.
    """
    default_path = os.path.join(
        judge_dir, "ICLR", f"compared_with_{baseline_name}", f"{model}.jsonl"
    )
    candidates = []
    if os.path.isdir(judge_dir):
        target_name = f"{model}.jsonl"
        for root, _dirs, files in os.walk(judge_dir):
            if target_name not in files:
                continue
            path = os.path.join(root, target_name)
            if not _judgment_baseline_matches(path, baseline_name):
                continue
            rel = os.path.relpath(path, judge_dir)
            # Prefer ICLR/compared_with_{baseline}, then compared_with_{baseline}, then others.
            if rel.startswith(os.path.join("ICLR", f"compared_with_{baseline_name}")):
                rank = 0
            elif rel.startswith(f"compared_with_{baseline_name}"):
                rank = 1
            elif os.path.dirname(rel) == "":
                rank = 3  # flat under judge dir
            else:
                rank = 2
            candidates.append((rank, path))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    return default_path


def load_existing_judgments_for_files(output_files):
    """Load uid→record maps for each model's resolved judgment file."""
    existing = {}
    for model, path in output_files.items():
        if not os.path.isfile(path):
            continue
        by_uid = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = row.get("uid")
                if uid is not None:
                    by_uid[uid] = row
        existing[model] = by_uid
    return existing


def get_score(judgment, patterns):
    import re
    for pattern in patterns:
        pattern = re.compile(pattern)
        
        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def pairwise_judgment(question, baseline, answer, reference, configs, settings):
    prompt_args = {
        "QUESTION": question['prompt'],
        "ANSWER_A": baseline["messages"][-1]["content"]['answer'],
        "ANSWER_B": answer["messages"][-1]["content"]['answer'],
    }
    
    if reference:
        prompt_args[f"REFERENCE"] = reference["messages"][-1]["content"]['answer']
        
    user_prompt = configs["prompt_template"].format(**prompt_args)
    messages = [
        {
            "role": "system", 
            "content": JUDGE_SETTINGS[question["category"]]["system_prompt"],
        },
        {
            "role": "user", 
            "content": user_prompt,
        }
    ]

    # build arguments for api completions
    kwargs = settings | {
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
    }
    kwargs['temperature'] = configs['temperature']
    kwargs['max_tokens'] = configs['max_tokens']
    
    api_completion_func = registered_api_completion[settings["api_type"]]
    output = api_completion_func(**kwargs)
    
    if output is None:
        return None

    score = get_score(output['answer'], configs["regex_patterns"])

    result = {
        "score": score,
        "judgment": output,
        "prompt": messages,
    }
    return result


def judgment(args):
    answer = args['answer']
    baseline = args['baseline']
    
    output = {
        "uid": args['question']["uid"],
        "category": args['question']["category"],
        "judge": args['configs']['judge_model'],
        "model": answer["model"],
        "baseline": baseline["model"],
        "games": []
    }

    # round 1
    result = pairwise_judgment(
        question=args['question'],
        baseline=baseline,
        answer=answer,
        reference=args['reference'],
        configs=args['configs'],
        settings=args['settings'],
    )
    output["games"].append(result)
        
    # round 2
    result = pairwise_judgment(
        question=args['question'],
        baseline=answer,
        answer=baseline,
        reference=args['reference'],
        configs=args['configs'],
        settings=args['settings'],
    )
    output["games"].append(result)

    with open(args['output_file'], "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/arena-hard-v2.0.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(f'judge model: {configs["judge_model"]}, reference: {configs["reference"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}')

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")

    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
        
    if configs["reference"]:
        assert not configs["reference"] in models, "ERROR: one of the models being evaluated is used as reference."
        ref_answers = [answer_dir[model] for model in configs["reference"]]
    else:
        ref_answers = None
    
    output_files = {}
    judge_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    # Resolve baseline from config (string or per-category dict)
    baseline_config = configs.get("baseline")

    def resolve_baseline_for_category(category):
        if isinstance(baseline_config, dict):
            return baseline_config.get(category)
        return baseline_config

    # Use a stable baseline label for output paths (string baseline, or first dict value).
    if isinstance(baseline_config, dict):
        baseline_label = next(iter(baseline_config.values()), None)
    else:
        baseline_label = baseline_config
    if not baseline_label:
        raise ValueError("Baseline must be set in the judgment setting file.")

    for model in models:
        output_files[model] = resolve_judgment_output_file(
            judge_dir, model, baseline_label
        )
        print(f"Judgment output for {model}: {output_files[model]}")

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_existing_judgments_for_files(output_files)

    endpoint_settings = endpoint_list[configs["judge_model"]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_settings["parallel"]) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                uid = question["uid"]

                kwargs = {}
                kwargs["question"] = question
                if model in model_answers and not uid in model_answers[model]:
                    print(f"Warning: {model} answer to {question['uid']} cannot be found.")
                    continue

                if model in existing_judgments and uid in existing_judgments[model]:
                    count += 1
                    continue

                kwargs["answer"] = model_answers[model][uid]
    
                baseline_name = resolve_baseline_for_category(question["category"])
                if not baseline_name:
                    raise ValueError(
                        f"Baseline not specified for category '{question['category']}' in setting file."
                    )
                kwargs["baseline"] = model_answers[baseline_name][uid]
                
                if ref_answers:
                    kwargs["reference"] = [ref_answer[uid] for ref_answer in ref_answers]
                else:
                    kwargs["reference"] = None
                    
                kwargs["configs"] = configs
                kwargs["settings"] = endpoint_settings
                kwargs["output_file"] = output_files[model]
                                
                future = executor.submit(judgment, kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgments")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
