"""
Stratified analysis

Usage:
    python stratify.py [results_dir] [--eval_type {em, llm}] [--model MODEL_NAME] [--inference-mode {0-shot,ICL}]

Examples:
    python stratify.py ../results --eval_type em                            # Perform stratification analysis on all models (exact match)
    python stratify.py ../results --eval_type llm                           # Perform stratification analysis on all models (LLM-as-a-Judge)
    python stratify.py ../results --eval_type em --model qwen72B            # Evaluate specific model, all inference modes
    python stratify.py ../results --eval_type em --inference-mode ICL       # Evaluate all models in ICL mode
    python stratify.py --eval_type em --model llama32_vision_90b --inference-mode 0-shot  # Specific model and inference mode
"""

import sys
import os
import glob
import argparse
import json
import numpy as np
from pathlib import Path
from evaluate_EM import get_model_info
from utils import load_data
from collections import defaultdict
from prettytable import PrettyTable
from rich import print

# Get HF_TOKEN from environment variable, with fallback
HF_TOKEN = os.environ.get('HF_TOKEN', '')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running this script.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate exact match accuracy for VLM models on SMMILE benchmark"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="../results",
        help="Directory containing result CSV files (default: ./results)"
    )
    parser.add_argument(
        "--eval_type",
        choices=["em", "llm"],
        help="Evaluation type (exact match or LLM-as-a-judge)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to evaluate (e.g., qwen72B, llama32_vision_90b)"
    )
    parser.add_argument(
        "--inference-mode",
        choices=["0-shot", "ICL"],
        help="Specific inference mode to evaluate"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        choices=["smmile/SMMILE-050525", "smmile/SMMILE-augmented-050825"],
        default="smmile/SMMILE-050525"
    )
    return parser.parse_args()

def print_stratification_results(valid_flags, all_results):
    # Display stratified results across each flag
    ordered_models = sorted([model for model in all_results if model.split(':')[1]=='0-shot']) + sorted([model for model in all_results if model.split(':')[1]=='ICL'])
    for flag in valid_flags: 
        print(f"=====DISPLAYING RESULTS FOR FLAG {flag.upper()}=====")
        table = PrettyTable()
        keys = sorted(list(set([k for model in all_results for k in all_results[model][f"{flag}_accuracy"]])))
        table.field_names = ["Model", "Mode"] + keys
        for model in ordered_models: 
            all_acc = all_results[model][f"{flag}_accuracy"]
            acc = [np.round(all_acc[k], 1) if k in all_acc else '--' for k in keys]
            table.add_row([model.split(':')[0], model.split(':')[1]] + acc)
        print(table)


def main():
    args = parse_arguments()

    # Find JSON files in the results directory
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists

    # Build file pattern based on model and inference-mode filters
    patterns = []

    if args.eval_type=="llm": ext="evaluationllm"
    elif args.eval_type=="em": ext="evaluation"
    else: 
        raise Exception("Invalid value of input parameter eval_type")

    # If specific model is requested
    if args.model:
        if args.inference_mode:
            # Both model and inference mode specified
            patterns.append(os.path.join(results_dir, f"result_{args.model}_{args.inference_mode}.csv"))
        else:
            # Only model specified, try both inference modes
            patterns.append(os.path.join(results_dir, f"result_{args.model}_0-shot_{ext}.json"))
            patterns.append(os.path.join(results_dir, f"result_{args.model}_0-shot_open_{ext}.json"))
            patterns.append(os.path.join(results_dir, f"result_{args.model}_ICL_{ext}.json"))
            patterns.append(os.path.join(results_dir, f"result_{args.model}_ICL_open_{ext}.json"))
            # Support legacy few-shot naming
            patterns.append(os.path.join(results_dir, f"result_{args.model}_few-shot_{ext}.json"))
    else:
        # No specific model, look for all supported models
        if args.inference_mode:
            # Only inference mode specified
            patterns.append(os.path.join(results_dir, f"result_*_{args.inference_mode}_{ext}.json"))
            patterns.append(os.path.join(results_dir, f"result_*_{args.inference_mode}_open_{ext}.json"))
            # Support legacy few-shot to ICL conversion
            if args.inference_mode == "ICL":
                patterns.append(os.path.join(results_dir, f"result_*_few-shot_{ext}.json"))
        else:
            # No filters, look for all result files
            patterns.append(os.path.join(results_dir, f"result_*_0-shot_{ext}.json"))
            patterns.append(os.path.join(results_dir, f"result_*_0-shot_open_{ext}.json"))
            patterns.append(os.path.join(results_dir, f"result_*_ICL_{ext}.json"))
            patterns.append(os.path.join(results_dir, f"result_*_ICL_open_{ext}.json"))
            # Legacy formats
            patterns.append(os.path.join(results_dir, f"result_*_few-shot_{ext}.json"))
            patterns.append(os.path.join(results_dir, f"result_*_with_answers_{ext}.json"))

    json_files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        json_files.extend(matches)

    # Also look in current directory if results_dir doesn't contain any files
    if not json_files:
        for pattern in [p.replace(results_dir + "/", "") for p in patterns]:
            matches = glob.glob(pattern)
            json_files.extend(matches)

    if not json_files:
        print(f"No result files found matching the criteria.")
        if args.model:
            print(f"Model filter: {args.model}")
        if args.inference_mode:
            print(f"Inference mode filter: {args.inference_mode}")
        print("Check the directory and file naming conventions.")
        sys.exit(1)

    print(f"Found {len(json_files)} result files to evaluate:")
    for f in json_files:
        print(f"  {os.path.basename(f)}")
    print()


    # Load original HF dataset and save flags
    dataset, problems_by_id = load_data(token=HF_TOKEN, dataset_id=args.dataset_id)
    print(len(problems_by_id))
    last_problem_per_set = {p: v[-1] for p,v in problems_by_id.items()} # query problems
    num_ICL_per_set = {p: len(v)-1 for p,v in problems_by_id.items()} # number of ICL examples per chunk
    first_ICL_per_set = {p: v[0] for p,v in problems_by_id.items()} # first ICL example per chunk
    last_ICL_per_set = {p: v[-2] for p,v in problems_by_id.items()} # last ICL example per chunk
    valid_flags = [k for k in dataset.features if k.split('_')[0]=='flag'] + ['speciality', 'flag_num_ICL', 'flag_first_ICL_match_problem', 'flag_last_ICL_match_problem']

    all_results = {}

    for json_path in json_files:
        model_name, inference_mode = get_model_info(json_path)
        if model_name == 'unknown':
            print(f"Skipping: {json_path}")
            continue
        model_id = f"{model_name}:{inference_mode}"  # Create unique identifier for model+inference mode combination

        print(f"Processing {model_name} model ({inference_mode})...")

        # Load outputs of evaluation scripts
        with open(json_path, 'r') as f:
            problem_accuracies = json.load(f)['problem_accuracies']

        stats = {}
        for flag_category in valid_flags:
            flags_to_accuracies = defaultdict(list)
            for pid in problem_accuracies: 
                # Stratify by number of ICL examplses
                if flag_category == 'flag_num_ICL': 
                    if inference_mode=="0-shot": flag_label = 0
                    else: flag_label = num_ICL_per_set[pid]
                 
                # Stratify by whether the first ICL example has an answer matching the query problem
                elif flag_category == 'flag_first_ICL_match_problem':
                    if inference_mode=="0-shot": flag_label = False 
                    else: flag_label = (last_problem_per_set[pid]['answer']==first_ICL_per_set[pid]['answer'])

                # Stratify by whether the last ICL example has an answer matching the query problem
                elif flag_category == 'flag_last_ICL_match_problem': 
                    if inference_mode=="0-shot": flag_label = False 
                    else: flag_label = (last_problem_per_set[pid]['answer']==last_ICL_per_set[pid]['answer'])

                # Stratify by pre-assigned flags
                else: 
                    flag_label = last_problem_per_set[pid][flag_category]
                flags_to_accuracies[flag_label].append(problem_accuracies[pid])


            stats[f"{flag_category}_accuracy"] = {k: np.mean(v) for k,v in flags_to_accuracies.items()}
            stats[f"{flag_category}_total"] = {k: int(len(v)) for k,v in flags_to_accuracies.items()}
            stats[f"{flag_category}_correct"] = {k: int(sum(np.array(v)==100)) for k,v in flags_to_accuracies.items()}

        all_results[model_id] = stats

    print_stratification_results(valid_flags, all_results)

    output_file = Path(results_dir) / f'model_stratification_{args.eval_type}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f)
    print(f"Stratification results saved to: {output_file}")


if __name__ == "__main__":
    main()