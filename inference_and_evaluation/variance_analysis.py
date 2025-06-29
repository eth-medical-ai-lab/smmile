import pandas as pd
import numpy as np
import argparse
import os
import re
import glob
from collections import defaultdict
import random
from typing import Dict, List, Tuple
from evaluate_EM import normalize_answer
import json


N_BOOTSTRAP_SAMPLES = 1000

def set_random_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def bootstrap_accuracy(is_correct: List[bool], n_bootstrap: int, mean_only_mode: bool) -> Tuple[float, float]:
    """
    Perform bootstrapping to calculate accuracy with confidence interval.

    Args:
        is_correct: List of boolean values indicating correctness
        n_bootstrap: Number of bootstrap samples
        mean_only_mode: False if computing variances; True for mean-only

    Returns:
        Tuple of (accuracy, standard_deviation)
    """
    is_correct_array = np.array(is_correct)
    n_samples = len(is_correct_array)

    if mean_only_mode:
        return np.mean(is_correct_array) * 100, None

    accuracies = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_sample = is_correct_array[indices]
        accuracy = np.mean(bootstrap_sample) * 100
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)

    return mean_accuracy, std_dev


def evaluate_mcqa_file(file_path: str, mean_only_mode: bool) -> Tuple[float, float]:
    """
    Evaluate MCQA file with is_correct column.

    Args:
        file_path: Path to the CSV file
        mean_only_mode: False if computing variances; True for mean-only

    Returns:
        Tuple of (accuracy, standard_deviation)
    """
    try:
        df = pd.read_csv(file_path)

        if 'is_correct' not in df.columns:
            print(f"Error: Missing 'is_correct' column in {file_path}")
            return np.inf, np.inf, np.inf

        is_correct = df['is_correct'].map(lambda x: True if str(x).lower() == 'true' else False)

        accuracy, std_dev = bootstrap_accuracy(is_correct.tolist(), N_BOOTSTRAP_SAMPLES, mean_only_mode)

        return accuracy, std_dev, df.shape[0]

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.inf, np.inf, np.inf


def evaluate_em_file(file_path: str, mean_only_mode: bool) -> Tuple[float, float]:
    """
    Evaluate Exact Match file with true_answer and generated_answer columns.

    Args:
        file_path: Path to the CSV file
        mean_only_mode: False if computing variances; True for mean-only

    Returns:
        Tuple of (accuracy, standard_deviation)
    """
    try:
        df = pd.read_csv(file_path)

        required_cols = ['true_answer', 'generated_answer']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"Error: Missing required columns in {file_path}: {', '.join(missing_cols)}")
            return np.inf, np.inf, np.inf

        df['normalized_true'] = df['true_answer'].apply(normalize_answer)
        df['normalized_generated'] = df['generated_answer'].apply(normalize_answer)
        is_correct = (df['normalized_true'] == df['normalized_generated']).tolist()

        accuracy, std_dev = bootstrap_accuracy(is_correct, N_BOOTSTRAP_SAMPLES, mean_only_mode)

        return accuracy, std_dev, df.shape[0]

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.inf, np.inf, np.inf


def evaluate_llm_file(file_path: str, mean_only_mode: bool) -> Tuple[float, float]:
    """
    Evaluate LLM-as-a-Judge file with true_answer and generated_answer columns.

    Args:
        file_path: Path to the CSV file
        mean_only_mode: False if computing variances; True for mean-only

    Returns:
        Tuple of (accuracy, standard_deviation)
    """
    try:
        # Load LLM-as-a-Judge results
        json_path = file_path[:-4] + '_evaluationllm.json'
        with open(json_path, 'r') as f:
            llm_results = json.load(f)

        is_correct = [llm_results['problem_accuracies'][pid] // 100 for pid in llm_results['problem_accuracies']]

        accuracy, std_dev = bootstrap_accuracy(is_correct, N_BOOTSTRAP_SAMPLES, mean_only_mode)

        return accuracy, std_dev, len(is_correct)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.inf, np.inf, np.inf


def extract_model_info(filename: str) -> Tuple[str, str, str]:
    """
    Extract model name, inference mode, and evaluation type from filename.

    Args:
        filename: The filename to parse

    Returns:
        Tuple of (model_name, inference_mode, eval_type)
    """
    base_filename = os.path.basename(filename)
    model_match = re.search(r'result_(.+?)_(0-shot|ICL|few-shot)', base_filename)
    if model_match:
        model_name = model_match.group(1)
        # Replace llama33textonly with text only
        if model_name == "llama33textonly":
            model_name = "text only"
    else:
        model_name = "unknown"

    if "_0-shot" in base_filename:
        inference_mode = "0-shot"
    elif "_ICL" in base_filename or "_few-shot" in base_filename:
        inference_mode = "ICL"
    else:
        inference_mode = "unknown"

    if base_filename.endswith("_mcqa.csv"):
        eval_type = "Closed"
    elif base_filename.endswith("_open.csv"):
        eval_type = "Open"
    else:
        eval_type = "Open"

    return model_name, inference_mode, eval_type


def create_results_table(results: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]], mean_only_mode: bool) -> str:
    """
    Create a formatted table from results.

    Args:
        results: Nested dictionary with structure {model_name: {eval_type: {inference_mode: (accuracy, std_dev)}}}
        mean_only_mode: False if computing variances; True for mean-only

    Returns:
        Formatted table as string
    """
    if not results:
        return "No results found."
    header = "Model Name || LLM, 0-shot || LLM, ICL || EM, 0-shot || EM, ICL || MCQA, 0-shot || MCQA, ICL"
    separator = "-" * len(header)

    table_lines = [header, separator]
    priority_models = ["majority", "random", "text only"]
    sorted_models = [model for model in priority_models if model in results]
    other_models = [model for model in sorted(results.keys()) if model not in priority_models]
    sorted_models.extend(other_models)

    for model in sorted_models:
        model_results = []
        model_results.append(model)
        for eval_type in ["LLM", "EM", "MCQA"]:
            for mode in ["0-shot", "ICL"]:
                if eval_type in results[model] and mode in results[model][eval_type]:
                    accuracy, std_dev = results[model][eval_type][mode]

                    if np.isinf(accuracy):
                        model_results.append("-")
                    elif mean_only_mode:
                        model_results.append(f"{accuracy:.2f}")
                    else:
                        if std_dev is None or np.isinf(std_dev):
                            model_results.append(f"{accuracy:.2f}")
                        else:
                            model_results.append(f"{accuracy:.2f} ± {std_dev:.2f}")
                else:
                    model_results.append("-")
        table_lines.append(" & ".join(model_results))

    return "\n".join(table_lines)


def create_latex_table(results: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]], output_path: str,
                       mean_only_mode: bool) -> None:
    """
    Create a LaTeX table with formatted values showing means and optionally standard deviations.

    Args:
        results: Nested dictionary with structure {model_name: {eval_type: {inference_mode: (accuracy, std_dev)}}}
        output_path: Path to save the LaTeX table
        mean_only_mode: If True, only means will be displayed without standard deviations
    """
    latex_table = []

    model_name_mapping = {
        "majority": "Majority",
        "random": "Random",
        "text only": "Text only$^*$",
        "MedVLM-R1": "MedVLM-R1",  # Already good
        "claude-3-7-sonnet": "Claude 3.7 Sonnet",
        "gpt-4o": "GPT-4o",
        "llama32_vision_90b": "Llama-3.2-Vision-90B",
        "llava_13b": "LLaVA-v1.5-13B",
        "llava_7b": "LLaVA-v1.5-7B",
        "llavanext_7b": "LLaVa-v1.6-Mistral-7B",
        "llavaonevision_0.5b": "LLaVA-Onevision-0.5B",
        "llavaonevision_7b": "LLaVA-Onevision-7B",
        "llavamed": "LLaVA-Med$^{**}$",
        "qwen32B": "Qwen2.5-VL-32B",
        "qwen72B": "Qwen2.5-VL-72B",
        "qwen3B": "Qwen2.5-VL-3B",
        "qwen7B": "Qwen2.5-VL-7B",
        "medgemma_4b": "MedGemma 4B Multimodal"
    }

    priority_models = ["majority", "random", "text only"]
    other_models = [model for model in sorted(results.keys()) if model not in priority_models]

    for model in priority_models:
        if model in results:
            model_results = []
            latex_model_name = model_name_mapping.get(model, model)
            latex_model_name = latex_model_name.replace('_', '-')
            model_results.append(latex_model_name)

            for eval_type in ["LLM", "EM", "MCQA"]:
                for mode in ["0-shot", "ICL"]:
                    if eval_type in results[model] and mode in results[model][eval_type]:
                        accuracy, std_dev = results[model][eval_type][mode]
                        if np.isinf(accuracy):
                            model_results.append("-")
                        elif mean_only_mode or std_dev is None:
                            model_results.append(f"${accuracy:.2f}$")
                        elif np.isinf(std_dev):
                            model_results.append(f"${accuracy:.2f}$")
                        else:
                            model_results.append(f"${accuracy:.2f} \\pm \\scriptstyle{{{std_dev:.2f}}}$")
                    else:
                        model_results.append("-")


            latex_table.append(" & ".join(model_results) + " \\\\")

    latex_table.append("\\midrule")

    for model in other_models:
        model_results = []

        latex_model_name = model_name_mapping.get(model, model)
        latex_model_name = latex_model_name.replace('_', '-')
        model_results.append(latex_model_name)

        for eval_type in ["LLM", "EM", "MCQA"]:
            for mode in ["0-shot", "ICL"]:
                if eval_type in results[model] and mode in results[model][eval_type]:
                    accuracy, std_dev = results[model][eval_type][mode]

                    if np.isinf(accuracy):
                        model_results.append("-")
                    elif mean_only_mode or std_dev is None:
                        model_results.append(f"${accuracy:.2f}$")
                    elif np.isinf(std_dev):
                        model_results.append(f"${accuracy:.2f}$")
                    else:
                        model_results.append(f"${accuracy:.2f} \\pm \\scriptstyle{{{std_dev:.2f}}}$")
                else:
                    model_results.append("-")

        latex_table.append(" & ".join(model_results) + " \\\\")

    latex_table.append("\\bottomrule")

    with open(output_path, "w") as f:
        f.write("\n".join(latex_table))

    print(f"LaTeX table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy on MCQA and exact match tasks")
    parser.add_argument("directory", help="Directory containing result CSV files")
    parser.add_argument("--output", default="evaluation_results.txt", help="Output file for results table")
    parser.add_argument('--mean_only', action='store_true', help='Compute only mean (no bootstrap variances)')
    parser.add_argument('--generate_table', action='store_true', help='Generate LaTeX table with formatted values')
    parser.add_argument('--latex_output', default="latex_table.tex", help='Output file for LaTeX table')
    args = parser.parse_args()

    set_random_seeds(42)

    if not os.path.isdir(args.directory):
        print(f"Error: Directory {args.directory} not found")
        return

    csv_files = glob.glob(os.path.join(args.directory, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.directory}")
        return

    print(f"Found {len(csv_files)} CSV files to evaluate")
    print(f"Using {N_BOOTSTRAP_SAMPLES} bootstrap samples for variance estimation")
    results = defaultdict(lambda: defaultdict(dict))

    for file_path in csv_files:
        model_name, inference_mode, eval_type = extract_model_info(file_path)
        print(f"Processing {os.path.basename(file_path)}: {model_name}, {inference_mode}, {eval_type}")

        if eval_type == "Closed":
            accuracy, std_dev, num_samples = evaluate_mcqa_file(file_path, args.mean_only)
            results[model_name]['MCQA'][inference_mode] = (accuracy, std_dev)
            if args.mean_only:
                print(f"  MCQA Accuracy ({num_samples} samples): {accuracy:.2f}")
            else:
                print(f"  MCQA Accuracy ({num_samples} samples): {accuracy:.2f} ± {std_dev:.2f}")
        else:  # EM
            accuracy, std_dev, num_samples = evaluate_em_file(file_path, args.mean_only)
            results[model_name]['EM'][inference_mode] = (accuracy, std_dev)
            if args.mean_only:
                print(f"  EM Accuracy ({num_samples} samples): {accuracy:.2f}")
            else:
                print(f"  EM Accuracy ({num_samples} samples): {accuracy:.2f} ± {std_dev:.2f}")

            accuracy, std_dev, num_samples = evaluate_llm_file(file_path, args.mean_only)
            results[model_name]['LLM'][inference_mode] = (accuracy, std_dev)
            if args.mean_only:
                print(f"  LLM-as-a-Judge Accuracy ({num_samples} samples): {accuracy:.2f}")
            else:
                print(f"  LLM-as-a-Judge Accuracy ({num_samples} samples): {accuracy:.2f} ± {std_dev:.2f}")

        table = create_results_table(results, args.mean_only)

        if table is None:
            print("Warning: Could not generate results table. No valid results found.")
            table = "No valid results found."

        output_path = os.path.join(args.directory, args.output)
        with open(output_path, "w") as f:
            f.write(table)

        print(f"\nResults saved to {output_path}")
        print("\nFinal Results Table:")
        print(table)

        if args.generate_table:
            if not results:
                print("Warning: Could not generate LaTeX table. No valid results found.")
            else:
                latex_output_path = os.path.join(args.directory, args.latex_output)
                create_latex_table(results, latex_output_path, args.mean_only)


if __name__ == "__main__":
    main()