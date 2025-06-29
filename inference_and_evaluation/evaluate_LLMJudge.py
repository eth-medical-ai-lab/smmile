"""
Evaluation script with LLM-as-a-judge for VLM models on the SMMILE benchmark.
Currently, this script utilizes Llama3.3 (70B) for evaluation. 

Setup Instructions: The steps below need to be followed in order to enable support for 
ollama, the toolkit we will be using for this script. Documentation on ollama is available
at https://github.com/ollama/ollama.
    - Step 1: In your Linux terminal, run: 
        curl -fsSL https://ollama.com/install.sh | sh
    - Step 2: Start the ollama server from the command line: 
        ollama serve
    - Step 3: Open a new terminal window. Download Llama3.3 weights (43GB):
        ollama pull llama3.3
    - Step 4: Install the ollama python package: 
        pip install ollama


Usage:
    python evaluate_LLMJudge.py [results_dir] [--model MODEL_NAME] [--inference-mode {0-shot,ICL}]

Examples:
    python evaluate_LLMJudge.py ../results                            # Evaluate all models
    python evaluate_LLMJudge.py ../results --model qwen72B            # Evaluate specific model, all inference modes
    python evaluate_LLMJudge.py ../results --inference-mode ICL       # Evaluate all models in ICL mode
    python evaluate_LLMJudge.py --model llama32_vision_90b --inference-mode 0-shot  # Specific model and inference mode
    python evaluate_LLMJudge.py ../results --visualize-only       # Generate visualizations using saved statistics; do not run LLM
"""

import pandas as pd
import numpy as np
import sys
import os
import glob
import argparse
from typing import Dict, Tuple, Set
import ollama
import json
from tqdm import tqdm
from evaluate_EM import get_model_info, print_results, compare_models

PROMPT = (
    "A medical AI model is provided with an image and asked the question \"{question}\". "
    "The correct answer to this question is: \"{answer}\". The AI model outputs \"{response}\" as its " + \
    "response. Is the AI model correct? Please output your answer as a single digit, where 1 " + \
    "indicates that the AI model is correct and 0 indicates that the AI model is incorrect with " + \
    "respect to the correct answer. Do not provide anything other than the digit in your response."
)


def calculate_match_with_llm(csv_path: str, exclude_problem_ids: Set[str] = None) -> Tuple[float, Dict]:
    """
    Calculate LLM-as-a-judge accuracy from a CSV file with true and generated answers.

    Args:
        csv_path: Path to CSV file with columns problem_id, true_answer, generated_answer
        exclude_problem_ids: Set of problem IDs to exclude from evaluation

    Returns:
        Tuple containing:
        - Accuracy as a percentage
        - Dictionary with detailed statistics
    """
    if exclude_problem_ids is None:
        exclude_problem_ids = set()

    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path)

        # Verify required columns exist
        required_cols = ['problem_id', 'true_answer', 'generated_answer']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"Error: Missing required columns: {', '.join(missing_cols)}")
            sys.exit(1)

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Filter out excluded problem IDs
    original_count = len(df)
    if exclude_problem_ids:
        df = df[~df['problem_id'].isin(exclude_problem_ids)]
        excluded_count = original_count - len(df)
        print(f"Excluded {excluded_count} examples with problem IDs in the exclusion list.")

    # Run LLM-as-a-judge scoring approach
    try: 
        prompts = df.apply(
            lambda x: PROMPT.format(question=x.final_question, answer=x.true_answer, response=x.generated_answer), 
            axis=1
        )
    except: 
        prompts = df.apply(
            lambda x: PROMPT.format(question=x.original_question, answer=x.true_answer, response=x.generated_answer), 
            axis=1
        )

    llm_judgment = []
    errors = []
    for p in tqdm(prompts): 
        llm_ans = ollama.chat(model='llama3.3', messages=[{'role': 'user', 'content': p}]).message.content
        # Check for formatting errors in llm response
        if llm_ans == '1' or llm_ans == '0': 
            llm_judgment.append(int(llm_ans))
            errors.append(0)
        else: 
            llm_judgment.append(0)
            errors.append(1)

    print('Number of correct responses:', sum(llm_judgment))
    print('Number of improperly formatted LLM responses:', sum(errors))

    df['llm_judgment'] = np.array(llm_judgment).astype(bool)
    total_examples = len(df)
    correct_examples = sum(df['llm_judgment'])
    accuracy = (correct_examples / total_examples) * 100 if total_examples > 0 else 0

    # Group by problem_id to analyze patterns
    problem_accuracy = df.groupby('problem_id')['llm_judgment'].mean() * 100
    
    err_count = df[df['generated_answer'].apply(lambda x: "ERROR" in str(x))].shape[0]

    stats = {
        'total_examples': total_examples,
        'original_count': original_count,
        'excluded_count': original_count - total_examples,
        'llm_err_count': sum(errors),
        'correct_examples': int(correct_examples),
        'accuracy': accuracy,
        'err_count': err_count,
        'problem_accuracies': problem_accuracy.to_dict(),
        'incorrect_examples': df[~(df['llm_judgment'])]['problem_id'].tolist(),
        'word_count': df['generated_answer'].apply(lambda x: len(str(x).split())).mean(),
    }

    return accuracy, stats

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-as-a-judge match accuracy for VLM models on SMMILE benchmark"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="../results",
        help="Directory containing result CSV files (default: ./results)"
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
        "--visualize-only",
        action='store_true',
        help="If set to true, load saved results and perform visualization (i.e. do not run scoring function)"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # List of problem IDs to exclude from evaluation
    exclude_problem_ids = {
        # Add any problem IDs you want to exclude here
    }

    # Find CSV files in the results directory
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists

    # Build file pattern based on model and inference-mode filters
    patterns = []

    # If specific model is requested
    if args.model:
        if args.inference_mode:
            # Both model and inference mode specified
            patterns.append(os.path.join(results_dir, f"result_{args.model}_{args.inference_mode}.csv"))
            patterns.append(os.path.join(results_dir, f"result_{args.model}_{args.inference_mode}_open.csv"))
        else:
            # Only model specified, try both inference modes
            patterns.append(os.path.join(results_dir, f"result_{args.model}_0-shot.csv"))
            patterns.append(os.path.join(results_dir, f"result_{args.model}_0-shot_open.csv"))
            patterns.append(os.path.join(results_dir, f"result_{args.model}_ICL.csv"))
            patterns.append(os.path.join(results_dir, f"result_{args.model}_ICL_open.csv"))
            # Support legacy few-shot naming
            patterns.append(os.path.join(results_dir, f"result_{args.model}_few-shot.csv"))
    else:
        # No specific model, look for all supported models
        if args.inference_mode:
            # Only inference mode specified
            patterns.append(os.path.join(results_dir, f"result_*_{args.inference_mode}.csv"))
            patterns.append(os.path.join(results_dir, f"result_*_{args.inference_mode}_open.csv"))
            # Support legacy few-shot to ICL conversion
            if args.inference_mode == "ICL":
                patterns.append(os.path.join(results_dir, f"result_*_few-shot.csv"))
        else:
            # No filters, look for all result files
            patterns.append(os.path.join(results_dir, "result_*_0-shot.csv"))
            patterns.append(os.path.join(results_dir, "result_*_0-shot_open.csv"))
            patterns.append(os.path.join(results_dir, "result_*_ICL.csv"))
            patterns.append(os.path.join(results_dir, "result_*_ICL_open.csv"))
            # Legacy formats
            patterns.append(os.path.join(results_dir, "result_*_few-shot.csv"))
            patterns.append(os.path.join(results_dir, "result_*_with_answers.csv"))

    csv_files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        csv_files.extend(matches)

    # Also look in current directory if results_dir doesn't contain any files
    if not csv_files:
        for pattern in [p.replace(results_dir + "/", "") for p in patterns]:
            matches = glob.glob(pattern)
            csv_files.extend(matches)

    if not csv_files:
        print(f"No result files found matching the criteria.")
        if args.model:
            print(f"Model filter: {args.model}")
        if args.inference_mode:
            print(f"Inference mode filter: {args.inference_mode}")
        print("Check the directory and file naming conventions.")
        sys.exit(1)

    print(f"Found {len(csv_files)} result files to evaluate:")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")
    print()

    all_results = {}

    for csv_path in csv_files:
        model_name, inference_mode = get_model_info(csv_path)
        if model_name == 'unknown':
            print(f"Skipping: {csv_path}")
            continue
        model_id = f"{model_name}:{inference_mode}"  # Create unique identifier for model+inference mode combination

        print(f"Processing {model_name} model ({inference_mode})...")

        if args.visualize_only: 
            input_file = os.path.splitext(csv_path)[0] + "_evaluationllm.json"
            try: 
                with open(input_file, 'r') as f:
                    stats = json.load(f)
            except: 
                print(f"WARNING: Missing results file for {model_id}. Run without --visualize-only flag to generate.")
            all_results[model_id] = stats
            print_results(stats['model'], stats['inference_mode'], stats['accuracy'], stats, mode="LLM as a Judge")
        else: 
            accuracy, stats = calculate_match_with_llm(csv_path, exclude_problem_ids)
            all_results[model_id] = stats
            print_results(model_name, inference_mode, accuracy, stats, mode="LLM as a Judge")

            # Save individual evaluation results
            output_file = os.path.splitext(csv_path)[0] + "_evaluationllm.json"
            with open(output_file, 'w') as f:
                json.dump({'model': model_name, 'inference_mode': inference_mode, **stats}, f)
            print(f"Evaluation results for {model_name} ({inference_mode}) saved to: {output_file}")

    # Only compare models if we have more than one result
    if len(all_results) > 1:
        comparison_file = os.path.join(results_dir, "model_comparison_llm.txt")
        compare_models(all_results, comparison_file, "LLM as a Judge")


if __name__ == "__main__":
    main()