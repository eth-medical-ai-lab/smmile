import pandas as pd
import sys
import os
import re
import glob
import argparse
from typing import Dict, Tuple, Set
import string
import json


def normalize_answer(text: str) -> str:
    """
    Normalize answer by removing punctuation, extra whitespace, and converting to lowercase.
    This allows for minor differences in formatting while preserving the core content.

    Args:
        text: The answer text to normalize

    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        # Convert non-string values to strings
        text = str(text)

    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def calculate_exact_match(csv_path: str, exclude_problem_ids: Set[str] = None,
                          flexible_matching: bool = True) -> Tuple[float, Dict]:
    """
    Calculate exact match accuracy from a CSV file with true and generated answers.

    Args:
        csv_path: Path to CSV file with columns problem_id, true_answer, generated_answer
        exclude_problem_ids: Set of problem IDs to exclude from evaluation
        flexible_matching: Whether to use normalized text comparison for more flexible matching

    Returns:
        Tuple containing:
        - Exact match accuracy as a percentage
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

    # Compare answers based on matching strategy
    if flexible_matching:
        # Apply normalization to both true and generated answers
        df['normalized_true'] = df['true_answer'].apply(normalize_answer)
        df['normalized_generated'] = df['generated_answer'].apply(normalize_answer)
        df['exact_match'] = df['normalized_true'] == df['normalized_generated']
    else:
        # Strict matching (exact string comparison)
        df['exact_match'] = df['true_answer'] == df['generated_answer']

    total_examples = len(df)
    correct_examples = df['exact_match'].sum()
    accuracy = (correct_examples / total_examples) * 100 if total_examples > 0 else 0

    # Group by problem_id to analyze patterns
    problem_accuracy = df.groupby('problem_id')['exact_match'].mean() * 100

    err_count = df[df['generated_answer'].apply(lambda x: "ERROR" in str(x))].shape[0]

    stats = {
        'total_examples': total_examples,
        'original_count': original_count,
        'excluded_count': original_count - total_examples,
        'correct_examples': int(correct_examples),
        'accuracy': accuracy,
        'err_count': err_count,
        'problem_accuracies': problem_accuracy.to_dict(),
        'incorrect_examples': df[~df['exact_match']]['problem_id'].tolist(), 
        'word_count': df['generated_answer'].apply(lambda x: len(str(x).split())).mean(),
    }

    return accuracy, stats


def print_results(model_name: str, inference_mode: str, accuracy: float, stats: Dict, mode: str = "Exact Match") -> None:
    """Print formatted results to console for a single model."""
    print("\n" + "=" * 60)
    print(f"{mode.upper()} EVALUATION RESULTS FOR {model_name} ({inference_mode})")
    print("=" * 60)
    print(f"Original examples: {stats['original_count']}")
    if stats['excluded_count'] > 0:
        print(f"Excluded examples: {stats['excluded_count']}")
    print(f"Total examples evaluated: {stats['total_examples']}")
    print(f"Correct answers: {stats['correct_examples']}")
    print(f"{mode} accuracy: {accuracy:.2f}%")
    print("-" * 60)

    if len(stats['problem_accuracies']) > 1:
        print("\nPer-problem accuracy:")
        for problem_id, acc in stats['problem_accuracies'].items():
            print(f"  Problem {problem_id}: {acc:.2f}%")

    # List problem IDs with incorrect answers if any
    if stats['incorrect_examples']:
        print("\nProblem IDs with incorrect answers:")
        incorrect_problem_counts = {}
        for problem_id in stats['incorrect_examples']:
            incorrect_problem_counts[problem_id] = incorrect_problem_counts.get(problem_id, 0) + 1

        for problem_id, count in sorted(incorrect_problem_counts.items()):
            print(f"  Problem {problem_id}: {count} incorrect")

    print("=" * 60 + "\n")


def compare_models(results: Dict[str, Dict], output_file: str = "model_comparison.txt", mode: str = "Exact Match") -> None:
    """
    Compare results across multiple models and inference modes.

    Args:
        results: Dictionary mapping model identifiers (model_name:inference_mode) to their statistics
        output_file: Path to save the comparison results
    """
    print("\n" + "=" * 80)
    print(f"MODEL COMPARISON SUMMARY ({mode.upper()})")
    print("=" * 80)

    print(f"{'Model':<26} | {'Mode':<10} | {'Accuracy (%)':<12} | {'Excluded/Total':<15} | {'Word Count':<10} | {'Error Count':<15}")
    print("-" * 80)

    # Sort by inference mode first (0-shot then ICL), then by accuracy (highest first)
    sorted_results = sorted(
        results.items(),
        key=lambda x: (
            # Sort by inference mode (0-shot first, then ICL)
            0 if "0-shot" in x[0] else 1,
            # Then by accuracy (descending)
            -x[1]['accuracy']
        )
    )

    # Group results by inference mode for better visualization
    current_inference_mode = None

    for model_id, stats in sorted_results:
        model_name, inference_mode = model_id.split(':')

        # Print a separator when switching inference modes
        if current_inference_mode != inference_mode:
            if current_inference_mode is not None:
                print("-" * 80)
            current_inference_mode = inference_mode

        acc = stats['accuracy']
        excluded_ratio = f"{stats['excluded_count']}/{stats['original_count']}"
        word_count = stats['word_count']
        error_count = f"{stats['err_count']}" if 'err_count' in stats else "None"
        print(f"{model_name:<26} | {inference_mode:<10} | {acc:<12.2f} | {excluded_ratio:<15} | {word_count:<10.2f} | {error_count:<15}")

    print("\n" + "=" * 80)

    # Write detailed results to file
    with open(output_file, 'w') as f:
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Overall accuracy table
        f.write(f"{'Model':<26} | {'Mode':<10} | {'Accuracy (%)':<12} | {'Excluded/Total':<15} | {'Word Count':<15}\n")
        f.write("-" * 70 + "\n")

        current_inference_mode = None
        for model_id, stats in sorted_results:
            model_name, inference_mode = model_id.split(':')

            # Print a separator when switching inference modes
            if current_inference_mode != inference_mode:
                if current_inference_mode is not None:
                    f.write("-" * 70 + "\n")
                current_inference_mode = inference_mode

            acc = stats['accuracy']
            excluded_ratio = f"{stats['excluded_count']}/{stats['original_count']}"
            f.write(f"{model_name:<26} | {inference_mode:<10} | {acc:<12.2f} | {excluded_ratio:<15} | {word_count:<10.2f}\n")

        f.write("\n")

        # Per-problem comparison - first for 0-shot, then for ICL
        for mode in ["0-shot", "ICL"]:
            # Filter results for this inference mode
            mode_results = {k.split(':')[0]: v for k, v in results.items() if k.split(':')[1] == mode}

            if len(mode_results) > 1:
                all_problem_ids = set()
                for model_stats in mode_results.values():
                    all_problem_ids.update(model_stats['problem_accuracies'].keys())

                if len(all_problem_ids) > 1:
                    f.write(f"\nPER-PROBLEM ACCURACY COMPARISON (%) - {mode}\n")
                    f.write("-" * 60 + "\n")

                    header = f"{'Problem ID':<12} | " + " | ".join(f"{model:<12}" for model in mode_results.keys())
                    f.write(header + "\n")
                    f.write("-" * len(header) + "\n")

                    for problem_id in sorted(all_problem_ids):
                        row = f"{problem_id:<12} | "
                        for model in mode_results.keys():
                            acc = mode_results[model]['problem_accuracies'].get(problem_id, 0)
                            row += f"{acc:<12.2f} | "
                        f.write(row.rstrip(" | ") + "\n")

                    f.write("\n")

    print(f"Comparison results saved to: {output_file}")


def get_model_info(filename: str) -> Tuple[str, str]:
    """
    Extract model name and inference mode from filename.

    Args:
        filename: The filename to parse

    Returns:
        Tuple of (model_name, inference_mode)
    """
    base_filename = os.path.basename(filename)

    # Default values if parsing fails
    model_name = "unknown"
    inference_mode = "unknown"

    # Pattern matching for different model naming conventions
    if "llama32_vision_90b" in base_filename:
        model_name = "llama32_vision_90b"

        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"
        # Legacy support for few-shot naming
        elif "_few-shot" in base_filename:
            inference_mode = "ICL"

    elif "qwen" in base_filename.lower():
        # Try to extract model size using regex
        model_size_match = re.search(r'qwen(\d+B)', base_filename, re.IGNORECASE)
        if model_size_match:
            model_name = f"qwen{model_size_match.group(1)}"
        else:
            model_name = "qwen"

        # Extract inference mode
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"
        # Legacy support for few-shot naming
        elif "_few-shot" in base_filename:
            inference_mode = "ICL"
        elif "_with_answers" in base_filename:
            # Legacy filename format
            inference_mode = "ICL"

    elif "llava_7b" in base_filename:
        model_name = "llava_7B"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"

    elif "llavamed" in base_filename:
        model_name = "llavamed_7B"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"

    elif "llava_13b" in base_filename:
        model_name = "llava_13B"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"

    elif "llavanext_7b" in base_filename:
        model_name = "llavanext_7B"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"

    elif "llavaonevision_7b" in base_filename:
        model_name = "llavaonevision_7B"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"

    elif "llavaonevision_0.5b" in base_filename:
        model_name = "llavaonevision_0.5B"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"

    elif "llama33textonly" in base_filename:
        model_name = "llama33textonly (baseline)"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"

    elif "random" in base_filename:
        model_name = "random (baseline)"
        if "_ICL" in base_filename:
            inference_mode = "ICL"

    elif "majority" in base_filename:
        model_name = "majority (baseline)"
        if "_ICL" in base_filename:
            inference_mode = "ICL"


    elif "aya_vision_32b" in base_filename:
        model_name = "aya_vision_32b"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"
        # Legacy support for few-shot naming
        elif "_few-shot" in base_filename:
            inference_mode = "ICL"

    elif "llama4_17b" in base_filename:
        model_name = "llama4_17b"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"
        # Legacy support for few-shot naming
        elif "_few-shot" in base_filename:
            inference_mode = "ICL"

    elif "gpt-4o" in base_filename:
        model_name = "gpt-4o"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"
        # Legacy support for few-shot naming
        elif "_few-shot" in base_filename:
            inference_mode = "ICL"

    elif "claude-3-7-sonnet" in base_filename:
        model_name = "claude-3-7-sonnet"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"
        # Legacy support for few-shot naming
        elif "_few-shot" in base_filename:
            inference_mode = "ICL"

    elif "MedVLM-R1" in base_filename:
        model_name = "MedVLM-R1"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"
        # Legacy support for few-shot naming
        elif "_few-shot" in base_filename:
            inference_mode = "ICL"

    elif "medgemma" in base_filename:
        model_name = "medgemma_4b"
        if "_0-shot" in base_filename:
            inference_mode = "0-shot"
        elif "_ICL" in base_filename:
            inference_mode = "ICL"
        # Legacy support for few-shot naming
        elif "_few-shot" in base_filename:
            inference_mode = "ICL"


    return model_name, inference_mode


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
        "--strict-match",
        action="store_true",
        help="Use strict exact matching (case and punctuation sensitive)"
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

        # Use flexible matching by default, unless strict-match flag is provided
        flexible_matching = not args.strict_match
        matching_mode = "flexible" if flexible_matching else "strict"
        print(f"Using {matching_mode} matching for answer comparison")

        accuracy, stats = calculate_exact_match(csv_path, exclude_problem_ids, flexible_matching)
        all_results[model_id] = stats
        print_results(model_name, inference_mode, accuracy, stats)

        # Save individual evaluation results
        output_file = os.path.splitext(csv_path)[0] + "_evaluation.json"
        with open(output_file, 'w') as f:
            json.dump({'model': model_name, 'inference_mode': inference_mode, **stats}, f)
        print(f"Evaluation results for {model_name} ({inference_mode}) saved to: {output_file}")

    # Only compare models if we have more than one result
    if len(all_results) > 1:
        comparison_file = os.path.join(results_dir, "model_comparison.txt")
        compare_models(all_results, comparison_file)


if __name__ == "__main__":
    main()
