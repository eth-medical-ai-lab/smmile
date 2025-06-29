import torch
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from utils import process_data, check_device
from collections import Counter 
from inference_llava_mcqa import prepare_mcqa_options

'''
# Description
Run random and majority baselines. 
- Majority: Given all the ICL examples, pick the most popular ICL answer and return that as the predicted answer.
- Random: Given all the ICL examples, pick one answer at random and return that as the predicted answer.
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with random/majority baselines on medical dataset")
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["random", "majority"],
        help="Baseline of interest"
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["ICL"],
        default="ICL",
        help="Mode for inference: ICL (with examples)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results",
        help="Directory to save result files"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="../missing_images",
        help="Directory containing manually downloaded images"
    )
    parser.add_argument(
        "--dataset",
        choices=["augmented"],
        help="Use augmented dataset (SMMILE-augmented-050825). If not specified, uses default SMMILE-050525"
    )
    parser.add_argument(
        "--task_format",
        type=str,
        choices=["open", "mcqa"],
        default="open",
        help="task format"
    )

    return parser.parse_args()

# Get HF_TOKEN from environment variable, with fallback
HF_TOKEN = os.environ.get('HF_TOKEN', '')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running this script.")

def run_inference(icl_questions_with_images, inference_mode, output_dir, baseline, task_format):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_device(device)

    # Prepare data for inference based on inference mode
    problem_ids = [chunk[0]['problem_id'] for chunk in icl_questions_with_images]
    final_questions = [chunk[-1]['question'] for chunk in icl_questions_with_images]
    true_answers = [chunk[-1]['answer'] for chunk in icl_questions_with_images]

    # Run inference
    full_responses = []
    generated_answers = []
    correct_options = []  # For MCQA
    selected_options = []  # For MCQA
    is_correct = []  # For MCQA

    for i, chunk in tqdm(enumerate(icl_questions_with_images), total=len(icl_questions_with_images)):
        # Prepare MCQA format if needed
        if task_format == "mcqa":
            options, correct_option = prepare_mcqa_options(chunk)
            correct_options.append(correct_option)
            if baseline == 'random': 
                selected_option = np.random.choice([x for x in options.keys()])
            elif baseline == 'majority': 
                options_to_letters = {v: k for k,v in options.items()}
                icl_answers = [options_to_letters[a['answer']] for a in chunk[:-1]]
                selected_option = Counter(icl_answers).most_common(1)[0][0]
            selected_options.append(selected_option)
            is_correct.append(selected_option == correct_option)
            ans = options[selected_option]
        else:
            # For in-context learning, we need to construct a conversation with examples
            possible_answers = [example['answer'] for example in chunk[:-1]]
            if baseline=='random': 
                ans = np.random.choice(possible_answers)
            elif baseline=='majority': 
                ans = Counter(possible_answers).most_common(1)[0][0]
        
        full_responses.append(ans)
        generated_answers.append(ans)


    if task_format == "mcqa": 
        results = pd.DataFrame({
            'problem_id': problem_ids,
            'final_question': final_questions,
            'true_answer': true_answers,
            'correct_option': correct_options,
            'selected_option': selected_options,
            'is_correct': is_correct,
            'generated_answer': generated_answers,
            'full_response': full_responses
        })
        output_filename = os.path.join(output_dir, f'result_{baseline}_{inference_mode}_mcqa.csv')
        results.to_csv(output_filename, index=False)
        print(f"Results saved to {output_filename}")
    else: 
        # Save results
        results = pd.DataFrame({
            'problem_id': problem_ids,
            'final_question': final_questions,
            'true_answer': true_answers,
            'generated_answer': generated_answers,
            'full_response': full_responses
        })
        output_filename = os.path.join(output_dir, f'result_{baseline}_{inference_mode}.csv')
        results.to_csv(output_filename, index=False)
        print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Running in {args.inference_mode} mode")
    if args.dataset == "augmented":
        print("This script will run analysis with augmented SMMILE dataset.")
        dataset_id = "smmile/SMMILE-augmented-050825"
        output_dir = args.output_dir + "_augmented"
    else:
        dataset_id = "smmile/SMMILE-050525"
        output_dir = args.output_dir

    data = process_data(image_dir=args.image_dir, token=HF_TOKEN, dataset_id=dataset_id)

    # Run inference
    print(f"Processing with {args.baseline} baseline approach...")
    run_inference(data, args.inference_mode, output_dir, args.baseline, args.task_format)