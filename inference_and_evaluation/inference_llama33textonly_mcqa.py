'''
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
'''

import torch
import os
import numpy as np
import random
import string
from tqdm import tqdm
import pandas as pd
import re
import argparse
import time
from utils import process_data, check_device

import ollama

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

'''
# Description
Inference with Llama 3.3 (70B) - Text only baseline
Supports both 0-shot and ICL inference modes
Supports both open-ended and multiple choice question answering (MCQA) formats
'''


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with LLaMA (text only) on medical dataset")
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["0-shot", "ICL"],
        default="ICL",
        help="Mode for inference: 0-shot (no examples) or ICL (with examples)"
    )
    parser.add_argument(
        "--task_format",
        type=str,
        choices=["open", "mcqa"],
        default="mcqa",
        help="Task format: open (free-form answers) or mcqa (multiple choice questions)"
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
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--dataset",
        choices=["augmented"],
        help="Use augmented dataset (SMMILE-augmented-050825). If not specified, uses default SMMILE-050525"
    )

    return parser.parse_args()


# Get HF_TOKEN from environment variable, with fallback
HF_TOKEN = os.environ.get('HF_TOKEN', '')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running this script.")


def prepare_mcqa_options(chunk):
    """Generate multiple choice options for a problem set"""
    all_answers = [example['answer'] for example in chunk]  # Collect all answers from the chunk including the target

    unique_answers = []
    for answer in all_answers:
        if answer not in unique_answers:
            unique_answers.append(answer)

    # mapping of answers to option letters
    options = {}
    for i, answer in enumerate(unique_answers):
        option_letter = string.ascii_uppercase[i]  # A, B, C, D, ...
        options[option_letter] = answer

    option_letters = list(options.keys())
    random.shuffle(option_letters)  # Randomly permute the options
    permuted_options = {letter: options[letter] for letter in option_letters}

    # Find which option letter corresponds to the correct answer
    target_answer = chunk[-1]['answer']
    correct_option = next(letter for letter, answer in permuted_options.items() if answer == target_answer)

    return permuted_options, correct_option


def format_mcqa_question(question, options):
    """Format a question with multiple choice options using a structured template with example"""
    options_text = "\n".join([f"{letter}. {answer}" for letter, answer in options.items()])

    formatted_question = f"""
{question}

Please select the best answer from the following options:
{options_text}

Your task:
1. Think through the question, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice (A, B, C, D, E, F, ...) inside <answer>...</answer> tags.
Example of the expected format:
<think>This is my thought.</think>
<answer>Correct letter.</answer>
"""

    return formatted_question


def extract_mcqa_answer(response, problem_id=None, options=None):
    """Extract the option letter from the model's response using tags"""
    # Check if response is None or empty
    if response is None or not isinstance(response, str):
        print(f"Warning: Response is None or not a string for problem {problem_id}")
        return "EXTRACTION_FAILED"

    # Check for the last assistant's response with various formats (case insensitive)
    assistant_markers = ["ASSISTANT:", "Assistant:", "assistant:", "ASSISTANT ", "Assistant ", "assistant "]
    last_assistant_idx = -1
    last_marker_used = ""

    for marker in assistant_markers:
        pos = response.rfind(marker)
        if pos > last_assistant_idx:
            last_assistant_idx = pos
            last_marker_used = marker

    if last_assistant_idx != -1:
        processed_response = response[last_assistant_idx + len(last_marker_used):].strip()
        # If there's another USER or user after this ASSISTANT, trim to that
        for user_marker in ["USER:", "User:", "user:", "USER ", "User ", "user "]:
            next_user_idx = processed_response.find(user_marker)
            if next_user_idx != -1:
                processed_response = processed_response[:next_user_idx].strip()
                break
    else:
        processed_response = response

    # Look for answer tag pattern first (preferred method)
    answer_match = re.search(r"<answer>(.*?)</answer>|<answer>(.*?)($|\n\n)", processed_response, re.DOTALL)
    if answer_match:
        if answer_match.group(1) is not None:
            answer = answer_match.group(1).strip()
        elif answer_match.group(2) is not None:
            answer = answer_match.group(2).strip()
        else:
            print(f"Warning: Regular expression matched but no capture groups found for problem {problem_id}")
            return "EXTRACTION_FAILED"

        # If it's multi-line or has extra text, just get the first character which should be A, B, C, etc.
        if answer and len(answer) >= 1:
            # Try to extract just the letter
            letter_match = re.search(r'([A-Z])', answer.upper())
            if letter_match:
                return letter_match.group(1)
            return answer[0].upper()  # Return just the first character and ensure uppercase

    # Fallback to looking for common patterns if no tags are found
    patterns = [
        r'answer is:?\s*([A-Z])[.\s]',  # "The answer is X" or "The answer is: X"
        r'\banswer:?\s*([A-Z])[.\s]',  # "Answer: X"

        # Common formatting patterns for "Answer: X" with various styling
        r'\*\*Answer:\*\*\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # Bold "Answer: X"
        r'\*\*Answer\*\*:\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # Bold "Answer": X
        r'\*Answer\*:\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # Italic *Answer*: X
        r'Answer:\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # Simple "Answer: X"

        # Other common patterns
        r'The correct answer is\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # "The correct answer is X"
        r'The answer(?:\s+is)?:?\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # "The answer is X"
        r'The correct answer(?:\s+is)?:?\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # "The correct answer is X"

        # Simpler patterns as fallback
        r'answer\s*is\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # "answer is X"
        r'([A-Za-z])\s*is the (?:correct )?answer',  # "X is the answer"
        r'option\s*([A-Za-z])\b',  # "Option X"
        r'choice\s*([A-Za-z])\b',  # "Choice X"
        r'select\s*([A-Za-z])\b',  # "select X"
        r'best answer is:?\s*([A-Z])[.\s]',  # "The best answer is X"
        r'choice\s*([A-Z])[.\s]',  # "Choice X"
        r'option\s*([A-Z])[.\s]',  # "Option X"
        r'select\s*([A-Z])[.\s]',  # "I select X"
        r'chose\s*([A-Z])[.\s]',  # "I chose X"
        r'choose\s*([A-Z])[.\s]',  # "I choose X"
        r'final answer:?\s*([A-Z])',  # "Final answer: X"
    ]

    response_clean = processed_response.strip()

    for pattern in patterns:
        match = re.search(pattern, response_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # If we still can't find anything, look for a letter with asterisks or other emphasis
    emphasis_match = re.search(r'[*_](([A-Z])[*_])', response_clean, re.IGNORECASE)
    if emphasis_match:
        return emphasis_match.group(2).upper()

    # If all extraction methods fail, look for any capital letter that might be an option
    any_letter_match = re.search(r'\b([A-Z])\b', response_clean)
    if any_letter_match:
        return any_letter_match.group(1).upper()

    # Check if the actual text of the answer is in the response
    # This handles cases where the model gives the full answer text instead of a letter
    if options:
        # Get the simplified text after the last assistant
        last_assistant_text = ""
        if last_assistant_idx != -1:
            last_assistant_text = response[last_assistant_idx + len(last_marker_used):].strip()
            # Trim to next user marker if exists
            for user_marker in ["USER:", "User:", "user:", "USER ", "User ", "user "]:
                next_user_idx = last_assistant_text.find(user_marker)
                if next_user_idx != -1:
                    last_assistant_text = last_assistant_text[:next_user_idx].strip()
                    break

        # Check if any of the option texts matches the model's response
        if last_assistant_text:
            for letter, answer_text in options.items():
                # First check for exact match (case insensitive)
                if answer_text.strip().lower() == last_assistant_text.lower():
                    print(f"Found exact match for problem {problem_id}: Option {letter} - '{answer_text}'")
                    return letter

                # Check if the answer text is contained within the response (for more robust matching)
                if answer_text.strip() and last_assistant_text and answer_text.strip().lower() in last_assistant_text.lower():
                    print(f"Found partial match for problem {problem_id}: Option {letter} - '{answer_text}'")
                    return letter

    return "EXTRACTION_FAILED"


def run_inference(icl_questions_with_images, args, output_dir):
    # Create output directory if it doesn't exist
    inference_mode = args.inference_mode
    task_format = args.task_format
    max_new_tokens = args.max_new_tokens

    os.makedirs(output_dir, exist_ok=True)

    # Check if CUDA is available (for other operations, but not for inference as we use ollama)
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

    print(f"\nStarting inference in {inference_mode} mode with {task_format} format...")
    inference_start_time = time.time()

    for i, chunk in tqdm(enumerate(icl_questions_with_images), total=len(icl_questions_with_images)):
        target_example = chunk[-1]  # The final example to query
        current_problem_id = problem_ids[i]

        # Prepare MCQA format if needed
        if task_format == "mcqa":
            options, correct_option = prepare_mcqa_options(chunk)
            correct_options.append(correct_option)
            formatted_target_question = format_mcqa_question(target_example['question'], options)
        else:
            # For open format, use the original question
            formatted_target_question = target_example['question']
            options = None  # No options for open-ended questions

        # For debugging the first example of each format type
        if i == 0:
            print(f"\nExample question format ({task_format}):")
            print(formatted_target_question[:300] + "..." if len(
                formatted_target_question) > 300 else formatted_target_question)

        # Construct prompt based on inference mode
        if inference_mode == "0-shot":
            if task_format == "mcqa":
                prompt = "You are an expert medical question-answering system. " + \
                         "Your task is to provide an answer to the following multiple-choice question: \n\n" + \
                         f"{formatted_target_question}"
            else:
                prompt = "You are an expert medical question-answering system. " + \
                         "Your task is to provide an answer to the following question: \n\n" + \
                         f"{target_example['question']}"
        else:
            # For in-context learning
            if task_format == "mcqa":
                prompt = "You are an expert medical question-answering system. " + \
                         "Here are some examples of questions and associated answers: \n\n"

                # Add examples but not in MCQA format since they're just examples
                for example in chunk[:-1]:
                    prompt += f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"

                prompt += "Now, your task is to provide an answer to the following multiple-choice question: \n\n" + \
                          f"{formatted_target_question}"
            else:
                prompt = "You are an expert medical question-answering system. " + \
                         "Here are some examples of questions and associated answers: \n\n"

                for example in chunk[:-1]:
                    prompt += f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"

                prompt += "Now, your task is to provide an answer to the following question: \n\n" + \
                          f"{target_example['question']}"

        # Generate output using ollama
        try:
            llm_ans = ollama.chat(model='llama3.3', messages=[{'role': 'user', 'content': prompt}]).message.content

            if i == 0 or i % 10 == 0:
                print(f"\nExample {i} Prompt (truncated):")
                print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
                print(f"\nExample {i} Response (truncated):")
                print(llm_ans[:300] + "..." if len(llm_ans) > 300 else llm_ans)

            # For MCQA, process and evaluate the answer
            if task_format == "mcqa":
                selected_option = extract_mcqa_answer(llm_ans, current_problem_id, options)

                # If extraction failed, try again with some additional context about format
                if selected_option == "EXTRACTION_FAILED" and i < len(icl_questions_with_images) - 1:
                    # Add a follow-up prompt to extract the answer
                    follow_up_prompt = f"Based on your analysis above, what is your final answer? Please respond with just the letter (A, B, C, etc.) of the correct option. Format your answer as: <answer>X</answer> where X is the letter."

                    follow_up_ans = ollama.chat(model='llama3.3', messages=[
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': llm_ans},
                        {'role': 'user', 'content': follow_up_prompt}
                    ]).message.content

                    # Combine the responses for full context
                    combined_response = llm_ans + "\n\nUser: " + follow_up_prompt + "\n\nAssistant: " + follow_up_ans
                    llm_ans = combined_response

                    # Try extraction again
                    selected_option = extract_mcqa_answer(follow_up_ans, current_problem_id, options)

                selected_options.append(selected_option)

                # Check if correct
                correct = (selected_option == correct_option)
                is_correct.append(correct)

                print(f"\nExample {i + 1}/{len(icl_questions_with_images)}:")
                print(f"  Problem ID: {current_problem_id}")
                print(f"  Correct option: {correct_option}")
                print(f"  Extracted option: {selected_option}")
                print(f"  Correct: {correct}")

                if selected_option in options:
                    answer_text = options[selected_option]
                else:
                    answer_text = f"INVALID_OPTION:{selected_option}"

                generated_answers.append(answer_text)
            else:
                # For open format, just store the raw response
                generated_answers.append(llm_ans)

            full_responses.append(llm_ans)

        except Exception as e:
            print(f"\nError generating response for problem {current_problem_id}: {e}")
            full_responses.append(f"ERROR: {str(e)}")
            generated_answers.append(f"ERROR: {str(e)}")
            if task_format == "mcqa":
                selected_options.append("ERROR")
                is_correct.append(False)

    inference_end_time = time.time()
    print(f"\nInference loop finished in {inference_end_time - inference_start_time:.2f} seconds.")

    # Calculate Accuracy for MCQA (if applicable)
    if task_format == "mcqa":
        # Calculate accuracy, excluding error cases
        valid_results = [result for result in is_correct if not isinstance(result, str)]
        if valid_results:
            accuracy = sum(valid_results) / len(valid_results)
            print(f"\nMCQA Accuracy: {accuracy:.4f} ({sum(valid_results)}/{len(valid_results)})")
        else:
            accuracy = 0.0
            print("\nMCQA Accuracy: 0.0 (no valid results)")

    # Save results
    print("Saving results...")
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
        output_filename = os.path.join(output_dir, f'result_llama33textonly_{inference_mode}_mcqa.csv')

        # Write accuracy to a summary file
        summary_file = os.path.join(output_dir, f'accuracy_summary_llama33textonly_{inference_mode}_mcqa.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Model: llama33textonly\n")
            f.write(f"Inference Mode: {inference_mode}\n")
            f.write(f"Task Format: {task_format}\n")
            if valid_results:
                f.write(f"Accuracy: {accuracy:.4f} ({sum(valid_results)}/{len(valid_results)})\n")
            else:
                f.write("Accuracy: 0.0 (no valid results)\n")
            f.write(f"Total Examples: {len(icl_questions_with_images)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Accuracy summary saved to {summary_file}")
    else:
        results = pd.DataFrame({
            'problem_id': problem_ids,
            'final_question': final_questions,
            'true_answer': true_answers,
            'generated_answer': generated_answers,
            'full_response': full_responses
        })
        output_filename = os.path.join(output_dir, f'result_llama33textonly_{inference_mode}_open.csv')

    results.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Running in {args.inference_mode} mode with {args.task_format} format")

    if args.dataset == "augmented":
        dataset_id = "smmile/SMMILE-augmented-050825"
        output_dir = args.output_dir + "_augmented"
    else:
        dataset_id = "smmile/SMMILE-050525"
        output_dir = args.output_dir

    # Process data
    data = process_data(image_dir=args.image_dir, token=HF_TOKEN, dataset_id=dataset_id)

    # Run inference
    print(f"Processing with Llama3.3 (Text Only)...")
    run_inference(data, args, output_dir)