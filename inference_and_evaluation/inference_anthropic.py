import asyncio
import argparse
import base64
import random
import string
import re
import os
import io
from tqdm import tqdm
import pandas as pd
from utils import process_data
import anthropic

# Get ANTHROPIC_API_KEY from environment variable
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set. Please set it before running this script.")

# Get HF_TOKEN from environment variable
HF_TOKEN = os.environ.get('HF_TOKEN', '')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running this script.")

# Define Claude model to use
MODEL_NAME = "claude-3-7-sonnet-20250219"
MAX_TOKENS = 512


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

    # Look for answer tag pattern first (preferred method)
    answer_match = re.search(r"<answer>(.*?)</answer>|<answer>(.*?)($|\n\n)", response, re.DOTALL)
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
        r'\*\*Answer:\*\*\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # Bold "Answer: X"
        r'\*\*Answer\*\*:\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # Bold "Answer": X
        r'\*Answer\*:\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # Italic *Answer*: X
        r'Answer:\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # Simple "Answer: X"
        r'The correct answer is\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # "The correct answer is X"
        r'The answer(?:\s+is)?:?\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # "The answer is X"
        r'The correct answer(?:\s+is)?:?\s*([A-Za-z])(?:\.\s*\w+|\.|\s+|$)',  # "The correct answer is X"
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

    response_clean = response.strip()

    for pattern in patterns:
        match = re.search(pattern, response_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    emphasis_match = re.search(r'[*_](([A-Z])[*_])', response_clean, re.IGNORECASE)
    if emphasis_match:
        return emphasis_match.group(2).upper()

    any_letter_match = re.search(r'\b([A-Z])\b', response_clean)
    if any_letter_match:
        return any_letter_match.group(1).upper()

    if options:
        for letter, answer_text in options.items():
            if answer_text.strip().lower() == response_clean.lower():
                print(f"Found exact match for problem {problem_id}: Option {letter} - '{answer_text}'")
                return letter
            if answer_text.strip() and response_clean and answer_text.strip().lower() in response_clean.lower():
                print(f"Found partial match for problem {problem_id}: Option {letter} - '{answer_text}'")
                return letter

    return "EXTRACTION_FAILED"


def image_to_base64(image):
    """Convert PIL Image to base64 encoding required by Anthropic API"""
    buffered = io.BytesIO()
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf8")


async def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Anthropic Claude 3.7 Sonnet on medical dataset"
    )
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
        default="open",
        help="Task format: open (free-form answers) or mcqa (multiple choice questions)"
    )
    parser.add_argument(
        "--dataset",
        choices=["augmented"],
        help="Use augmented dataset (SMMILE-augmented-050825). If not specified, uses default SMMILE-050525"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="../missing_images",
        help="Directory containing manually downloaded images"
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="/home/.cache/",
        help="Cache directory for HuggingFace datasets"
    )
    args = parser.parse_args()

    # Set HuggingFace cache directory
    os.environ["HF_HOME"] = args.hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.hf_cache_dir, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(args.hf_cache_dir, "datasets")

    if args.dataset == "augmented":
        dataset_id = "smmile/SMMILE-augmented-050825"
        output_dir = "../results_augmented"
    else:
        # Default case (when --dataset is not specified or None)
        dataset_id = "smmile/SMMILE-050525"
        output_dir = "../results"

    print(f"Using dataset: {dataset_id}")
    print(f"Results will be saved to: {output_dir}")
    print(f"Using HuggingFace cache directory: {args.hf_cache_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Load data
    icl_questions_with_images = process_data(image_dir=args.image_dir, token=HF_TOKEN, dataset_id=dataset_id)

    # Prepare data for inference
    problem_ids = [chunk[0]['problem_id'] for chunk in icl_questions_with_images]
    final_questions = [chunk[-1]['question'] for chunk in icl_questions_with_images]
    true_answers = [chunk[-1]['answer'] for chunk in icl_questions_with_images]

    # Run inference
    full_responses = []
    generated_answers = []

    # For MCQA format
    correct_options = []
    selected_options = []
    is_correct = []

    print(
        f"\nStarting inference in {args.inference_mode} mode with {args.task_format} format using Claude 3.7 Sonnet...")

    for i, chunk in enumerate(tqdm(icl_questions_with_images, total=len(icl_questions_with_images))):
        current_problem_id = problem_ids[i]
        target_example = chunk[-1]  # The final example to query

        # Prepare for MCQA if required
        if args.task_format == "mcqa":
            options, correct_option = prepare_mcqa_options(chunk)
            correct_options.append(correct_option)
            formatted_target_question = format_mcqa_question(target_example['question'], options)
        else:
            # For open format, use the original question
            formatted_target_question = target_example['question']
            options = None
            correct_option = None

        try:
            message_list = []

            # Set up system message
            system_message = "You are a highly skilled physician and an excellent writer."

            # Convert images to the format required by Anthropic API
            target_image_b64 = image_to_base64(target_example['image'])

            if args.inference_mode == "0-shot":
                # Zero-shot mode (no examples)
                # Prepare a message with the image and question
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": formatted_target_question if args.task_format == "mcqa" else target_example[
                             'question']},
                        {"type": "image",
                         "source": {"type": "base64", "media_type": "image/png", "data": target_image_b64}}
                    ]
                }

                message_list = [user_message]

            else:
                # ICL mode (with examples)
                # Build a conversation with examples
                for idx, example in enumerate(chunk[:-1]):
                    # Convert example image to base64
                    example_image_b64 = image_to_base64(example['image'])

                    # Add user message with image and question
                    message_list.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": example['question']},
                            {"type": "image",
                             "source": {"type": "base64", "media_type": "image/png", "data": example_image_b64}}
                        ]
                    })

                    # Add assistant response
                    message_list.append({
                        "role": "assistant",
                        "content": example['answer']
                    })

                # Add the final test question
                message_list.append({
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": formatted_target_question if args.task_format == "mcqa" else target_example[
                             'question']},
                        {"type": "image",
                         "source": {"type": "base64", "media_type": "image/png", "data": target_image_b64}}
                    ]
                })

            # Send request to Anthropic API
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                system=system_message,
                messages=message_list
            )

            # Get the response text
            response_text = response.content[0].text if response.content else ""

            # Record the response
            print(f"Example {i + 1}/{len(icl_questions_with_images)}: Response for problem {current_problem_id}")
            print(f"Response: {response_text[:100]}...")  # Print first 100 chars

            full_responses.append(response_text)

            if args.task_format == "mcqa":
                # Extract the selected option from the response
                selected_option = extract_mcqa_answer(response_text, current_problem_id, options)
                selected_options.append(selected_option)

                if selected_option == "EXTRACTION_FAILED":
                    is_correct.append(False)
                    generated_answers.append("EXTRACTION_FAILED")
                else:
                    # Check if the selected option matches the correct option
                    correct = (selected_option == correct_option)
                    is_correct.append(correct)

                    # Store the answer text rather than just the option letter
                    if selected_option in options:
                        answer_text = options[selected_option]
                    else:
                        answer_text = f"INVALID_OPTION:{selected_option}"

                    generated_answers.append(answer_text)

                if i % 5 == 0 or i == len(icl_questions_with_images) - 1:
                    print(f"\nExample {i + 1}/{len(icl_questions_with_images)}:")
                    print(f"  Problem ID: {current_problem_id}")
                    print(f"  Correct option: {correct_option}")
                    print(f"  Extracted option: {selected_option}")
                    print(f"  Correct: {correct if selected_option != 'EXTRACTION_FAILED' else False}")
            else:
                generated_answers.append(response_text)

        except Exception as e:
            error_msg = f"ERROR: API Request Failed - {str(e)}"
            print(f"Error for problem {current_problem_id}: {error_msg}")
            full_responses.append(error_msg)
            generated_answers.append(error_msg)

            if args.task_format == "mcqa":
                selected_options.append("ERROR")
                is_correct.append(False)

    # Calculate accuracy for MCQA
    if args.task_format == "mcqa":
        valid_results = [result for result in is_correct if not isinstance(result, str)]
        if valid_results:
            accuracy = sum(valid_results) / len(valid_results)
            print(f"\nMCQA Accuracy: {accuracy:.4f} ({sum(valid_results)}/{len(valid_results)})")
        else:
            accuracy = 0.0
            print("\nMCQA Accuracy: 0.0 (no valid results)")

        # Save results for MCQA
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
        output_filename = os.path.join(output_dir, f'result_claude-3-7-sonnet_{args.inference_mode}_mcqa.csv')
    else:
        # Save results for open-ended format
        results = pd.DataFrame({
            'problem_id': problem_ids,
            'final_question': final_questions,
            'true_answer': true_answers,
            'generated_answer': generated_answers,
            'full_response': full_responses
        })
        output_filename = os.path.join(output_dir, f'result_claude-3-7-sonnet_{args.inference_mode}_open.csv')

    results.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    asyncio.run(main())