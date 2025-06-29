import os
import argparse
import random
import string
import time
import numpy as np
import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from utils import print_gpu_memory, check_device, process_data

'''
# Description
Inference with Qwen-2.5 series (3B, 7B, 32B, 72B)
Supports both 0-shot and ICL (in-context learning) inference modes
Supports both open-ended and multiple choice question answering (MCQA) formats
'''

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with Qwen 2.5 VL models on medical dataset")
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["0-shot", "ICL"],
        default="ICL",
        help="Mode for inference: 0-shot (no examples) or ICL (in-context learning with examples)"
    )
    parser.add_argument(
        "--task_format",
        type=str,
        choices=["open", "mcqa"],
        default="open",
        help="Task format: open (free-form answers) or mcqa (multiple choice questions)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["3B", "7B", "32B", "72B", "all"],
        default=["all"],
        help="Specify which models to run (3B, 7B, 32B, 72B, or all)"
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
        "--gpu_devices",
        type=str,
        default="0,1",
        help="Comma-separated list of GPU devices to use (e.g., '0,1')"
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="/home/.cache/",
        help="Cache directory for HuggingFace models"
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


def get_model_list(model_selection):
    qwen_models = {
        "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
        "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
        "32B": "Qwen/Qwen2.5-VL-32B-Instruct",
        "72B": "Qwen/Qwen2.5-VL-72B-Instruct"
    }

    if "all" in model_selection:
        return list(qwen_models.values())

    selected_models = []
    for model in model_selection:
        if model in qwen_models:
            selected_models.append(qwen_models[model])

    return selected_models


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


def run_inference(model_name, icl_questions_with_images, inference_mode, task_format, output_dir, hf_cache_dir):
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_device(device)
    print_gpu_memory("Before model load")

    print(f"Loading model: {model_name}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir=hf_cache_dir,
    )

    if hasattr(model, 'hf_device_map'):
        print("Model device mapping:")
        for module_name, device_id in model.hf_device_map.items():
            print(f"  {module_name}: device_{device_id}")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache_dir,)

    try:
        model_size = re.search(r'(\d+B)', model_name)
        if model_size:
            model_size = model_size.group(1)  # "3B", "7B", etc.
        else:
            model_size = model_name.split('/')[-1]
    except Exception:
        model_size = "unknown"

    # Prepare data for inference
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

    for i, chunk in enumerate(tqdm(icl_questions_with_images, total=len(icl_questions_with_images))):
        current_problem_id = problem_ids[i]
        target_example = chunk[-1]  # The final example to query

        if task_format == "mcqa":
            options, correct_option = prepare_mcqa_options(chunk)
            correct_options.append(correct_option)
            formatted_target_question = format_mcqa_question(target_example['question'], options)
        else:
            # For open format, use the original question
            formatted_target_question = target_example['question']
            options = None


        try:
            if inference_mode == "0-shot":
                conversation = [
                    {'role': 'user',
                     'content': [
                         {'type': 'image', 'image': target_example['image']},
                         {'type': 'text', 'text': formatted_target_question}
                     ]}
                ]
                chat = processor.apply_chat_template(conversation, add_generation_prompt=True)
                image = [target_example['image']]  # Only use the test example's image

            else:  # ICL mode
                conversation = []
                for idx, example in enumerate(chunk):
                    conversation.append({
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': example['image']},
                            {'type': 'text',
                             'text': example['question'] if idx < len(chunk) - 1 else formatted_target_question}
                        ]
                    })
                    if idx < len(chunk) - 1:
                        conversation.append({'role': 'assistant', 'content': example['answer']})

                chat = processor.apply_chat_template(conversation, add_generation_prompt=True)
                image = [example['image'] for example in chunk]

            inputs = processor(text=chat, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            output = model.generate(**inputs, max_new_tokens=512)

            full_response = processor.batch_decode(output, skip_special_tokens=True)[0]
            full_responses.append(full_response)

            # Extract only the generated answer (the part after the last "assistant")
            try:
                last_assistant_index = full_response.rfind("assistant")
                if last_assistant_index != -1:
                    generated_answer = full_response[last_assistant_index + len("assistant"):].strip()

                    for keyword in ["user", "system"]:
                        next_keyword = generated_answer.find(keyword)
                        if next_keyword != -1:
                            generated_answer = generated_answer[:next_keyword].strip()
                else:
                    matches = re.findall(r"assistant\s*([\s\S]*?)(?=user|system|$)", full_response, re.DOTALL)
                    if matches:
                        generated_answer = matches[-1].strip()
                    else:
                        generated_answer = full_response

            except Exception as e:
                print(f"Error extracting answer: {e}")
                generated_answer = full_response

            if task_format == "mcqa":
                selected_option = extract_mcqa_answer(generated_answer, current_problem_id, options)

                if selected_option == "EXTRACTION_FAILED":
                    selected_option = extract_mcqa_answer(full_response, current_problem_id, options)

                selected_options.append(selected_option)

                correct = (selected_option == correct_option)
                is_correct.append(correct)

                if i % 5 == 0 or i == len(icl_questions_with_images) - 1:
                    print(f"\nExample {i + 1}/{len(icl_questions_with_images)}:")
                    print(f"  Problem ID: {current_problem_id}")
                    print(f"  Correct option: {correct_option}")
                    print(f"  Extracted option: {selected_option}")
                    print(f"  Correct: {correct}")

                # Store the answer text rather than just the option letter
                if selected_option in options:
                    answer_text = options[selected_option]
                else:
                    answer_text = f"INVALID_OPTION:{selected_option}"

                generated_answers.append(answer_text)
            else:
                generated_answers.append(generated_answer)

        except Exception as e:
            print(f"Error generating response for example {i}: {e}")
            full_responses.append(f"ERROR: {str(e)}")
            generated_answers.append(f"ERROR: {str(e)}")
            if task_format == "mcqa":
                selected_options.append("ERROR")
                is_correct.append(False)

        if 'inputs' in locals():
            del inputs
        if 'output' in locals():
            del output

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(icl_questions_with_images)} examples")
            print_gpu_memory(f"After example {i + 1}")
            torch.cuda.empty_cache()

    inference_end_time = time.time()
    print(f"\nInference completed in {inference_end_time - inference_start_time:.2f} seconds.")

    if task_format == "mcqa":
        valid_results = [result for result in is_correct if not isinstance(result, str)]
        if valid_results:
            accuracy = sum(valid_results) / len(valid_results)
            print(f"\nMCQA Accuracy: {accuracy:.4f} ({sum(valid_results)}/{len(valid_results)})")
        else:
            accuracy = 0.0
            print("\nMCQA Accuracy: 0.0 (no valid results)")

    if task_format == "mcqa":
        results = pd.DataFrame({
            'problem_id': problem_ids,
            'final_question': final_questions,
            'true_answer': true_answers,
            'correct_option': correct_options,
            'selected_option': selected_options,
            'is_correct': is_correct,
            'generated_answer': generated_answers,
        })
        output_filename = os.path.join(output_dir, f'result_qwen{model_size}_{inference_mode}_mcqa.csv')

    else:
        results = pd.DataFrame({
            'problem_id': problem_ids,
            'final_question': final_questions,
            'true_answer': true_answers,
            'generated_answer': generated_answers,
            'full_response': full_responses
        })
        output_filename = os.path.join(output_dir, f'result_qwen{model_size}_{inference_mode}_open.csv')

    results.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")

    # GPU cleanup
    del model
    del processor
    if 'results' in locals():
        del results
    torch.cuda.empty_cache()
    print_gpu_memory("After model cleanup")


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Running in {args.inference_mode} mode with {args.task_format} format")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    print(f"Using GPU devices: {args.gpu_devices}")

    models_to_run = get_model_list(args.models)

    # Process data
    if args.dataset == "augmented":
        dataset_id = "smmile/SMMILE-augmented-050825"
        output_dir = args.output_dir + "_augmented"
    else:
        dataset_id = "smmile/SMMILE-050525"
        output_dir = args.output_dir

    print(f"Dataset: {'augmented' if args.dataset == 'augmented' else 'default (smmile)'} -> {dataset_id}")
    data = process_data(image_dir=args.image_dir, token=HF_TOKEN, dataset_id=dataset_id)

    #Run inference
    for model_name in models_to_run:
        print(f"\n========================")
        print(f"Processing with {model_name}...")
        print(f"========================")
        run_inference(
            model_name,
            data,
            args.inference_mode,
            args.task_format,
            output_dir,
            args.hf_cache_dir
        )
        torch.cuda.empty_cache()