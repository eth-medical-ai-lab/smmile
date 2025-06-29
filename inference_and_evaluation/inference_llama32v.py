import os
import sys
import argparse
import random
import string
import re
import time
import torch
import datasets
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import print_gpu_memory


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


'''
# Description
Inference with Meta Llama 3.2 Vision 90B from Hugging Face
Filters out problems with missing images
Supports both 0-shot and ICL (in-context learning) inference modes
Supports both open-ended and multiple-choice question answering formats
'''



def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with Llama 3.2 Vision 90B on smmile dataset")
    parser.add_argument(
        "--dataset",
        choices=["augmented"],
        help="Use augmented dataset (SMMILE-augmented-050825). If not specified, uses default SMMILE-050525"
    )
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
        "--image_dir",
        type=str,
        default="../missing_images",
        help="Directory containing manually downloaded images"
    )
    parser.add_argument(
        "--gpu_devices",
        type=str,
        default="0,1",
        help="Comma-separated list of GPU devices to use (e.g., '0,1'). If not provided, will use all available GPUs."
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="/home/.cache/",
        help="Cache directory for HuggingFace models"
    )
    return parser.parse_args()


# Get HF_TOKEN from environment variable, with fallback
HF_TOKEN = os.environ.get('HF_TOKEN', '')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running this script.")


def process_data(image_dir, hf_token, dataset_id):
    """Loads dataset, handles missing images, and groups by problem_id."""
    print(f"Loading dataset {dataset_id}...")
    try:
        dataset = datasets.load_dataset(dataset_id, token=hf_token)['train']
    except Exception as e:
        print(
            f"Error loading dataset. Make sure HF_TOKEN is valid and you have accepted the terms for the dataset if necessary.")
        print(f"Error details: {e}")
        try:
            from huggingface_hub import login
            login(token=hf_token)
            dataset = datasets.load_dataset(dataset_id, token=hf_token)['train']
        except:
            print("direct login also didn't work!")
            sys.exit(1)

    print(f"Checking for and loading missing images from: {image_dir}")
    os.makedirs(image_dir, exist_ok=True)
    processed_count = 0
    missing_urls = []

    # Try to load images for each example
    for i, example in enumerate(tqdm(dataset, desc="Processing dataset examples")):
        if example['image'] is None:
            if example['image_url']:
                try:
                    url_part = example['image_url'].split('/')[-1]
                    safe_filename = re.sub(r'[\\/*?:"<>|]', "", url_part)
                    image_savepath = os.path.join(image_dir, safe_filename)

                    if os.path.exists(image_savepath):
                        try:
                            img = Image.open(image_savepath)
                            img.verify()
                            example['image'] = Image.open(image_savepath).convert("RGB")
                            processed_count += 1
                        except Exception as e:
                            print(
                                f"\nWarning: Error loading or verifying image from {image_savepath}: {e}. Treating as missing.")
                            example['image'] = None
                            missing_urls.append(example['image_url'])
                    else:
                        missing_urls.append(example['image_url'])
                        example['image'] = None  # Ensure it's None
                except Exception as e:
                    print(f"\nError processing image URL or path for example {i}: {e}")
                    example['image'] = None  # Ensure it's None if URL processing failed
                    if example['image_url']:
                        missing_urls.append(example['image_url'])
            else:
                example['image'] = None
        else:
            try:
                example['image'] = example['image'].convert("RGB")
                processed_count += 1
            except Exception as e:
                print(f"\nWarning: Error converting existing image to RGB for example {i}: {e}. Treating as missing.")
                example['image'] = None

    if missing_urls:
        print(f"\n--- Missing Image Summary ---")
        print(f"{len(set(missing_urls))} unique image URLs seem to be missing or unloadable.")
        print(f"Please check the '{image_dir}' directory.")
        print(f"----------------------------")

    print("Grouping examples by problem_id...")
    problems_by_id = {}
    for example in dataset:
        pid = example['problem_id']
        if pid not in problems_by_id:
            problems_by_id[pid] = []
        problems_by_id[pid].append(example)
    for pid in problems_by_id:
        problems_by_id[pid] = sorted(problems_by_id[pid], key=lambda x: x['order'])

    print("Filtering problems with complete image sets...")
    icl_questions_with_images = []
    skipped_problems = []
    for problem_id, examples in problems_by_id.items():
        has_all_images = True
        missing_image_count = 0
        for example in examples:
            if example['image'] is None:
                has_all_images = False
                missing_image_count += 1
                break

        if has_all_images:
            icl_questions_with_images.append(examples)
        else:
            skipped_problems.append(problem_id)

    print(f"Successfully processed {processed_count} examples with images.")
    print(f'Total problems in dataset: {len(problems_by_id)}')
    print(f'Number of problems included (all images present): {len(icl_questions_with_images)}')
    print(f'Number of problems skipped (missing images): {len(skipped_problems)}')

    if not icl_questions_with_images:
        print("\nError: No problems found with all required images present. Cannot proceed.")
        print("Please ensure images are downloaded correctly into the specified image directory.")
        sys.exit(1)

    return icl_questions_with_images


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
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice (A, B, C, D, E, ...) inside <answer>...</answer> tags.
Example of the expected format:
<think>This is my thought.</think>
<answer>X</answer>
Alternatively:
The correct answer is X.
"""
    # Debug print to see the formatted question
    #print("\n--- Formatted MCQA Question ---")
    #print(formatted_question)
    print("----------------------------------------")
    return formatted_question


def extract_mcqa_answer_2(response):
    """Extract the option letter from the model's response using various patterns"""
    # Check if response is None or empty
    if response is None or not isinstance(response, str) or not response.strip():
        print("Warning: Response is empty or not a string")
        return "EXTRACTION_FAILED"

    # Extract only the last assistant's response if possible
    last_assistant_idx = response.rfind("assistant")
    if last_assistant_idx != -1:
        processed_response = response[last_assistant_idx + len("assistant"):].strip()
        #print("stripped response after last assistant: ", processed_response)
    else:
        processed_response = response

    # CHECK FOR ANSWER TAG IN THE PROCESSED RESPONSE FIRST
    tag_pattern = r'<answer>([A-Za-z])</answer>'
    tag_matches = re.findall(tag_pattern, processed_response, re.IGNORECASE)
    if tag_matches:
        return tag_matches[0].upper()

    # Define patterns to try in order of preference (more specific to less specific)
    patterns = [
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
    ]

    # Try each pattern in order on the processed response
    for pattern in patterns:
        matches = re.findall(pattern, processed_response, re.IGNORECASE)
        if matches:
            return matches[0].upper()

    print("No answer pattern matched. Extraction failed.")
    return "EXTRACTION_FAILED"



def run_inference(icl_questions_with_images, output_dir, args):
    """Run inference with the specified parameters."""
    inference_mode = args.inference_mode
    task_format = args.task_format
    hf_cache_dir = args.hf_cache_dir

    os.makedirs(output_dir, exist_ok=True)

    # --- Device Setup ---
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This model requires GPU acceleration. Exiting.")
        sys.exit(1)

    if args.gpu_devices:
        print(f"Using device_map='auto' with visible GPUs: {args.gpu_devices}")
    else:
        print(f"Using device_map='auto' with all available GPUs")

    print(f"Number of visible GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    print_gpu_memory("Before Model Load")

    # --- Model Loading ---
    model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    print(f"Loading model: {model_id}...")

    try:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=HF_TOKEN,
            cache_dir=hf_cache_dir,
        )
        print(f"Model loaded successfully onto devices.")

        # Print device mapping if available
        if hasattr(model, 'hf_device_map'):
            print("Model device mapping:")
            for module_name, device_id in model.hf_device_map.items():
                print(f"  {module_name}: device_{device_id}")

    except Exception as e:
        print(f"\nError loading model {model_id}: {e}")
        print("This could be due to insufficient GPU memory, incorrect model ID, or network issues.")
        print_gpu_memory("After Failed Model Load Attempt")
        sys.exit(1)

    try:
        processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN, cache_dir=hf_cache_dir)
    except Exception as e:
        print(f"\nError loading processor for {model_id}: {e}")
        sys.exit(1)

    print_gpu_memory("After Model and Processor Load")

    # --- Inference ---
    problem_ids = [chunk[0]['problem_id'] for chunk in icl_questions_with_images]
    final_questions = [chunk[-1]['question'] for chunk in icl_questions_with_images]
    true_answers = [chunk[-1]['answer'] for chunk in icl_questions_with_images]

    full_responses = []
    generated_answers = []
    correct_options = []  # For MCQA
    selected_options = []  # For MCQA
    is_correct = []  # For MCQA

    extraction_failed_count = 0  # Counter for extraction failures

    print(f"\nStarting inference in {inference_mode} mode with {task_format} format...")
    inference_start_time = time.time()

    for i, chunk in enumerate(tqdm(icl_questions_with_images, desc="Generating Responses")):
        messages = []
        target_example = chunk[-1]  # The final example to query

        # --- Handle task format specific preparation ---
        if task_format == "mcqa":
            options, correct_option = prepare_mcqa_options(chunk)
            correct_options.append(correct_option)
            formatted_target_question = format_mcqa_question(target_example['question'], options)
        else:
            # For open format, use the original question
            formatted_target_question = target_example['question']

        # --- Construct messages based on inference mode and task format ---
        if inference_mode == "0-shot":
            # For 0-shot, just include the target question
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": target_example['image']},
                    {"type": "text", "text": formatted_target_question}
                ]}
            ]
        elif inference_mode == "ICL" and len(chunk) > 1:
            # For ICL, include previous examples as context
            for example in chunk[:-1]:
                # Add proper spacing to ensure correct formatting
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example['image']},
                        {"type": "text", "text": example['question']}
                    ]
                })
                # Fix: Add proper space between assistant's message and the next user's message
                messages.append({
                    "role": "assistant",
                    "content": example['answer'] + "\n"
                })

            # Add the target question
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": target_example['image']},
                    {"type": "text", "text": formatted_target_question}
                ]
            })

        # --- Process inputs ---
        try:
            # Extract all images from the messages in correct order
            images = []
            for msg in messages:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content["type"] == "image" and "image" in content:
                            images.append(content["image"])

            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                text=input_text,
                images=images,
                return_tensors="pt",
            ).to(model.device)

        except Exception as e:
            print(f"\nError processing inputs for problem {problem_ids[i]} (index {i}): {e}")
            full_responses.append(f"ERROR: Input processing failed - {str(e)}")
            generated_answers.append(f"ERROR: Input processing failed - {str(e)}")
            if task_format == "mcqa":
                selected_options.append("ERROR")
                is_correct.append(False)
            continue

        # --- Generate output ---
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            input_token_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            output_token_len = outputs.shape[1]
            new_tokens_len = output_token_len - input_token_len

            # Check if any new tokens were generated
            if new_tokens_len <= 0:
                print(f"  Warning: No new tokens generated!")
                response = ""
            else:
                try:
                    decoded_responses = processor.batch_decode(
                        outputs,
                        skip_special_tokens=True
                    )

                    # More safely extract the first response
                    if decoded_responses and len(decoded_responses) > 0:
                        response = decoded_responses[0]
                        if response is None:
                            print(f"  Warning: First decoded response is None!")
                            response = ""
                    else:
                        print(f"  Warning: No decoded responses returned!")
                        response = ""
                except Exception as decode_err:
                    print(f"  Error during decoding: {decode_err}")
                    response = ""

            if task_format == "mcqa":
                selected_option = extract_mcqa_answer_2(response)
                if selected_option == "EXTRACTION_FAILED":
                    extraction_failed_count += 1
                selected_options.append(selected_option)

                correct = (selected_option == correct_option)
                is_correct.append(correct)

                print(f"\nExample {i + 1}/{len(icl_questions_with_images)}:")
                print(f"  Problem ID: {problem_ids[i]}")
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
                # Extract only the assistant's response (last part)
                try:
                    # Try to find the last assistant's response
                    response_parts = response.split("assistant")
                    if len(response_parts) > 1:
                        # Get the last assistant's response
                        generated_answer = response_parts[-1].strip()
                    else:
                        generated_answer = response.strip()
                except Exception:
                    generated_answer = response.strip()

                generated_answers.append(generated_answer)

            full_responses.append(response if isinstance(response, str) else "")  # Ensure we store a string

        except torch.cuda.OutOfMemoryError as e:
            print(f"\nERROR: CUDA Out of Memory during generation for problem {problem_ids[i]} (index {i})!")
            print("Try reducing the number of few-shot examples, using GPUs with more VRAM,")
            print("or potentially enabling quantization.")
            print_gpu_memory(f"OOM on index {i}")
            full_responses.append("ERROR: CUDA OOM")
            generated_answers.append("ERROR: CUDA OOM")
            if task_format == "mcqa":
                selected_options.append("ERROR")
                is_correct.append(False)
            if 'inputs' in locals(): del inputs
            if 'outputs' in locals(): del outputs
            torch.cuda.empty_cache()
            time.sleep(1)
            continue

        except Exception as e:
            print(f"\nError generating response for problem {problem_ids[i]} (index {i}): {e}")
            full_responses.append(f"ERROR: Generation failed - {str(e)}")
            generated_answers.append(f"ERROR: Generation failed - {str(e)}")
            if task_format == "mcqa":
                selected_options.append("ERROR")
                is_correct.append(False)
            if 'inputs' in locals(): del inputs
            if 'outputs' in locals(): del outputs
            torch.cuda.empty_cache()
            time.sleep(1)

        # --- Periodic Cleanup & Status ---
        if 'inputs' in locals():
            del inputs
        if 'outputs' in locals():
            del outputs
        if (i + 1) % 5 == 0:
            print(f"\nProcessed {i + 1}/{len(icl_questions_with_images)} examples...")
            print_gpu_memory(f"After example {i + 1}")
            torch.cuda.empty_cache()

    inference_end_time = time.time()
    print(f"\nInference loop finished in {inference_end_time - inference_start_time:.2f} seconds.")

    # --- Save Results ---
    print("Saving results...")

    # Verify all arrays have the same length before creating DataFrame
    if task_format == "mcqa":
        data_lengths = {
            'problem_ids': len(problem_ids),
            'final_questions': len(final_questions),
            'true_answers': len(true_answers),
            'correct_options': len(correct_options),
            'selected_options': len(selected_options),
            'is_correct': len(is_correct),
            'full_responses': len(full_responses)
        }

        print(f"Array lengths check: {data_lengths}")

        min_length = min(data_lengths.values())
        results = pd.DataFrame({
            'problem_id': problem_ids[:min_length],
            'final_question': final_questions[:min_length],
            'true_answer': true_answers[:min_length],
            'correct_option': correct_options[:min_length],
            'selected_option': selected_options[:min_length],
            'is_correct': is_correct[:min_length],
            'full_response': full_responses[:min_length]
        })
        output_filename = os.path.join(output_dir, f'result_llama32_vision_90b_{inference_mode}_mcqa.csv')
    else:
        data_lengths = {
            'problem_ids': len(problem_ids),
            'final_questions': len(final_questions),
            'true_answers': len(true_answers),
            'generated_answers': len(generated_answers),
            'full_responses': len(full_responses)
        }

        print(f"Array lengths check: {data_lengths}")
        min_length = min(data_lengths.values())

        results = pd.DataFrame({
            'problem_id': problem_ids[:min_length],
            'final_question': final_questions[:min_length],
            'true_answer': true_answers[:min_length],
            'generated_answer': generated_answers[:min_length],
            'full_response': full_responses[:min_length]
        })
        output_filename = os.path.join(output_dir, f'result_llama32_vision_90b_{inference_mode}_open.csv')

    results.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")

    # --- GPU Cleanup ---
    del model
    del processor
    if 'results' in locals(): del results  # Free DataFrame memory
    torch.cuda.empty_cache()
    print_gpu_memory("After Final Cleanup")


if __name__ == "__main__":
    args = parse_arguments()

    if args.dataset == "augmented":
        dataset_id = "smmile/SMMILE-augmented-050825"
        output_dir = "../results_augmented"
    else:
        dataset_id = "smmile/SMMILE-050525"
        output_dir = "../results"

    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
        print(f"Setting CUDA_VISIBLE_DEVICES to: '{args.gpu_devices}'")

    print(f"Running inference with Llama 3.2 Vision 90B in {args.inference_mode} mode")
    print(f"Task format: {args.task_format}")

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This model requires GPU acceleration. Exiting.")
        sys.exit(1)
    print(f"PyTorch sees {torch.cuda.device_count()} CUDA devices")

    # --- Data Processing ---
    print("\n--- Starting Data Processing ---")
    data = process_data(args.image_dir, HF_TOKEN, dataset_id=dataset_id)
    print("--- Data Processing Finished ---")

    # --- Inference ---
    print("\n--- Starting Inference ---")
    run_inference(data, output_dir, args)
    print("--- Inference Finished ---")