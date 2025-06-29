import os
import sys
import argparse
import torch
import datasets
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm
import pandas as pd
import re
import time
from huggingface_hub import login
from utils import print_gpu_memory, check_environment
import string


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with MedVLM-R1 on smmile dataset")
    parser.add_argument(
        "--dataset",
        choices=["augmented"],
        help="Use augmented dataset (SMMILE-augmented-050825). If not specified, uses default SMMILE-050525"
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["0-shot", "ICL"],
        default="0-shot",
        help="Mode for inference: 0-shot (no examples) or ICL (in-context learning with examples)"
    )
    parser.add_argument(
        "--task_format",
        type=str,
        choices=["open", "mcqa"],
        default="mcqa",
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
                    # Sanitize URL to create a safe filename
                    url_part = example['image_url'].split('/')[-1]
                    # Remove potentially problematic characters for filenames
                    safe_filename = re.sub(r'[\\/*?:"<>|]', "", url_part)
                    image_savepath = os.path.join(image_dir, safe_filename)

                    if os.path.exists(image_savepath):
                        try:
                            # Verify image can be opened
                            img = Image.open(image_savepath)
                            img.verify()  # Check if image data is corrupt
                            # Reopen after verify
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
                    example['image'] = None
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
                example['image'] = None  # Mark as None if conversion fails

    if missing_urls:
        print(f"\n--- Missing Image Summary ---")
        print(f"{len(set(missing_urls))} unique image URLs seem to be missing or unloadable.")
        print(f"Please check the '{image_dir}' directory.")
        print(f"----------------------------")

    # Group examples by problem_id
    print("Grouping examples by problem_id...")
    problems_by_id = {}
    for example in dataset:
        pid = example['problem_id']
        if pid not in problems_by_id:
            problems_by_id[pid] = []
        problems_by_id[pid].append(example)
    for pid in problems_by_id:
        problems_by_id[pid] = sorted(problems_by_id[pid], key=lambda x: x['order'])

    # Filter problem sets that have all images (either original or manually loaded)
    print("Filtering problems with complete image sets...")
    questions_with_images = []
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
            questions_with_images.append(examples)
        else:
            skipped_problems.append(problem_id)

    print(f"Successfully processed {processed_count} examples with images.")
    print(f'Total problems in dataset: {len(problems_by_id)}')
    print(f'Number of problems included (all images present): {len(questions_with_images)}')
    print(f'Number of problems skipped (missing images): {len(skipped_problems)}')

    if not questions_with_images:
        print("\nError: No problems found with all required images present. Cannot proceed.")
        print("Please ensure images are downloaded correctly into the specified image directory.")
        sys.exit(1)

    return questions_with_images


def format_mcqa(problem_set):
    """
    Format a problem set as a multiple-choice question where the options include
    all unique answers from the problem set.
    """
    target_example = problem_set[-1]  # The final example is the one we're testing on
    context_examples = problem_set[:-1]  # Earlier examples

    # all unique answers across all examples in this problem set (including the final target qu.)
    all_answers = []
    for example in problem_set:
        if example['answer'] not in all_answers:
            all_answers.append(example['answer'])

    options = []
    for i, answer in enumerate(all_answers):
        letter = string.ascii_uppercase[i]
        options.append(f"{letter}) {answer}")

    # Format the multiple-choice question, find which option corresponds to the correct answer
    mcqa_format = target_example['question'] + "\n" + "\n".join(options)

    correct_letter = None
    for i, option in enumerate(options):
        if target_example['answer'] in option:
            correct_letter = string.ascii_uppercase[i]
            break

    formatted_context_examples = []
    for example in context_examples:
        # Create options for this example (using the same all_answers)
        example_mcqa_format = example['question'] + "\n" + "\n".join(options)

        # Find correct letter for this example
        example_correct_letter = None
        for i, option in enumerate(options):
            if example['answer'] in option:
                example_correct_letter = string.ascii_uppercase[i]
                break

        formatted_context_examples.append({
            'image': example['image'],
            'question': example_mcqa_format,
            'correct_letter': example_correct_letter,
            'correct_answer': example['answer']
        })

    return {
        'problem_id': target_example['problem_id'],
        'image': target_example['image'],
        'question': mcqa_format,
        'original_question': target_example['question'],
        'options': options,
        'correct_letter': correct_letter,
        'correct_answer': target_example['answer'],
        'context_examples': formatted_context_examples  # Add context examples for ICL
    }


def format_open_ended(problem_set):
    """
    Format a problem set for open-ended response.
    """
    target_example = problem_set[-1]  # The final example is the one we're testing on
    context_examples = problem_set[:-1]  # Earlier examples

    formatted_context_examples = []
    for example in context_examples:
        formatted_context_examples.append({
            'image': example['image'],
            'question': example['question'],
            'correct_answer': example['answer']
        })

    return {
        'problem_id': target_example['problem_id'],
        'image': target_example['image'],
        'question': target_example['question'],
        'original_question': target_example['question'],
        'correct_answer': target_example['answer'],
        'context_examples': formatted_context_examples  # Add context examples for ICL
    }


def extract_mcqa_answer(response):
    """Extract the answer from the model's response."""
    # Look for answer tag pattern
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        # Extract and clean up the answer
        answer = answer_match.group(1).strip()
        # If it's multi-line or has extra text, just get the first character which should be A, B, C, etc.
        if answer and len(answer) >= 1:
            return answer[0].upper()  # Return just the letter and ensure uppercase

    # If we can't find explicit tags, look for just a lone letter answer
    lone_letter_match = re.search(r'\b([A-J])\b', response)
    if lone_letter_match:
        return lone_letter_match.group(1).upper()

    return "EXTRACTION_FAILED"


def run_inference(questions_with_images, output_dir, inference_mode, task_format, hf_cache_dir, hf_token):
    model_id = "JZPeterPan/MedVLM-R1"

    os.makedirs(output_dir, exist_ok=True)

    # --- Device Setup ---
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This model requires GPU acceleration. Exiting.")
        sys.exit(1)

    # Use specified GPUs or all available GPUs
    if args.gpu_devices:
        print(f"Using device_map='auto' with visible GPUs: {args.gpu_devices}")
    else:
        print(f"Using device_map='auto' with all available GPUs")

    # --- Model Loading ---
    print_gpu_memory("Before Model Load")
    print(f"Loading model: {model_id}...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
            cache_dir=hf_cache_dir,
        )
        print(f"Model loaded successfully onto devices.")

        if hasattr(model, 'hf_device_map'):
            print("Model device mapping:")
            for module_name, device_id in model.hf_device_map.items():
                print(f"  {module_name}: device_{device_id}")

    except ImportError as e:
        print(f"\nError during model loading: {e}")
        print("Check model ID, Hugging Face token, and dependencies.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading model {model_id}: {e}")
        print("This could be due to insufficient GPU memory, incorrect model ID, or network issues.")
        print_gpu_memory("After Failed Model Load Attempt")
        sys.exit(1)

    print("Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token, cache_dir=hf_cache_dir)
    except Exception as e:
        print(f"\nError loading processor for {model_id}: {e}")
        sys.exit(1)


    print_gpu_memory("After Model and Processor Load")

    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        temperature=1,
        num_return_sequences=1,
        pad_token_id=151643,
    )

    # Format questions based on task format
    print(f"Formatting questions for {task_format} task format...")
    if task_format == "mcqa":
        formatted_questions = [format_mcqa(problem_set) for problem_set in questions_with_images]

        # Template for MCQA questions
        QUESTION_TEMPLATE = """
{Question}

Your task:
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
3. No extra information or text outside of these tags.
"""
    else:  # Open-ended format
        formatted_questions = [format_open_ended(problem_set) for problem_set in questions_with_images]

        # Template for open-ended questions
        QUESTION_TEMPLATE = """
        {Question}

        Please provide your response to this question.
        """

    # Template for ICL examples (adjusted for both formats)
    if task_format == "mcqa":
        ICL_EXAMPLE_TEMPLATE = """
Question:
{Question}
Your task:
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
3. No extra information or text outside of these tags.
Answer: {Answer}
"""
    else:  # Open-ended
        ICL_EXAMPLE_TEMPLATE = """
Question:
{Question}
Answer: {Answer}
"""

    # --- Inference ---
    problem_ids = [q['problem_id'] for q in formatted_questions]
    original_questions = [q['original_question'] for q in formatted_questions]
    if task_format == "mcqa":
        correct_letters = [q['correct_letter'] for q in formatted_questions]
    correct_answers = [q['correct_answer'] for q in formatted_questions]

    full_responses = []
    extracted_answers = []
    is_correct = []

    print(f"\nStarting inference in {inference_mode} mode with {task_format} format...")
    inference_start_time = time.time()

    for i, question in enumerate(tqdm(formatted_questions, desc="Generating Responses")):
        try:
            if inference_mode == "0-shot":
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": question['image']},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question['question'])}
                    ]
                }]
            else:  # ICL mode
                # For ICL, we need to create a single message with all the context embedded in the text
                context_str = ""
                if question['context_examples']:
                    for idx, example in enumerate(question['context_examples']):
                        # Format each example with its question and answer
                        context_str += f"Example {idx + 1}:\n"
                        if task_format == "mcqa":
                            context_str += f"Question: {example['question']}\n"
                            context_str += f"Answer: {example['correct_letter']}\n\n"
                        else:  # Open-ended
                            context_str += f"Question: {example['question']}\n"
                            context_str += f"Answer: {example['correct_answer']}\n\n"

                # Add the final question with context
                full_prompt = f"{context_str}Now answer this question:\n{question['question']}"

                message = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": question['image']},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=full_prompt)}
                    ]
                }]

            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

            # Process vision info (required for MedVLM-R1)
            image_inputs, video_inputs = process_vision_info(message)

            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    use_cache=True,
                    generation_config=generation_config
                )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            if task_format == "mcqa":
                extracted_answer = extract_mcqa_answer(output_text)
                answer_correct = (extracted_answer == question['correct_letter'])
                is_correct.append(answer_correct)
            else:  # Open-ended
                extracted_answer = output_text
                is_correct.append(None)

            full_responses.append(output_text)
            extracted_answers.append(extracted_answer)

        except torch.cuda.OutOfMemoryError as e:
            print(f"\nERROR: CUDA Out of Memory during generation for problem {problem_ids[i]} (index {i})!")
            print("Try using GPUs with more VRAM or enabling quantization.")
            print_gpu_memory(f"OOM on index {i}")
            full_responses.append("ERROR: CUDA OOM")
            extracted_answers.append("ERROR")
            is_correct.append(False if task_format == "mcqa" else None)
            if 'inputs' in locals(): del inputs
            if 'generated_ids' in locals(): del generated_ids
            torch.cuda.empty_cache()
            time.sleep(1)
            continue

        except Exception as e:
            print(f"\nError generating response for problem {problem_ids[i]} (index {i}): {e}")
            full_responses.append(f"ERROR: Generation failed - {str(e)}")
            extracted_answers.append("ERROR")
            is_correct.append(False if task_format == "mcqa" else None)
            if 'inputs' in locals(): del inputs
            if 'generated_ids' in locals(): del generated_ids
            torch.cuda.empty_cache()
            time.sleep(1)

        # --- Periodic Cleanup & Status ---
        if 'inputs' in locals():
            del inputs
        if 'generated_ids' in locals():
            del generated_ids
        if 'image_inputs' in locals():
            del image_inputs
        if 'video_inputs' in locals():
            del video_inputs
        if (i + 1) % 10 == 0:
            print(f"\nProcessed {i + 1}/{len(formatted_questions)} examples...")
            print_gpu_memory(f"After example {i + 1}")
            torch.cuda.empty_cache()

    inference_end_time = time.time()
    print(f"\nInference loop finished in {inference_end_time - inference_start_time:.2f} seconds.")

    # --- Save Results ---
    print("Saving results...")

    # Extract model name/size for file naming
    model_name_short = model_id.split('/')[-1]

    if task_format == "mcqa":
        results = pd.DataFrame({
            'problem_id': problem_ids,
            'original_question': original_questions,
            'correct_letter': correct_letters,
            'correct_answer': correct_answers,
            'extracted_answer': extracted_answers,
            'is_correct': is_correct,
            'full_response': full_responses
        })

    else:  # Open-ended
        results = pd.DataFrame({
            'problem_id': problem_ids,
            'original_question': original_questions,
            'true_answer': correct_answers,
            'generated_answer': extracted_answers,
            'full_response': full_responses
        })

    output_filename = os.path.join(output_dir, f'result_{model_name_short}_{inference_mode}_{task_format}.csv')
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
    image_dir = args.image_dir
    inference_mode = args.inference_mode
    task_format = args.task_format
    hf_cache_dir = args.hf_cache_dir
    check_environment()

    HF_TOKEN = os.environ.get('HF_TOKEN')
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set. Please set it before running this script.")

    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
        print(f"Setting CUDA_VISIBLE_DEVICES to: '{args.gpu_devices}'")

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This model requires GPU acceleration. Exiting.")
        sys.exit(1)
    print(f"PyTorch sees {torch.cuda.device_count()} CUDA devices")

    print(f"\nRunning with MedVLM-R1 model")
    print(f"Running in {inference_mode} mode with {task_format} format.")
    print(f"Dataset: {'augmented' if args.dataset == 'augmented' else 'default (smmile)'} -> {dataset_id}")
    print(f"Output directory: {output_dir}")
    print(f"Image directory: {image_dir}")

    # --- Data Processing ---
    print("\n--- Starting Data Processing ---")
    data = process_data(image_dir, HF_TOKEN, dataset_id)
    print("--- Data Processing Finished ---")

    # --- Inference ---
    print("\n--- Starting Inference ---")
    run_inference(data, output_dir, inference_mode, task_format, hf_cache_dir, HF_TOKEN)
    print("--- Inference Finished ---")