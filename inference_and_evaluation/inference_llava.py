'''
Note: This script uses transformers==4.46.3
'''

import torch
import os
from tqdm import tqdm
import pandas as pd
import argparse
from transformers import (
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    AutoProcessor,
)
from utils import process_data, check_device

'''
# Description
Inference with LLaVA (Currently supports regular LLaVA and LLaVA-NEXT)
Supports both 0-shot and ICL inference modes
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA on medical dataset")
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["0-shot", "ICL"],
        default="ICL",
        help="Mode for inference: 0-shot (no examples) or ICL (with examples)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["llava_7b", "llava_13b", "llavanext_7b", "llavaonevision_7b", "llavaonevision_0.5b"],
        required=True,
        help="LLaVA model variant"
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
        "--gpu-devices",
        type=str,
        default="0",
        help="Comma-separated list of GPU devices to use (e.g., '0,1')"
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="/home/.cache/",
        help="Cache directory for HuggingFace models"
    )
    parser.add_argument(
        "--noisy",
        type=str,
        choices=["AddOneExample-BySpeciality", "AddOneExample"],
        default=None,
        help="Noisy mode"
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

def run_inference(icl_questions_with_images, inference_mode, output_dir, output_name, hf_cache_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_device(device)

    # Load model and processor
    if output_name == 'llava_7b': 
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", 
            torch_dtype=torch.float16, 
            device_map="auto",
            cache_dir=hf_cache_dir,
        )
        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", 
            cache_dir=hf_cache_dir,
        )
    if output_name == 'llava_13b': 
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-13b-hf", 
            torch_dtype=torch.float16, 
            device_map="auto",
            cache_dir=hf_cache_dir,
        )
        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-13b-hf", 
            cache_dir=hf_cache_dir,
        )
    elif output_name == 'llavanext_7b': 
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", 
            torch_dtype=torch.float16, 
            device_map="auto",
            cache_dir=hf_cache_dir,
        )
        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", 
            cache_dir=hf_cache_dir,
        )
    elif output_name == "llavaonevision_7b": 
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            torch_dtype=torch.float16,
            device_map="auto", 
            cache_dir=hf_cache_dir
        )
        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf", 
            cache_dir=hf_cache_dir
        ) 
    elif output_name == "llavaonevision_0.5b": 
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            torch_dtype=torch.float16,
            device_map="auto", 
            cache_dir=hf_cache_dir
        )
        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", 
            cache_dir=hf_cache_dir
        ) 


    # Prepare data for inference based on inference mode
    problem_ids = [chunk[0]['problem_id'] for chunk in icl_questions_with_images]
    final_questions = [chunk[-1]['question'] for chunk in icl_questions_with_images]
    true_answers = [chunk[-1]['answer'] for chunk in icl_questions_with_images]

    # Run inference
    full_responses = []
    generated_answers = []

    for i, chunk in tqdm(enumerate(icl_questions_with_images), total=len(icl_questions_with_images)):
        if inference_mode == "0-shot":
            # For 0-shot, include only the final test example with no prior examples
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": chunk[-1]['image']},
                        {"type": "text", "text": chunk[-1]['question']}
                    ],
                },
            ]

            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                text=input_text, 
                images=chunk[-1]['image'],
                return_tensors="pt"
            )
        else:
            # For in-context learning, we need to construct a conversation with examples
            messages = []

            # Add all examples except the last one
            for idx, example in enumerate(chunk[:-1]):
                # Add user message with image and question
                messages.append({"role": "user", "content": [
                    {"type": "image", "image": example['image']},
                    {"type": "text", "text": example['question']}
                ]})
                # Add assistant response
                messages.append({"role": "assistant", "content": [
                    {"type": "text", "text": example['answer']}
                ]})

            # Add the final test question with image
            messages.append({"role": "user", "content": [
                {"type": "image", "image": chunk[-1]['image']},
                {"type": "text", "text": chunk[-1]['question']}
            ]})

            # Prepare inputs - for ICL, we need to use the last image only
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

            # Extract all images from the messages in correct order
            images = []
            for msg in messages:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content["type"] == "image" and "image" in content:
                            images.append(content["image"])

            inputs = processor(
                text=input_text,
                images=images,
                return_tensors="pt",
            )
        # Generate output
        try:
            if output_name in ["llavaonevision_7b", "llavaonevision_0.5b"]: 
                inputs = inputs.to(model.device, torch.float16)
            else: inputs = inputs.to(model.device)

            output = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                pad_token_id=processor.tokenizer.pad_token_id
            )

            # Decode output
            response = processor.decode(output[0], skip_special_tokens=True)

            # Extract the model's answer
            generated_answer = processor.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

            full_responses.append(response)
            generated_answers.append(generated_answer)

        except Exception as e:
            print(f"Error generating response for problem {problem_ids[i]}: {e}")
            full_responses.append(f"ERROR: {str(e)}")
            generated_answers.append(f"ERROR: {str(e)}")

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(icl_questions_with_images)} examples")
            check_device(device)  # Check memory usage periodically

    # Save results
    results = pd.DataFrame({
        'problem_id': problem_ids,
        'final_question': final_questions,
        'true_answer': true_answers,
        'generated_answer': generated_answers,
        'full_response': full_responses
    })

    output_filename = os.path.join(output_dir, f'result_{output_name}_{inference_mode}.csv')
    results.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")

    # GPU cleanup
    del model
    del processor
    if 'inputs' in locals():
        del inputs
    if 'output' in locals():
        del output
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Running in {args.inference_mode} mode")

    # Set CUDA device for GPU operations
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    print(f"Using GPU devices: {args.gpu_devices}")

    # Process data
    if args.dataset == "augmented":
        dataset_id = "smmile/SMMILE-augmented-050825"
        output_dir = args.output_dir + "_augmented"
    else:
        dataset_id = "smmile/SMMILE-050525"
        output_dir = args.output_dir
    
    data = process_data(args.image_dir, HF_TOKEN, dataset_id=dataset_id)
    
    # Run inference
    print(f"Processing with {args.model_name}...")
    run_inference(data, args.inference_mode, output_dir, args.model_name, args.hf_cache_dir)
