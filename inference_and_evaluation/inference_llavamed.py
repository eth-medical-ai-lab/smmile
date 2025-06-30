'''
Note: This script requires transformers==4.36.2
'''

import torch
import os
from tqdm import tqdm
import pandas as pd
import argparse
from utils import process_data, check_device
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images

'''
# Description
Inference with LLaVA-Med
Supports both 0-shot and ICL inference modes
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA-Med on medical dataset")
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["0-shot", "ICL"],
        default="ICL",
        help="Mode for inference: 0-shot (no examples) or ICL (with examples)"
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
        "--dataset",
        choices=["augmented"],
        help="Use augmented dataset (SMMILE-augmented-050825). If not specified, uses default SMMILE-050525"
    )
    return parser.parse_args()

# Get HF_TOKEN from environment variable, with fallback
HF_TOKEN = os.environ.get('HF_TOKEN', '')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running this script.")

def run_inference(icl_questions_with_images, inference_mode, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_device(device)

    # Load model and processor
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        "microsoft/llava-med-v1.5-mistral-7b",
        None, 
        'llava-med-v1.5-mistral-7b'
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
            conv = conv_templates['mistral_instruct'].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + chunk[-1]['question'])
            conv.append_message(conv.roles[1], None)
            input_text = conv.get_prompt()

            input_ids = tokenizer_image_token(input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image_tensor = process_images([chunk[-1]['image']], image_processor, model.config)[0].unsqueeze(0)

        else:
            # For in-context learning, we need to construct a conversation with examples
            conv = conv_templates['mistral_instruct'].copy()
            for example in chunk[:-1]: 
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + example['question'])
                conv.append_message(conv.roles[1], example['answer'])

            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + chunk[-1]['question'])
            conv.append_message(conv.roles[1], None)
            input_text = conv.get_prompt()

            input_ids = tokenizer_image_token(input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image_tensor = process_images([example['image'] for example in chunk], image_processor, model.config)

        # Generate output
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(),
                    do_sample=False,
                    max_new_tokens=512, 
                )

            # Decode output
            response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Extract the model's answer
            generated_answer = response

            full_responses.append(response)
            generated_answers.append(generated_answer)

            print("Response:", response)

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

    output_filename = os.path.join(output_dir, f'result_llavamed_{inference_mode}.csv')
    results.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")

    # GPU cleanup
    del model
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
    run_inference(data, args.inference_mode, output_dir)
