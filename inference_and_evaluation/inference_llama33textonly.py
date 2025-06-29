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
from tqdm import tqdm
import pandas as pd
import argparse
from utils import process_data, check_device
import ollama

'''
# Description
Inference with Llama 3.3 (70B) - Text only baseline
Supports both 0-shot and ICL inference modes
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
            prompt = "You are an expert medical question-answering system. " + \
                "Your task is to provide an answer to the following question: \"{question}\"."
            prompt = prompt.format(question=chunk[-1]['question'])
            
        else:
            # For in-context learning, we need to construct a conversation with examples
            prompt = "You are an expert medical question-answering system. " + \
                "Here are some examples of questions and associated answers: \n"
            for example in chunk[:-1]: 
                prompt += f"{example['question']} {example['answer']}\n"
            prompt += f"Your task is to provide an answer to the following question: \"{chunk[-1]['question']}\"."
        # Generate output
        try:
            llm_ans = ollama.chat(model='llama3.3', messages=[{'role': 'user', 'content': prompt}]).message.content
            print("Prompt:", prompt)
            print("Question:", chunk[-1]['question'])
            print("Response:", llm_ans)

            full_responses.append(llm_ans)
            generated_answers.append(llm_ans)

        except Exception as e:
            print(f"Error generating response for problem {problem_ids[i]}: {e}")
            full_responses.append(f"ERROR: {str(e)}")
            generated_answers.append(f"ERROR: {str(e)}")

    # Save results
    results = pd.DataFrame({
        'problem_id': problem_ids,
        'final_question': final_questions,
        'true_answer': true_answers,
        'generated_answer': generated_answers,
        'full_response': full_responses
    })

    output_filename = os.path.join(output_dir, f'result_llama33textonly_{inference_mode}.csv')
    results.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Running in {args.inference_mode} mode")

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
    run_inference(data, args.inference_mode, output_dir)