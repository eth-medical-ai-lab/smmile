import datasets
import os
import sys
from PIL import Image
import torch
from huggingface_hub import login

def load_data(token, dataset_id="smmile/SMMILE-050525"):
    print(f"Loading dataset '{dataset_id}'...")
    try:
        dataset = datasets.load_dataset(dataset_id, token=token)['train']
    except Exception as e:
        print(f"Error loading dataset. Make sure HF_TOKEN is valid and you have accepted the terms for the dataset if necessary.")
        print(f"Error details: {e}")
        try:
            login(token=token)
            dataset = datasets.load_dataset(dataset_id, token=token)['train']
        except:
            print("direct login also didn't work!")
        sys.exit(1)

    # Try to load images for each example
    for idx, example in enumerate(dataset):
        if example['image'] is None:
            print(f'Image missing in the dataset. Please update the HuggingFace dataset.')
            print(example['problem_id'], example['image_url'])

    # Group examples by problem_id
    problems_by_id = {}
    for example in dataset:
        pid = example['problem_id']
        if pid not in problems_by_id:
            problems_by_id[pid] = []
        problems_by_id[pid].append(example)
    for pid in problems_by_id: 
        problems_by_id[pid] = sorted(problems_by_id[pid], key=lambda x: x['order'])

    return dataset, problems_by_id

def process_data(image_dir, token, dataset_id="smmile/SMMILE-050525"):
    """Loads dataset, handles missing images, and groups by problem_id."""
    dataset, problems_by_id = load_data(token, dataset_id)

    # Remove problem sets that do not have all images (either original or manually loaded)
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

    print(f'Total problems in dataset: {len(problems_by_id)}')
    print(f'Number of problems included (all images present): {len(icl_questions_with_images)}')
    print(f'Number of problems skipped (missing images): {len(skipped_problems)}')
    return icl_questions_with_images

def check_device(device): 
    if device == "cpu":
        print("WARNING: CUDA is not available. Running on CPU which will be very slow!")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Optional: monitor GPU memory usage
        def print_gpu_memory():
            if torch.cuda.is_available():
                print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        print_gpu_memory()  # Initial memory usage

def print_gpu_memory(stage=""):
    """Prints current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"--- GPU Memory Usage ({stage}) ---")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Total: {total:.2f} GB")
        print("-------------------------------")
    else:
        print("CUDA not available, cannot print GPU memory usage.")


def check_environment():
    print("--- Environment Check ---")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Pytorch CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA available: False")
    print("-------------------------")