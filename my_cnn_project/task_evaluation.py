# task_evaluation.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast # Import AMP
import numpy as np
import os
import time

# --- Configuration for MNIST Evaluation ---

# Directory to download/load MNIST dataset
DATA_ROOT = './mnist_data'

# --- OPTIMIZATION: Batch Size ---
# Increase this value based on your GPU VRAM. Larger batches generally lead to
# faster evaluation if they fit in memory. Start with 256 or 512.
# Monitor GPU memory usage (nvidia-smi) and reduce if you get CUDA OOM errors.
BATCH_SIZE = 256 # Increased from 128

# --- OPTIMIZATION: Subset Evaluation ---
# Set to an integer to evaluate only on that many samples from the validation set.
# Set to None to evaluate on the full validation set (10,000 samples for MNIST).
# Using a subset makes evaluation MUCH faster but gives a NOISIER fitness score.
# Good for rapid testing, might require more generations or tuning for best results.
# SAMPLES_TO_EVALUATE = 2048 # Example: Evaluate on ~2k samples
SAMPLES_TO_EVALUATE = None # Evaluate on the full dataset by default

# --- Global variable to cache dataset/loader (simple caching for this example) ---
_validation_loader = None

def get_mnist_validation_loader(device):
    """ Loads or retrieves the MNIST validation dataloader. """
    global _validation_loader
    if _validation_loader is not None:
        # Check if the device matches, recreate if necessary (edge case)
        if _validation_loader.pin_memory != (device.type == 'cuda'):
             print("    (Device changed, reloading DataLoader)")
             _validation_loader = None
        else:
            return _validation_loader

    print(f"    (Loading MNIST validation dataset [Batch Size: {BATCH_SIZE}]...)")
    # Define transformations (normalize MNIST images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific mean/std
    ])

    # Download and load the validation dataset
    try:
        validation_dataset = torchvision.datasets.MNIST(
            root=DATA_ROOT,
            train=False,        # Load the validation/test set
            download=True,      # Download if not present
            transform=transform
        )
    except Exception as e:
        print(f"\n!!! ERROR: Failed to download or load MNIST dataset from {DATA_ROOT}.")
        print("!!! Please check your internet connection and directory permissions.")
        print(f"!!! Error details: {e}\n")
        raise

    # Create the DataLoader
    # Use pin_memory=True if using CUDA
    pin_memory_setting = (device.type == 'cuda')
    # Use num_workers for parallel data loading (adjust based on your CPU cores)
    num_workers_setting = min(4, os.cpu_count() // 2) if os.cpu_count() else 2 # Heuristic

    _validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers_setting,
        pin_memory=pin_memory_setting
    )
    dataset_size = len(validation_dataset)
    subset_info = f"(evaluating subset: {SAMPLES_TO_EVALUATE})" if SAMPLES_TO_EVALUATE is not None else "(evaluating full dataset)"
    print(f"    (MNIST validation dataset loaded: {dataset_size} samples {subset_info})")
    return _validation_loader


# --- The Fitness Function ---

def evaluate_network_on_task(model_instance, device):
    """
    Evaluates the performance (accuracy) of the given model instance
    on the MNIST validation dataset (or a subset). Uses AMP if on CUDA.

    Args:
        model_instance (torch.nn.Module): An instance of your network with weights loaded.
        device (torch.device): The device ('cuda' or 'cpu') to run evaluation on.

    Returns:
        float: The validation accuracy (between 0.0 and 1.0). Higher is better.
               Returns -float('inf') if evaluation fails.
    """
    eval_start_time = time.time()
    fitness = 0.0 # Default fitness

    try:
        # 1. Get the validation data loader
        validation_loader = get_mnist_validation_loader(device)

        # 2. Set up model for evaluation
        model_instance.to(device)
        model_instance.eval()

        correct_predictions = 0
        total_samples = 0
        samples_processed_count = 0 # Counter for subset evaluation

        # 3. Iterate through the validation dataset without calculating gradients
        with torch.no_grad():
            for inputs, labels in validation_loader:
                # --- OPTIMIZATION: Subset Evaluation Check ---
                if SAMPLES_TO_EVALUATE is not None and samples_processed_count >= SAMPLES_TO_EVALUATE:
                    # print(f"    (Reached subset limit: {samples_processed_count} >= {SAMPLES_TO_EVALUATE})") # Debug print
                    break # Stop iterating early

                # Move data to the specified device
                inputs, labels = inputs.to(device), labels.to(device)

                # --- OPTIMIZATION: Automatic Mixed Precision (AMP) ---
                # Use autocast only when running on CUDA device
                # Alternative syntax (if device MUST be cuda)
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda', enabled=True): # Use literal 'cuda'
                        outputs = model_instance(inputs)
                else: # Handle CPU case explicitly if needed
                    outputs = model_instance(inputs) # Run without autocast on CPU


                # Get predictions (class with the highest logit)
                # Perform argmax outside autocast if outputs might be float16
                _, predicted_classes = torch.max(outputs.data, 1)

                # Update counts
                batch_size_actual = labels.size(0)
                total_samples += batch_size_actual
                correct_predictions += (predicted_classes == labels).sum().item()
                samples_processed_count += batch_size_actual # Update subset counter

        # 4. Calculate accuracy based on samples evaluated
        if total_samples > 0:
            accuracy = correct_predictions / total_samples
            fitness = float(accuracy)
        else:
            # This might happen if SAMPLES_TO_EVALUATE is 0 or data loader is empty
            print("    Warning: Zero samples evaluated. Check SAMPLES_TO_EVALUATE and dataset.")
            fitness = 0.0

        eval_time = time.time() - eval_start_time
        amp_info = "(AMP enabled)" if device.type == 'cuda' else "(AMP disabled/CPU)"
        subset_info = f"on {total_samples} samples" if SAMPLES_TO_EVALUATE is not None else "on full dataset"
        # print(f"    (Eval took {eval_time:.3f}s {amp_info}, Accuracy: {fitness:.4f} {subset_info})") # Optional detailed print

    except Exception as e:
        print(f"    ERROR during fitness evaluation: {e}")
        # Handle potential OOM errors or other issues during evaluation
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return -float('inf')

    # Ensure the return value is a float
    return float(fitness)

# --- Optional: Add a main block for testing this file directly ---
if __name__ == '__main__':
    # This allows you to test the evaluation function independently
    print("--- Testing task_evaluation.py directly ---")
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using test device: {test_device}")

    # You need a model definition available to test
    try:
        # Assuming model_definition.py is in the parent directory relative to this test run
        # Adjust path if necessary
        import sys
        sys.path.append('..') # Add parent directory to path temporarily
        from model_definition import SimpleCNN # Or your actual model class
        print("Successfully imported model definition.")

        # Instantiate a dummy model
        test_model = SimpleCNN(input_channels=1, num_classes=10)
        print("Model instantiated.")

        # Evaluate the randomly initialized model
        print("Running evaluation function...")
        test_fitness = evaluate_network_on_task(test_model, test_device)
        print(f"\nDirect test result: Fitness (Accuracy) = {test_fitness:.4f}")

        # Test again to check caching
        print("\nRunning evaluation function again (testing loader cache)...")
        test_fitness_2 = evaluate_network_on_task(test_model, test_device)
        print(f"Second run result: Fitness (Accuracy) = {test_fitness_2:.4f}")

    except ImportError as imp_err:
        print(f"\nError: Could not import model definition for testing.")
        print(f"Ensure 'model_definition.py' exists and is runnable.")
        print(f"Import Error: {imp_err}")
    except Exception as e:
        print(f"\nAn error occurred during direct testing: {e}")
        import traceback
        traceback.print_exc()

