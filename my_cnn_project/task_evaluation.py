# task_evaluation.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import time # Optional: for timing

# --- Configuration for MNIST Evaluation ---
# NOTE: It's inefficient to load the dataset every single time this function is called.
# For a real application, you'd load the dataset ONCE outside the main evolution loop
# and pass the data_loader into this function. But for simplicity as a self-contained
# example file, we load it here. Be aware of the performance implication.
DATA_ROOT = './mnist_data' # Directory to download/load MNIST dataset
BATCH_SIZE = 128           # Batch size for evaluation (adjust based on VRAM)

# --- Global variable to store dataset/loader to avoid reloading every time ---
# This is a simple caching mechanism for this example file. Better solutions exist.
_validation_loader = None

def get_mnist_validation_loader(device):
    """ Loads or retrieves the MNIST validation dataloader. """
    global _validation_loader
    if _validation_loader is not None:
        return _validation_loader

    print("    (Loading MNIST validation dataset for the first time...)")
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
        raise # Re-raise the exception to stop the process cleanly

    # Create the DataLoader
    # Pin memory if using GPU for potentially faster data transfer
    pin_memory = (device.type == 'cuda')
    _validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # No need to shuffle for validation
        num_workers=2, # Use background workers to load data (adjust as needed)
        pin_memory=pin_memory
    )
    print(f"    (MNIST validation dataset loaded: {len(validation_dataset)} samples)")
    return _validation_loader


# --- The Fitness Function ---

def evaluate_network_on_task(model_instance, device):
    """
    Evaluates the performance (accuracy) of the given model instance
    on the MNIST validation dataset.

    Args:
        model_instance (torch.nn.Module): An instance of your network with weights loaded.
        device (torch.device): The device ('cuda' or 'cpu') to run evaluation on.

    Returns:
        float: The validation accuracy (between 0.0 and 1.0). Higher is better.
               Returns -float('inf') if evaluation fails.
    """
    eval_start_time = time.time()
    fitness = 0.0 # Default fitness is 0 accuracy

    try:
        # 1. Get the validation data loader
        validation_loader = get_mnist_validation_loader(device)

        # 2. Set up model for evaluation
        model_instance.to(device) # Ensure model is on the correct device
        model_instance.eval()     # Set model to evaluation mode (important!)

        correct_predictions = 0
        total_samples = 0

        # 3. Iterate through the validation dataset without calculating gradients
        with torch.no_grad():
            for inputs, labels in validation_loader:
                # Move data to the specified device
                inputs, labels = inputs.to(device), labels.to(device)

                # Get model outputs (logits)
                outputs = model_instance(inputs)

                # Get predictions (class with the highest logit)
                _, predicted_classes = torch.max(outputs.data, 1)

                # Update counts
                total_samples += labels.size(0)
                correct_predictions += (predicted_classes == labels).sum().item()

        # 4. Calculate accuracy
        if total_samples > 0:
            accuracy = correct_predictions / total_samples
            fitness = float(accuracy) # Use accuracy as the fitness score
        else:
            print("    Warning: No samples found in validation loader during evaluation.")
            fitness = 0.0

        eval_time = time.time() - eval_start_time
        # print(f"    (Eval took {eval_time:.3f}s, Accuracy: {fitness:.4f})") # Optional detailed print

    except Exception as e:
        print(f"    ERROR during fitness evaluation: {e}")
        # It's important to handle errors, e.g., by returning a very low fitness
        # This prevents a single failing individual from crashing the whole evolution
        return -float('inf')

    # Ensure the return value is a float
    return float(fitness)
