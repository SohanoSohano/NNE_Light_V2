# final_evolve_script_optimized.py

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import importlib.util
import copy
import random
import time

# Check PyTorch version for compile feature
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile') and callable(torch.compile)
if TORCH_COMPILE_AVAILABLE:
    print("Note: torch.compile available (PyTorch 2.0+ detected).")
else:
    print("Note: torch.compile not available (requires PyTorch 2.0+).")


# --- Utility Functions (No changes needed here) ---

def flatten_weights(model):
    """ Flattens all model parameters into a single numpy vector. """
    try:
        weights = []
        for param in model.parameters():
            if param.requires_grad:
                weights.append(param.data.cpu().numpy().flatten())
        if not weights:
            raise ValueError("No trainable parameters found in the model to flatten.")
        return np.concatenate(weights)
    except Exception as e:
        print(f"Error during weight flattening: {e}")
        raise

def load_weights_from_flat(model, flat_weights):
    """ Loads flattened weights back into a model instance. """
    try:
        offset = 0
        if not isinstance(flat_weights, np.ndarray):
             flat_weights = np.array(flat_weights)

        flat_weights_tensor = torch.from_numpy(flat_weights).float()
        model_device = next(model.parameters()).device

        for param in model.parameters():
             if param.requires_grad:
                numel = param.numel()
                param_shape = param.size()
                if offset + numel > len(flat_weights_tensor):
                    raise ValueError(f"Shape mismatch: Not enough data in flat_weights to fill parameter {param.shape} (offset {offset}, numel {numel}, flat_weights len {len(flat_weights_tensor)})")

                param_slice = flat_weights_tensor[offset:offset + numel].view(param_shape).to(model_device)
                param.data.copy_(param_slice)
                offset += numel

        if offset != len(flat_weights_tensor):
            print(f"Warning: Size mismatch after loading weights. Offset {offset} != flat_weights length {len(flat_weights_tensor)}. Check model definition correspondence.")

    except Exception as e:
        print(f"Error loading weights from flat vector: {e}")
        raise

def load_pytorch_model(model_definition_path, class_name, state_dict_path, device, *model_args, **model_kwargs):
    """ Loads the model class, instantiates it, and loads the state_dict. """
    try:
        spec = importlib.util.spec_from_file_location("model_module", model_definition_path)
        if spec is None or spec.loader is None:
             raise ImportError(f"Could not load spec for module at {model_definition_path}")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        if not hasattr(model_module, class_name):
             raise AttributeError(f"Class '{class_name}' not found in {model_definition_path}")
        ModelClass = getattr(model_module, class_name)

        print(f"Instantiating model '{class_name}' with args: {model_args}, kwargs: {model_kwargs}")
        model = ModelClass(*model_args, **model_kwargs)
        model.to(device)

        if state_dict_path and os.path.exists(state_dict_path):
            print(f"Loading state_dict from: {state_dict_path}")
            try:
                state_dict = torch.load(state_dict_path, map_location=device)
                model.load_state_dict(state_dict)
                print("State_dict loaded successfully.")
            except Exception as load_err:
                print(f"Error loading state_dict: {load_err}. Check architecture.")
                raise
        elif state_dict_path:
            print(f"Warning: state_dict path '{state_dict_path}' provided but not found. Using initial model weights.")
        else:
             print("No state_dict path provided. Using initial model weights.")

        model.eval()
        return model

    except Exception as e:
        print(f"Error in load_pytorch_model: {e}")
        raise

def load_task_eval_function(task_module_path):
    """ Loads the fitness evaluation function from a specified file. """
    try:
        spec = importlib.util.spec_from_file_location("task_module", task_module_path)
        if spec is None or spec.loader is None:
             raise ImportError(f"Could not load spec for module at {task_module_path}")
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)

        if not hasattr(task_module, 'evaluate_network_on_task'):
             raise AttributeError(f"Function 'evaluate_network_on_task(model_instance, device)' not found in {task_module_path}")

        return getattr(task_module, 'evaluate_network_on_task')
    except Exception as e:
        print(f"Error loading task evaluation function: {e}")
        raise


# --- Genetic Algorithm Components ---

# MODIFIED: Added torch.compile() attempt
def evaluate_population(population_weights, model_definition_path, class_name, task_eval_func, device, model_args, model_kwargs):
    """ Evaluates the fitness of each individual (weight vector) in the population. """
    fitness_scores = []
    num_individuals = len(population_weights)

    for i, flat_weights in enumerate(population_weights):
        individual_start_time = time.time()
        current_model = None # Ensure defined in outer scope for cleanup
        compiled_model_obj = None # Store compiled object reference if needed

        # Add a flag to track if compile was attempted/successful for logging
        compile_attempted = False
        compile_successful = False

        print(f"Evaluating individual {i+1}/{num_individuals}...")
        try:
            # 1. Create & Load Model
            current_model = load_pytorch_model(model_definition_path, class_name, None, device, *model_args, **model_kwargs)
            load_weights_from_flat(current_model, flat_weights)
            current_model.to(device)
            current_model.eval()

            # --- Optimization: Attempt torch.compile (PyTorch 2.0+) ---
            model_to_evaluate = current_model # Default to uncompiled
            if False:
                compile_attempted = True
                try:
                    print(f"    (Attempting torch.compile on model instance {i+1}...)")
                    # Use mode="reduce-overhead" for potentially faster compilation than default
                    # Use mode="max-autotune" for potentially best runtime but longer compile time
                    compiled_model_obj = torch.compile(current_model, mode="reduce-overhead")
                    model_to_evaluate = compiled_model_obj # Use compiled version if successful
                    compile_successful = True
                    print(f"    (Model instance {i+1} compiled successfully)")
                except Exception as compile_err:
                    print(f"    Warning: torch.compile() failed for individual {i+1}: {compile_err}. Using uncompiled model.")
            # --- End torch.compile ---


            # 2. Call the user-defined evaluation function (passing the potentially compiled model)
            #    -> AMP needs to be applied INSIDE task_eval_func <---
            fitness = task_eval_func(model_to_evaluate, device)


            # 3. Process Fitness Score
            if not isinstance(fitness, (float, int)):
                 print(f"Warning: Fitness function returned non-numeric value ({type(fitness)}). Setting fitness to -inf.")
                 fitness = -float('inf')

            fitness_scores.append(float(fitness))
            eval_time = time.time() - individual_start_time
            compile_info = "(compiled)" if compile_successful else "(compile failed/skipped)" if compile_attempted else ""
            print(f"Individual {i+1} fitness: {fitness:.4f} {compile_info} (eval time: {eval_time:.2f}s)")

        except Exception as e:
            print(f"Error evaluating individual {i+1}: {e}")
            fitness_scores.append(-float('inf')) # Assign very low fitness on error
        finally:
            # 4. Cleanup: Explicitly delete models and clear cache
            #    Make sure to delete both original and potentially compiled versions
            del model_to_evaluate # This might be original or compiled
            if current_model is not None and compiled_model_obj is not None and current_model is not compiled_model_obj:
                # Only delete current_model if it's different from compiled_model_obj
                 del current_model
            elif current_model is not None and compiled_model_obj is None:
                 # Delete if compilation wasn't attempted or failed
                 del current_model

            # Clear potentially compiled object reference
            compiled_model_obj = None

            if device.type == 'cuda':
                torch.cuda.empty_cache()


    return fitness_scores


# --- selection, crossover, mutate functions (No changes needed) ---
def select_parents(population_weights, fitness_scores, num_parents):
    """ Selects parents using tournament selection. """
    parents = []
    population_size = len(population_weights)
    if population_size == 0: return []

    tournament_size = max(2, min(population_size, 5))

    valid_indices = [i for i, f in enumerate(fitness_scores) if f > -float('inf')]
    if not valid_indices:
         print("Warning: All individuals failed evaluation. Cannot select parents.")
         return []


    for _ in range(num_parents):
        tournament_candidate_indices = random.sample(valid_indices, min(len(valid_indices), tournament_size))
        best_fitness_in_tournament = -float('inf')
        winner_index_in_population = -1
        for idx in tournament_candidate_indices:
             if fitness_scores[idx] > best_fitness_in_tournament:
                  best_fitness_in_tournament = fitness_scores[idx]
                  winner_index_in_population = idx

        if winner_index_in_population != -1:
            parents.append(population_weights[winner_index_in_population])
        else:
             print("Warning: Could not select a winner in tournament. Picking random valid individual.")
             parents.append(population_weights[random.choice(valid_indices)])

    return parents

def crossover(parent1, parent2):
    """ Simple average crossover for weight vectors. """
    p1 = np.array(parent1)
    p2 = np.array(parent2)
    if p1.shape != p2.shape:
        raise ValueError(f"Parent shapes do not match for crossover: {p1.shape} vs {p2.shape}")
    child = (p1 + p2) / 2.0
    return child

def mutate(weights, mutation_rate, mutation_strength):
    """ Adds Gaussian noise to a fraction of weights based on mutation rate. """
    if mutation_rate <= 0 or mutation_strength <= 0:
         return weights

    mutated_weights = weights.copy()
    num_weights_to_mutate = int(len(weights) * mutation_rate)
    if num_weights_to_mutate == 0 and mutation_rate > 0:
        num_weights_to_mutate = 1

    indices_to_mutate = np.random.choice(len(weights), num_weights_to_mutate, replace=False)

    noise = np.random.normal(0, mutation_strength, size=num_weights_to_mutate)
    mutated_weights[indices_to_mutate] += noise.astype(mutated_weights.dtype)
    return mutated_weights


# --- Main Evolution Loop (No significant changes needed) ---

def run_evolution(args):
    """ Executes the main evolutionary process. """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"--- Starting Evolution for {args.model_type} ---")

    # --- Validate Paths ---
    if not os.path.isdir(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
    model_definition_path = os.path.join(args.model_dir, "model_definition.py")
    weights_path = os.path.join(args.model_dir, args.weights_file)
    task_module_path = os.path.join(args.model_dir, "task_evaluation.py")

    if not os.path.exists(model_definition_path):
        print(f"Error: model_definition.py not found in {args.model_dir}")
        return
    if not os.path.exists(weights_path) and args.weights_file: # Allow running without initial weights
        print(f"Warning: Weights file '{args.weights_file}' not found in {args.model_dir}. Starting from random init.")
        weights_path = None # Set path to None if file not found
    elif not args.weights_file:
         print("No weights file specified. Starting from random init.")
         weights_path = None
    if not os.path.exists(task_module_path):
        print(f"Error: task_evaluation.py not found in {args.model_dir}")
        return

    # --- Load User Modules & Initial Weights ---
    try:
        task_eval_func = load_task_eval_function(task_module_path)
        # Define model args/kwargs - adjust if your model needs specific ones
        model_args = []
        model_kwargs = {}
        # Load model once to get weight shape, use weights_path (which might be None)
        initial_model = load_pytorch_model(
            model_definition_path, args.model_class, weights_path, device, *model_args, **model_kwargs
        )
        initial_weights = flatten_weights(initial_model)
        print(f"Initial model '{args.model_class}' loaded. Weight vector size: {initial_weights.shape[0]}")
        del initial_model # Free memory
        if device.type == 'cuda': torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during initial loading: {e}")
        return

    # --- Initialize Population ---
    population = [initial_weights.copy()]
    print(f"Initializing population of size {args.pop_size}...")
    for i in range(args.pop_size - 1):
        mutated_initial = mutate(initial_weights, args.init_mutation_rate, args.init_mutation_strength)
        population.append(mutated_initial)
        # print(f"  Generated individual {i+2}/{args.pop_size} via initial mutation.") # Reduce print noise

    # --- Evolution Cycle ---
    best_fitness_overall = -float('inf')
    best_weights_overall = initial_weights.copy()

    for generation in range(args.generations):
        gen_start_time = time.time()
        print(f"\n--- Generation {generation + 1}/{args.generations} ---")

        # 1. Evaluate Population (Now potentially uses torch.compile)
        fitness_scores = evaluate_population(
            population, model_definition_path, args.model_class, task_eval_func, device, model_args, model_kwargs
        )

        # 2. Process Results
        valid_fitness_scores = [f for f in fitness_scores if f > -float('inf')]
        if not valid_fitness_scores:
             print("Error: All individuals failed evaluation in this generation. Stopping evolution.")
             break

        max_fitness = np.max(valid_fitness_scores)
        avg_fitness = np.mean(valid_fitness_scores)
        best_idx_current_gen = np.argmax(fitness_scores)
        print(f"Generation {generation + 1} Stats: Max Fitness = {max_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")

        if fitness_scores[best_idx_current_gen] > best_fitness_overall:
            best_fitness_overall = fitness_scores[best_idx_current_gen]
            best_weights_overall = population[best_idx_current_gen].copy()
            print(f"*** New best overall fitness found: {best_fitness_overall:.4f} ***")

        # 3. Selection
        num_parents_to_select = max(2, args.pop_size // 2)
        parents = select_parents(population, fitness_scores, num_parents_to_select)

        if not parents:
            print("Warning: Parent selection failed. Re-initializing population.")
            population = [best_weights_overall.copy()] + [mutate(best_weights_overall, args.mutation_rate, args.mutation_strength) for _ in range(args.pop_size - 1)]
            continue

        # 4. Reproduction
        next_population = []
        next_population.append(population[best_idx_current_gen].copy()) # Elitism
        # print(f"  Added elite individual (fitness {fitness_scores[best_idx_current_gen]:.4f})")

        while len(next_population) < args.pop_size:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            try:
                child = crossover(p1, p2)
                mutated_child = mutate(child, args.mutation_rate, args.mutation_strength)
                next_population.append(mutated_child)
            except ValueError as cross_err:
                 print(f"Warning: Crossover/Mutation error: {cross_err}. Skipping offspring.")
                 if len(next_population) < args.pop_size:
                      next_population.append(mutate(random.choice(parents), args.mutation_rate, args.mutation_strength))

        population = next_population
        gen_time = time.time() - gen_start_time
        print(f"Generation {generation + 1} finished in {gen_time:.2f}s")

    # --- End of Evolution ---
    total_time = time.time() - start_time
    print("\n--- Evolution Finished ---")
    print(f"Total evolution time: {total_time:.2f}s ({total_time/60:.1f} mins)")
    print(f"Best fitness achieved overall: {best_fitness_overall:.4f}")

    # Save the best weights found
    output_filename_base = args.output_weights_file or f"evolved_best_{args.model_class}.pth"
    output_path = os.path.join(args.model_dir, output_filename_base)
    print(f"Saving best weights to {output_path}...")
    try:
        final_best_model = load_pytorch_model(
            model_definition_path, args.model_class, None, device, *model_args, **model_kwargs
        )
        load_weights_from_flat(final_best_model, best_weights_overall)
        torch.save(final_best_model.state_dict(), output_path)
        print("Best weights saved successfully.")
    except Exception as e:
        print(f"Error saving final weights: {e}")

# --- Argument Parser and Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolve weights of a pre-loaded PyTorch model using a Genetic Algorithm.")

    # Required Arguments
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing model_definition.py, task_evaluation.py, and optionally the weights file.")
    parser.add_argument("--model-class", type=str, required=True, help="Name of the model class within model_definition.py (e.g., SimpleCNN).")
    parser.add_argument("--model-type", type=str, required=True, choices=['cnn', 'rnn'], help="Type of the model ('cnn' or 'rnn').")
    # Made weights file optional
    parser.add_argument("--weights-file", type=str, default=None, help="[Optional] Filename of the initial model weights (.pt or .pth) within the model directory. If omitted, starts from random init.")

    # User Modules Path (Consolidated)
    parser.add_argument("--task-module", type=str, default="task_evaluation.py", help="Filename of the Python file containing the 'evaluate_network_on_task' function (within model directory).")


    # Optional Arguments - GA Parameters
    parser.add_argument("--generations", type=int, default=50, help="Number of generations.")
    parser.add_argument("--pop-size", type=int, default=30, help="Population size.")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Fraction of weights to mutate.")
    parser.add_argument("--mutation-strength", type=float, default=0.05, help="Std deviation of mutation noise.")
    parser.add_argument("--init-mutation-rate", type=float, default=0.2, help="Mutation rate for initial population.")
    parser.add_argument("--init-mutation-strength", type=float, default=0.1, help="Mutation strength for initial population.")
    parser.add_argument("--output-weights-file", type=str, default=None, help="Filename for saving the final best evolved weights. Defaults to 'evolved_best_<ModelClass>.pth'.")
    parser.add_argument("--cpu", action='store_true', help="Force using CPU.")

    args = parser.parse_args()

    # Update task_module path to be relative to model_dir
    # This simplifies command line, assuming task_evaluation.py is in the model_dir
    args.task_module = os.path.join(args.model_dir, args.task_module)


    # Basic validation
    if args.pop_size < 2: print("Error: Population size must be >= 2."); exit()
    if args.generations < 1: print("Error: Generations must be >= 1."); exit()

    run_evolution(args)

