# final_evolve_script_bokeh_server.py

# --- Core Libraries ---
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import importlib.util
import copy
import random
import time
import logging
from pathlib import Path
import threading # For running evolution in background
import queue     # For passing data between threads
import sys       # For accessing command line args with bokeh serve

# --- Bokeh Server Libraries ---
try:
    from bokeh.plotting import figure, ColumnDataSource, curdoc
    from bokeh.layouts import column
    from bokeh.models import NumeralTickFormatter # For formatting axes
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

# --- TensorBoard ---
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# --- Configure Logging ---
# MODIFICATION: Set level to DEBUG to see detailed trace messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- PyTorch Compile Check ---
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile') and callable(torch.compile)
if TORCH_COMPILE_AVAILABLE:
    logger.info("torch.compile available (PyTorch 2.0+ detected).")
else:
    logger.info("torch.compile not available (requires PyTorch 2.0+).")

# --- Utility, Model Loading, GA Functions ---
# --- START OF RE-INSERTED MISSING FUNCTIONS (from previous fix) ---

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
        logger.error(f"Error during weight flattening: {e}", exc_info=True)
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
            logger.warning(f"Size mismatch after loading weights. Offset {offset} != flat_weights length {len(flat_weights_tensor)}. Check model definition correspondence.")

    except Exception as e:
        logger.error(f"Error loading weights from flat vector: {e}", exc_info=True)
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

        # Use DEBUG for instantiation logs as they can be frequent
        logger.debug(f"Instantiating model '{class_name}' with args: {model_args}, kwargs: {model_kwargs}")
        model = ModelClass(*model_args, **model_kwargs)
        model.to(device)

        if state_dict_path and os.path.exists(state_dict_path):
            logger.info(f"Loading state_dict from: {state_dict_path}")
            try:
                state_dict = torch.load(state_dict_path, map_location=device)
                model.load_state_dict(state_dict)
                logger.info("State_dict loaded successfully.")
            except Exception as load_err:
                logger.error(f"Error loading state_dict: {load_err}. Check architecture.", exc_info=True)
                raise
        elif state_dict_path:
            logger.warning(f"state_dict path '{state_dict_path}' provided but not found. Using initial model weights.")
        else:
             # This log might be less useful now, change to DEBUG
             logger.debug("No state_dict path provided. Using initial model weights.")

        model.eval()
        return model

    except Exception as e:
        logger.error(f"Error in load_pytorch_model: {e}", exc_info=True)
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
        logger.error(f"Error loading task evaluation function: {e}", exc_info=True)
        raise

def select_parents(population_weights, fitness_scores, num_parents):
    """ Selects parents using tournament selection. """
    parents = []
    population_size = len(population_weights)
    if population_size == 0: return []
    tournament_size = max(2, min(population_size, 5))
    valid_indices = [i for i, f in enumerate(fitness_scores) if f > -float('inf')]
    if not valid_indices:
        logger.warning("All individuals failed evaluation. Cannot select parents.")
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
            logger.warning("Could not select a winner in tournament. Picking random valid individual.")
            parents.append(population_weights[random.choice(valid_indices)])
    return parents

def crossover(parent1, parent2):
    """ Simple average crossover for weight vectors. """
    p1 = np.array(parent1)
    p2 = np.array(parent2)
    if p1.shape != p2.shape:
        # Log error before raising might be helpful
        logger.error(f"Parent shapes do not match for crossover: {p1.shape} vs {p2.shape}")
        raise ValueError(f"Parent shapes do not match for crossover: {p1.shape} vs {p2.shape}")
    child = (p1 + p2) / 2.0
    return child

def mutate(weights, mutation_rate, mutation_strength):
    """ Adds Gaussian noise to a fraction of weights based on mutation rate. """
    if mutation_rate <= 0 or mutation_strength <= 0:
         return weights
    mutated_weights = weights.copy()
    num_weights_to_mutate = int(len(weights) * mutation_rate)
    if num_weights_to_mutate == 0 and mutation_rate > 0: num_weights_to_mutate = 1
    indices_to_mutate = np.random.choice(len(weights), num_weights_to_mutate, replace=False)
    noise = np.random.normal(0, mutation_strength, size=num_weights_to_mutate)
    mutated_weights[indices_to_mutate] += noise.astype(mutated_weights.dtype)
    return mutated_weights

# --- END OF RE-INSERTED MISSING FUNCTIONS ---


# --- Small modification to evaluate_population to put results in queue instead of returning ---
def evaluate_population(population_weights, model_definition_path, class_name, task_eval_func, device, model_args, model_kwargs, result_queue):
    """ Evaluates fitness and puts results (scores, avg_time) into the queue. """
    fitness_scores = []
    eval_times = []
    num_individuals = len(population_weights)
    try: # Wrap the whole process
        for i, flat_weights in enumerate(population_weights):
            individual_start_time = time.time()
            current_model = None
            compiled_model_obj = None # Keep variable defined even if compile disabled
            compile_attempted = False
            compile_successful = False

            # Use \r for progress indicator, but maybe log less frequently or use tqdm if preferred
            print(f"\rEvaluating individual {i+1}/{num_individuals}...", end="") # Keep terminal progress

            try:
                current_model = load_pytorch_model(model_definition_path, class_name, None, device, *model_args, **model_kwargs)
                load_weights_from_flat(current_model, flat_weights)
                current_model.to(device)
                current_model.eval()

                model_to_evaluate = current_model
                # --- torch.compile attempt (kept disabled for now) ---
                if False: # torch.compile disabled
                    compile_attempted = True
                    # ... compile logic ...
            # --- End compile block ---

                fitness = task_eval_func(model_to_evaluate, device)

                if not isinstance(fitness, (float, int)):
                     logger.warning(f"Fitness func non-numeric ({type(fitness)}) for ind {i+1}. Setting -inf.")
                     fitness = -float('inf')

                fitness_scores.append(float(fitness))
                eval_time = time.time() - individual_start_time
                eval_times.append(eval_time)

            except Exception as e:
                logger.error(f"Error evaluating individual {i+1}: {e}", exc_info=True)
                fitness_scores.append(-float('inf'))
                eval_times.append(time.time() - individual_start_time) # Record time on error too
            finally:
                # Cleanup
                del model_to_evaluate
                if current_model is not None and compiled_model_obj is not None and current_model is not compiled_model_obj: del current_model
                elif current_model is not None and compiled_model_obj is None: del current_model
                compiled_model_obj = None
                if device.type == 'cuda': torch.cuda.empty_cache()

        print() # Newline after progress
        avg_eval_time = np.mean(eval_times) if eval_times else 0
        logger.info(f"Finished evaluating {num_individuals} individuals. Avg eval time: {avg_eval_time:.2f}s")
        # Put results into the queue for the main thread
        result_queue.put((fitness_scores, avg_eval_time))

    except Exception as outer_e:
         logger.error(f"Critical error during population evaluation: {outer_e}", exc_info=True)
         result_queue.put(([-float('inf')] * num_individuals, 0)) # Signal failure


# --- Evolution Loop Modified for Threading ---
def run_evolution_thread(args, data_queue, stop_event):
    """ Runs the evolution loop in a separate thread and sends data via queue. """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"[Thread] Using device: {device}")
    # --- Paths and Loading ---
    try:
        model_definition_path = str(Path(args.model_dir) / "model_definition.py")
        weights_path = Path(args.model_dir) / args.weights_file if args.weights_file else None
        task_module_path = str(Path(args.model_dir) / args.task_module)
        if not os.path.exists(task_module_path): raise FileNotFoundError(task_module_path)

        task_eval_func = load_task_eval_function(task_module_path)
        model_args = []; model_kwargs = {}
        initial_model = load_pytorch_model(model_definition_path, args.model_class, weights_path, device, *model_args, **model_kwargs)
        initial_weights = flatten_weights(initial_model)
        logger.info(f"[Thread] Initial model loaded. Weight vector size: {initial_weights.shape[0]}")
        del initial_model
        if device.type == 'cuda': torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"[Thread] Error during initial loading: {e}", exc_info=True)
        data_queue.put({'status': 'error', 'message': str(e)}); return

    # --- Population Init ---
    population = [initial_weights.copy()]
    logger.info(f"[Thread] Initializing population of size {args.pop_size}...")
    for i in range(args.pop_size - 1):
        mutated_initial = mutate(initial_weights, args.init_mutation_rate, args.init_mutation_strength)
        population.append(mutated_initial)

    best_fitness_overall = -float('inf')
    best_weights_overall = initial_weights.copy()
    evaluation_result_queue = queue.Queue(maxsize=1)

    # --- Evolution Cycle ---
    try:
        for generation in range(args.generations):
            if stop_event.is_set(): logger.info("[Thread] Stop event received, terminating."); break

            gen_start_time = time.time()
            logger.info(f"[Thread] --- Generation {generation + 1}/{args.generations} ---")

            # 1. Evaluate Population
            evaluate_population(population, model_definition_path, args.model_class, task_eval_func, device, model_args, model_kwargs, evaluation_result_queue)
            fitness_scores, avg_eval_time = evaluation_result_queue.get()

            # 2. Process Results
            valid_fitness_scores = [f for f in fitness_scores if f > -float('inf')]
            if not valid_fitness_scores: logger.error("[Thread] All individuals failed evaluation. Stopping."); break
            max_fitness = np.max(valid_fitness_scores)
            avg_fitness = np.mean(valid_fitness_scores)
            min_fitness = np.min(valid_fitness_scores)
            best_idx_current_gen = np.argmax(fitness_scores)
            logger.info(f"[Thread] Generation {generation + 1} Stats: Max={max_fitness:.4f}, Avg={avg_fitness:.4f}, Min={min_fitness:.4f}")

            if fitness_scores[best_idx_current_gen] > best_fitness_overall:
                best_fitness_overall = fitness_scores[best_idx_current_gen]
                best_weights_overall = population[best_idx_current_gen].copy()
                logger.info(f"[Thread] *** New best overall fitness found: {best_fitness_overall:.4f} ***")

            # 3. Selection
            num_parents_to_select = max(2, args.pop_size // 2)
            parents = select_parents(population, fitness_scores, num_parents_to_select)

            if not parents:
                logger.warning("[Thread] Parent selection failed. Re-initializing population from best.")
                population = [best_weights_overall.copy()] + [mutate(best_weights_overall, args.mutation_rate, args.mutation_strength) for _ in range(args.pop_size - 1)]
                gen_time = time.time() - gen_start_time
                # MODIFICATION: Log data put on queue
                data_to_send = {'generation': generation + 1, 'max_fitness': max_fitness, 'avg_fitness': avg_fitness, 'min_fitness': min_fitness, 'gen_time': gen_time, 'avg_eval_time': avg_eval_time, 'status': 'running'}
                logger.debug(f"[Thread] Putting data onto queue (after selection fail) for Gen {generation + 1}: {data_to_send}")
                data_queue.put(data_to_send)
                continue # Skip reproduction

            # 4. Reproduction
            next_population = [population[best_idx_current_gen].copy()] # Elitism
            while len(next_population) < args.pop_size:
                p1 = random.choice(parents); p2 = random.choice(parents)
                try:
                    child = crossover(p1, p2)
                    mutated_child = mutate(child, args.mutation_rate, args.mutation_strength)
                    next_population.append(mutated_child)
                except ValueError as cross_err:
                    logger.warning(f"[Thread] Crossover/Mutation error: {cross_err}. Skipping offspring.")
                    if len(next_population) < args.pop_size: next_population.append(mutate(random.choice(parents), args.mutation_rate, args.mutation_strength))
            population = next_population
            gen_time = time.time() - gen_start_time
            logger.info(f"[Thread] Generation {generation + 1} finished in {gen_time:.2f}s")

            # MODIFICATION: Log data put on queue
            data_to_send = {
                'generation': generation + 1,
                'max_fitness': max_fitness,
                'avg_fitness': avg_fitness,
                'min_fitness': min_fitness,
                'gen_time': gen_time,
                'avg_eval_time': avg_eval_time,
                'status': 'running'
            }
            logger.debug(f"[Thread] Putting data onto queue for Gen {generation + 1}: {data_to_send}")
            data_queue.put(data_to_send)

        # --- Finished Loop ---
        logger.info("[Thread] Evolution loop finished.")
        # MODIFICATION: Log data put on queue
        finish_data = {'status': 'finished', 'best_fitness': best_fitness_overall, 'best_weights': best_weights_overall}
        logger.debug(f"[Thread] Putting finish data onto queue: {finish_data.keys()}") # Log keys only for brevity
        data_queue.put(finish_data)


    except Exception as e:
        logger.error(f"[Thread] Exception in evolution loop: {e}", exc_info=True)
        # MODIFICATION: Log data put on queue
        error_data = {'status': 'error', 'message': str(e)}
        logger.debug(f"[Thread] Putting error data onto queue: {error_data}")
        data_queue.put(error_data)
    except KeyboardInterrupt: # Allow thread to be interrupted
         logger.warning("[Thread] KeyboardInterrupt received in thread.")
         # MODIFICATION: Log data put on queue
         interrupt_data = {'status': 'interrupted'}
         logger.debug(f"[Thread] Putting interrupt data onto queue: {interrupt_data}")
         data_queue.put(interrupt_data)
    finally:
        total_time = time.time() - start_time
        logger.info(f"[Thread] Evolution thread finished. Total time: {total_time:.2f}s")


# --- Bokeh Server Application ---

# --- Argument Parsing (Corrected for 'bokeh serve') ---
parser = argparse.ArgumentParser(description="Evolve weights (Bokeh Server version).")
# --- Add ALL arguments ---
parser.add_argument("--model-dir", type=str, required=True, help="Directory containing model_definition.py, task_evaluation.py, etc.")
parser.add_argument("--model-class", type=str, required=True, help="Name of the model class.")
parser.add_argument("--model-type", type=str, required=True, choices=['cnn', 'rnn'], help="Model type.")
parser.add_argument("--weights-file", type=str, default=None, help="[Optional] Initial weights filename.")
parser.add_argument("--task-module", type=str, default="task_evaluation.py", help="Task evaluation filename.")
parser.add_argument("--generations", type=int, default=50, help="Number of generations.")
parser.add_argument("--pop-size", type=int, default=30, help="Population size.")
parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation rate.")
parser.add_argument("--mutation-strength", type=float, default=0.05, help="Mutation strength.")
parser.add_argument("--init-mutation-rate", type=float, default=0.2, help="Initial mutation rate.")
parser.add_argument("--init-mutation-strength", type=float, default=0.1, help="Initial mutation strength.")
parser.add_argument("--output-weights-file", type=str, default=None, help="Output weights filename.")
parser.add_argument("--cpu", action='store_true', help="Force CPU.")
parser.add_argument("--enable-tensorboard", action='store_true', help="Enable logging to TensorBoard.")
parser.add_argument("--tensorboard-logdir", type=str, default=None, help="Custom TensorBoard log directory.")

def filter_bokeh_args(argv):
    """ Separates Bokeh server args from script args passed after '--args'. """
    try:
        args_index = argv.index('--args')
        script_args = argv[args_index + 1:]
        return script_args
    except ValueError: return argv[1:]

ARGS = parser.parse_args(filter_bokeh_args(sys.argv))
# --- End Argument Parsing Correction ---

# --- Global variables ---
data_queue = queue.Queue()
stop_event = threading.Event()
evolution_thread = None
tb_writer = None
bokeh_sources = None

# --- Bokeh Document Setup Function ---
def modify_doc(doc):
    """Setup the Bokeh document, plots, callbacks, and start the evolution thread."""
    global ARGS, evolution_thread, tb_writer, bokeh_sources

    if not BOKEH_AVAILABLE: logger.error("Bokeh library not found."); return

    # --- Setup TensorBoard ---
    if ARGS.enable_tensorboard:
        if TENSORBOARD_AVAILABLE:
            log_dir = ARGS.tensorboard_logdir or os.path.join("runs", f"{ARGS.model_class}_bokeh_{int(time.time())}")
            os.makedirs(log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging enabled. Run: tensorboard --logdir {os.path.abspath(log_dir)}")
        else: logger.warning("TensorBoard library not found. Disabling.")

    # --- Setup Bokeh Data Sources ---
    fitness_source = ColumnDataSource(data={'generation': [], 'max_fitness': [], 'avg_fitness': [], 'min_fitness': []})
    time_source = ColumnDataSource(data={'generation': [], 'gen_time': [], 'avg_eval_time': []})
    bokeh_sources = {'fitness': fitness_source, 'time': time_source}

    # --- Setup Bokeh Plots ---
    p_fitness = figure(height=300, title="Fitness over Generations", x_axis_label="Generation", y_axis_label="Fitness", sizing_mode="stretch_width")
    p_fitness.line(x='generation', y='max_fitness', source=fitness_source, legend_label="Max", color="green", line_width=2)
    p_fitness.line(x='generation', y='avg_fitness', source=fitness_source, legend_label="Avg", color="blue", line_width=2)
    p_fitness.line(x='generation', y='min_fitness', source=fitness_source, legend_label="Min", color="red", line_dash="dotted", line_width=2)
    p_fitness.legend.location = "top_left"; p_fitness.legend.click_policy="hide"
    p_time = figure(height=250, title="Timings per Generation", x_axis_label="Generation", y_axis_label="Seconds", sizing_mode="stretch_width", x_range=p_fitness.x_range)
    p_time.line(x='generation', y='gen_time', source=time_source, legend_label="Gen Time", color="orange", line_width=2)
    p_time.line(x='generation', y='avg_eval_time', source=time_source, legend_label="Avg Eval Time", color="purple", line_dash="dashed", line_width=2)
    p_time.legend.location = "top_left"; p_time.legend.click_policy="hide"

    # --- Define Update Callback ---
    def update_plot(data_point):
        """Callback to update Bokeh data sources."""
        # MODIFICATION: Log start
        logger.debug(f"Callback update_plot running for Gen {data_point.get('generation')}...")
        new_fitness_data = {'generation': [data_point['generation']],'max_fitness': [data_point['max_fitness']],'avg_fitness': [data_point['avg_fitness']],'min_fitness': [data_point['min_fitness']]}
        new_time_data = {'generation': [data_point['generation']],'gen_time': [data_point['gen_time']],'avg_eval_time': [data_point['avg_eval_time']]}
        # MODIFICATION: Log before stream
        logger.debug(f"Streaming fitness data: {new_fitness_data}")
        logger.debug(f"Streaming time data: {new_time_data}")
        try:
            fitness_source.stream(new_fitness_data, rollover=ARGS.generations + 10)
            time_source.stream(new_time_data, rollover=ARGS.generations + 10)
            # MODIFICATION: Log success
            logger.debug(f"Stream successful for Gen {data_point.get('generation')}")
        except Exception as e:
             logger.error(f"Error during CDS stream: {e}", exc_info=True)

        if tb_writer:
             current_gen_num = data_point['generation']
             tb_writer.add_scalar('Fitness/Max', data_point['max_fitness'], current_gen_num); tb_writer.add_scalar('Fitness/Average', data_point['avg_fitness'], current_gen_num); tb_writer.add_scalar('Fitness/Min', data_point['min_fitness'], current_gen_num)
             tb_writer.add_scalar('Timings/Generation_Seconds', data_point['gen_time'], current_gen_num); tb_writer.add_scalar('Timings/Avg_Eval_Seconds', data_point['avg_eval_time'], current_gen_num)
             tb_writer.flush()

    # --- Define Periodic Check Function ---
    final_results = {}
    def check_queue():
        """Periodically check the queue for new data from the evolution thread."""
        # MODIFICATION: Log start
        logger.debug("Callback check_queue running...")
        try:
            data = data_queue.get_nowait()
            # MODIFICATION: Log data received
            logger.debug(f"Callback check_queue got data: {data}")
            if data['status'] == 'running':
                 # MODIFICATION: Log scheduling
                 logger.debug(f"Callback check_queue scheduling update_plot for Gen {data.get('generation')}")
                 doc.add_next_tick_callback(lambda data=data: update_plot(data))
            elif data['status'] in ['finished', 'error', 'interrupted']:
                logger.info(f"Evolution thread status: {data['status']}")
                final_results['status'] = data['status']
                if data['status'] == 'finished':
                     final_results['best_fitness'] = data.get('best_fitness')
                     final_results['best_weights'] = data.get('best_weights')
                stop_callback()
                if data['status'] == 'finished' and final_results.get('best_weights') is not None:
                     # Ensure saving happens *after* potential final plot update
                     doc.add_next_tick_callback(lambda: save_final_results(ARGS, final_results['best_weights'], final_results['best_fitness'], bokeh_sources))


        except queue.Empty: pass
        except Exception as e: logger.error(f"Error in check_queue callback: {e}", exc_info=True); stop_callback()

    # --- Define Callback Stop Function ---
    callback_id = None
    def stop_callback():
        nonlocal callback_id
        if callback_id:
            try: doc.remove_periodic_callback(callback_id); logger.info("Stopped periodic queue check."); callback_id = None
            except Exception as e: logger.error(f"Error stopping callback: {e}")

    # --- Add Plots to Document ---
    layout = column(p_fitness, p_time, sizing_mode="stretch_width")
    doc.add_root(layout)

    # --- Start Evolution Thread ---
    logger.info("Starting evolution thread...")
    evolution_thread = threading.Thread(target=run_evolution_thread, args=(ARGS, data_queue, stop_event), name="EvolutionThread", daemon=True) # Give thread a name
    evolution_thread.start()

    # --- Register Periodic Callback ---
    callback_id = doc.add_periodic_callback(check_queue, 500) # Check queue every 500ms

    # --- Handle Session Destruction ---
    def cleanup_session(session_context):
        logger.warning("Bokeh session destroyed. Signaling evolution thread to stop.")
        stop_event.set()
        if evolution_thread and evolution_thread.is_alive():
             evolution_thread.join(timeout=10)
             if evolution_thread.is_alive(): logger.warning("Evolution thread did not exit cleanly after 10s.")
        if tb_writer: tb_writer.close(); logger.info("TensorBoard writer closed during cleanup.")
        if final_results.get('status') == 'finished' and final_results.get('best_weights') is not None:
             logger.info("Attempting to save results during session cleanup...")
             save_final_results(ARGS, final_results['best_weights'], final_results['best_fitness'], bokeh_sources)
        elif not final_results: logger.info("No final results recorded, likely interrupted before finish.")

    doc.on_session_destroyed(cleanup_session)

def save_final_results(args, best_weights_overall, best_fitness_overall, bokeh_sources):
    """Saves final weights and optional HTML plot."""
    if getattr(save_final_results, 'executed', False): logger.info("save_final_results already executed, skipping."); return
    setattr(save_final_results, 'executed', True)

    logger.info("\n--- Saving Final Results ---")
    logger.info(f"Best fitness achieved overall: {best_fitness_overall:.4f}")

    # Save weights
    output_filename_base = args.output_weights_file or f"evolved_best_{args.model_class}.pth"
    output_path = Path(args.model_dir) / output_filename_base
    logger.info(f"Saving best weights to {output_path}...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model_definition_path = str(Path(args.model_dir) / "model_definition.py")
        model_args=[]; model_kwargs={}
        final_best_model = load_pytorch_model(model_definition_path, args.model_class, None, device, *model_args, **model_kwargs)
        load_weights_from_flat(final_best_model, best_weights_overall)
        torch.save(final_best_model.state_dict(), str(output_path))
        logger.info("Best weights saved successfully.")
    except Exception as e: logger.error(f"Error saving final weights: {e}", exc_info=True)

    # Save HTML Plot
    if bokeh_sources:
         try:
             final_fitness_source = bokeh_sources['fitness']; final_time_source = bokeh_sources['time']
             p_fitness_final = figure(height=300, title="Fitness (Final)", x_axis_label="Generation", y_axis_label="Fitness", sizing_mode="stretch_width")
             p_fitness_final.line(x='generation', y='max_fitness', source=final_fitness_source, legend_label="Max", color="green"); p_fitness_final.line(x='generation', y='avg_fitness', source=final_fitness_source, legend_label="Avg", color="blue"); p_fitness_final.line(x='generation', y='min_fitness', source=final_fitness_source, legend_label="Min", color="red", line_dash="dotted")
             p_time_final = figure(height=250, title="Timings (Final)", x_axis_label="Generation", y_axis_label="Seconds", sizing_mode="stretch_width", x_range=p_fitness_final.x_range)
             p_time_final.line(x='generation', y='gen_time', source=final_time_source, legend_label="Gen Time", color="orange"); p_time_final.line(x='generation', y='avg_eval_time', source=final_time_source, legend_label="Avg Eval Time", color="purple", line_dash="dashed")
             final_layout = column(p_fitness_final, p_time_final, sizing_mode="stretch_width")
             html_filename = f"evolution_plot_{args.model_class}_bokeh.html"
             html_filepath = Path(args.model_dir) / html_filename
             from bokeh.io import output_file, save
             output_file(str(html_filepath), title="Evolution Progress")
             save(final_layout)
             logger.info(f"Saved final Bokeh graph to {html_filepath}")
         except Exception as plot_save_err: logger.error(f"Could not save final Bokeh plot to HTML: {plot_save_err}")


# --- Start Bokeh Application ---
if BOKEH_AVAILABLE:
    logger.info("Setting up Bokeh document...")
    # The 'bokeh serve' command executes the script and uses this call
    # to get the application instance.
    modify_doc(curdoc())
    logger.info("Bokeh document setup complete. Server is running.")
else:
    logger.error("Bokeh is not installed. Please run 'pip install bokeh'.")

