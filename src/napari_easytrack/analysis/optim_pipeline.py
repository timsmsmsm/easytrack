import gc
import json
from multiprocessing import Process, Pipe
from multiprocessing import Semaphore

import btrack
# misc imports
import numpy as np
import optuna
import pandas as pd
from btrack import utils, config
from joblib import parallel_backend
from traccuracy._tracking_graph import TrackingGraph
# import traccuracy
from traccuracy.loaders._ctc import ctc_to_graph, _get_node_attributes
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics


def write_best_params_to_config(params, config_file_path):
    """
    Write the best parameters to the config file in JSON format.

    Args:
        params (dict): Dictionary containing the best parameters.
        config_file_path (str): Path to the config file where parameters will be written.

    """
    config = {
        "TrackerConfig": {
            "MotionModel": {
                "name": "cell_motion",
                "dt": params.get('dt', 1.0),
                "measurements": params.get('measurements', 3),
                "states": params.get('states', 6),
                "accuracy": params.get('accuracy', 7.5),
                "prob_not_assign": params.get('prob_not_assign', 0.1),
                "max_lost": params.get('max_lost', 5),
                "max_search_radius": params.get('max_search_radius', 100),
                "A": {
                    "matrix": [
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1
                    ]
                },
                "H": {
                    "matrix": [
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0
                    ]
                },
                "P": {
                    "sigma": params.get('p_sigma', 150.0),
                    "matrix": [
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1
                    ]
                },
                "G": {
                    "sigma": params.get('g_sigma', 15.0),
                    "matrix": [
                        0.5,
                        0.5,
                        0.5,
                        1,
                        1,
                        1
                    ]
                },
                "R": {
                    "sigma": params.get('r_sigma', 5.0),
                    "matrix": [
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1
                    ]
                }
            },
            "ObjectModel": {},
            "HypothesisModel": {
                "name": "cell_hypothesis",
                "hypotheses": [
                    "P_FP",
                    "P_init",
                    "P_term",
                    "P_link",
                    "P_branch",
                    "P_dead"
                ],
                "lambda_time": params.get('lambda_time', 5.0),
                "lambda_dist": params.get('lambda_dist', 3.0),
                "lambda_link": params.get('lambda_link', 10.0),
                "lambda_branch": params.get('lambda_branch', 50.0),
                "eta": params.get('eta', 1e-10),
                "theta_dist": params.get('theta_dist', 20.0),
                "theta_time": params.get('theta_time', 5.0),
                "dist_thresh": params.get('dist_thresh', 40),
                "time_thresh": params.get('time_thresh', 2),
                "apop_thresh": params.get('apop_thresh', 5),
                "segmentation_miss_rate": params.get('segmentation_miss_rate', 0.1),
                "apoptosis_rate": params.get('apoptosis_rate', 0.001),
                "relax": params.get('relax', True),
            }
        }
    }

    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=4)

def wrapper_func(conn, func, args, kwargs):
    """
    Wrapper function to execute a function with arguments and send the result through a pipe.

    Args:
        conn (multiprocessing.Pipe): Pipe for sending results.
        func (callable): Function to be executed.
        args (tuple): Arguments to pass to the function.
        kwargs (dict): Keyword arguments to pass to the function.
    """
    try:
        result = func(*args, **kwargs)
        conn.send(result)
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()

def run_with_timeout(func, args, kwargs, timeout, objectives, sem, timeout_penalty):
    """
    Run a function with a timeout and handle termination if it exceeds the timeout.

    Args:
        func (callable): Function to be executed.
        args (tuple): Arguments to pass to the function.
        kwargs (dict): Keyword arguments to pass to the function.
        timeout (int): Timeout in seconds.
        objectives (str): Objectives type, '1obj' or '2obj'.
        sem (multiprocessing.Semaphore): Semaphore to control concurrent execution.
        timeout_penalty (float): Penalty value for timeouts.

    Returns:
        result: Result of the function execution or a default value if timed out.
    """
    acquired = sem.acquire(timeout=timeout)
    if not acquired:
        print("Failed to acquire semaphore due to timeout.")
        return None  # Or handle the case where semaphore was not acquired

    parent_conn, child_conn = Pipe()
    p = Process(target=wrapper_func, args=(child_conn, func, args, kwargs))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print("Trial terminated due to timeout.")
        if objectives == '1obj':
            result = timeout_penalty
        elif objectives == '2obj':
            result = (timeout_penalty, 0.0)
    else:
        result = parent_conn.recv() if parent_conn.poll() else None
        if isinstance(result, Exception):
            print(f"Exception in child process: {result}")
            result = None

    parent_conn.close()
    child_conn.close()

    # Release semaphore
    sem.release()
    return result

def objective_with_timeout(trial, dataset, gt_data, objectives, timeout, sem, timeout_penalty):
    """
    Wrapper for running the optimization objective with a timeout.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        dataset: Dataset to be optimized.
        gt_data: Ground truth data.
        objectives (str): Objectives type, '1obj' or '2obj'.
        timeout (int): Timeout in seconds.
        sem (multiprocessing.Semaphore): Semaphore to control concurrent execution.
        timeout_penalty (float): Penalty value for timeouts.

    Returns:
        result: Result of the optimization objective.
    """
    try:
        return run_with_timeout(objective, (trial, dataset, gt_data, objectives), {}, timeout, objectives, sem, timeout_penalty)
    except Exception as e:
        print(f"Exception occurred during trial execution: {e}")
        return None

def optimize_dataset_with_timeout(dataset, gt_data, objectives, study_name, n_trials=64, timeout=300, timeout_penalty=100000, use_parallel_backend=True, sampler='tpe', search_space=None):
    """
    Optimize a dataset with a timeout for each trial.

    Args:
        dataset: Dataset to be optimized.
        gt_data: Ground truth data.
        objectives (str): Objectives type, '1obj' or '2obj'.
        study_name (str): Name of the Optuna study.
        n_trials (int, optional): Number of trials to run. Defaults to 64.
        timeout (int, optional): Timeout for each trial in seconds. Defaults to 300.
        timeout_penalty (float, optional): Penalty value for timeouts. Defaults to 100000.
        use_parallel_backend (bool, optional): Whether to use parallel backend. Defaults to True.
        sampler (str, optional): Sampler to use for the optimization. Defaults to 'tpe'.
        search_space (dict, optional): Search space for grid sampling.

    Returns:
        optuna.study.Study: The Optuna study object.
    """
    if sampler == 'NSGA-II' and objectives == '1obj':
        raise ValueError("NSGA-II sampler is not supported for single-objective optimisation.")
    if sampler == 'cmaes' and objectives == '2obj':
        raise ValueError("CMA-ES sampler is not supported for multi-objective optimisation.")

    sampler_options = {
        'random': optuna.samplers.RandomSampler(),
        'grid': optuna.samplers.GridSampler(search_space) if search_space else None,
        'cmaes': optuna.samplers.CmaEsSampler(),
        'tpe': optuna.samplers.TPESampler(),
        'NSGA-II': optuna.samplers.NSGAIISampler()
    }

    if sampler not in sampler_options or sampler_options[sampler] is None:
        raise ValueError("Invalid or unspecified sampler for the provided configuration.")
    
    storage = optuna.storages.RDBStorage(url="sqlite:///btrack.db", engine_kwargs={"connect_args": {"timeout": 100}})

    direction = ["minimize"] if objectives == '1obj' else ["minimize", "maximize"]

    study = optuna.create_study(directions=direction, study_name=study_name, storage=storage, load_if_exists=True, sampler=sampler_options[sampler])

    # Counter for completed trials to enable staggering only at start
    completed_trials = [0]  # Use list to make it mutable in closure
    
    def objective_with_stagger(trial):
        """Wrapper that adds stagger delay for first few parallel trials."""
        # Only stagger the first batch of parallel trials (first 8 trials)
        if use_parallel_backend and completed_trials[0] < 8:
            import time
            import random
            # Stagger trials 0-3 with 0-3 second delays
            delay = (trial.number % 4) * 1.0 + random.uniform(0, 0.5)
            print(f"Trial {trial.number}: Stagger delay {delay:.2f}s")
            time.sleep(delay)
        
        result = objective_with_timeout(trial, dataset, gt_data, objectives, timeout, sem, timeout_penalty)
        completed_trials[0] += 1
        return result

    if use_parallel_backend:
        sem = Semaphore(value=4)
        with parallel_backend('multiprocessing'):
            study.optimize(objective_with_stagger, n_trials=n_trials, n_jobs=4, gc_after_trial=True)
    else:
        sem = Semaphore(value=1)
        study.optimize(lambda trial: objective_with_timeout(trial, dataset, gt_data, objectives, timeout, sem, timeout_penalty), n_trials=n_trials, n_jobs=1, gc_after_trial=True)
    return study


def read_config_params(config_file_path):
    """
    Read parameters from a config file.

    Args:
        config_file_path (str): Path to the config file.

    Returns:
        dict: Dictionary containing the config parameters.
    """
    with open(config_file_path, 'r') as file:
        config_params = json.load(file)
    return config_params

def add_config_params_to_dict(params_dict, config_file_path, dataset_name):
    """
    Add default parameters from the config file to the dictionary.

    Args:
        params_dict (dict): Dictionary to add parameters to.
        config_file_path (str): Path to the config file.
        dataset_name (str): Name of the dataset.

    Returns:
        dict: Updated dictionary with added parameters.
    """
    config_params = read_config_params(config_file_path)
    hypothesis_model = config_params['TrackerConfig']['HypothesisModel']
    motion_model = config_params['TrackerConfig']['MotionModel']
    params_to_add = {
        'theta_dist': hypothesis_model['theta_dist'],
        'lambda_dist': hypothesis_model['lambda_dist'],
        'lambda_time': hypothesis_model['lambda_time'],
        'lambda_link': hypothesis_model['lambda_link'],
        'lambda_branch': hypothesis_model['lambda_branch'],
        'theta_time': hypothesis_model['theta_time'],
        'dist_thresh': hypothesis_model['dist_thresh'],
        'time_thresh': hypothesis_model['time_thresh'],
        'apop_thresh': hypothesis_model['apop_thresh'],
        'segmentation_miss_rate': hypothesis_model['segmentation_miss_rate'],
        'p_sigma': motion_model['P']['sigma'],
        'g_sigma': motion_model['G']['sigma'],
        'r_sigma': motion_model['R']['sigma'],
        'max_lost': motion_model['max_lost'],
        'prob_not_assign': motion_model['prob_not_assign'],
        'max_search_radius': 100,
        'div_hypothesis': 1
    }
    params_to_add = pd.DataFrame([params_to_add], index=[dataset_name])
    params_dict = pd.concat([params_dict, params_to_add])
    return params_dict


def scale_matrix(matrix: np.ndarray, original_sigma: float, new_sigma: float) -> np.ndarray:
    """
    Scales a matrix by first reverting the original scaling and then applying a new sigma value.

    Parameters:
        matrix (np.ndarray): The matrix to be scaled.
        original_sigma (float): The original sigma value used to scale the matrix.
        new_sigma (float): The new sigma value to scale the matrix.

    Returns:
        np.ndarray: The rescaled matrix.
    """
    if original_sigma != 0:
        unscaled_matrix = matrix / original_sigma
    else:
        unscaled_matrix = matrix.copy()  # Avoid division by zero
    rescaled_matrix = unscaled_matrix * new_sigma
    return rescaled_matrix

def add_missing_attributes(graph):
    """
    Add attributes to nodes without any attributes, essentially inserting dummy nodes where necessary.
    
    Args:
        graph (networkx.Graph): The graph to update.
    """
    # Count nodes with missing attributes
    nodes_without_attrs = [node for node in graph.nodes if not graph.nodes[node]]
    
    if not nodes_without_attrs:
        # All nodes have attributes, nothing to do
        return
    
    # If we reach here, some nodes are missing attributes
    # Try to infer based on node type
    for node in nodes_without_attrs:
        if isinstance(node, int):
            # For integer nodes, set minimal default attributes
            graph.nodes[node].update({
                'segmentation_id': node,
                'x': None,
                'y': None,
                't': 0
            })
        elif isinstance(node, str) and '_' in node:
            # For string nodes in "cell_timestep" format
            try:
                segmentation_id_str, time_str = node.split('_')
                segmentation_id = int(segmentation_id_str)
                time = int(time_str)
                prev_node = f"{segmentation_id}_{time - 1}"
                x, y = None, None
                if graph.has_node(prev_node) and 'x' in graph.nodes[prev_node] and 'y' in graph.nodes[prev_node]:
                    x = graph.nodes[prev_node]['x']
                    y = graph.nodes[prev_node]['y']
                attributes = {
                    'segmentation_id': segmentation_id,
                    'x': x,
                    'y': y,
                    't': time
                }
                graph.nodes[node].update(attributes)
            except (ValueError, AttributeError):
                # If splitting fails, set default attributes
                graph.nodes[node].update({
                    'segmentation_id': None,
                    'x': None,
                    'y': None,
                    't': 0
                })

def run_cell_tracking_algorithm(objects, config, volume, max_search_radius=100, use_napari=False):
    """
    Run the cell tracking algorithm.

    Args:
        objects: Objects to track.
        config: Tracking configuration.
        volume: Volume of the tracking space.
        max_search_radius (int, optional): Maximum search radius. Defaults to 100.
        use_napari (bool, optional): Whether to use napari for visualization. Defaults to False.

    Returns:
        tuple: Tracking results including LBEP and tracks, and napari data if use_napari is True.
    """
    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(config)
        tracker.max_search_radius = max_search_radius
        tracker.append(objects)
        tracker.volume = volume[::-1]
        tracker.track(step_size=100)
        tracker.optimize()
        tracks = tracker.tracks
        lbep = tracker.LBEP
        if use_napari:
            nap_data, nap_properties, nap_graph = tracker.to_napari()
            return lbep, tracks, nap_data, nap_properties, nap_graph
        else:
            return lbep, tracks

def calculate_accuracy(lbep, segm, gt_data):
    """
    Calculate the accuracy of the tracking algorithm.

    Args:
        lbep: LBEP data from tracking.
        segm: Segmentation data.
        gt_data: Ground truth data.

    Returns:
        dict: Dictionary containing accuracy metrics.
    """
    tracks_df = pd.DataFrame({
        "Cell_ID": lbep[:, 0],
        "Start": lbep[:, 1],
        "End": lbep[:, 2],
        "Parent_ID": [0 if lbep[idx, 3] == lbep[idx, 0] else lbep[idx, 3] for idx in range(lbep.shape[0])],
    })
    detections_df = _get_node_attributes(segm)
    G = ctc_to_graph(tracks_df, detections_df)
    add_missing_attributes(G)
    pred_data = TrackingGraph(G, segm)
    
    # Create matcher
    matcher = CTCMatcher()
    
    # Match the graphs
    matched = matcher.compute_mapping(gt_data, pred_data)
    
    # Compute metrics
    ctc_metrics = CTCMetrics()
    div_metrics = DivisionMetrics()
    
    # Compute CTC metrics - returns a Results object
    ctc_results = ctc_metrics.compute(matched)
    
    # Initialize results dictionary
    results = {}

    # Extract CTC metrics - Results objects have a 'results' attribute that's a dict
    if hasattr(ctc_results, 'results'):
        ctc_data = ctc_results.results
        keys_to_check = ['fp_nodes', 'fn_nodes', 'ns_nodes', 'fp_edges', 'fn_edges', 'ws_edges', 'TRA', 'DET', 'AOGM']
        for key in keys_to_check:
            if key in ctc_data:
                results[key] = ctc_data[key]

    # Compute Division metrics - returns a Results object
    # Try to compute division metrics, but skip if matcher is incompatible
    try:
        div_results = div_metrics.compute(matched)
        
        # Extract Division metrics - they're nested under 'Frame Buffer 0'
        if hasattr(div_results, 'results'):
            div_data = div_results.results
            if 'Frame Buffer 0' in div_data:
                frame_buffer_data = div_data['Frame Buffer 0']
                keys_to_extract = ['Mitotic Branching Correctness', 'False Positive Divisions', 'False Negative Divisions']
                for key in keys_to_extract:
                    if key in frame_buffer_data:
                        results[key] = frame_buffer_data[key]
    except TypeError:
        # Division metrics not compatible with this matcher (e.g., for 3D spatial tracking)
        # Continue without division metrics
        pass

    return results

def objective(trial, dataset, gt_data, objectives):
    """
    Objective function for Bayesian Optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        dataset: Dataset to be optimized.
        gt_data: Ground truth data.
        objectives (str): Objectives type, '1obj' or '2obj'.

    Returns:
        float or tuple: The objective value(s) for the trial.
    """
    try:
        gc.collect()

        param_ranges = {
            'theta_dist': (0, 99.99),
            'lambda_time': (0, 99.99),
            'lambda_dist': (0, 99.99),
            'lambda_link': (0, 99.99),
            'lambda_branch': (0, 99.99),
            'theta_time': (0, 99.99),
            'dist_thresh': (0, 99.99),
            'time_thresh': (0, 99.99),
            'apop_thresh': (0, 99, 'int'),
            'segmentation_miss_rate': (0, 1.0),
            'p_sigma': (0, 500),
            'g_sigma': (0, 500),
            'r_sigma': (0, 500),
            'accuracy': (0.1, 10),
            'max_lost': (1, 10, 'int'),
            'prob_not_assign': (0.0, 1.0),
            'max_search_radius': (0, 1000, 'int'),
            'div_hypothesis': (0, 1, 'int')
        }
        params = {}
        for param, (low, high, *type_) in param_ranges.items():
            if type_ and type_[0] == 'int':
                params[param] = trial.suggest_int(param, low, high)
            elif type_ and type_[0] == 'categorical':
                params[param] = trial.suggest_categorical(param, [low, high])
            else:
                params[param] = trial.suggest_float(param, low, high)

        objects = utils.segmentation_to_objects(dataset.segmentation, properties=('area',))
        conf = config.load_config('cell_config.json')
        volume = dataset.volume
        attributes = {
            'theta_dist': params['theta_dist'],
            'lambda_time': params['lambda_time'],
            'lambda_dist': params['lambda_dist'],
            'lambda_link': params['lambda_link'],
            'lambda_branch': params['lambda_branch'],
            'theta_time': params['theta_time'],
            'dist_thresh': params['dist_thresh'],
            'time_thresh': params['time_thresh'],
            'apop_thresh': params['apop_thresh'],
            'segmentation_miss_rate': params['segmentation_miss_rate'],
            'P': scale_matrix(conf.motion_model.P, 150.0, params['p_sigma']),
            'G': scale_matrix(conf.motion_model.G, 15.0, params['g_sigma']),
            'R': scale_matrix(conf.motion_model.R, 5.0, params['r_sigma']),
            'accuracy': params['accuracy'],
            'max_lost': params['max_lost'],
            'prob_not_assign': params['prob_not_assign']
        }

        for attr, value in attributes.items():
            if attr in ['P', 'G', 'R', 'max_lost', 'prob_not_assign', 'accuracy']:
                setattr(conf.motion_model, attr, value)
            else:
                setattr(conf.hypothesis_model, attr, value)

        if params['div_hypothesis'] == 1:
            setattr(conf.hypothesis_model, 'hypotheses', [
                    "P_FP",
                    "P_init",
                    "P_term",
                    "P_link",
                    "P_branch",
                    "P_dead"
                ])
        elif params['div_hypothesis'] == 0:
            setattr(conf.hypothesis_model, 'hypotheses', [
                    "P_FP",
                    "P_init",
                    "P_term",
                    "P_link",
                    "P_dead"
                ])
        else:
            raise ValueError(f"Invalid value for div_hypothesis: {params['div_hypothesis']}. It should be 0 or 1.")
        
        conf.enable_optimisation = True 

        lbep, tracks = run_cell_tracking_algorithm(objects, conf, volume, params['max_search_radius'])
        segm = utils.update_segmentation(np.asarray(dataset.segmentation), tracks)
        results = calculate_accuracy(lbep, segm, gt_data)  
        
        for attr, value in results.items():
            trial.set_user_attr(attr, value)

        if objectives == '1obj':
            return results['AOGM']
        elif objectives == '2obj':
            return results['AOGM'], results['Mitotic Branching Correctness']
    except Exception as e:  # Changed: capture the actual exception
        import traceback
        print(f"Exception occurred: {type(e).__name__}: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()  # This will print the full error trace
        
        # Return a penalty value instead of None
        if objectives == '1obj':
            return 100000.0  # Large penalty value
        elif objectives == '2obj':
            return 100000.0, 0.0
    
def optimize_dataset(dataset, gt_data, objectives, study_name, n_trials=64, use_parallel_backend=True, sampler='tpe', search_space=None):
    """
    Optimize a dataset without timeout for each trial.

    Args:
        dataset: Dataset to be optimized.
        gt_data: Ground truth data.
        objectives (str): Objectives type, '1obj' or '2obj'.
        study_name (str): Name of the Optuna study.
        n_trials (int, optional): Number of trials to run. Defaults to 64.
        use_parallel_backend (bool, optional): Whether to use parallel backend. Defaults to True.
        sampler (str, optional): Sampler to use for the optimization. Defaults to 'tpe'.
        search_space (dict, optional): Search space for grid sampling.

    Returns:
        optuna.study.Study: The Optuna study object.
    """
    if sampler == 'NSGA-II' and objectives == '1obj':
        raise ValueError("NSGA-II sampler is not supported for single-objective optimisation.")
    if sampler == 'cmaes' and objectives == '2obj':
        raise ValueError("CMA-ES sampler is not supported for multi-objective optimisation.")

    if sampler == 'random':
        sampler = optuna.samplers.RandomSampler()
        print("Random sampler selected.")
    elif sampler == 'grid':
        if search_space is None:
            raise ValueError("A search space must be provided for grid search.")
        sampler = optuna.samplers.GridSampler(search_space)
    elif sampler == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler()
        print("CMA-ES sampler selected.")
    elif sampler == 'tpe':
        sampler = optuna.samplers.TPESampler()
        print("TPE sampler selected.")
    elif sampler == 'NSGA-II':
        sampler = optuna.samplers.NSGAIISampler()
        print("NSGA-II sampler selected.")
    else:
        raise ValueError("unknown sampler selected")

    if objectives == '1obj':
        study = optuna.create_study(directions=["minimize"], study_name=study_name, storage="sqlite:///btrack.db", load_if_exists=True, sampler=sampler)
    if objectives == '2obj':
        study = optuna.create_study(directions=["minimize", "maximize"], study_name=study_name, storage="sqlite:///btrack.db", load_if_exists=True, sampler=sampler)

    if use_parallel_backend:
        with parallel_backend('multiprocessing'):
            study.optimize(lambda trial: objective(trial, dataset, gt_data, objectives), timeout=1800, n_trials=n_trials, n_jobs=4, gc_after_trial=True)
    else:
        study.optimize(lambda trial: objective(trial, dataset, gt_data, objectives), timeout=1800, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
    return study

def compute_scaling_factors(voxel_sizes):
    """
    Compute the scaling factors to make the TIFF image isotropic.

    Args:
        voxel_sizes (tuple): A tuple of three voxel sizes in micrometres (vx, vy, vz).

    Returns:
        tuple: A tuple of three scaling factors (sx, sy, sz) to make the image isotropic.
    """
    vx, vy, vz = voxel_sizes
    avg_voxel_size = (vx + vy + vz) / 3.0
    sx = avg_voxel_size / vx
    sy = avg_voxel_size / vy
    sz = avg_voxel_size / vz
    return sx, sy, sz
