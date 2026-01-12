"""
Unified tracking module for both optimization and preset widgets.

Provides:
- Proper config loading (base config + parameter modification)
- Tracking execution matching optimization approach
- Full napari output (tracks layer support)
- Process-based execution for cancellability
"""

import multiprocessing
import sys

# Fix for macOS CoreFoundation fork issue
if sys.platform == 'darwin':  # Only on macOS
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

import os
import json
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable
from multiprocessing import Process, Queue, Value
import shutil

import numpy as np
from qtpy.QtCore import QThread, Signal
import btrack
from btrack import utils, config


# ============= CONFIG PATH RESOLUTION =============

def get_default_config_path() -> str:
    """
    Get the absolute path to the default cell_config.json file.
    Works correctly whether running from source or installed package.
    
    Returns:
        Absolute path to cell_config.json
    """
    try:
        # Try importlib.resources (Python 3.9+)
        from importlib import resources
        
        if hasattr(resources, 'files'):
            # Python 3.9+
            config_file = resources.files('napari_easytrack').joinpath('configs/cell_config.json')
            
            # Need to handle Traversable -> Path conversion
            if hasattr(config_file, '__fspath__'):
                # Python 3.12+ returns a Path-like directly
                return str(config_file)
            else:
                # Python 3.9-3.11 needs as_file context manager
                from importlib.resources import as_file
                with as_file(config_file) as path:
                    # For spawned processes, we need a persistent path
                    # Copy to temp location that persists
                    temp_config = Path(tempfile.gettempdir()) / 'napari_easytrack_cell_config.json'
                    if not temp_config.exists() or temp_config.stat().st_size == 0:
                        shutil.copy2(path, temp_config)
                    return str(temp_config)
        else:
            # Older Python - use pkg_resources
            import pkg_resources
            return pkg_resources.resource_filename('napari_easytrack', 'configs/cell_config.json')
            
    except Exception as e:
        # Fallback: try relative to this file (development mode)
        fallback_path = Path(__file__).parent.parent / 'configs' / 'cell_config.json'
        if fallback_path.exists():
            print(f"[CONFIG] Using development path: {fallback_path}")
            return str(fallback_path.resolve())
        
        raise FileNotFoundError(
            f"Could not locate cell_config.json. "
            f"Tried package resources and {fallback_path}. "
            f"Original error: {e}"
        )


# ============= MATRIX SCALING =============

def scale_matrix(matrix: np.ndarray, original_sigma: float, new_sigma: float) -> np.ndarray:
    """
    Scales a matrix by reverting original scaling and applying new sigma.
    
    CRITICAL: Must match the scale_matrix used in optimization.
    """
    if original_sigma != 0:
        unscaled_matrix = matrix / original_sigma
    else:
        unscaled_matrix = matrix.copy()
    rescaled_matrix = unscaled_matrix * new_sigma
    return rescaled_matrix


# ============= CORE TRACKING FUNCTION =============

def run_tracking_core(
    segmentation: np.ndarray,
    params: Dict[str, Any],
    base_config_path: Optional[str] = None
) -> Tuple[np.ndarray, list, Dict, np.ndarray, Dict, Any]:
    """
    Core tracking function using the optimization approach.
    
    This loads base config and modifies it, matching what happens during optimization.
    
    Args:
        segmentation: 3D or 4D array (T,Y,X) or (T,Z,Y,X)
        params: Parameter dictionary
        base_config_path: Path to config file. If None, uses package default.
        
    Returns:
        Tuple of (tracked_seg, tracks, track_info, napari_data, napari_properties, napari_graph)
    """
    print(f"[TRACKING] Starting with segmentation shape: {segmentation.shape}")
    
    # Use default config if none provided
    if base_config_path is None:
        base_config_path = get_default_config_path()
    
    # Ensure absolute path
    config_path = Path(base_config_path)
    if not config_path.is_absolute():
        config_path = config_path.resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    base_config_path = str(config_path)
    
    # Ensure correct data type
    segmentation = np.ascontiguousarray(
        segmentation.astype(segmentation.dtype.newbyteorder('='))
    )
    
    # Determine dimensionality
    if segmentation.ndim == 3:
        T, Y, X = segmentation.shape
        is_2d = True
        print(f"[TRACKING] 2D+T data: T={T}, Y={Y}, X={X}")
    elif segmentation.ndim == 4:
        T, Z, Y, X = segmentation.shape
        is_2d = False
        print(f"[TRACKING] 3D+T data: T={T}, Z={Z}, Y={Y}, X={X}")
    else:
        raise ValueError(f"Unsupported shape: {segmentation.shape}")
    
    # CRITICAL: Load base config and modify (matches optimization)
    print(f"[TRACKING] Loading base config: {base_config_path}")
    conf = config.load_config(base_config_path)
    
    # Extract objects
    print(f"[TRACKING] Extracting objects...")
    objects = utils.segmentation_to_objects(
        segmentation,
        properties=('area',)
    )
    print(f"[TRACKING] Found {len(objects)} objects")
    
    # Calculate volume (spatial dimensions only, not time)
    if is_2d:
        volume = ((0, X), (0, Y))
    else:
        volume = ((0, X), (0, Y), (0, Z))
    
    print(f"[TRACKING] Volume: {volume}")
    
    # Apply parameters to config (matches optimization exactly)
    print(f"[TRACKING] Applying optimized parameters...")
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
    
    # Handle division hypothesis
    if params.get('div_hypothesis', 1) == 1:
        setattr(conf.hypothesis_model, 'hypotheses', [
            "P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"
        ])
    else:
        setattr(conf.hypothesis_model, 'hypotheses', [
            "P_FP", "P_init", "P_term", "P_link", "P_dead"
        ])
    
    # Enable optimization
    conf.enable_optimisation = True
    
    max_search_radius = params.get('max_search_radius', 100)
    print(f"[TRACKING] max_search_radius: {max_search_radius}")
    
    # Run tracking
    print(f"[TRACKING] Running btrack...")
    with btrack.BayesianTracker(verbose=True) as tracker:
        tracker.configure(conf)
        tracker.volume = volume
        tracker.max_search_radius = max_search_radius
        tracker.append(objects)
        
        tracker.track(step_size=100)
        tracker.optimize()
        
        print(f"[TRACKING] Found {len(tracker.tracks)} tracks")
        
        # Update segmentation
        tracked_seg = utils.update_segmentation(segmentation, tracker.tracks)
        
        # Get napari format
        napari_data, napari_properties, napari_graph = tracker.to_napari()
        
        # Convert graph to dict format for napari
        if isinstance(napari_graph, np.ndarray):
            if napari_graph.size == 0:
                napari_graph = {}
            else:
                graph_dict = {}
                if napari_graph.ndim == 2 and napari_graph.shape[1] == 2:
                    for child, parent in napari_graph:
                        child_id = int(child)
                        parent_id = int(parent)
                        if child_id not in graph_dict:
                            graph_dict[child_id] = []
                        graph_dict[child_id].append(parent_id)
                napari_graph = graph_dict
        
        # Fix dimensionality for 2D+T data
        # btrack returns [track_id, t, z, y, x] (5 cols) always
        # For 2D+T input (T,Y,X), we need [track_id, t, y, x] (4 cols)
        if napari_data.shape[1] == 5 and is_2d:
            print(f"[TRACKING] Removing Z column for 2D+T data")
            napari_data = np.column_stack([
                napari_data[:, 0],  # track_id
                napari_data[:, 1],  # t
                napari_data[:, 3],  # y
                napari_data[:, 4]   # x
            ])
        
        # Calculate track statistics
        track_info = {
            'total_tracks': len(tracker.tracks),
            'tracks_gt_1': sum(1 for t in tracker.tracks if len(t) > 1),
            'tracks_gt_5': sum(1 for t in tracker.tracks if len(t) > 5),
            'tracks_gt_10': sum(1 for t in tracker.tracks if len(t) > 10),
        }
        
        tracks = tracker.tracks
    
    print(f"[TRACKING] Complete!")
    return tracked_seg, tracks, track_info, napari_data, napari_properties, napari_graph


# ============= SIMPLE SYNCHRONOUS API (for optimization widget) =============

def run_tracking_with_params(
    segmentation: np.ndarray,
    params: Dict[str, Any],
    voxel_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    base_config_path: Optional[str] = None,
    return_napari: bool = False
) -> Tuple:
    """
    Simple synchronous tracking for optimization widget.
    
    Args:
        segmentation: 3D or 4D array
        params: Parameter dictionary
        voxel_scale: Voxel scaling (currently unused but kept for compatibility)
        base_config_path: Path to config file. If None, uses package default.
        return_napari: If True, returns napari tracks data as well
        
    Returns:
        If return_napari=False: (tracked_seg, tracks, track_info)
        If return_napari=True: (tracked_seg, tracks, track_info, napari_data, napari_properties, napari_graph)
    """
    print("\n" + "=" * 60)
    print("RUNNING TRACKING WITH OPTIMIZED PARAMETERS")
    print("=" * 60)
    
    # Use default if not provided
    if base_config_path is None:
        base_config_path = get_default_config_path()
    
    tracked_seg, tracks, track_info, napari_data, napari_properties, napari_graph = run_tracking_core(
        segmentation, params, base_config_path
    )
    
    print("\nTracking Statistics:")
    print(f"  Total tracks: {track_info['total_tracks']}")
    print(f"  Tracks > 1 frame: {track_info['tracks_gt_1']}")
    print(f"  Tracks > 5 frames: {track_info['tracks_gt_5']}")
    print(f"  Tracks > 10 frames: {track_info['tracks_gt_10']}")
    print("=" * 60 + "\n")
    
    if return_napari:
        return tracked_seg, tracks, track_info, napari_data, napari_properties, napari_graph
    else:
        return tracked_seg, tracks, track_info


# ============= ASYNC PROCESS-BASED API (for preset widget) =============

def run_tracking_process(
    input_file: str,
    output_file: str,
    params: Dict[str, Any],
    progress_queue: Queue,
    status_flag: Value,
    base_config_path: Optional[str] = None
):
    """
    Run tracking in separate process with file-based I/O.
    """
    try:
        print(f"[CHILD {os.getpid()}] Process started")
        
        # Resolve config path in child process
        if base_config_path is None:
            base_config_path = get_default_config_path()
        
        progress_queue.put("Loading segmentation...")
        
        # Load segmentation
        segmentation = np.load(input_file)
        print(f"[CHILD] Loaded: {segmentation.shape}")
        
        progress_queue.put(f"Extracting objects from {segmentation.shape[0]} frames...")
        
        # Run tracking
        tracked_seg, tracks, track_info, napari_data, napari_properties, napari_graph = \
            run_tracking_core(segmentation, params, base_config_path)
        
        progress_queue.put("Saving results...")
        
        # Save results
        np.save(output_file, tracked_seg)
        
        # Save napari data
        napari_output_file = output_file.replace('.npy', '_napari.npz')
        properties_dict = {f'prop_{k}': v for k, v in napari_properties.items()}
        
        np.savez(
            napari_output_file,
            data=napari_data,
            properties_keys=list(napari_properties.keys()),
            graph=napari_graph,
            **properties_dict
        )
        
        # Save track info
        info_file = output_file.replace('.npy', '_info.json')
        with open(info_file, 'w') as f:
            json.dump(track_info, f)
        
        progress_queue.put("Tracking complete!")
        status_flag.value = 1
        print(f"[CHILD] Success")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[CHILD] ERROR: {error_msg}")
        progress_queue.put(f"ERROR: {str(e)}")
        status_flag.value = -1


class TrackingMonitor(QThread):
    """Monitor tracking process and relay results."""
    
    finished = Signal(object, object, object, object, object)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(
        self,
        process: Process,
        progress_queue: Queue,
        status_flag: Value,
        output_file: str,
        temp_dir: str
    ):
        super().__init__()
        self.process = process
        self.progress_queue = progress_queue
        self.status_flag = status_flag
        self.output_file = output_file
        self.temp_dir = temp_dir
        self._is_cancelled = False
    
    def cancel(self):
        """Cancel the tracking process."""
        print(f"[MONITOR] Cancel requested")
        self._is_cancelled = True
        if self.process.is_alive():
            print(f"[MONITOR] Terminating process...")
            self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                print(f"[MONITOR] Killing process...")
                self.process.kill()
                self.process.join()
    
    def run(self):
        """Monitor the process."""
        try:
            print(f"[MONITOR] Monitoring PID: {self.process.pid}")
            
            while True:
                # Relay progress messages
                while not self.progress_queue.empty():
                    msg = self.progress_queue.get_nowait()
                    print(f"[MONITOR] Progress: {msg}")
                    self.progress.emit(msg)
                
                # Check if done
                if not self.process.is_alive():
                    print(f"[MONITOR] Process ended")
                    
                    # Get final messages
                    while not self.progress_queue.empty():
                        msg = self.progress_queue.get_nowait()
                        self.progress.emit(msg)
                    
                    if self.status_flag.value == 1:
                        # Success - load results
                        tracked_seg = np.load(self.output_file)
                        
                        info_file = self.output_file.replace('.npy', '_info.json')
                        with open(info_file, 'r') as f:
                            track_info = json.load(f)
                        
                        napari_file = self.output_file.replace('.npy', '_napari.npz')
                        napari_npz = np.load(napari_file, allow_pickle=True)
                        napari_data = napari_npz['data']
                        
                        properties_keys = napari_npz['properties_keys']
                        napari_properties = {
                            key: napari_npz[f'prop_{key}'] 
                            for key in properties_keys
                        } if len(properties_keys) > 0 else {}
                        
                        napari_graph = napari_npz['graph']
                        
                        self.finished.emit(
                            tracked_seg, track_info, 
                            napari_data, napari_properties, napari_graph
                        )
                    elif self.status_flag.value == -1:
                        self.error.emit("Tracking failed. Check console.")
                    elif not self._is_cancelled:
                        self.error.emit("Process ended unexpectedly")
                    break
                
                if self._is_cancelled:
                    break
                
                self.msleep(100)
                
        except Exception as e:
            if not self._is_cancelled:
                error_msg = f"Monitor error: {str(e)}\n{traceback.format_exc()}"
                print(f"[MONITOR] ERROR: {error_msg}")
                self.error.emit(error_msg)
        finally:
            # Clean up
            print(f"[MONITOR] Cleaning up: {self.temp_dir}")
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass


class TrackingManager:
    """High-level interface for starting and managing tracking."""
    
    def __init__(self):
        self.current_process: Optional[Process] = None
        self.current_monitor: Optional[TrackingMonitor] = None
    
    def start_tracking(
        self,
        segmentation: np.ndarray,
        params: Dict[str, Any],
        on_progress: Optional[Callable] = None,
        on_finished: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        base_config_path: Optional[str] = None
    ) -> TrackingMonitor:
        """
        Start tracking in background process.
        
        Args:
            segmentation: 3D or 4D array
            params: Parameter dictionary
            on_progress: Callback for progress (str)
            on_finished: Callback for completion (tracked_seg, track_info, napari_data, napari_properties, napari_graph)
            on_error: Callback for errors (error_msg)
            base_config_path: Path to config file. If None, uses package default.
            
        Returns:
            TrackingMonitor that can be cancelled
        """
        # Resolve config path BEFORE spawning process
        if base_config_path is None:
            base_config_path = get_default_config_path()
        else:
            # Ensure it's absolute
            config_path = Path(base_config_path)
            if not config_path.is_absolute():
                config_path = config_path.resolve()
            base_config_path = str(config_path)
        
        # Verify it exists
        if not Path(base_config_path).exists():
            raise FileNotFoundError(f"Config file not found: {base_config_path}")
        
        print(f"[MAIN] Using config: {base_config_path}")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix='btrack_')
        input_file = Path(temp_dir) / 'input_seg.npy'
        output_file = Path(temp_dir) / 'output_seg.npy'
        
        print(f"[MAIN] Saving to {input_file}")
        np.save(input_file, segmentation)
        
        # Shared state
        status_flag = Value('i', 0)
        progress_queue = Queue()
        
        # Start process
        print(f"[MAIN] Starting tracking process...")
        self.current_process = Process(
            target=run_tracking_process,
            args=(str(input_file), str(output_file), params, 
                  progress_queue, status_flag, base_config_path)
        )
        self.current_process.start()
        
        # Start monitor
        self.current_monitor = TrackingMonitor(
            self.current_process,
            progress_queue,
            status_flag,
            str(output_file),
            temp_dir
        )
        
        # Connect callbacks
        if on_progress:
            self.current_monitor.progress.connect(on_progress)
        if on_finished:
            self.current_monitor.finished.connect(on_finished)
        if on_error:
            self.current_monitor.error.connect(on_error)
        
        self.current_monitor.start()
        
        return self.current_monitor
    
    def cancel_current(self):
        """Cancel current tracking operation."""
        if self.current_monitor:
            self.current_monitor.cancel()


# ============= UTILITY FUNCTIONS =============

def format_params_summary(params: Dict) -> str:
    """Format parameters into readable summary."""
    key_params = [
        'max_search_radius', 'dist_thresh', 'theta_dist',
        'lambda_link', 'segmentation_miss_rate', 'prob_not_assign'
    ]
    
    lines = []
    for param in key_params:
        if param in params:
            value = params[param]
            if isinstance(value, float):
                lines.append(f"{param}: {value:.2f}")
            else:
                lines.append(f"{param}: {value}")
    
    return ", ".join(lines)


def validate_params(params: Dict) -> Tuple[bool, str]:
    """Validate parameters dictionary."""
    required_params = [
        'max_search_radius', 'dist_thresh', 'theta_dist',
        'lambda_link', 'segmentation_miss_rate', 'prob_not_assign',
        'theta_time', 'lambda_time', 'lambda_dist', 'lambda_branch',
        'time_thresh', 'apop_thresh', 'p_sigma', 'g_sigma', 'r_sigma',
        'accuracy', 'max_lost', 'div_hypothesis'
    ]
    
    missing = [p for p in required_params if p not in params]
    
    if missing:
        return False, f"Missing required parameters: {', '.join(missing)}"
    
    return True, ""