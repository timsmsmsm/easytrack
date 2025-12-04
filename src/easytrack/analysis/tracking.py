"""
Tracking engine and process management for btrack.

Handles:
- Running btrack tracking in a separate process
- Monitoring progress and managing cancellation
- File-based communication to avoid Queue size limits
- Exporting tracks to napari
"""

import os
import sys
import json
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from multiprocessing import Process, Queue, Value
import shutil

import numpy as np
from qtpy.QtCore import QThread, Signal
import btrack
from btrack import utils, config

from presets import create_btrack_config_dict


# ============= TRACKING PROCESS =============

def run_tracking_process(
    input_file: str,
    output_file: str,
    params: Dict[str, Any],
    progress_queue: Queue,
    status_flag: Value
):
    """
    Run tracking in a separate process using file-based I/O.
    
    Args:
        input_file: Path to input segmentation .npy file
        output_file: Path to save output tracked segmentation
        params: Tracking parameters
        progress_queue: Queue to send progress updates
        status_flag: Shared value to indicate completion (0=running, 1=success, -1=error)
    """
    try:
        print(f"[CHILD {os.getpid()}] Process started")
        progress_queue.put("Loading segmentation from file...")
        
        # Load segmentation from file
        segmentation = np.load(input_file)
        print(f"[CHILD] Loaded segmentation: shape={segmentation.shape}, dtype={segmentation.dtype}")
        
        # Ensure correct data type
        segmentation = np.ascontiguousarray(
            segmentation.astype(segmentation.dtype.newbyteorder('='))
        )
        
        # Handle both 2D+T and 3D+T data
        if segmentation.ndim == 3:
            # 2D+T: (T, Y, X)
            T, Y, X = segmentation.shape
            print(f"[CHILD] Detected 2D+T data: T={T}, Y={Y}, X={X}")
        elif segmentation.ndim == 4:
            # 3D+T: (T, Z, Y, X)
            T, Z, Y, X = segmentation.shape
            print(f"[CHILD] Detected 3D+T data: T={T}, Z={Z}, Y={Y}, X={X}")
        else:
            raise ValueError(f"Unsupported segmentation shape: {segmentation.shape}")
        
        progress_queue.put(f"Extracting objects from {T} frames...")
        print(f"[CHILD] Extracting objects...")
        
        # Extract objects
        objects = utils.segmentation_to_objects(
            segmentation,
            properties=('area',),
            num_workers=1
        )
        
        print(f"[CHILD] Found {len(objects)} objects")
        progress_queue.put(f"Found {len(objects)} objects. Starting tracking...")
        
        # Create temporary config file with parameters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = create_btrack_config_dict(params)
            json.dump(config_data, f, indent=2)
            config_path = f.name
        
        print(f"[CHILD] Loading config from {config_path}")
        # Load config
        conf = config.load_config(config_path)
        
        # Enable optimization
        conf.enable_optimisation = True
        
        # Define volume based on dimensionality
        # Note: btrack's ImagingVolume is for SPATIAL dimensions only (not time)
        # For 2D: (X, Y) or for 3D: (X, Y, Z)
        # Time is handled separately by the tracker
        
        if segmentation.ndim == 3:
            # 2D+T: spatial volume is (X, Y) in btrack's reversed order
            volume = ((0, X), (0, Y))
            print(f"[CHILD] Spatial volume (2D): X=[0,{X}], Y=[0,{Y}]")
        else:
            # 3D+T: spatial volume is (X, Y, Z) in btrack's reversed order
            volume = ((0, X), (0, Y), (0, Z))
            print(f"[CHILD] Spatial volume (3D): X=[0,{X}], Y=[0,{Y}], Z=[0,{Z}]")
        
        progress_queue.put("Running btrack tracking and optimization...")
        print(f"[CHILD] Starting btrack...")
        
        # Track
        with btrack.BayesianTracker(verbose=True) as tracker:
            tracker.configure(conf)
            tracker.volume = volume  # Set spatial volume (not including time)
            tracker.max_search_radius = params['max_search_radius']
            tracker.append(objects)
            
            print(f"[CHILD] Running tracker.track()...")
            tracker.track(step_size=100)
            
            print(f"[CHILD] Running tracker.optimize()...")
            tracker.optimize()
            
            print(f"[CHILD] Updating segmentation with {len(tracker.tracks)} tracks...")
            # Get results
            tracked_seg = utils.update_segmentation(segmentation, tracker.tracks)
            
            # Get tracks in napari format
            print(f"[CHILD] Exporting tracks to napari...")
            napari_data, napari_properties, napari_graph = tracker.to_napari()
            
            # Convert graph from array to dict if necessary
            # Napari expects graph as a dict: {node_id: [parent_ids]}
            if isinstance(napari_graph, np.ndarray):
                if napari_graph.size == 0:
                    print(f"[CHILD] Graph is empty array, converting to empty dict")
                    napari_graph = {}
                else:
                    # btrack might return graph as array - need to convert
                    # Typically graph is (N, 2) array of [child, parent] pairs
                    print(f"[CHILD] Converting graph array to dict for napari")
                    graph_dict = {}
                    if napari_graph.ndim == 2 and napari_graph.shape[1] == 2:
                        for child, parent in napari_graph:
                            child_id = int(child)
                            parent_id = int(parent)
                            if child_id not in graph_dict:
                                graph_dict[child_id] = []
                            graph_dict[child_id].append(parent_id)
                    napari_graph = graph_dict
                    print(f"[CHILD] Converted graph dict has {len(napari_graph)} nodes")
            
            # Fix dimensionality mismatch between btrack output and input data
            # btrack always returns [track_id, t, z, y, x] (5 cols)
            # but for 2D+T data (shape: T, Y, X), we need [track_id, t, y, x] (4 cols)
            # and for 3D+T data (shape: T, Z, Y, X), we keep all 5 cols
            
            if napari_data.shape[1] == 5 and segmentation.ndim == 3:
                # Input is 2D+T (T, Y, X), so remove the Z column
                z_column = napari_data[:, 2]
                z_range = z_column.max() - z_column.min()
                print(f"[CHILD] Input is 2D+T (shape: {segmentation.shape})")
                print(f"[CHILD] Z column range: {z_range} (should be ~0 for 2D tracking)")
                print(f"[CHILD] Removing Z column to match 2D+T format...")
                
                # Remove Z column: [track_id, t, z, y, x] -> [track_id, t, y, x]
                napari_data = np.column_stack([
                    napari_data[:, 0],  # track_id
                    napari_data[:, 1],  # t
                    napari_data[:, 3],  # y (skip z at index 2)
                    napari_data[:, 4]   # x
                ])
                print(f"[CHILD] After removing Z: shape={napari_data.shape}")
                
            elif napari_data.shape[1] == 5 and segmentation.ndim == 4:
                # Input is 3D+T (T, Z, Y, X), keep all columns
                print(f"[CHILD] Input is 3D+T (shape: {segmentation.shape})")
                print(f"[CHILD] Keeping all 5 columns [track_id, t, z, y, x]")
                # napari_data stays as is
            
            else:
                print(f"[CHILD] Unexpected data format: napari shape={napari_data.shape}, seg shape={segmentation.shape}")
            
            track_info = {
                'total_tracks': len(tracker.tracks),
                'tracks_gt_1': sum(1 for t in tracker.tracks if len(t) > 1),
                'tracks_gt_5': sum(1 for t in tracker.tracks if len(t) > 5),
                'tracks_gt_10': sum(1 for t in tracker.tracks if len(t) > 10),
            }
        
        # Clean up temp config file
        Path(config_path).unlink()
        
        print(f"[CHILD] Saving results to {output_file}...")
        progress_queue.put("Saving results...")
        
        # Save results to files
        np.save(output_file, tracked_seg)
        
        # Save napari tracks data
        napari_output_file = output_file.replace('.npy', '_napari.npz')
        
        # Save properties dict more carefully
        # Properties is a dict where each key maps to an array
        properties_dict = {}
        for key, value in napari_properties.items():
            properties_dict[f'prop_{key}'] = value
        
        # Save with allow_pickle for graph (could be dict)
        np.savez(
            napari_output_file,
            data=napari_data,
            properties_keys=list(napari_properties.keys()),
            graph=napari_graph,  # This could be dict or array
            **properties_dict
        )
        
        # Save track info
        info_file = output_file.replace('.npy', '_info.json')
        with open(info_file, 'w') as f:
            json.dump(track_info, f)
        
        print(f"[CHILD] Results saved successfully")
        progress_queue.put("Tracking complete!")
        
        # Signal success
        status_flag.value = 1
        print(f"[CHILD] Process completed successfully")
        
    except Exception as e:
        error_msg = f"Error during tracking: {str(e)}\n{traceback.format_exc()}"
        print(f"[CHILD] ERROR: {error_msg}")
        progress_queue.put(f"ERROR: {str(e)}")
        status_flag.value = -1


# ============= TRACKING MONITOR THREAD =============

class TrackingMonitor(QThread):
    """
    Monitor a tracking process and relay progress/results.
    
    Runs in a Qt thread to avoid blocking the UI while monitoring
    a separate process that's doing the actual tracking work.
    """
    
    finished = Signal(object, object, object, object, object)  # tracked_seg, track_info, napari_data, napari_properties, napari_graph
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
            print(f"[MONITOR] Process terminated")
        
    def run(self):
        """Monitor the process and relay messages."""
        try:
            print(f"[MONITOR] Started monitoring process PID: {self.process.pid}")
            
            while True:
                # Check for progress messages
                while not self.progress_queue.empty():
                    msg = self.progress_queue.get_nowait()
                    print(f"[MONITOR] Progress: {msg}")
                    self.progress.emit(msg)
                
                # Check if process is done
                if not self.process.is_alive():
                    print(f"[MONITOR] Process ended with exit code: {self.process.exitcode}")
                    
                    # Get final progress messages
                    while not self.progress_queue.empty():
                        msg = self.progress_queue.get_nowait()
                        print(f"[MONITOR] Final progress: {msg}")
                        self.progress.emit(msg)
                    
                    # Check status
                    if self.status_flag.value == 1:
                        print(f"[MONITOR] Success! Loading results from {self.output_file}")
                        # Load results from file
                        tracked_seg = np.load(self.output_file)
                        info_file = self.output_file.replace('.npy', '_info.json')
                        with open(info_file, 'r') as f:
                            track_info = json.load(f)
                        
                        # Load napari tracks data
                        napari_file = self.output_file.replace('.npy', '_napari.npz')
                        napari_npz = np.load(napari_file, allow_pickle=True)
                        napari_data = napari_npz['data']
                        
                        # Reconstruct properties dict
                        properties_keys = napari_npz['properties_keys']
                        if len(properties_keys) == 0:
                            # Empty properties dict
                            napari_properties = {}
                        else:
                            napari_properties = {
                                key: napari_npz[f'prop_{key}'] 
                                for key in properties_keys
                            }
                        
                        napari_graph = napari_npz['graph']
                        
                        print(f"[MONITOR] Loaded results: {track_info}")
                        print(f"[MONITOR] Napari tracks shape: {napari_data.shape}")
                        print(f"[MONITOR] Napari properties keys: {list(napari_properties.keys())}")
                        self.finished.emit(tracked_seg, track_info, napari_data, napari_properties, napari_graph)
                    elif self.status_flag.value == -1:
                        print(f"[MONITOR] Process reported error")
                        self.error.emit("Tracking process reported an error. Check console for details.")
                    elif not self._is_cancelled:
                        print(f"[MONITOR] Process ended without setting status flag")
                        self.error.emit("Process ended unexpectedly without result")
                    break
                
                # Check for cancellation
                if self._is_cancelled:
                    print(f"[MONITOR] Cancelled by user")
                    break
                
                # Small sleep to avoid busy waiting
                self.msleep(100)
                
        except Exception as e:
            if not self._is_cancelled:
                error_msg = f"Monitor error: {str(e)}\n{traceback.format_exc()}"
                print(f"[MONITOR] ERROR: {error_msg}")
                self.error.emit(error_msg)
        finally:
            # Clean up temp files
            print(f"[MONITOR] Cleaning up temp directory: {self.temp_dir}")
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass


# ============= HIGH-LEVEL TRACKING INTERFACE =============

class TrackingManager:
    """
    High-level interface for starting and managing tracking operations.
    
    Handles the complexity of multiprocessing, file I/O, and progress monitoring.
    """
    
    def __init__(self):
        self.current_process: Optional[Process] = None
        self.current_monitor: Optional[TrackingMonitor] = None
        
    def start_tracking(
        self,
        segmentation: np.ndarray,
        params: Dict[str, Any],
        on_progress=None,
        on_finished=None,
        on_error=None
    ) -> TrackingMonitor:
        """
        Start a tracking operation.
        
        Args:
            segmentation: 3D numpy array (T, Y, X)
            params: Tracking parameters
            on_progress: Callback for progress updates (str)
            on_finished: Callback for completion (tracked_seg, track_info, napari_data, napari_properties, napari_graph)
            on_error: Callback for errors (error_msg)
            
        Returns:
            TrackingMonitor instance that can be cancelled
        """
        # Create temp directory for I/O
        temp_dir = tempfile.mkdtemp(prefix='btrack_')
        input_file = Path(temp_dir) / 'input_seg.npy'
        output_file = Path(temp_dir) / 'output_seg.npy'
        
        print(f"[MAIN] Saving segmentation to {input_file}")
        # Save input segmentation
        np.save(input_file, segmentation)
        
        # Create shared status flag
        status_flag = Value('i', 0)  # 0=running, 1=success, -1=error
        
        # Create queue for progress
        progress_queue = Queue()
        
        print(f"[MAIN] Starting tracking process...")
        # Create and start tracking process
        self.current_process = Process(
            target=run_tracking_process,
            args=(str(input_file), str(output_file), params, progress_queue, status_flag)
        )
        self.current_process.start()
        
        # Create and start monitor thread
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
        """Cancel the current tracking operation if any."""
        if self.current_monitor:
            self.current_monitor.cancel()