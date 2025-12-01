"""
Tracking engine and process management for btrack.

Handles:
- Running btrack tracking in a separate process
- Monitoring progress and managing cancellation
- File-based communication to avoid Queue size limits
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
        
        T, Y, X = segmentation.shape
        
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
        
        # Define volume
        volume = ((0, T), (0, X), (0, Y))
        
        progress_queue.put("Running btrack tracking and optimization...")
        print(f"[CHILD] Starting btrack...")
        
        # Track
        with btrack.BayesianTracker(verbose=True) as tracker:
            tracker.configure(conf)
            tracker.volume = volume[::-1]  # Reverse for btrack
            tracker.max_search_radius = params['max_search_radius']
            tracker.append(objects)
            
            print(f"[CHILD] Running tracker.track()...")
            tracker.track(step_size=100)
            
            print(f"[CHILD] Running tracker.optimize()...")
            tracker.optimize()
            
            print(f"[CHILD] Updating segmentation with {len(tracker.tracks)} tracks...")
            # Get results
            tracked_seg = utils.update_segmentation(segmentation, tracker.tracks)
            
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
    
    finished = Signal(object, object)  # tracked_segmentation, track_info
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
                        print(f"[MONITOR] Loaded results: {track_info}")
                        self.finished.emit(tracked_seg, track_info)
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
            on_finished: Callback for completion (tracked_seg, track_info)
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
