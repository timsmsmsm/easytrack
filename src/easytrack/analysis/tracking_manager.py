# ============= HIGH-LEVEL TRACKING INTERFACE =============
#
# TODO: Possible refactors to improve code quality:
#
# 1. Consider extracting `run_tracking_process` into a dedicated TrackingProcess class.
#    This would encapsulate the tracking logic and make it easier to test and extend.
#    The class could have methods like: load_segmentation(), extract_objects(),
#    configure_tracker(), run_tracker(), and save_results().
#
# 2. The dimensionality handling (2D+T vs 3D+T) appears in multiple places.
#    Consider creating a DimensionalityHandler class or utility functions to
#    centralize this logic and reduce code duplication.
#
# 3. The napari data post-processing (fixing column order, handling graph format)
#    could be extracted into a separate NapariExporter class or module.
#
# 4. Consider using a dataclass for TrackingParams to replace the Dict[str, Any]
#    parameter passing. This would provide type safety and better documentation.
#
# 5. The file I/O operations (save_tracking_results, loading in TrackingMonitor)
#    could be consolidated into a TrackingResultsIO class for consistency.
#
import json
import os
import tempfile
import traceback
from multiprocessing import Process, Queue, Value
from pathlib import Path
from typing import Dict, Any, Optional

import btrack
import numpy as np
from btrack import utils, config
from qtpy.QtCore import QThread, Signal

from easytrack import logger
from easytrack.analysis.tracking_monitor import TrackingMonitor
from easytrack.presets import create_btrack_config_dict


def save_tracking_results(napari_data: np.ndarray[tuple[Any, ...], np.dtype[np._ScalarT]] | np.ndarray[tuple[Any, ...], np.dtype[Any]],
                          napari_graph: dict | dict[Any, Any],
                          napari_properties: dict,
                          output_file: str,
                          track_info: dict[str, int],
                          tracked_seg: np.ndarray[tuple[Any, ...], np.dtype[Any]]):
    """
    Save tracking results to files for napari consumption.
    Args:
        napari_data:
        napari_graph:
        napari_properties:
        output_file:
        track_info:
        tracked_seg:

    Returns:

    """
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

    logger.info(f"[CHILD] Results saved successfully")


def graph_array_to_dict(graph_dict: dict[Any, Any],
                        napari_graph: np.ndarray[tuple[Any, ...],
                        np.dtype[Any]]):
    """
    Convert napari graph from array format to dict format.
    Args:
        graph_dict: Dictionary to populate
        napari_graph: Array of shape (N, 2) with [child, parent] pairs

    Returns:
    Populate graph_dict with child-parent relationships.

    """
    for child, parent in napari_graph:
        child_id = int(child)
        parent_id = int(parent)
        if child_id not in graph_dict:
            graph_dict[child_id] = []
        graph_dict[child_id].append(parent_id)

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
        input_file: Path to input temporary segmentation .npy file
        output_file: Path to save temporary output tracked segmentation
        params: Tracking parameters
        progress_queue: Queue to send progress updates
        status_flag: Shared value to indicate completion (0=running, 1=success, -1=error)

    TODO: This function is quite long and handles multiple responsibilities:
          - Loading and validating segmentation data
          - Extracting objects from segmentation
          - Configuring and running the btrack tracker
          - Post-processing napari data format
          - Saving results to files
          Consider breaking this into smaller, focused functions or a class with
          clear method boundaries. This would improve testability and maintainability.
    """
    try:
        logger.info(f"[CHILD {os.getpid()}] Process started")
        progress_queue.put("Loading segmentation from file...")

        # Load segmentation from file
        segmentation = np.load(input_file)
        logger.info(f"[CHILD] Loaded segmentation: shape={segmentation.shape}, dtype={segmentation.dtype}")

        # Ensure correct data type
        segmentation = np.ascontiguousarray(
            segmentation.astype(segmentation.dtype.newbyteorder('='))
        )

        # Handle both 2D+T and 3D+T data
        if segmentation.ndim == 3:
            # 2D+T: (T, Y, X)
            T, Y, X = segmentation.shape
            logger.info(f"[CHILD] Detected 2D+T data: T={T}, Y={Y}, X={X}")
        elif segmentation.ndim == 4:
            # 3D+T: (T, Z, Y, X)
            T, Z, Y, X = segmentation.shape
            logger.info(f"[CHILD] Detected 3D+T data: T={T}, Z={Z}, Y={Y}, X={X}")
        else:
            raise ValueError(f"Unsupported segmentation shape: {segmentation.shape}")

        progress_queue.put(f"Extracting objects from {T} frames...")
        logger.info(f"[CHILD] Extracting objects...")

        # Extract objects
        objects = utils.segmentation_to_objects(
            segmentation,
            properties=('area',),
            num_workers=1
        )

        logger.info(f"[CHILD] Found {len(objects)} objects")
        progress_queue.put(f"Found {len(objects)} objects. Starting tracking...")

        # Create temporary config file with parameters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = create_btrack_config_dict(params)
            json.dump(config_data, f, indent=2)
            config_path = f.name

        logger.info(f"[CHILD] Loading config from {config_path}")
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
            logger.info(f"[CHILD] Spatial volume (2D): X=[0,{X}], Y=[0,{Y}]")
        elif segmentation.ndim == 4:
            # 3D+T: spatial volume is (X, Y, Z) in btrack's reversed order
            volume = ((0, X), (0, Y), (0, Z))
            logger.info(f"[CHILD] Spatial volume (3D): X=[0,{X}], Y=[0,{Y}], Z=[0,{Z}]")

        progress_queue.put("Running btrack tracking and optimization...")
        logger.info(f"[CHILD] Starting btrack...")

        # Track
        with btrack.BayesianTracker(verbose=True) as tracker:
            # Configure tracker
            tracker.configure(conf)
            tracker.volume = volume  # Set spatial volume (not including time)
            tracker.max_search_radius = params['max_search_radius']
            tracker.append(objects)

            logger.info(f"[CHILD] Running tracker.track()...")
            tracker.track(step_size=100)

            logger.info(f"[CHILD] Running tracker.optimize()...")
            tracker.optimize()

            logger.info(f"[CHILD] Updating segmentation with {len(tracker.tracks)} tracks...")

            # Get results
            tracked_seg = utils.update_segmentation(segmentation, tracker.tracks)

            # Get tracks in napari format
            logger.info(f"[CHILD] Exporting tracks to napari...")
            napari_data, napari_properties, napari_graph = tracker.to_napari()

            # Convert graph from array to dict if necessary
            # Napari expects graph as a dict: {node_id: [parent_ids]}
            if isinstance(napari_graph, np.ndarray):
                graph_dict = {}
                if napari_graph.size == 0:
                    logger.info(f"[CHILD] Graph is empty array, converting to empty dict")
                elif napari_graph.ndim == 2 and napari_graph.shape[1] == 2:
                    # btrack might return graph as array - need to convert
                    # Typically graph is (N, 2) array of [child, parent] pairs
                    logger.info(f"[CHILD] Converting graph array to dict for napari")
                    graph_array_to_dict(graph_dict, napari_graph)
                    logger.info(f"[CHILD] Converted graph dict has {len(graph_dict)} nodes")
                else:
                    logger.info(f"[CHILD] Unexpected graph array shape: {napari_graph.shape}, converting to empty dict")

                napari_graph = graph_dict

            # Fix dimensionality mismatch between btrack output and input data
            # btrack always returns [track_id, t, z, y, x] (5 cols)
            # but for 2D+T data (shape: T, Y, X), we need [track_id, t, y, x] (4 cols)
            # and for 3D+T data (shape: T, Z, Y, X), we keep all 5 cols

            if napari_data.shape[1] == 5 and segmentation.ndim == 3:
                # Input is 2D+T (T, Y, X), so remove the Z column
                z_column = napari_data[:, 2]
                z_range = z_column.max() - z_column.min()
                logger.info(f"[CHILD] Input is 2D+T (shape: {segmentation.shape})")
                logger.info(f"[CHILD] Z column range: {z_range} (should be ~0 for 2D tracking)")
                logger.info(f"[CHILD] Removing Z column to match 2D+T format...")

                # Remove Z column: [track_id, t, z, y, x] -> [track_id, t, y, x]
                napari_data = np.column_stack([
                    napari_data[:, 0],  # track_id
                    napari_data[:, 1],  # t
                    napari_data[:, 3],  # y (skip z at index 2)
                    napari_data[:, 4]  # x
                ])
                logger.info(f"[CHILD] After removing Z: shape={napari_data.shape}")

            elif napari_data.shape[1] == 5 and segmentation.ndim == 4:
                # Input is 3D+T (T, Z, Y, X), keep all columns
                logger.info(f"[CHILD] Input is 3D+T (shape: {segmentation.shape})")
                logger.info(f"[CHILD] Keeping all 5 columns [track_id, t, z, y, x]")
                # napari_data stays as is

            else:
                logger.info(
                    f"[CHILD] Unexpected data format: napari shape={napari_data.shape}, seg shape={segmentation.shape}")

            track_info = {
                'total_tracks': len(tracker.tracks),
                'tracks_gt_1': sum(1 for t in tracker.tracks if len(t) > 1),
                'tracks_gt_5': sum(1 for t in tracker.tracks if len(t) > 5),
                'tracks_gt_10': sum(1 for t in tracker.tracks if len(t) > 10),
            }

        # Clean up temp config file
        Path(config_path).unlink()

        logger.info(f"[CHILD] Saving results to {output_file}...")
        progress_queue.put("Saving results...")

        # Save results to files
        save_tracking_results(napari_data, napari_graph, napari_properties, output_file, track_info, tracked_seg)

        progress_queue.put("Tracking complete!")

        # Signal success
        status_flag.value = 1
        logger.info(f"[CHILD] Process completed successfully")

    except Exception as e:
        error_msg = f"Error during tracking: {str(e)}\n{traceback.format_exc()}"
        logger.info(f"[CHILD] ERROR: {error_msg}")
        progress_queue.put(f"ERROR: {str(e)}")
        status_flag.value = -1


class TrackingManager:
    """
    High-level interface for starting and managing tracking operations.

    Handles the complexity of multiprocessing, file I/O, and progress monitoring.

    TODO: Consider adding the following improvements:
          - Add tracking state management (idle, running, cancelled, completed, error)
          - Implement a tracking history to allow reviewing past tracking results
          - Add support for batch tracking of multiple segmentations
          - Consider using asyncio instead of QThread for better integration with
            modern Python async patterns
    """

    def __init__(self):
        """
        Initialize the TrackingManager.
        """
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

        logger.info(f"[MAIN] Saving segmentation to {input_file}")
        # Save input segmentation
        np.save(input_file, segmentation)

        # Create shared status flag
        status_flag = Value('i', 0)  # 0=running, 1=success, -1=error

        # Create queue for progress
        progress_queue = Queue()

        logger.info(f"[MAIN] Starting tracking process...")
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