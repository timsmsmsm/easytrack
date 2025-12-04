"""
Tracking engine and process management for btrack.

Handles:
- Running btrack tracking in a separate process
- Monitoring progress and managing cancellation
- File-based communication to avoid Queue size limits
- Exporting tracks to napari
"""
from __future__ import annotations

import json
import shutil
import traceback
from multiprocessing import Process, Queue, Value

import numpy as np
from qtpy.QtCore import QThread, Signal

from easytrack import logger


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
        logger.info(f"[MONITOR] Cancel requested")
        self._is_cancelled = True
        if self.process.is_alive():
            logger.info(f"[MONITOR] Terminating process...")
            self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                logger.info(f"[MONITOR] Killing process...")
                self.process.kill()
                self.process.join()
            logger.info(f"[MONITOR] Process terminated")

    def run(self):
        """Monitor the process and relay messages."""
        try:
            logger.info(f"[MONITOR] Started monitoring process PID: {self.process.pid}")

            while True:
                # Check for progress messages
                while not self.progress_queue.empty():
                    msg = self.progress_queue.get_nowait()
                    logger.info(f"[MONITOR] Progress: {msg}")
                    self.progress.emit(msg)

                # Check if process is done
                if not self.process.is_alive():
                    logger.info(f"[MONITOR] Process ended with exit code: {self.process.exitcode}")

                    # Get final progress messages
                    while not self.progress_queue.empty():
                        msg = self.progress_queue.get_nowait()
                        logger.info(f"[MONITOR] Final progress: {msg}")
                        self.progress.emit(msg)

                    # Check status
                    if self.status_flag.value == 1:
                        logger.info(f"[MONITOR] Success! Loading results from {self.output_file}")
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

                        logger.info(f"[MONITOR] Loaded results: {track_info}")
                        logger.info(f"[MONITOR] Napari tracks shape: {napari_data.shape}")
                        logger.info(f"[MONITOR] Napari properties keys: {list(napari_properties.keys())}")
                        self.finished.emit(tracked_seg, track_info, napari_data, napari_properties, napari_graph)
                    elif self.status_flag.value == -1:
                        logger.info(f"[MONITOR] Process reported error")
                        self.error.emit("Tracking process reported an error. Check console for details.")
                    elif not self._is_cancelled:
                        logger.info(f"[MONITOR] Process ended without setting status flag")
                        self.error.emit("Process ended unexpectedly without result")
                    break

                # Check for cancellation
                if self._is_cancelled:
                    logger.info(f"[MONITOR] Cancelled by user")
                    break

                # Small sleep to avoid busy waiting
                self.msleep(100)

        except Exception as e:
            if not self._is_cancelled:
                error_msg = f"Monitor error: {str(e)}\n{traceback.format_exc()}"
                logger.info(f"[MONITOR] ERROR: {error_msg}")
                self.error.emit(error_msg)
        finally:
            # Clean up temp files
            logger.info(f"[MONITOR] Cleaning up temp directory: {self.temp_dir}")
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass


