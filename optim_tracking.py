"""
Tracking executor for applying optimized btrack parameters.

This module handles:
- Converting optimized trial parameters to btrack config
- Running btrack with optimized parameters
- Returning tracked segmentation and statistics
"""

import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Dict, Tuple, Optional
import btrack
from btrack import utils, config

# Import your config writing function
from optim_pipeline import write_best_params_to_config

def run_tracking_with_params(
    segmentation: np.ndarray,
    params: Dict,
    voxel_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[np.ndarray, list, Dict]:
    """
    Run btrack tracking with optimized parameters.
    
    Args:
        segmentation: 3D numpy array (T, Y, X) with cell labels
        params: Dictionary of optimized parameters from Optuna trial
        voxel_scale: Tuple of (t, y, x) scaling factors
        
    Returns:
        Tuple of:
        - tracked_segmentation: 3D array with track IDs
        - tracks: List of btrack track objects
        - stats: Dictionary with tracking statistics
    """
    print("\n" + "=" * 60)
    print("RUNNING TRACKING WITH OPTIMIZED PARAMETERS")
    print("=" * 60)
    
    # Ensure contiguous array with native byte order
    segmentation = np.ascontiguousarray(
        segmentation.astype(segmentation.dtype.newbyteorder('='))
    )
    
    T, Y, X = segmentation.shape
    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Voxel scale: {voxel_scale}")
    
    # Create temporary config file with optimized parameters
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_config_path = f.name
    
    try:
        # Write optimized parameters to config
        print("\nWriting optimized config...")
        write_best_params_to_config(params, temp_config_path)
        
        # Load the config
        conf = config.load_config(temp_config_path)
        conf.enable_optimisation = True
        
        # Convert segmentation to btrack objects
        print("Converting segmentation to objects...")
        objects = utils.segmentation_to_objects(
            segmentation, 
            properties=('area',),
            num_workers=1
        )
        print(f"  Found {len(objects)} cell detections")
        
        # Set up volume
        volume = ((0, X), (0, Y), (0, T))
        
        # Get max_search_radius from params
        max_search_radius = params.get('max_search_radius', 100)
        print(f"  Max search radius: {max_search_radius}")
        
        # Run tracking
        print("\nRunning btrack...")
        with btrack.BayesianTracker(verbose=False) as tracker:
            tracker.configure(conf)
            tracker.max_search_radius = max_search_radius
            tracker.append(objects)
            tracker.volume = volume[::-1]
            tracker.track(step_size=100)
            tracker.optimize()
            tracks = tracker.tracks
            lbep = tracker.LBEP
            print(f"  Tracking complete: {len(tracks)} tracks found")
            
            # Update segmentation with track IDs
            print("Updating segmentation with track IDs...")
            tracked_segmentation = utils.update_segmentation(segmentation, tracks)
        
        # Calculate statistics
        track_lengths = [len(track) for track in tracks]
        tracks_gt_1 = sum(1 for length in track_lengths if length > 1)
        tracks_gt_5 = sum(1 for length in track_lengths if length > 5)
        tracks_gt_10 = sum(1 for length in track_lengths if length > 10)
        
        stats = {
            'total_tracks': len(tracks),
            'tracks_gt_1': tracks_gt_1,
            'tracks_gt_5': tracks_gt_5,
            'tracks_gt_10': tracks_gt_10,
            'min_length': min(track_lengths) if track_lengths else 0,
            'max_length': max(track_lengths) if track_lengths else 0,
            'mean_length': np.mean(track_lengths) if track_lengths else 0.0,
            'median_length': np.median(track_lengths) if track_lengths else 0.0
        }
        
        print("\nTracking Statistics:")
        print(f"  Total tracks: {stats['total_tracks']}")
        print(f"  Tracks > 1 frame: {stats['tracks_gt_1']}")
        print(f"  Tracks > 5 frames: {stats['tracks_gt_5']}")
        print(f"  Tracks > 10 frames: {stats['tracks_gt_10']}")
        print(f"  Track length range: {stats['min_length']}-{stats['max_length']}")
        print(f"  Mean length: {stats['mean_length']:.1f}")
        print(f"  Median length: {stats['median_length']:.1f}")
        print("=" * 60 + "\n")
        
        return tracked_segmentation, tracks, stats
        
    finally:
        # Clean up temporary config file
        Path(temp_config_path).unlink(missing_ok=True)


def format_params_summary(params: Dict) -> str:
    """
    Format parameters into a readable summary string.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Formatted string with key parameters
    """
    key_params = [
        'max_search_radius',
        'dist_thresh', 
        'theta_dist',
        'lambda_link',
        'segmentation_miss_rate',
        'prob_not_assign'
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
    """
    Validate that parameters dictionary has required fields.
    
    Args:
        params: Parameters dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_params = [
        'max_search_radius',
        'dist_thresh',
        'theta_dist',
        'lambda_link',
        'segmentation_miss_rate',
        'prob_not_assign',
        'theta_time',
        'lambda_time',
        'lambda_dist',
        'lambda_branch',
        'time_thresh',
        'apop_thresh',
        'p_sigma',
        'g_sigma',
        'r_sigma',
        'accuracy',
        'max_lost',
        'div_hypothesis'
    ]
    
    missing = [p for p in required_params if p not in params]
    
    if missing:
        return False, f"Missing required parameters: {', '.join(missing)}"
    
    return True, ""


def get_default_params() -> Dict:
    """
    Get default btrack parameters as fallback.
    
    Returns:
        Dictionary of default parameters
    """
    return {
        'dt': 1.0,
        'measurements': 3,
        'states': 6,
        'accuracy': 7.5,
        'prob_not_assign': 0.1,
        'max_lost': 5,
        'max_search_radius': 100,
        'p_sigma': 150.0,
        'g_sigma': 15.0,
        'r_sigma': 5.0,
        'lambda_time': 5.0,
        'lambda_dist': 3.0,
        'lambda_link': 10.0,
        'lambda_branch': 50.0,
        'eta': 1e-10,
        'theta_dist': 20.0,
        'theta_time': 5.0,
        'dist_thresh': 40,
        'time_thresh': 2,
        'apop_thresh': 5,
        'segmentation_miss_rate': 0.1,
        'apoptosis_rate': 0.001,
        'relax': True,
        'div_hypothesis': 1
    }
