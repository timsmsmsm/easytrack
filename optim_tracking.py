"""
Tracking executor for applying optimized btrack parameters.

This module handles:
- Loading base config and modifying parameters (matching optimization exactly)
- Running btrack with optimized parameters
- Returning tracked segmentation and statistics

CRITICAL: This version loads cell_config.json and modifies it, just like
the optimization does, rather than creating a new config from scratch.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import btrack
from btrack import utils, config


def scale_matrix(matrix: np.ndarray, original_sigma: float, new_sigma: float) -> np.ndarray:
    """
    Scales a matrix by first reverting the original scaling and then applying a new sigma value.
    
    This MUST match the scale_matrix function used in optimization.
    """
    if original_sigma != 0:
        unscaled_matrix = matrix / original_sigma
    else:
        unscaled_matrix = matrix.copy()
    rescaled_matrix = unscaled_matrix * new_sigma
    return rescaled_matrix


def run_tracking_with_params(
    segmentation: np.ndarray,
    params: Dict,
    voxel_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    base_config_path: str = 'cell_config.json'
) -> Tuple[np.ndarray, list, Dict]:
    """
    Run btrack tracking with optimized parameters.
    
    This function EXACTLY matches the tracking procedure in objective() function
    during optimization.
    
    Args:
        segmentation: 3D numpy array (T, Y, X) with cell labels
        params: Dictionary of optimized parameters from Optuna trial
        voxel_scale: Tuple of (t, y, x) scaling factors
        base_config_path: Path to base cell_config.json file
        
    Returns:
        Tuple of:
        - tracked_segmentation: 3D array with track IDs
        - tracks: List of btrack track objects
        - stats: Dictionary with tracking statistics
    """
    print("\n" + "=" * 60)
    print("RUNNING TRACKING WITH OPTIMIZED PARAMETERS")
    print("=" * 60)
    
    T, Y, X = segmentation.shape
    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Voxel scale: {voxel_scale}")
    
    # CRITICAL: Load base config first, then modify - matches optimization!
    print(f"\nLoading base config: {base_config_path}")
    conf = config.load_config(base_config_path)
    
    # Calculate scaled volume - matches optimization
    try:
        from optim_pipeline import compute_scaling_factors
    except ImportError:
        def compute_scaling_factors(voxel_sizes):
            vt, vy, vx = voxel_sizes
            avg_voxel_size = (vt + vy + vx) / 3.0
            st = avg_voxel_size / vt
            sy = avg_voxel_size / vy
            sx = avg_voxel_size / vx
            return st, sy, sx
    
    scale = compute_scaling_factors(voxel_scale)
    scaled_dims = [segmentation.shape[idx+1] * scale[idx+1] for idx in range(2)]  
    volume = tuple(zip([0]*2, scaled_dims))
    
    print(f"Scale factors: {scale}")
    print(f"Volume (scaled): {volume}")
    
    # Convert segmentation to objects - matches optimization
    print("\nConverting segmentation to objects...")
    objects = utils.segmentation_to_objects(
        segmentation, 
        properties=('area',)
    )
    print(f"  Found {len(objects)} cell detections")
    
    # Build attributes dict EXACTLY like optimization does
    print("\nApplying optimized parameters...")
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
    
    # Apply attributes to config - matches optimization
    for attr, value in attributes.items():
        if attr in ['P', 'G', 'R', 'max_lost', 'prob_not_assign', 'accuracy']:
            setattr(conf.motion_model, attr, value)
        else:
            setattr(conf.hypothesis_model, attr, value)
    
    # Handle division hypothesis - matches optimization
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
    
    # Enable optimization - matches optimization
    conf.enable_optimisation = True
    
    # Get max_search_radius
    max_search_radius = params.get('max_search_radius', 100)
    print(f"  Max search radius: {max_search_radius}")
    
    # Print key config values for debugging
    print(f"\nKey config values:")
    print(f"  max_lost: {conf.motion_model.max_lost}")
    print(f"  prob_not_assign: {conf.motion_model.prob_not_assign}")
    print(f"  dist_thresh: {conf.hypothesis_model.dist_thresh}")
    print(f"  theta_dist: {conf.hypothesis_model.theta_dist}")
    print(f"  lambda_link: {conf.hypothesis_model.lambda_link}")
    
    # Run tracking - matches optimization exactly
    print("\nRunning btrack...")
    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(conf)
        tracker.max_search_radius = max_search_radius
        tracker.append(objects)
        tracker.volume = volume[::-1]  # Reverse for btrack convention
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


def format_params_summary(params: Dict) -> str:
    """Format parameters into a readable summary string."""
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
    """Validate that parameters dictionary has required fields."""
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
    """Get default btrack parameters as fallback."""
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