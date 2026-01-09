"""
Configuration and preset management for btrack tracking.

Handles:
- Loading preset configurations
- Loading custom JSON configs
- Creating btrack-compatible config dictionaries
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """
    Load all parameters from a btrack config JSON file.
    
    Args:
        config_path: Path to the config JSON file
        
    Returns:
        Dictionary with all the btrack parameters
    """
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    motion_model = full_config['TrackerConfig']['MotionModel']
    hypothesis_model = full_config['TrackerConfig']['HypothesisModel']
    
    params = {
        # Motion model parameters
        'max_search_radius': motion_model.get('max_search_radius', 100),
        'prob_not_assign': motion_model.get('prob_not_assign', 0.1),
        'max_lost': motion_model.get('max_lost', 5),
        'accuracy': motion_model.get('accuracy', 7.5),
        'dt': motion_model.get('dt', 1.0),
        'measurements': motion_model.get('measurements', 3),
        'states': motion_model.get('states', 6),
        
        # Sigma values for matrix scaling
        'p_sigma': motion_model['P'].get('sigma', 150.0),
        'g_sigma': motion_model['G'].get('sigma', 15.0),
        'r_sigma': motion_model['R'].get('sigma', 5.0),
        
        # Hypothesis model parameters
        'theta_dist': hypothesis_model.get('theta_dist', 20.0),
        'theta_time': hypothesis_model.get('theta_time', 5.0),
        'lambda_time': hypothesis_model.get('lambda_time', 5.0),
        'lambda_dist': hypothesis_model.get('lambda_dist', 3.0),
        'lambda_link': hypothesis_model.get('lambda_link', 10.0),
        'lambda_branch': hypothesis_model.get('lambda_branch', 50.0),
        'eta': hypothesis_model.get('eta', 1e-10),
        'dist_thresh': hypothesis_model.get('dist_thresh', 40),
        'time_thresh': hypothesis_model.get('time_thresh', 2),
        'apop_thresh': hypothesis_model.get('apop_thresh', 5),
        'segmentation_miss_rate': hypothesis_model.get('segmentation_miss_rate', 0.1),
        'apoptosis_rate': hypothesis_model.get('apoptosis_rate', 0.001),
        'relax': hypothesis_model.get('relax', True),
        
        # Division hypothesis (check if P_branch is in hypotheses list)
        'div_hypothesis': 1 if 'P_branch' in hypothesis_model.get('hypotheses', []) else 0,
    }
    
    return params


def create_btrack_config_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a btrack-compatible config dictionary from parameters.
    
    Args:
        params: Dictionary of tracking parameters
        
    Returns:
        Full btrack configuration dictionary
    """
    # Set hypotheses list based on div_hypothesis parameter
    if params.get('div_hypothesis', 1) == 1:
        hypotheses_list = ["P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"]
    else:
        hypotheses_list = ["P_FP", "P_init", "P_term", "P_link", "P_dead"]
    
    return {
        "TrackerConfig": {
            "MotionModel": {
                "name": "cell_motion",
                "dt": params.get('dt', 1.0),
                "measurements": params.get('measurements', 3),
                "states": params.get('states', 6),
                "accuracy": params.get('accuracy', 7.5),
                "prob_not_assign": params['prob_not_assign'],
                "max_lost": params['max_lost'],
                "max_search_radius": params['max_search_radius'],
                "A": {"matrix": [1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1]},
                "H": {"matrix": [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0]},
                "P": {"sigma": params.get('p_sigma', 150.0), "matrix": [0.1,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0.1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1]},
                "G": {"sigma": params.get('g_sigma', 15.0), "matrix": [0.5,0.5,0.5,1,1,1]},
                "R": {"sigma": params.get('r_sigma', 5.0), "matrix": [1,0,0,0,1,0,0,0,1]}
            },
            "ObjectModel": {},
            "HypothesisModel": {
                "name": "cell_hypothesis",
                "hypotheses": hypotheses_list,
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
                "relax": params.get('relax', True)
            }
        }
    }


def load_preset_if_exists(config_path: str) -> Dict[str, Any]:
    """
    Load a preset config file if it exists, otherwise return empty dict.
    
    Args:
        config_path: Path to config file (relative or absolute)
        
    Returns:
        Parameter dictionary or empty dict if file doesn't exist
    """
    path = Path(config_path)
    if path.exists():
        return load_config_from_json(str(path))
    return {}


# ============= PRESET DEFINITIONS =============

def get_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get all available presets.
    
    Returns:
        Dictionary mapping preset names to their configs and descriptions
    """

    MODULE_DIR = Path(__file__).parent
    disc2_params = load_preset_if_exists(str(MODULE_DIR / 'configs/disc2_config.json'))
    _3d_params = load_preset_if_exists(str(MODULE_DIR / 'configs/optimized_config_3d.json'))

    presets = {
        "Epithelial Cells (Default)": {
            "description": """
Optimized for epithelial tissue imaging:
• Slow-moving cells in dense tissue
• Minimal divisions
• Strong cell-cell adhesion
• Trained on Wing disc epithelium data
            """,
            "config": disc2_params
        },
        
        "Epithelial Cells (Z-tracking)": {
            "description": """
• For stitching of 3D stacks over Z-axis
• Trained on 3D Wing disc data
            """,
            "config": _3d_params
        },
        "Custom JSON": {
            "description": """
Load parameters from your own JSON config file:
• Use the file browser below to select a config file
• Must be in correct JSON format
• All parameters will be loaded from the file
            """,
            "config": {}  # Empty, will be populated from file
        },
    }
    
    return presets
