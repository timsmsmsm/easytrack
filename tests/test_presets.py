"""Tests for presets module."""

import json

from pathlib import Path
import tempfile

from presets import (
    load_config_from_json,
    create_btrack_config_dict,
    load_preset_if_exists,
    get_presets,
)


class TestLoadConfigFromJson:
    """Tests for load_config_from_json function."""

    def test_load_valid_config(self):
        """Test loading a valid btrack config JSON."""
        config_data = {
            "TrackerConfig": {
                "MotionModel": {
                    "max_search_radius": 100,
                    "prob_not_assign": 0.1,
                    "max_lost": 5,
                    "accuracy": 7.5,
                    "dt": 1.0,
                    "measurements": 3,
                    "states": 6,
                    "P": {"sigma": 150.0},
                    "G": {"sigma": 15.0},
                    "R": {"sigma": 5.0},
                },
                "HypothesisModel": {
                    "theta_dist": 20.0,
                    "theta_time": 5.0,
                    "lambda_time": 5.0,
                    "lambda_dist": 3.0,
                    "lambda_link": 10.0,
                    "lambda_branch": 50.0,
                    "eta": 1e-10,
                    "dist_thresh": 40,
                    "time_thresh": 2,
                    "apop_thresh": 5,
                    "segmentation_miss_rate": 0.1,
                    "apoptosis_rate": 0.001,
                    "relax": True,
                    "hypotheses": ["P_FP", "P_init", "P_term", "P_link", "P_dead"],
                },
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            params = load_config_from_json(temp_path)
            
            # Check expected keys are present
            assert 'max_search_radius' in params
            assert 'prob_not_assign' in params
            assert 'theta_dist' in params
            assert 'lambda_link' in params
            assert 'segmentation_miss_rate' in params
            
            # Check values
            assert params['max_search_radius'] == 100
            assert params['prob_not_assign'] == 0.1
            assert params['theta_dist'] == 20.0
            assert params['p_sigma'] == 150.0
            assert params['g_sigma'] == 15.0
            assert params['r_sigma'] == 5.0
            
        finally:
            Path(temp_path).unlink()

    def test_load_config_with_division(self):
        """Test loading config with division hypothesis enabled."""
        config_data = {
            "TrackerConfig": {
                "MotionModel": {
                    "max_search_radius": 100,
                    "prob_not_assign": 0.1,
                    "max_lost": 5,
                    "accuracy": 7.5,
                    "dt": 1.0,
                    "measurements": 3,
                    "states": 6,
                    "P": {"sigma": 150.0},
                    "G": {"sigma": 15.0},
                    "R": {"sigma": 5.0},
                },
                "HypothesisModel": {
                    "hypotheses": ["P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"],
                },
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            params = load_config_from_json(temp_path)
            assert params['div_hypothesis'] == 1
        finally:
            Path(temp_path).unlink()

    def test_load_config_without_division(self):
        """Test loading config without division hypothesis."""
        config_data = {
            "TrackerConfig": {
                "MotionModel": {
                    "max_search_radius": 100,
                    "prob_not_assign": 0.1,
                    "max_lost": 5,
                    "accuracy": 7.5,
                    "dt": 1.0,
                    "measurements": 3,
                    "states": 6,
                    "P": {"sigma": 150.0},
                    "G": {"sigma": 15.0},
                    "R": {"sigma": 5.0},
                },
                "HypothesisModel": {
                    "hypotheses": ["P_FP", "P_init", "P_term", "P_link", "P_dead"],
                },
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            params = load_config_from_json(temp_path)
            assert params['div_hypothesis'] == 0
        finally:
            Path(temp_path).unlink()


class TestCreateBtrackConfigDict:
    """Tests for create_btrack_config_dict function."""

    def test_basic_config_creation(self):
        """Test creating a basic btrack config dictionary."""
        params = {
            'prob_not_assign': 0.1,
            'max_lost': 5,
            'max_search_radius': 100,
            'dt': 1.0,
            'measurements': 3,
            'states': 6,
            'accuracy': 7.5,
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
            'div_hypothesis': 1,
        }
        
        config = create_btrack_config_dict(params)
        
        # Check structure
        assert 'TrackerConfig' in config
        assert 'MotionModel' in config['TrackerConfig']
        assert 'HypothesisModel' in config['TrackerConfig']
        
        motion = config['TrackerConfig']['MotionModel']
        assert motion['max_search_radius'] == 100
        assert motion['prob_not_assign'] == 0.1
        assert motion['P']['sigma'] == 150.0
        
        hypothesis = config['TrackerConfig']['HypothesisModel']
        assert 'P_branch' in hypothesis['hypotheses']
        assert hypothesis['theta_dist'] == 20.0

    def test_config_with_division_disabled(self):
        """Test config creation with division hypothesis disabled."""
        params = {
            'prob_not_assign': 0.1,
            'max_lost': 5,
            'max_search_radius': 100,
            'div_hypothesis': 0,
        }
        
        config = create_btrack_config_dict(params)
        
        hypothesis = config['TrackerConfig']['HypothesisModel']
        assert 'P_branch' not in hypothesis['hypotheses']
        assert 'P_link' in hypothesis['hypotheses']

    def test_config_with_division_enabled(self):
        """Test config creation with division hypothesis enabled."""
        params = {
            'prob_not_assign': 0.1,
            'max_lost': 5,
            'max_search_radius': 100,
            'div_hypothesis': 1,
        }
        
        config = create_btrack_config_dict(params)
        
        hypothesis = config['TrackerConfig']['HypothesisModel']
        assert 'P_branch' in hypothesis['hypotheses']

    def test_config_has_required_matrices(self):
        """Test that config includes required matrix definitions."""
        params = {
            'prob_not_assign': 0.1,
            'max_lost': 5,
            'max_search_radius': 100,
        }
        
        config = create_btrack_config_dict(params)
        
        motion = config['TrackerConfig']['MotionModel']
        assert 'A' in motion and 'matrix' in motion['A']
        assert 'H' in motion and 'matrix' in motion['H']
        assert 'P' in motion and 'matrix' in motion['P']
        assert 'G' in motion and 'matrix' in motion['G']
        assert 'R' in motion and 'matrix' in motion['R']


class TestLoadPresetIfExists:
    """Tests for load_preset_if_exists function."""

    def test_load_existing_config(self):
        """Test loading an existing config file."""
        config_data = {
            "TrackerConfig": {
                "MotionModel": {
                    "max_search_radius": 100,
                    "prob_not_assign": 0.1,
                    "max_lost": 5,
                    "accuracy": 7.5,
                    "dt": 1.0,
                    "measurements": 3,
                    "states": 6,
                    "P": {"sigma": 150.0},
                    "G": {"sigma": 15.0},
                    "R": {"sigma": 5.0},
                },
                "HypothesisModel": {
                    "hypotheses": ["P_FP", "P_init"],
                },
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            params = load_preset_if_exists(temp_path)
            assert 'max_search_radius' in params
            assert params['max_search_radius'] == 100
        finally:
            Path(temp_path).unlink()

    def test_nonexistent_returns_empty(self):
        """Test that non-existent file returns empty dict."""
        result = load_preset_if_exists("/nonexistent/path/to/config.json")
        assert result == {}


class TestGetPresets:
    """Tests for get_presets function."""

    def test_returns_dict(self):
        """Test that get_presets returns a dictionary."""
        presets = get_presets()
        assert isinstance(presets, dict)

    def test_presets_have_required_keys(self):
        """Test that each preset has required keys."""
        presets = get_presets()
        
        for name, preset in presets.items():
            assert 'description' in preset, f"Preset '{name}' missing 'description'"
            assert 'config' in preset, f"Preset '{name}' missing 'config'"
            assert isinstance(preset['description'], str)
            assert isinstance(preset['config'], dict)

    def test_custom_json_preset_exists(self):
        """Test that Custom JSON preset exists."""
        presets = get_presets()
        assert "Custom JSON" in presets

    def test_epithelial_default_preset_exists(self):
        """Test that the default epithelial preset exists."""
        presets = get_presets()
        assert "Epithelial Cells (Default)" in presets

    def test_preset_descriptions_not_empty(self):
        """Test that preset descriptions are not empty."""
        presets = get_presets()
        
        for name, preset in presets.items():
            assert len(preset['description'].strip()) > 0, f"Preset '{name}' has empty description"


class TestRealConfigFiles:
    """Tests for loading real config files from the configs directory."""

    def test_load_disc2_config(self):
        """Test loading the disc2 config if it exists."""
        config_path = Path(__file__).parent.parent / "configs" / "disc2_config.json"
        if config_path.exists():
            params = load_config_from_json(str(config_path))
            assert 'max_search_radius' in params
            assert 'prob_not_assign' in params

    def test_load_3d_config(self):
        """Test loading the 3D config if it exists."""
        config_path = Path(__file__).parent.parent / "configs" / "optimized_config_3d.json"
        if config_path.exists():
            params = load_config_from_json(str(config_path))
            assert 'max_search_radius' in params
            assert 'prob_not_assign' in params

    def test_roundtrip_config(self):
        """Test that a config can be loaded, converted, and still be valid JSON."""
        config_path = Path(__file__).parent.parent / "configs" / "disc2_config.json"
        if config_path.exists():
            params = load_config_from_json(str(config_path))
            config_dict = create_btrack_config_dict(params)
            
            # Should be valid JSON
            json_str = json.dumps(config_dict)
            parsed = json.loads(json_str)
            
            assert 'TrackerConfig' in parsed
            assert 'MotionModel' in parsed['TrackerConfig']
