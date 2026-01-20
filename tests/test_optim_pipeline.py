"""Tests for optimization pipeline module."""

import pytest
from pathlib import Path
import tempfile
import json

from src.napari_easytrack.analysis.optim_pipeline import (
    compute_scaling_factors,
    read_config_params,
    write_best_params_to_config,
    add_config_params_to_dict,
    add_missing_attributes,
)


class TestComputeScalingFactors:
    """Tests for compute_scaling_factors function."""

    def test_isotropic_voxels(self):
        """Test with isotropic voxels (all same size)."""
        voxel_sizes = (1.0, 1.0, 1.0)
        result = compute_scaling_factors(voxel_sizes)
        
        # All scaling factors should be 1.0
        assert result == (1.0, 1.0, 1.0)

    def test_anisotropic_voxels(self):
        """Test with anisotropic voxels."""
        voxel_sizes = (2.0, 1.0, 0.5)
        result = compute_scaling_factors(voxel_sizes)
        
        # Average = (2.0 + 1.0 + 0.5) / 3 = 1.1667
        # sx = 1.1667 / 2.0 = 0.5833...
        # sy = 1.1667 / 1.0 = 1.1667
        # sz = 1.1667 / 0.5 = 2.3333...
        expected_avg = (2.0 + 1.0 + 0.5) / 3.0
        expected = (
            expected_avg / 2.0,
            expected_avg / 1.0,
            expected_avg / 0.5
        )
        
        assert len(result) == 3
        for i in range(3):
            assert abs(result[i] - expected[i]) < 1e-10

    def test_z_larger_than_xy(self):
        """Test when Z voxel size is larger (common in microscopy)."""
        voxel_sizes = (0.1, 0.1, 0.5)
        result = compute_scaling_factors(voxel_sizes)
        
        # Average = (0.1 + 0.1 + 0.5) / 3 = 0.2333...
        # Z should have scaling factor < 1 (shrink)
        # X, Y should have scaling factor > 1 (expand)
        assert result[2] < 1.0  # Z should shrink
        assert result[0] > 1.0  # X should expand
        assert result[1] > 1.0  # Y should expand

    def test_preserves_relative_ratios(self):
        """Test that scaling preserves relative ratios."""
        voxel_sizes = (1.0, 2.0, 4.0)
        sx, sy, sz = compute_scaling_factors(voxel_sizes)
        
        # After scaling, all dimensions should have the same effective size
        # vx * sx, vy * sy, vz * sz should all be equal to avg
        avg = (1.0 + 2.0 + 4.0) / 3.0
        assert abs(1.0 * sx - avg) < 1e-10
        assert abs(2.0 * sy - avg) < 1e-10
        assert abs(4.0 * sz - avg) < 1e-10


class TestReadConfigParams:
    """Tests for read_config_params function."""

    def test_read_valid_config(self):
        """Test reading a valid JSON config file."""
        config_data = {
            "TrackerConfig": {
                "MotionModel": {
                    "max_search_radius": 100,
                    "prob_not_assign": 0.1
                },
                "HypothesisModel": {
                    "theta_dist": 20.0
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            result = read_config_params(temp_path)
            
            assert "TrackerConfig" in result
            assert result["TrackerConfig"]["MotionModel"]["max_search_radius"] == 100
            assert result["TrackerConfig"]["MotionModel"]["prob_not_assign"] == 0.1
        finally:
            Path(temp_path).unlink()

    def test_read_empty_config(self):
        """Test reading an empty JSON config."""
        config_data = {}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            result = read_config_params(temp_path)
            assert result == {}
        finally:
            Path(temp_path).unlink()

    def test_nonexistent_file_raises(self):
        """Test that non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_config_params("/nonexistent/config.json")


class TestWriteBestParamsToConfig:
    """Tests for write_best_params_to_config function."""

    def test_writes_valid_json_config(self):
        """Test that a valid JSON config is written."""
        params = {
            'prob_not_assign': 0.15,
            'max_lost': 7,
            'max_search_radius': 120,
            'dt': 1.0,
            'measurements': 3,
            'states': 6,
            'accuracy': 8.0,
            'p_sigma': 200.0,
            'g_sigma': 20.0,
            'r_sigma': 6.0,
            'lambda_time': 6.0,
            'lambda_dist': 4.0,
            'lambda_link': 12.0,
            'lambda_branch': 60.0,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            write_best_params_to_config(params, temp_path)
            
            # Read back and verify
            with open(temp_path, 'r') as f:
                config = json.load(f)
            
            assert 'TrackerConfig' in config
            assert 'MotionModel' in config['TrackerConfig']
            assert 'HypothesisModel' in config['TrackerConfig']
            
            motion = config['TrackerConfig']['MotionModel']
            assert motion['max_search_radius'] == 120
            assert motion['prob_not_assign'] == 0.15
            assert motion['max_lost'] == 7
            
        finally:
            Path(temp_path).unlink()

    def test_handles_minimal_params(self):
        """Test writing config with minimal parameters using defaults."""
        params = {
            'max_search_radius': 100,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            write_best_params_to_config(params, temp_path)
            
            # Should use defaults for missing params
            with open(temp_path, 'r') as f:
                config = json.load(f)
            
            motion = config['TrackerConfig']['MotionModel']
            assert motion['max_search_radius'] == 100
            # These should have defaults
            assert 'dt' in motion
            assert 'measurements' in motion
            
        finally:
            Path(temp_path).unlink()

    def test_includes_required_matrices(self):
        """Test that required matrices are included in config."""
        params = {'max_search_radius': 100}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            write_best_params_to_config(params, temp_path)
            
            with open(temp_path, 'r') as f:
                config = json.load(f)
            
            motion = config['TrackerConfig']['MotionModel']
            # Check that matrices are present
            assert 'A' in motion and 'matrix' in motion['A']
            assert 'H' in motion and 'matrix' in motion['H']
            assert 'P' in motion
            assert 'G' in motion
            assert 'R' in motion
            
        finally:
            Path(temp_path).unlink()


class TestAddConfigParamsToDict:
    """Tests for add_config_params_to_dict function."""

    def test_adds_params_from_config(self):
        """Test adding parameters from config file to dictionary."""
        import pandas as pd
        
        # Create a config file
        config_data = {
            "TrackerConfig": {
                "MotionModel": {
                    "max_lost": 5,
                    "prob_not_assign": 0.1,
                    "P": {"sigma": 150.0},
                    "G": {"sigma": 15.0},
                    "R": {"sigma": 5.0},
                },
                "HypothesisModel": {
                    "theta_dist": 20.0,
                    "lambda_dist": 3.0,
                    "lambda_time": 5.0,
                    "lambda_link": 10.0,
                    "lambda_branch": 50.0,
                    "theta_time": 5.0,
                    "dist_thresh": 40,
                    "time_thresh": 2,
                    "apop_thresh": 5,
                    "segmentation_miss_rate": 0.1,
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Start with empty dataframe
            params_dict = pd.DataFrame()
            
            # Add config params
            result = add_config_params_to_dict(params_dict, temp_path, "test_dataset")
            
            # Check that parameters were added
            assert len(result) == 1
            assert "test_dataset" in result.index
            assert result.loc["test_dataset", "theta_dist"] == 20.0
            assert result.loc["test_dataset", "p_sigma"] == 150.0
            
        finally:
            Path(temp_path).unlink()


class TestAddMissingAttributes:
    """Tests for add_missing_attributes function."""

    def test_no_missing_attributes(self):
        """Test graph with all attributes present."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")
        
        graph = nx.Graph()
        graph.add_node(1, segmentation_id=1, x=10, y=20, t=0)
        graph.add_node(2, segmentation_id=2, x=15, y=25, t=1)
        
        # Should not raise error
        add_missing_attributes(graph)
        
        # Attributes should still be there
        assert graph.nodes[1]['segmentation_id'] == 1

    def test_adds_default_attributes_for_integer_nodes(self):
        """Test adding attributes to integer nodes without attributes."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")
        
        graph = nx.Graph()
        graph.add_node(1)  # Node without attributes
        graph.add_node(2, segmentation_id=2, x=10, y=20, t=0)  # Node with attributes
        
        add_missing_attributes(graph)
        
        # Node 1 should now have default attributes
        assert 'segmentation_id' in graph.nodes[1]
        assert graph.nodes[1]['segmentation_id'] == 1
        assert graph.nodes[1]['t'] == 0

    def test_adds_attributes_for_string_nodes(self):
        """Test adding attributes to string nodes in cell_timestep format."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")
        
        graph = nx.Graph()
        # Add a node with attributes
        graph.add_node("5_0", segmentation_id=5, x=10, y=20, t=0)
        # Add a node without attributes that should inherit from previous
        graph.add_node("5_1")
        
        add_missing_attributes(graph)
        
        # Node 5_1 should have attributes
        assert 'segmentation_id' in graph.nodes["5_1"]
        assert graph.nodes["5_1"]['segmentation_id'] == 5
        assert graph.nodes["5_1"]['t'] == 1
        # Should try to inherit x, y from previous timestep
        assert graph.nodes["5_1"]['x'] == 10
        assert graph.nodes["5_1"]['y'] == 20

    def test_handles_invalid_string_nodes(self):
        """Test handling of string nodes that don't match expected format."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")
        
        graph = nx.Graph()
        graph.add_node("invalid_node_name")
        
        add_missing_attributes(graph)
        
        # Should have default attributes
        assert 'segmentation_id' in graph.nodes["invalid_node_name"]
        assert graph.nodes["invalid_node_name"]['segmentation_id'] is None


