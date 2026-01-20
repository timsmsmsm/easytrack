"""Tests for optimization pipeline module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json

from src.napari_easytrack.analysis.optim_pipeline import (
    compute_scaling_factors,
    read_config_params,
    scale_matrix,
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


class TestScaleMatrix:
    """Tests for scale_matrix function in optim_pipeline."""

    def test_scale_with_nonzero_original(self):
        """Test scaling matrix with non-zero original sigma."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        original_sigma = 2.0
        new_sigma = 4.0
        
        result = scale_matrix(matrix, original_sigma, new_sigma)
        
        # Should unscale by dividing by 2.0, then rescale by multiplying by 4.0
        expected = matrix * (new_sigma / original_sigma)
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_with_zero_original(self):
        """Test scaling matrix when original sigma is zero."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        original_sigma = 0.0
        new_sigma = 3.0
        
        result = scale_matrix(matrix, original_sigma, new_sigma)
        
        # Should just multiply by new_sigma
        expected = matrix * new_sigma
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_identity_matrix(self):
        """Test scaling an identity matrix."""
        matrix = np.eye(3)
        original_sigma = 1.0
        new_sigma = 5.0
        
        result = scale_matrix(matrix, original_sigma, new_sigma)
        
        expected = np.eye(3) * 5.0
        np.testing.assert_array_almost_equal(result, expected)
