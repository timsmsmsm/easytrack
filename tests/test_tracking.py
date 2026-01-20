"""Tests for tracking module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json

from src.napari_easytrack.analysis.tracking import (
    scale_matrix,
    get_default_config_path,
)


class TestScaleMatrix:
    """Tests for the scale_matrix function."""

    def test_scale_with_nonzero_original(self):
        """Test scaling matrix with non-zero original sigma."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        original_sigma = 2.0
        new_sigma = 4.0
        
        result = scale_matrix(matrix, original_sigma, new_sigma)
        
        # Should unscale by dividing by 2.0, then rescale by multiplying by 4.0
        # Effective multiplication by 2.0
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

    def test_scale_preserves_shape(self):
        """Test that scaling preserves matrix shape."""
        matrix = np.random.rand(4, 5)
        original_sigma = 2.5
        new_sigma = 1.5
        
        result = scale_matrix(matrix, original_sigma, new_sigma)
        
        assert result.shape == matrix.shape

    def test_scale_with_same_sigma(self):
        """Test scaling when new and original sigma are the same."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        sigma = 3.0
        
        result = scale_matrix(matrix, sigma, sigma)
        
        # Should return essentially the same matrix
        np.testing.assert_array_almost_equal(result, matrix)


class TestGetDefaultConfigPath:
    """Tests for get_default_config_path function."""

    def test_returns_string_path(self):
        """Test that function returns a string path."""
        result = get_default_config_path()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_path_contains_cell_config(self):
        """Test that path contains cell_config.json."""
        result = get_default_config_path()
        assert 'cell_config.json' in result

    def test_config_file_exists_or_in_temp(self):
        """Test that the returned path exists (config file is accessible)."""
        result = get_default_config_path()
        path = Path(result)
        
        # The path should either exist directly, or be in a temp location
        temp_dir = Path(tempfile.gettempdir())
        assert path.exists() or path.parent == temp_dir
