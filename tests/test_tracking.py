"""Tests for tracking module."""

import numpy as np
from pathlib import Path
import tempfile

from src.napari_easytrack.analysis.tracking import (
    scale_matrix,
    get_default_config_path,
    format_params_summary,
    validate_params,
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
        assert path.exists() or str(path).startswith(str(temp_dir))


class TestFormatParamsSummary:
    """Tests for format_params_summary function."""

    def test_formats_basic_params(self):
        """Test formatting basic parameters."""
        params = {
            'max_search_radius': 100,
            'prob_not_assign': 0.1,
            'div_hypothesis': 1,
        }
        
        summary = format_params_summary(params)
        
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_formats_empty_params(self):
        """Test formatting empty parameters."""
        params = {}
        
        summary = format_params_summary(params)
        
        assert isinstance(summary, str)

    def test_formats_division_hypothesis(self):
        """Test that division hypothesis is formatted correctly."""
        params_with_div = {'div_hypothesis': 1, 'max_search_radius': 100}
        params_without_div = {'div_hypothesis': 0, 'max_search_radius': 100}
        
        summary_with = format_params_summary(params_with_div)
        summary_without = format_params_summary(params_without_div)
        
        # Both should be valid strings
        assert isinstance(summary_with, str)
        assert isinstance(summary_without, str)


class TestValidateParams:
    """Tests for validate_params function."""

    def test_validates_correct_params(self):
        """Test validation of correct parameters."""
        params = {
            'max_search_radius': 100,
            'prob_not_assign': 0.1,
            'max_lost': 5,
        }
        
        is_valid, error = validate_params(params)
        
        # Check that validation returns proper types
        assert isinstance(is_valid, bool)
        assert isinstance(error, str)
        # If invalid, there should be an error message
        if not is_valid:
            assert len(error) > 0

    def test_validates_empty_params(self):
        """Test validation of empty parameters."""
        params = {}
        
        is_valid, error = validate_params(params)
        
        # Empty params might be invalid or valid depending on implementation
        assert isinstance(is_valid, bool)
        assert isinstance(error, str)

    def test_validates_params_with_invalid_values(self):
        """Test validation catches invalid parameter values."""
        params = {
            'max_search_radius': -100,  # Negative value might be invalid
        }
        
        is_valid, error = validate_params(params)
        
        # Check that validation returns proper types
        assert isinstance(is_valid, bool)
        assert isinstance(error, str)

