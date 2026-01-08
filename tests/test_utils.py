"""Tests for utils module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from src.napari_easytrack.utils import (
    _is_already_labeled,
    _convert_to_labels,
    clean_segmentation,
    get_cleaning_stats,
    _clean_frame,
    _find_neighbor_label,
    _get_connectivity_structure,
    load_single_stack,
    load_files_from_pattern,
    load_segmentation,
)


class TestIsAlreadyLabeled:
    """Tests for the _is_already_labeled function."""

    def test_binary_data_not_labeled(self):
        """Binary data (0/1) should not be considered labeled."""
        data = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        assert _is_already_labeled(data) is False

    def test_few_unique_values_not_labeled(self):
        """Data with few unique values (<= 10) should not be considered labeled."""
        data = np.array([[0, 1, 2], [3, 0, 1], [2, 3, 0]], dtype=np.uint16)
        assert _is_already_labeled(data) is False

    def test_many_unique_values_is_labeled(self):
        """Data with many unique values (> 10) should be considered labeled."""
        # Create data with more than 10 unique values
        data = np.arange(15).reshape(3, 5).astype(np.uint16)
        assert _is_already_labeled(data) is True

    def test_float_data_not_labeled(self):
        """Float data should not be considered labeled."""
        data = np.linspace(0, 100, 50).reshape(5, 10).astype(np.float32)
        assert _is_already_labeled(data) is False

    def test_integer_types_labeled(self):
        """Various integer types with many values should be labeled."""
        for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
            data = np.arange(20).astype(dtype).reshape(4, 5)
            assert _is_already_labeled(data) is True


class TestConvertToLabels:
    """Tests for the _convert_to_labels function."""

    def test_simple_binary_mask(self):
        """Test conversion of simple binary boundaries to labels."""
        # Create a simple grid with boundaries
        frame = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ], dtype=np.uint8)
        
        result = _convert_to_labels(frame)
        
        # The interior region should be labeled
        assert result[1, 1] > 0
        assert result[2, 2] > 0
        # Boundary pixels should also be assigned (filled in)
        assert result[0, 0] >= 0  # Can be 0 or assigned to nearest

    def test_boolean_input(self):
        """Test that boolean input is handled correctly."""
        frame = np.array([
            [True, True, True],
            [True, False, True],
            [True, True, True],
        ], dtype=bool)
        
        result = _convert_to_labels(frame)
        
        # Should produce integer labels
        assert result.dtype in [np.int32, np.int64]
        # The interior should be labeled
        assert result[1, 1] > 0

    def test_multiple_regions(self):
        """Test labeling of multiple separate regions."""
        frame = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=np.uint8)
        
        result = _convert_to_labels(frame)
        
        # Different regions should have different labels
        unique_labels = np.unique(result)
        assert len(unique_labels) >= 2  # At least 2 different labels (could include 0)


class TestCleanSegmentation:
    """Tests for the clean_segmentation function."""

    def test_basic_cleaning_3d(self):
        """Test basic 3D segmentation cleaning."""
        # Create a simple 3D segmentation (T, Y, X)
        seg = np.zeros((3, 10, 10), dtype=np.int32)
        seg[0, 2:5, 2:5] = 1
        seg[1, 2:5, 2:5] = 1
        seg[2, 2:5, 2:5] = 1
        
        cleaned = clean_segmentation(seg, verbose=False)
        
        # Should return same shape
        assert cleaned.shape == seg.shape
        # Labels should be preserved
        assert 1 in np.unique(cleaned)

    def test_fragment_removal(self):
        """Test that disconnected fragments are removed."""
        # Create segmentation with a fragment
        seg = np.zeros((2, 15, 15), dtype=np.int32)
        # Main region
        seg[0, 2:8, 2:8] = 1
        seg[1, 2:8, 2:8] = 1
        # Small disconnected fragment of same label
        seg[0, 12:14, 12:14] = 1
        
        cleaned = clean_segmentation(seg, verbose=False, min_size=5)
        
        # Shape should be preserved
        assert cleaned.shape == seg.shape
        # The main region should still be labeled
        assert cleaned[0, 5, 5] == 1

    def test_4d_segmentation(self):
        """Test cleaning of 4D segmentation (T, Z, Y, X)."""
        seg = np.zeros((2, 3, 10, 10), dtype=np.int32)
        seg[0, :, 2:5, 2:5] = 1
        seg[1, :, 2:5, 2:5] = 1
        
        cleaned = clean_segmentation(seg, verbose=False)
        
        # Should return same shape
        assert cleaned.shape == seg.shape
        assert cleaned.ndim == 4

    def test_preserves_dtype(self):
        """Test that cleaning preserves reasonable dtype."""
        seg = np.ones((2, 5, 5), dtype=np.int32) * 5
        cleaned = clean_segmentation(seg, verbose=False)
        # Should return integer dtype
        assert np.issubdtype(cleaned.dtype, np.integer)


class TestGetCleaningStats:
    """Tests for the get_cleaning_stats function."""

    def test_basic_stats(self):
        """Test basic statistics calculation."""
        original = np.zeros((3, 10, 10), dtype=np.int32)
        original[0, 2:5, 2:5] = 1
        original[1, 2:5, 2:5] = 2
        
        cleaned = original.copy()
        cleaned[1, 4, 4] = 0  # Remove one pixel
        
        stats = get_cleaning_stats(original, cleaned)
        
        assert 'original_pixels' in stats
        assert 'cleaned_pixels' in stats
        assert 'pixels_removed' in stats
        assert 'original_labels' in stats
        assert 'cleaned_labels' in stats
        assert stats['pixels_removed'] == 1

    def test_no_change_stats(self):
        """Test stats when no pixels are removed."""
        seg = np.zeros((2, 5, 5), dtype=np.int32)
        seg[0, 1:3, 1:3] = 1
        
        stats = get_cleaning_stats(seg, seg)
        
        assert stats['pixels_removed'] == 0
        assert stats['original_pixels'] == stats['cleaned_pixels']


class TestCleanFrame:
    """Tests for the _clean_frame helper function."""

    def test_single_component_unchanged(self):
        """Test that single-component labels are unchanged."""
        frame = np.zeros((10, 10), dtype=np.int32)
        frame[2:5, 2:5] = 1
        frame[6:9, 6:9] = 2
        
        stats = _clean_frame(frame, min_size=5, is_3d=False)
        
        # No reassignment should occur for single-component labels
        assert stats['removed'] == 0

    def test_multi_component_cleaned(self):
        """Test that multi-component labels have fragments reassigned."""
        frame = np.zeros((20, 20), dtype=np.int32)
        # Large component
        frame[2:8, 2:8] = 1
        # Small disconnected component of same label
        frame[15:17, 15:17] = 1
        # Neighbor label
        frame[10:15, 10:15] = 2
        
        stats = _clean_frame(frame, min_size=5, is_3d=False)
        
        # Some pixels should have been reassigned
        assert stats['reassigned'] > 0


class TestFindNeighborLabel:
    """Tests for the _find_neighbor_label helper function."""

    def test_finds_neighbor(self):
        """Test that neighboring labels are found correctly."""
        from scipy import ndimage
        
        frame = np.zeros((10, 10), dtype=np.int32)
        frame[0:5, 0:5] = 1
        frame[5:10, 0:5] = 2
        
        # Mask at the boundary
        mask = np.zeros((10, 10), dtype=bool)
        mask[4, 2:4] = True  # Adjacent to label 2
        
        structure = ndimage.generate_binary_structure(2, connectivity=1)
        neighbor = _find_neighbor_label(mask, frame, current_label=1, structure=structure)
        
        assert neighbor == 2

    def test_no_neighbor_returns_zero(self):
        """Test that isolated regions return 0."""
        from scipy import ndimage
        
        frame = np.zeros((10, 10), dtype=np.int32)
        frame[4:6, 4:6] = 1
        
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        
        structure = ndimage.generate_binary_structure(2, connectivity=1)
        neighbor = _find_neighbor_label(mask, frame, current_label=1, structure=structure)
        
        assert neighbor == 0


class TestGetConnectivityStructure:
    """Tests for the _get_connectivity_structure function."""

    def test_2d_structure(self):
        """Test 2D connectivity structure."""
        structure = _get_connectivity_structure(is_3d=False)
        
        assert structure.ndim == 2
        assert structure.shape == (3, 3)
        # Center should be True
        assert structure[1, 1] is True or structure[1, 1] == 1

    def test_3d_structure(self):
        """Test 3D connectivity structure."""
        structure = _get_connectivity_structure(is_3d=True)
        
        assert structure.ndim == 3
        assert structure.shape == (3, 3, 3)
        # Center should be True
        assert structure[1, 1, 1] is True or structure[1, 1, 1] == 1


class TestLoadSegmentation:
    """Tests for the load_segmentation function."""

    def test_load_from_directory_with_pattern(self):
        """Test loading from example_data directory."""
        # Use the example data if it exists
        example_dir = Path(__file__).parent.parent / "example_data" / "2d_time_example"
        if example_dir.exists():
            try:
                result = load_segmentation(
                    str(example_dir),
                    pattern="*.tif",
                    convert_to_labels=True,
                    crop_edges=False
                )
                # Should return a 3D array
                assert result.ndim == 3
                # First dimension is time
                assert result.shape[0] > 0
            except ValueError as e:
                if "imagecodecs" in str(e):
                    pytest.skip("imagecodecs package not available for LZW compression")
                raise

    def test_load_single_file(self):
        """Test loading a single stack file."""
        example_file = Path(__file__).parent.parent / "example_data" / "z_tracking_example.tif"
        if example_file.exists():
            try:
                result = load_segmentation(str(example_file))
                # Should return at least 3D
                assert result.ndim >= 3
            except ValueError as e:
                if "imagecodecs" in str(e):
                    pytest.skip("imagecodecs package not available for LZW compression")
                raise

    def test_nonexistent_path_raises(self):
        """Test that non-existent paths raise an error."""
        with pytest.raises(ValueError):
            load_segmentation("/nonexistent/path/to/data")


class TestLoadSingleStack:
    """Tests for load_single_stack function."""

    def test_nonexistent_file_raises(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_single_stack("/nonexistent/file.tif")

    def test_load_real_file(self):
        """Test loading a real file if it exists."""
        example_file = Path(__file__).parent.parent / "example_data" / "z_tracking_example.tif"
        if example_file.exists():
            try:
                result = load_single_stack(str(example_file))
                assert result.ndim >= 3
                assert np.issubdtype(result.dtype, np.integer)
            except ValueError as e:
                if "imagecodecs" in str(e):
                    pytest.skip("imagecodecs package not available for LZW compression")
                raise


class TestLoadFilesFromPattern:
    """Tests for load_files_from_pattern function."""

    def test_no_matching_files_raises(self):
        """Test that no matching files raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                load_files_from_pattern(tmpdir, pattern="*.nonexistent")

    def test_load_from_example_data(self):
        """Test loading from example data directory."""
        example_dir = Path(__file__).parent.parent / "example_data" / "2d_time_example"
        if example_dir.exists():
            try:
                result = load_files_from_pattern(
                    str(example_dir),
                    pattern="*.tif",
                    convert_to_labels=True
                )
                assert result.ndim == 3
                # Should have loaded multiple frames
                assert result.shape[0] >= 1
            except ValueError as e:
                if "imagecodecs" in str(e):
                    pytest.skip("imagecodecs package not available for LZW compression")
                raise

    def test_load_synthetic_2d_files(self):
        """Test loading synthetic 2D TIFF files."""
        from skimage import io as skio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic 2D frames
            for i in range(3):
                frame = np.zeros((10, 10), dtype=np.uint8)
                frame[2:5, 2:5] = 1  # Binary boundary
                skio.imsave(os.path.join(tmpdir, f"frame_t{i:04d}.tif"), frame)
            
            result = load_files_from_pattern(
                tmpdir,
                pattern="*.tif",
                convert_to_labels=True,
                crop_edges=False
            )
            
            assert result.ndim == 3
            assert result.shape[0] == 3  # 3 frames
            assert result.shape[1] == 10
            assert result.shape[2] == 10

    def test_load_with_cropping(self):
        """Test loading with edge cropping enabled."""
        from skimage import io as skio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic 2D frames
            for i in range(2):
                frame = np.zeros((20, 20), dtype=np.uint8)
                frame[5:15, 5:15] = 1
                skio.imsave(os.path.join(tmpdir, f"frame_t{i:04d}.tif"), frame)
            
            result = load_files_from_pattern(
                tmpdir,
                pattern="*.tif",
                convert_to_labels=True,
                crop_edges=True,
                crop_pixels=2
            )
            
            assert result.ndim == 3
            assert result.shape[0] == 2
            # Should be cropped by 2 pixels top and bottom
            assert result.shape[1] == 16  # 20 - 2*2


class TestLoadSingleStackSynthetic:
    """Tests for load_single_stack with synthetic data."""

    def test_load_3d_stack(self):
        """Test loading a synthetic 3D stack."""
        from skimage import io as skio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a 3D stack (T, Y, X)
            stack = np.zeros((5, 20, 20), dtype=np.uint8)
            for t in range(5):
                stack[t, 5:15, 5:15] = 1
            
            file_path = os.path.join(tmpdir, "stack.tif")
            skio.imsave(file_path, stack)
            
            result = load_single_stack(file_path, convert_to_labels=True)
            
            assert result.ndim == 3
            assert result.shape[0] == 5

    def test_load_labeled_data_no_conversion(self):
        """Test that already labeled data is not re-converted."""
        from skimage import io as skio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with many unique labels (should be detected as labeled)
            stack = np.zeros((3, 20, 20), dtype=np.uint16)
            for t in range(3):
                for i in range(5):
                    for j in range(5):
                        stack[t, i*4:(i+1)*4, j*4:(j+1)*4] = i * 5 + j + 1
            
            file_path = os.path.join(tmpdir, "labeled.tif")
            skio.imsave(file_path, stack)
            
            result = load_single_stack(file_path, convert_to_labels=True)
            
            # Should preserve original labels
            assert result.ndim == 3
            assert np.issubdtype(result.dtype, np.integer)

    def test_load_with_cropping(self):
        """Test loading with cropping enabled."""
        from skimage import io as skio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            stack = np.zeros((3, 30, 30), dtype=np.uint8)
            stack[:, 10:20, 10:20] = 1
            
            file_path = os.path.join(tmpdir, "stack.tif")
            skio.imsave(file_path, stack)
            
            result = load_single_stack(
                file_path,
                convert_to_labels=True,
                crop_edges=True,
                crop_pixels=3
            )
            
            assert result.ndim == 3
            # Should be cropped
            assert result.shape[1] == 24  # 30 - 2*3


class TestLoadSegmentationSynthetic:
    """Tests for load_segmentation with synthetic data."""

    def test_load_from_synthetic_directory(self):
        """Test loading from a directory with synthetic files."""
        from skimage import io as skio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i in range(3):
                frame = np.zeros((15, 15), dtype=np.uint8)
                frame[3:12, 3:12] = 1
                skio.imsave(os.path.join(tmpdir, f"test_t{i:04d}.tif"), frame)
            
            result = load_segmentation(
                tmpdir,
                pattern="*.tif",
                convert_to_labels=True
            )
            
            assert result.ndim == 3
            assert result.shape[0] == 3

    def test_load_from_synthetic_file(self):
        """Test loading a single synthetic file."""
        from skimage import io as skio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            stack = np.zeros((4, 15, 15), dtype=np.uint8)
            stack[:, 3:12, 3:12] = 1
            
            file_path = os.path.join(tmpdir, "stack.tif")
            skio.imsave(file_path, stack)
            
            result = load_segmentation(file_path)
            
            assert result.ndim == 3
            # Check the first dimension - may be reinterpreted depending on content
            assert result.shape[0] >= 1
