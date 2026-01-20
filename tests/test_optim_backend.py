"""Tests for optimization backend module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from src.napari_easytrack.analysis.optim_backend import (
    _build_label_timepoints,
    _clean_ctc_directory,
    _fill_gaps_in_segmentation,
    CellTrackingChallengeDataset,
)


class TestBuildLabelTimepoints:
    """Tests for _build_label_timepoints function."""

    def test_simple_continuous_labels(self):
        """Test building timepoints for continuous labels."""
        # Create simple segmentation with 2 labels across 3 timepoints
        seg = np.zeros((3, 10, 10), dtype=np.uint16)
        seg[0, 2:5, 2:5] = 1
        seg[1, 2:5, 2:5] = 1
        seg[2, 2:5, 2:5] = 1
        seg[0, 6:8, 6:8] = 2
        seg[1, 6:8, 6:8] = 2
        
        result = _build_label_timepoints(seg)
        
        assert 1 in result
        assert 2 in result
        assert result[1] == [0, 1, 2]
        assert result[2] == [0, 1]

    def test_labels_with_gaps(self):
        """Test labels that have temporal gaps."""
        seg = np.zeros((5, 10, 10), dtype=np.uint16)
        # Label 1 appears at t=0, t=2, t=4 (with gaps)
        seg[0, 2:5, 2:5] = 1
        seg[2, 2:5, 2:5] = 1
        seg[4, 2:5, 2:5] = 1
        
        result = _build_label_timepoints(seg)
        
        assert result[1] == [0, 2, 4]

    def test_background_ignored(self):
        """Test that background (label 0) is ignored."""
        seg = np.zeros((2, 10, 10), dtype=np.uint16)
        seg[0, 2:5, 2:5] = 1
        
        result = _build_label_timepoints(seg)
        
        assert 0 not in result
        assert 1 in result

    def test_empty_segmentation(self):
        """Test with empty segmentation (all zeros)."""
        seg = np.zeros((3, 10, 10), dtype=np.uint16)
        
        result = _build_label_timepoints(seg)
        
        assert len(result) == 0

    def test_multiple_labels(self):
        """Test with multiple labels appearing at different times."""
        seg = np.zeros((4, 10, 10), dtype=np.uint16)
        seg[0, 2:4, 2:4] = 1
        seg[1, 2:4, 2:4] = 1
        seg[1, 5:7, 5:7] = 2
        seg[2, 5:7, 5:7] = 2
        seg[3, 5:7, 5:7] = 2
        seg[2, 8:10, 8:10] = 3
        
        result = _build_label_timepoints(seg)
        
        assert result[1] == [0, 1]
        assert result[2] == [1, 2, 3]
        assert result[3] == [2]


class TestCleanCtcDirectory:
    """Tests for _clean_ctc_directory function."""

    def test_removes_tracking_files(self):
        """Test that tracking files are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tra_dir = Path(tmpdir)
            
            # Create dummy tracking files
            (tra_dir / "man_track000.tif").touch()
            (tra_dir / "man_track001.tif").touch()
            (tra_dir / "man_track.txt").touch()
            
            # Clean directory
            _clean_ctc_directory(tra_dir)
            
            # Verify files are removed
            assert not (tra_dir / "man_track000.tif").exists()
            assert not (tra_dir / "man_track001.tif").exists()
            assert not (tra_dir / "man_track.txt").exists()

    def test_does_not_remove_other_files(self):
        """Test that non-tracking files are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tra_dir = Path(tmpdir)
            
            # Create tracking and other files
            (tra_dir / "man_track000.tif").touch()
            (tra_dir / "other_file.tif").touch()
            (tra_dir / "data.txt").touch()
            
            _clean_ctc_directory(tra_dir)
            
            # Verify only tracking files are removed
            assert not (tra_dir / "man_track000.tif").exists()
            assert (tra_dir / "other_file.tif").exists()
            assert (tra_dir / "data.txt").exists()

    def test_handles_empty_directory(self):
        """Test cleaning an empty directory doesn't raise errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tra_dir = Path(tmpdir)
            
            # Should not raise any errors
            _clean_ctc_directory(tra_dir)


class TestCellTrackingChallengeDataset:
    """Tests for CellTrackingChallengeDataset class."""

    def test_dataset_creation(self):
        """Test creating a dataset object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = CellTrackingChallengeDataset(
                name="test_dataset",
                path=Path(tmpdir),
                experiment="01",
                scale=(1.0, 1.0, 1.0)
            )
            
            assert dataset.name == "test_dataset"
            assert dataset.experiment == "01"
            assert dataset.scale == (1.0, 1.0, 1.0)

    def test_segmentation_loading(self):
        """Test loading segmentation from TIF files."""
        from skimage import io
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create experiment directory
            exp_dir = Path(tmpdir) / "01"
            exp_dir.mkdir()
            
            # Create dummy segmentation files
            for i in range(3):
                frame = np.zeros((10, 10), dtype=np.uint16)
                frame[2:5, 2:5] = i + 1
                io.imsave(exp_dir / f"mask{i:03d}.tif", frame)
            
            dataset = CellTrackingChallengeDataset(
                name="test_dataset",
                path=Path(tmpdir),
                experiment="01"
            )
            
            seg = dataset.segmentation
            
            assert seg.shape == (3, 10, 10)
            assert seg.dtype == np.uint16

    def test_segmentation_caching(self):
        """Test that segmentation is cached after first load."""
        from skimage import io
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "01"
            exp_dir.mkdir()
            
            # Create a single file
            frame = np.zeros((10, 10), dtype=np.uint16)
            io.imsave(exp_dir / "mask000.tif", frame)
            
            dataset = CellTrackingChallengeDataset(
                name="test_dataset",
                path=Path(tmpdir),
                experiment="01"
            )
            
            # First access
            seg1 = dataset.segmentation
            # Second access should use cache
            seg2 = dataset.segmentation
            
            # Should be the same object (cached)
            assert seg1 is seg2

    def test_volume_property(self):
        """Test volume property calculation."""
        from skimage import io
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "01"
            exp_dir.mkdir()
            
            # Create a file
            frame = np.zeros((20, 30), dtype=np.uint16)
            io.imsave(exp_dir / "mask000.tif", frame)
            
            dataset = CellTrackingChallengeDataset(
                name="test_dataset",
                path=Path(tmpdir),
                experiment="01",
                scale=(2.0, 3.0, 1.0)  # Different scaling
            )
            
            volume = dataset.volume
            
            # Should be ((0, scaled_y), (0, scaled_x))
            assert volume == ((0, 40.0), (0, 90.0))  # 20*2.0, 30*3.0

    def test_no_tif_files_raises(self):
        """Test that missing TIF files raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "01"
            exp_dir.mkdir()
            
            dataset = CellTrackingChallengeDataset(
                name="test_dataset",
                path=Path(tmpdir),
                experiment="01"
            )
            
            with pytest.raises(ValueError, match="No TIF files found"):
                _ = dataset.segmentation


class TestFillGapsInSegmentation:
    """Tests for _fill_gaps_in_segmentation function."""

    def test_no_gaps(self):
        """Test segmentation with no gaps."""
        seg = np.zeros((5, 20, 20), dtype=np.uint16)
        # Label 1 appears continuously
        for t in range(5):
            seg[t, 5:10, 5:10] = 1
        
        result = _fill_gaps_in_segmentation(seg)
        
        # Should be unchanged (no gaps to fill)
        np.testing.assert_array_equal(result, seg)

    def test_simple_gap(self):
        """Test filling a simple gap."""
        seg = np.zeros((5, 20, 20), dtype=np.uint16)
        # Label 1 appears at t=0, t=1, t=3, t=4 (gap at t=2)
        seg[0, 5:10, 5:10] = 1
        seg[1, 5:10, 5:10] = 1
        # t=2 is missing (gap)
        seg[3, 5:10, 5:10] = 1
        seg[4, 5:10, 5:10] = 1
        
        result = _fill_gaps_in_segmentation(seg)
        
        # Gap at t=2 should be filled with at least one pixel of label 1
        assert np.any(result[2] == 1)
        # Other frames should have label 1
        assert np.any(result[0] == 1)
        assert np.any(result[1] == 1)
        assert np.any(result[3] == 1)
        assert np.any(result[4] == 1)

    def test_multiple_gaps(self):
        """Test filling multiple gaps for the same label."""
        seg = np.zeros((6, 20, 20), dtype=np.uint16)
        # Label 1 appears at t=0, t=3, t=5 (gaps at t=1, t=2, t=4)
        seg[0, 5:10, 5:10] = 1
        seg[3, 5:10, 5:10] = 1
        seg[5, 5:10, 5:10] = 1
        
        result = _fill_gaps_in_segmentation(seg)
        
        # All gaps should be filled
        assert np.any(result[1] == 1)  # Gap 1
        assert np.any(result[2] == 1)  # Gap 2
        assert np.any(result[4] == 1)  # Gap 3

    def test_multiple_labels_with_gaps(self):
        """Test filling gaps for multiple labels."""
        seg = np.zeros((5, 20, 20), dtype=np.uint16)
        # Label 1: appears at t=0, t=2, t=4 (gaps at t=1, t=3)
        seg[0, 2:7, 2:7] = 1
        seg[2, 2:7, 2:7] = 1
        seg[4, 2:7, 2:7] = 1
        # Label 2: appears at t=0, t=3 (gaps at t=1, t=2)
        seg[0, 10:15, 10:15] = 2
        seg[3, 10:15, 10:15] = 2
        
        result = _fill_gaps_in_segmentation(seg)
        
        # Gaps for label 1 should be filled
        assert np.any(result[1] == 1)
        assert np.any(result[3] == 1)
        # Gaps for label 2 should be filled
        assert np.any(result[1] == 2)
        assert np.any(result[2] == 2)

    def test_single_timepoint_label_unchanged(self):
        """Test that labels appearing only once are unchanged."""
        seg = np.zeros((3, 20, 20), dtype=np.uint16)
        # Label 1 appears only at t=1
        seg[1, 5:10, 5:10] = 1
        
        result = _fill_gaps_in_segmentation(seg)
        
        # Should be unchanged (no gaps possible with single timepoint)
        np.testing.assert_array_equal(result, seg)

    def test_preserves_existing_labels(self):
        """Test that existing labels are preserved."""
        seg = np.zeros((4, 20, 20), dtype=np.uint16)
        seg[0, 2:7, 2:7] = 1
        seg[1, 2:7, 2:7] = 1
        # Gap at t=2
        seg[3, 2:7, 2:7] = 1
        # Another label present at all times
        seg[0, 10:15, 10:15] = 2
        seg[1, 10:15, 10:15] = 2
        seg[2, 10:15, 10:15] = 2
        seg[3, 10:15, 10:15] = 2
        
        result = _fill_gaps_in_segmentation(seg)
        
        # Label 2 should be preserved at all times
        assert np.any(result[0] == 2)
        assert np.any(result[1] == 2)
        assert np.any(result[2] == 2)
        assert np.any(result[3] == 2)

