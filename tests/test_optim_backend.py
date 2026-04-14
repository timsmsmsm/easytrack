"""Tests for optimization backend module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.napari_easytrack.analysis.optim_backend import (
    _build_label_timepoints,
    _clean_directory,
    _clean_ctc_directory,
    _fill_gaps_in_segmentation,
    _build_track_mapping_split_at_gaps,
    _build_track_mapping_continuous,
    _write_ctc_files,
    prepare_ground_truth_ctc,
    create_dataset_structure,
    prepare_layer_for_optimization,
    validate_segmentation,
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

    def test_4d_no_gaps(self):
        """Test 4D (T, Z, Y, X) segmentation with no gaps is unchanged."""
        seg = np.zeros((4, 5, 20, 20), dtype=np.uint16)
        for t in range(4):
            seg[t, 1:3, 5:10, 5:10] = 1
        
        result = _fill_gaps_in_segmentation(seg)
        
        np.testing.assert_array_equal(result, seg)

    def test_4d_simple_gap(self):
        """Test filling a gap in 4D (T, Z, Y, X) segmentation."""
        seg = np.zeros((5, 5, 20, 20), dtype=np.uint16)
        # Label 1 present at t=0, t=1, t=3, t=4 (gap at t=2)
        seg[0, 1:3, 5:10, 5:10] = 1
        seg[1, 1:3, 5:10, 5:10] = 1
        seg[3, 1:3, 5:10, 5:10] = 1
        seg[4, 1:3, 5:10, 5:10] = 1
        
        result = _fill_gaps_in_segmentation(seg)
        
        # Gap at t=2 should have been filled with at least one pixel of label 1
        assert np.any(result[2] == 1)

    def test_gap_filling_centroid_occupied_2d(self):
        """Test 2D gap filling when centroid position is occupied (forces radius>0 search)."""
        seg = np.zeros((3, 10, 10), dtype=np.uint16)
        # Label 1 at [4:7, 4:7] → centroid (5, 5); present at t=0 and t=2 (gap at t=1)
        seg[0, 4:7, 4:7] = 1
        seg[2, 4:7, 4:7] = 1
        # At gap time t=1, occupy only the centroid position with label 2
        # This forces the perimeter (radius>0) search for label 1
        seg[1, 5, 5] = 2

        result = _fill_gaps_in_segmentation(seg)

        # Label 1 must be placed somewhere at t=1 (at perimeter around centroid)
        assert np.any(result[1] == 1)

    def test_gap_filling_centroid_occupied_4d(self):
        """Test 4D gap filling when centroid is occupied (forces radius>0 3D surface search)."""
        seg = np.zeros((3, 10, 10, 10), dtype=np.uint16)
        # Label 1 at [4:7, 4:7, 4:7] → centroid (5, 5, 5); gap at t=1
        seg[0, 4:7, 4:7, 4:7] = 1
        seg[2, 4:7, 4:7, 4:7] = 1
        # Occupy only the centroid voxel at t=1 with label 2
        seg[1, 5, 5, 5] = 2

        result = _fill_gaps_in_segmentation(seg)

        # Label 1 must be placed somewhere at t=1 via the 3D surface search
        assert np.any(result[1] == 1)

    def test_gap_filling_many_gaps_print(self):
        """Test that gap filling with more than 5 gaps triggers the summary print."""
        seg = np.zeros((14, 10, 10), dtype=np.uint16)
        # Label 1 appears only at even timepoints: 0, 2, 4, 6, 8, 10, 12
        # → gaps at 1, 3, 5, 7, 9, 11 (6 gaps, more than the 5 printed explicitly)
        for t in range(0, 13, 2):
            seg[t, 4:6, 4:6] = 1

        result = _fill_gaps_in_segmentation(seg)

        # All odd timepoints up to the last label appearance should have label 1 filled in
        for t in range(1, 13, 2):
            assert np.any(result[t] == 1)


class TestValidateSegmentation:
    """Tests for validate_segmentation function."""

    def test_valid_segmentation(self):
        """Test validation of a valid segmentation."""
        seg = np.zeros((5, 20, 20), dtype=np.uint16)
        seg[0, 2:7, 2:7] = 1
        seg[1, 2:7, 2:7] = 1
        
        is_valid, error = validate_segmentation(seg)
        
        assert is_valid is True
        assert error == ""

    def test_wrong_dimensions_2d(self):
        """Test that 2D array is rejected."""
        seg = np.zeros((20, 20), dtype=np.uint16)
        
        is_valid, error = validate_segmentation(seg)
        
        assert is_valid is False
        assert "must be 3D" in error

    def test_wrong_dimensions_5d(self):
        """Test that 5D array is rejected."""
        seg = np.zeros((5, 3, 10, 20, 20), dtype=np.uint16)
        seg[0, 0, 2:5, 2:5, 2:5] = 1
        
        is_valid, error = validate_segmentation(seg)
        
        assert is_valid is False
        assert "must be 3D (T, Y, X) or 4D" in error

    def test_wrong_dimensions_4d(self):
        """Test that 4D array (T, Z, Y, X) is now valid (3D+time support)."""
        seg = np.zeros((5, 10, 20, 20), dtype=np.uint16)
        seg[0, 2:5, 2:5, 2:5] = 1
        seg[1, 2:5, 2:5, 2:5] = 1
        
        is_valid, error = validate_segmentation(seg)
        
        assert is_valid is True
        assert error == ""

    def test_4d_too_few_timepoints(self):
        """Test that 4D array with single timepoint is rejected."""
        seg = np.zeros((1, 5, 20, 20), dtype=np.uint16)
        seg[0, 2:4, 5:10, 5:10] = 1
        
        is_valid, error = validate_segmentation(seg)
        
        assert is_valid is False
        assert "at least 2 time frames" in error

    def test_4d_no_labels(self):
        """Test that all-zero 4D segmentation is rejected."""
        seg = np.zeros((5, 5, 20, 20), dtype=np.uint16)
        
        is_valid, error = validate_segmentation(seg)
        
        assert is_valid is False
        assert "No labels found" in error

    def test_too_few_timepoints(self):
        """Test that single timepoint is rejected."""
        seg = np.zeros((1, 20, 20), dtype=np.uint16)
        seg[0, 5:10, 5:10] = 1
        
        is_valid, error = validate_segmentation(seg)
        
        assert is_valid is False
        assert "at least 2 time frames" in error

    def test_no_labels(self):
        """Test that all-zero segmentation is rejected."""
        seg = np.zeros((5, 20, 20), dtype=np.uint16)
        
        is_valid, error = validate_segmentation(seg)
        
        assert is_valid is False
        assert "No labels found" in error

    def test_non_integer_dtype(self):
        """Test that float dtype is rejected."""
        seg = np.zeros((5, 20, 20), dtype=np.float32)
        seg[0, 5:10, 5:10] = 1.0
        
        is_valid, error = validate_segmentation(seg)
        
        assert is_valid is False
        assert "must be integer type" in error


class TestBuildTrackMappingContinuous:
    """Tests for _build_track_mapping_continuous function."""

    def test_simple_continuous_track(self):
        """Test building mapping for continuous tracks."""
        seg = np.zeros((5, 20, 20), dtype=np.uint16)
        # Label 1 appears continuously
        for t in range(5):
            seg[t, 5:10, 5:10] = 1
        
        mapping = _build_track_mapping_continuous(seg)
        
        assert len(mapping) == 1
        assert mapping[0]['new_id'] == 1
        assert mapping[0]['original_label'] == 1
        assert mapping[0]['start'] == 0
        assert mapping[0]['end'] == 4

    def test_multiple_labels(self):
        """Test mapping multiple labels."""
        seg = np.zeros((5, 20, 20), dtype=np.uint16)
        # Label 1
        for t in range(3):
            seg[t, 2:7, 2:7] = 1
        # Label 2
        for t in range(2, 5):
            seg[t, 10:15, 10:15] = 2
        
        mapping = _build_track_mapping_continuous(seg)
        
        assert len(mapping) == 2
        # Find mappings for each label
        label1_map = [m for m in mapping if m['original_label'] == 1][0]
        label2_map = [m for m in mapping if m['original_label'] == 2][0]
        
        assert label1_map['start'] == 0
        assert label1_map['end'] == 2
        assert label2_map['start'] == 2
        assert label2_map['end'] == 4

    def test_labels_with_gaps(self):
        """Test that labels with gaps are treated as continuous from first to last appearance."""
        seg = np.zeros((6, 20, 20), dtype=np.uint16)
        # Label 1 appears at t=0, t=2, t=5 (with gaps)
        seg[0, 5:10, 5:10] = 1
        seg[2, 5:10, 5:10] = 1
        seg[5, 5:10, 5:10] = 1
        
        mapping = _build_track_mapping_continuous(seg)
        
        assert len(mapping) == 1
        assert mapping[0]['start'] == 0
        assert mapping[0]['end'] == 5  # Full span from first to last

    def test_new_track_ids_are_sequential(self):
        """Test that new track IDs are sequential starting from 1."""
        seg = np.zeros((3, 20, 20), dtype=np.uint16)
        seg[0, 2:5, 2:5] = 5  # Original label 5
        seg[1, 8:11, 8:11] = 10  # Original label 10
        seg[2, 14:17, 14:17] = 3  # Original label 3
        
        mapping = _build_track_mapping_continuous(seg)
        
        new_ids = [m['new_id'] for m in mapping]
        # Should be 1, 2, 3 (sequential, starting from 1)
        assert sorted(new_ids) == [1, 2, 3]


class TestWriteCtcFiles:
    """Tests for _write_ctc_files function."""

    def test_writes_track_file(self):
        """Test that man_track.txt is created with correct format."""
        from skimage import io as skio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tra_dir = Path(tmpdir)
            
            # Create simple segmentation
            seg = np.zeros((3, 20, 20), dtype=np.uint16)
            seg[0, 5:10, 5:10] = 1
            seg[1, 5:10, 5:10] = 1
            seg[2, 5:10, 5:10] = 1
            
            # Create mapping
            mapping = [{'new_id': 1, 'original_label': 1, 'start': 0, 'end': 2}]
            
            _write_ctc_files(seg, mapping, tra_dir)
            
            # Check that track file exists
            track_file = tra_dir / 'man_track.txt'
            assert track_file.exists()
            
            # Check content
            with open(track_file, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 1
            assert lines[0].strip() == "1 0 2 0"

    def test_writes_tif_files(self):
        """Test that TIF files are created for each timepoint."""
        from skimage import io as skio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tra_dir = Path(tmpdir)
            
            # Create segmentation
            seg = np.zeros((3, 20, 20), dtype=np.uint16)
            seg[0, 5:10, 5:10] = 1
            seg[1, 5:10, 5:10] = 1
            seg[2, 5:10, 5:10] = 1
            
            mapping = [{'new_id': 1, 'original_label': 1, 'start': 0, 'end': 2}]
            
            _write_ctc_files(seg, mapping, tra_dir)
            
            # Check that TIF files exist
            assert (tra_dir / 'man_track0000.tif').exists()
            assert (tra_dir / 'man_track0001.tif').exists()
            assert (tra_dir / 'man_track0002.tif').exists()

    def test_relabels_segmentation(self):
        """Test that segmentation is relabeled according to mapping."""
        from skimage import io as skio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tra_dir = Path(tmpdir)
            
            # Create segmentation with label 5
            seg = np.zeros((2, 20, 20), dtype=np.uint16)
            seg[0, 5:10, 5:10] = 5
            seg[1, 5:10, 5:10] = 5
            
            # Map label 5 to track ID 1
            mapping = [{'new_id': 1, 'original_label': 5, 'start': 0, 'end': 1}]
            
            _write_ctc_files(seg, mapping, tra_dir)
            
            # Load and check relabeled TIF
            relabeled = skio.imread(tra_dir / 'man_track0000.tif')
            
            # Should have label 1 instead of 5
            assert np.any(relabeled == 1)
            assert not np.any(relabeled == 5)

    def test_handles_multiple_tracks(self):
        """Test writing files with multiple tracks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tra_dir = Path(tmpdir)
            
            # Create segmentation with 2 labels
            seg = np.zeros((3, 20, 20), dtype=np.uint16)
            seg[0, 2:7, 2:7] = 1
            seg[1, 2:7, 2:7] = 1
            seg[1, 10:15, 10:15] = 2
            seg[2, 10:15, 10:15] = 2
            
            mapping = [
                {'new_id': 1, 'original_label': 1, 'start': 0, 'end': 1},
                {'new_id': 2, 'original_label': 2, 'start': 1, 'end': 2}
            ]
            
            _write_ctc_files(seg, mapping, tra_dir)
            
            # Check track file has 2 lines
            with open(tra_dir / 'man_track.txt', 'r') as f:
                lines = f.readlines()
            assert len(lines) == 2


class TestCleanDirectory:
    """Tests for _clean_directory function."""

    def test_handles_nonexistent_directory(self):
        """Test that a non-existent directory is handled gracefully."""
        import tempfile as _tf
        nonexistent = Path(_tf.gettempdir()) / "nonexistent_clean_dir_abc123"
        _clean_directory(nonexistent)  # Should not raise

    def test_removes_tif_files(self):
        """Test that TIF files matching the pattern are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "foo.tif").touch()
            (d / "bar.tif").touch()
            _clean_directory(d)
            assert not (d / "foo.tif").exists()
            assert not (d / "bar.tif").exists()

    def test_removes_man_track_txt(self):
        """Test that man_track.txt is also removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "man_track.txt").touch()
            _clean_directory(d)
            assert not (d / "man_track.txt").exists()

    def test_leaves_non_tif_files_intact(self):
        """Test that non-TIF files (excluding man_track.txt) are not removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "data.csv").touch()
            _clean_directory(d)
            assert (d / "data.csv").exists()


class TestCleanCtcDirectoryExtra:
    """Additional edge-case tests for _clean_ctc_directory."""

    def test_handles_nonexistent_directory(self):
        """Test that a non-existent directory is handled gracefully."""
        import tempfile as _tf
        nonexistent = Path(_tf.gettempdir()) / "nonexistent_ctc_dir_xyz789"
        _clean_ctc_directory(nonexistent)  # Should not raise


class TestFillGapsEdgeCases:
    """Edge-case tests for _fill_gaps_in_segmentation."""

    def test_no_space_for_placeholder_discards_label(self):
        """Test that a label is discarded when no pixel is free at the gap frame."""
        # 3-frame, 1×1 spatial. Label 1 at t=0 and t=2 (gap at t=1).
        # The only pixel at t=1 is occupied by label 2, leaving no room.
        seg = np.zeros((3, 1, 1), dtype=np.uint16)
        seg[0, 0, 0] = 1
        seg[2, 0, 0] = 1
        seg[1, 0, 0] = 2  # Occupies the only available cell

        result = _fill_gaps_in_segmentation(seg)

        # Label 1 must be removed from all timepoints because it couldn't be placed
        assert not np.any(result == 1)
        # Label 2 at t=1 should still be present
        assert result[1, 0, 0] == 2


class TestBuildTrackMappingSplitAtGaps:
    """Tests for _build_track_mapping_split_at_gaps function."""

    def test_continuous_label_produces_one_segment(self):
        """Test that a continuously present label produces a single track segment."""
        seg = np.zeros((4, 10, 10), dtype=np.uint16)
        for t in range(4):
            seg[t, 2:5, 2:5] = 1

        mapping = _build_track_mapping_split_at_gaps(seg)

        assert len(mapping) == 1
        assert mapping[0]['original_label'] == 1
        assert mapping[0]['start'] == 0
        assert mapping[0]['end'] == 3
        assert mapping[0]['parent_id'] == 0

    def test_label_with_one_gap_produces_two_segments(self):
        """Test that a label with one gap is split into two linked segments."""
        seg = np.zeros((5, 10, 10), dtype=np.uint16)
        seg[0, 2:5, 2:5] = 1
        seg[1, 2:5, 2:5] = 1
        # gap at t=2
        seg[3, 2:5, 2:5] = 1
        seg[4, 2:5, 2:5] = 1

        mapping = _build_track_mapping_split_at_gaps(seg)

        assert len(mapping) == 2
        seg1 = mapping[0]
        seg2 = mapping[1]
        assert seg1['start'] == 0 and seg1['end'] == 1
        assert seg2['start'] == 3 and seg2['end'] == 4
        # Second segment must reference the first via parent_id
        assert seg2['parent_id'] == seg1['new_id']
        assert seg1['parent_id'] == 0

    def test_multiple_gaps_produce_multiple_segments(self):
        """Test that two gaps on the same label create three segments."""
        seg = np.zeros((7, 10, 10), dtype=np.uint16)
        seg[0, 2:5, 2:5] = 1        # segment 1: t=0
        # gap at t=1
        seg[2, 2:5, 2:5] = 1        # segment 2: t=2
        # gap at t=3
        seg[4, 2:5, 2:5] = 1        # segment 3: t=4-5
        seg[5, 2:5, 2:5] = 1

        mapping = _build_track_mapping_split_at_gaps(seg)

        assert len(mapping) == 3
        # Verify parent linkage chain: 0 → id1 → id2
        assert mapping[0]['parent_id'] == 0
        assert mapping[1]['parent_id'] == mapping[0]['new_id']
        assert mapping[2]['parent_id'] == mapping[1]['new_id']

    def test_multiple_labels(self):
        """Test that multiple independent labels each get their own segments."""
        seg = np.zeros((3, 10, 10), dtype=np.uint16)
        for t in range(3):
            seg[t, 2:5, 2:5] = 1  # Continuous
        seg[0, 6:9, 6:9] = 2      # Label 2 only at t=0

        mapping = _build_track_mapping_split_at_gaps(seg)

        assert len(mapping) == 2
        orig_labels = {m['original_label'] for m in mapping}
        assert orig_labels == {1, 2}

    def test_new_ids_are_sequential(self):
        """Test that new track IDs are unique sequential integers starting from 1."""
        seg = np.zeros((4, 10, 10), dtype=np.uint16)
        seg[0, 2:5, 2:5] = 3    # Original label 3
        seg[1, 6:9, 6:9] = 7    # Original label 7
        seg[2, 2:5, 2:5] = 10   # Original label 10
        seg[3, 2:5, 2:5] = 10   # label 10 also at t=3

        mapping = _build_track_mapping_split_at_gaps(seg)

        new_ids = sorted(m['new_id'] for m in mapping)
        assert new_ids == list(range(1, len(new_ids) + 1))


class TestPrepareGroundTruthCtc:
    """Tests for prepare_ground_truth_ctc function."""

    def test_creates_expected_directory_and_files(self):
        """Test that the function creates the TRA directory and expected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seg = np.zeros((3, 10, 10), dtype=np.uint16)
            for t in range(3):
                seg[t, 2:5, 2:5] = 1

            tra_dir, returned_seg = prepare_ground_truth_ctc(seg, Path(tmpdir))

            assert tra_dir.exists()
            assert (tra_dir / 'man_track.txt').exists()
            for t in range(3):
                assert (tra_dir / f'man_track{t:04d}.tif').exists()

    def test_returns_segmentation_unchanged(self):
        """Test that the original segmentation is returned unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seg = np.zeros((3, 10, 10), dtype=np.uint16)
            seg[0, 2:5, 2:5] = 1
            seg[2, 2:5, 2:5] = 1   # gap at t=1

            _, returned_seg = prepare_ground_truth_ctc(seg, Path(tmpdir))

            np.testing.assert_array_equal(returned_seg, seg)

    def test_overwrites_previous_run(self):
        """Test that a second call cleanly overwrites files from the first call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seg = np.zeros((2, 10, 10), dtype=np.uint16)
            seg[0, 2:5, 2:5] = 1
            seg[1, 2:5, 2:5] = 1

            prepare_ground_truth_ctc(seg, Path(tmpdir))
            prepare_ground_truth_ctc(seg, Path(tmpdir))  # Second call should not fail

            assert (Path(tmpdir) / '01_GT' / 'TRA' / 'man_track.txt').exists()


class TestCreateDatasetStructure:
    """Tests for create_dataset_structure function."""

    def test_creates_tif_files_for_each_timepoint(self):
        """Test that one TIF per timepoint is created in the 01 subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seg = np.zeros((3, 10, 10), dtype=np.uint16)
            for t in range(3):
                seg[t, 2:5, 2:5] = t + 1

            output_path = create_dataset_structure(seg, Path(tmpdir))

            dataset_dir = Path(tmpdir) / '01'
            assert dataset_dir.exists()
            for t in range(3):
                assert (dataset_dir / f't{t:03d}.tif').exists()
            assert output_path == Path(tmpdir)

    def test_overwrites_previous_files(self):
        """Test that pre-existing TIF files in the dataset dir are replaced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / '01'
            dataset_dir.mkdir()
            (dataset_dir / 't000.tif').touch()  # Pre-existing file

            seg = np.zeros((2, 10, 10), dtype=np.uint16)
            seg[0, 2:5, 2:5] = 1
            seg[1, 2:5, 2:5] = 1

            create_dataset_structure(seg, Path(tmpdir))

            assert (dataset_dir / 't000.tif').exists()
            assert (dataset_dir / 't001.tif').exists()


class TestPrepareLayerForOptimization:
    """Tests for prepare_layer_for_optimization using a lightweight mock layer."""

    class _MockLayer:
        """Minimal stand-in for a napari Labels layer."""
        def __init__(self, data, name="mock"):
            self.data = data
            self.name = name

    def test_valid_layer_returns_dataset_and_gt(self):
        """Test that a valid layer produces a dataset, gt_data and work_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seg = np.zeros((3, 20, 20), dtype=np.uint16)
            for t in range(3):
                seg[t, 5:10, 5:10] = 1

            layer = self._MockLayer(seg)
            dataset, gt_data, work_dir = prepare_layer_for_optimization(
                layer, Path(tmpdir)
            )

            assert dataset is not None
            assert gt_data is not None
            assert work_dir.exists()

    def test_invalid_segmentation_raises_value_error(self):
        """Test that a 2-D (non-temporal) segmentation raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seg = np.zeros((20, 20), dtype=np.uint16)  # 2D — invalid
            layer = self._MockLayer(seg)

            with pytest.raises(ValueError, match="Invalid segmentation"):
                prepare_layer_for_optimization(layer, Path(tmpdir))


