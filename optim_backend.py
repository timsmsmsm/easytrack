"""
Backend for preparing napari Labels layers for btrack optimization.

This module handles:
- Converting napari Labels layers to CTC format
- Creating temporary directory structures
- Preparing ground truth data (filling gaps, no divisions)
- Creating dataset objects for optimization
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from skimage import io
import dataclasses
import numpy.typing as npt
import warnings
from traccuracy.loaders import load_ctc_data
from optim_pipeline import compute_scaling_factors

# Suppress low contrast warnings from skimage (normal for label images)
warnings.filterwarnings('ignore', message='.*is a low contrast image.*')


@dataclasses.dataclass
class CellTrackingChallengeDataset:
    """Simple CTC dataset for optimization."""
    name: str 
    path: Path 
    experiment: str
    scale: Tuple[float, float, float] = dataclasses.field(default_factory=lambda: (1.0, 1.0, 1.0))
    _segmentation: Optional[npt.ArrayLike] = None

    @property 
    def segmentation(self) -> npt.ArrayLike:
        if self._segmentation is None:
            filepath = self.path / self.experiment
            tif_files = sorted(filepath.glob("*.tif"))
            if not tif_files:
                raise ValueError(f"No TIF files found in {filepath}")
            frames = [io.imread(f) for f in tif_files]
            self._segmentation = np.stack(frames, axis=0)
        return self._segmentation
    
    @property 
    def volume(self) -> tuple:
        dims = self.segmentation.shape[1:]
        ndim = len(dims)
        scaled_dims = [dims[idx]*self.scale[idx] for idx in range(ndim)]
        return tuple(zip([0]*ndim, scaled_dims))

def _clean_ctc_directory(tra_dir: Path):
    """Remove any existing CTC files from previous runs."""
    for old_file in tra_dir.glob("man_track*.tif"):
        old_file.unlink()
    if (tra_dir / 'man_track.txt').exists():
        (tra_dir / 'man_track.txt').unlink()


def _build_label_timepoints(segmentation):
    """
    Build a dict mapping each label to the timepoints where it appears.
    
    Args:
        segmentation: 3D array (T, Y, X)
        
    Returns:
        Dict[int, List[int]]: {label_id: [t0, t1, t2, ...]}
    """
    label_timepoints = {}
    
    for t in range(segmentation.shape[0]):
        unique_labels = np.unique(segmentation[t])
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            if label not in label_timepoints:
                label_timepoints[label] = []
            label_timepoints[label].append(t)
    
    return label_timepoints


def _fill_gaps_in_segmentation(segmentation):
    """
    Fill temporal gaps in segmentation with placeholder pixels.
    
    For each label with gaps, place a single placeholder pixel at the centroid
    of its last known position. This maintains track continuity for CTC format
    without requiring the full mask.
    
    Args:
        segmentation: 3D array (T, Y, X) with integer labels
        
    Returns:
        3D array with gaps filled
    """
    filled = segmentation.copy()
    label_timepoints = _build_label_timepoints(segmentation)
    
    print(f"  Filling temporal gaps for {len(label_timepoints)} labels...")
    
    total_gaps_filled = 0
    gap_details = []
    
    for label, timepoints in label_timepoints.items():
        if len(timepoints) < 2:
            continue
        
        # Find gaps
        first_t = min(timepoints)
        last_t = max(timepoints)
        all_times = set(range(first_t, last_t + 1))
        gaps = sorted(all_times - set(timepoints))
        
        if not gaps:
            continue
        
        # Fill each gap with a placeholder pixel
        for gap_t in gaps:
            # Find most recent frame where this label appeared
            prev_t = max([t for t in timepoints if t < gap_t])
            
            # Get centroid of previous mask
            mask = (segmentation[prev_t] == label)
            if not np.any(mask):
                continue
            
            y_coords, x_coords = np.where(mask)
            centroid_y = int(np.mean(y_coords))
            centroid_x = int(np.mean(x_coords))
            
            # Try to place placeholder at centroid or nearby
            placed = False
            search_radius = 10
            
            for radius in range(search_radius + 1):
                if placed:
                    break
                    
                if radius == 0:
                    # Try exact centroid
                    candidates = [(centroid_y, centroid_x)]
                else:
                    # Try in expanding square around centroid
                    candidates = []
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if abs(dy) == radius or abs(dx) == radius:  # Only perimeter
                                y, x = centroid_y + dy, centroid_x + dx
                                if 0 <= y < filled.shape[1] and 0 <= x < filled.shape[2]:
                                    candidates.append((y, x))
                
                # Try to place at first available position
                for y, x in candidates:
                    if filled[gap_t, y, x] == 0:
                        filled[gap_t, y, x] = label
                        gap_details.append(f"Label {label}: t={gap_t} placeholder at ({y},{x})")
                        placed = True
                        break
            
            if not placed:
                # Last resort: find ANY empty pixel
                empty_pixels = np.argwhere(filled[gap_t] == 0)
                if len(empty_pixels) > 0:
                    y, x = empty_pixels[0]
                    filled[gap_t, y, x] = label
                    gap_details.append(f"Label {label}: t={gap_t} placeholder at ({y},{x}) [fallback]")
                    placed = True
            
            if placed:
                total_gaps_filled += 1
            else:
                # This is bad - no space to place placeholder
                print(f"  WARNING: Could not place placeholder for label {label} at t={gap_t}")
    
    if total_gaps_filled > 0:
        print(f"  Filled {total_gaps_filled} temporal gaps with placeholder pixels")
        # Print first few details
        for detail in gap_details[:5]:
            print(f"    {detail}")
        if len(gap_details) > 5:
            print(f"    ... and {len(gap_details) - 5} more")
    else:
        print(f"  No gaps found - segmentation is already continuous")
    
    return filled


def _build_track_mapping_continuous(segmentation):
    """
    Build mapping from original labels to new track IDs WITHOUT splitting.
    
    Each original label becomes one continuous track from its first to last appearance.
    Assumes gaps have already been filled.
    
    Args:
        segmentation: 3D array (T, Y, X) with integer labels (gaps already filled)
        
    Returns:
        List of dicts: [{'new_id': 1, 'original_label': 5, 'start': 0, 'end': 10}, ...]
    """
    n_timepoints = segmentation.shape[0]
    label_timepoints = _build_label_timepoints(segmentation)
    
    n_labels = len(label_timepoints)
    print(f"  Processing {n_labels} unique labels...")
    print(f"  Segmentation has {n_timepoints} timepoints (indices 0-{n_timepoints-1})")
    
    track_mapping = []
    new_track_id = 1
    
    for original_label, timepoints in label_timepoints.items():
        # Don't split! Each label is one continuous track
        start = min(timepoints)
        end = max(timepoints)
        
        track_mapping.append({
            'new_id': new_track_id,
            'original_label': original_label,
            'start': start,
            'end': end
        })
        new_track_id += 1
    
    print(f"  Created {len(track_mapping)} continuous tracks (no splits)")
    return track_mapping


def _write_ctc_files(segmentation, track_mapping, tra_dir: Path):
    """Write both man_track.txt and segmentation TIFFs."""
    # Write tracking file
    with open(tra_dir / 'man_track.txt', 'w') as f:
        for track in track_mapping:
            f.write(f"{track['new_id']} {track['start']} {track['end']} 0\n")
    
    # Group tracks by timepoint for efficient relabeling
    tracks_by_time = {}
    for track in track_mapping:
        for t in range(track['start'], track['end'] + 1):
            if t not in tracks_by_time:
                tracks_by_time[t] = []
            tracks_by_time[t].append(track)
    
    # Write TIFFs
    n_timepoints = segmentation.shape[0]
    for t in range(n_timepoints):
        relabeled = np.zeros_like(segmentation[t])
        
        # Only process tracks active at this timepoint
        if t in tracks_by_time:
            for track in tracks_by_time[t]:
                mask = (segmentation[t] == track['original_label'])
                relabeled[mask] = track['new_id']
        
        filename = tra_dir / f"man_track{t:04d}.tif"
        io.imsave(filename, relabeled.astype(np.uint16), check_contrast=False)
    
    print(f"  Created {n_timepoints} TIF files (man_track0000.tif to man_track{n_timepoints-1:04d}.tif)")


def prepare_ground_truth_ctc(segmentation, output_dir):
    """
    Convert segmentation to CTC format ground truth.
    
    Fills temporal gaps and creates continuous tracks for optimization.
    
    Args:
        segmentation: 3D array (T, Y, X) with integer labels
        output_dir: Path to output directory
        
    Returns:
        Path to TRA directory
    """
    output_dir = Path(output_dir)
    tra_dir = output_dir / '01_GT' / 'TRA'
    tra_dir.mkdir(parents=True, exist_ok=True)
    
    _clean_ctc_directory(tra_dir)
    
    # FIRST: Fill any temporal gaps
    filled_segmentation = _fill_gaps_in_segmentation(segmentation)
    
    # THEN: Build continuous track mapping (no splits)
    track_mapping = _build_track_mapping_continuous(filled_segmentation)
    
    # Write CTC files with filled segmentation
    _write_ctc_files(filled_segmentation, track_mapping, tra_dir)
    
    return tra_dir


def create_dataset_structure(segmentation: np.ndarray, output_dir: Path) -> Path:
    """
    Create dataset structure for optimization.
    
    Uses the ORIGINAL segmentation (without gap filling) as the input for tracking.
    
    Args:
        segmentation: 3D numpy array (T, Y, X) with cell labels
        output_dir: Directory to save dataset files
        
    Returns:
        Path to dataset root directory
    """
    output_dir = Path(output_dir)
    dataset_dir = output_dir / '01'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    T = segmentation.shape[0]
    
    # Save frames as t*.tif (original segmentation, not gap-filled)
    for t in range(T):
        io.imsave(
            dataset_dir / f't{t:03d}.tif', 
            segmentation[t].astype(np.uint16),
            check_contrast=False
        )
    
    return output_dir


def validate_segmentation(segmentation: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that segmentation is suitable for optimization.
    
    Args:
        segmentation: Numpy array to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
    """
    # Check dimensions
    if segmentation.ndim != 3:
        return False, f"Segmentation must be 3D (T, Y, X), got {segmentation.ndim}D"
    
    T, Y, X = segmentation.shape
    
    # Check minimum frames
    if T < 2:
        return False, f"Need at least 2 time frames, got {T}"
    
    # Check for labels
    unique_labels = np.unique(segmentation)
    num_labels = len(unique_labels[unique_labels > 0])
    
    if num_labels == 0:
        return False, "No labels found in segmentation (all zeros)"
    
    # Check data type
    if not np.issubdtype(segmentation.dtype, np.integer):
        return False, f"Segmentation must be integer type, got {segmentation.dtype}"
    
    return True, ""


def prepare_layer_for_optimization(
    labels_layer,
    output_dir: Path,
    voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[CellTrackingChallengeDataset, object, Path]:
    """
    Prepare napari Labels layer for btrack optimization.
    
    Creates CTC format structure and returns dataset + gt_data objects.
    Ground truth has gaps filled to create continuous tracks with edges.
    Dataset uses original segmentation (without gap filling).
    
    Args:
        labels_layer: napari Labels layer containing segmentation
        output_dir: Directory to create temporary optimization files
        voxel_sizes: Tuple of (t, y, x) voxel sizes for scaling
        
    Returns:
        tuple: (dataset, gt_data, work_dir_path)
            - dataset: CellTrackingChallengeDataset object
            - gt_data: TrackingGraph loaded from CTC format
            - work_dir_path: Path to temporary working directory
            
    Raises:
        ValueError: If segmentation is invalid
    """
    # Extract segmentation from layer
    segmentation = labels_layer.data
    
    # Validate
    is_valid, error_msg = validate_segmentation(segmentation)
    if not is_valid:
        raise ValueError(f"Invalid segmentation: {error_msg}")
    
    # Create temporary working directory
    output_dir = Path(output_dir)
    work_dir = output_dir / 'optimization_temp'
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR OPTIMIZATION")
    print("=" * 60)
    print(f"Layer: {labels_layer.name}")
    print(f"Shape: {segmentation.shape} (T, Y, X)")
    print(f"Unique labels: {len(np.unique(segmentation)) - 1}")
    print(f"Voxel sizes: {voxel_sizes}")
    
    # Create ground truth in CTC format (fills gaps, creates continuous tracks)
    print("\nCreating ground truth (CTC format with gap filling)...")
    gt_dir = work_dir / 'GT'
    tra_dir = prepare_ground_truth_ctc(segmentation, gt_dir)
    
    # Create dataset structure (original segmentation, no gap filling)
    print("\nCreating dataset structure (original segmentation)...")
    dataset_root = create_dataset_structure(segmentation, work_dir / 'dataset')
    
    # Load ground truth data
    print("\nLoading ground truth data...")
    gt_data = load_ctc_data(
        str(tra_dir), 
        str(tra_dir / 'man_track.txt'), 
        name='ground_truth'
    )
    print(f"  Ground truth: {len(gt_data.nodes())} cell detections")
    print(f"  Ground truth edges: {len(gt_data.edges())} links")
    
    # Create dataset object
    print("\nCreating dataset object...")
    scale = compute_scaling_factors(voxel_sizes)
    dataset = CellTrackingChallengeDataset(
        name='dataset',
        path=dataset_root,
        experiment='01',
        scale=scale
    )
    print(f"  Dataset: {dataset.segmentation.shape}")
    print(f"  Scaling factors: {scale}")
    
    print("=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60 + "\n")
    
    return dataset, gt_data, work_dir