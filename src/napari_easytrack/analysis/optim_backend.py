"""Optimization backend --- :mod:`napari_easytrack.analysis.optim_backend`
=========================================================================
Backend for preparing napari Labels layers for btrack optimization.

This module handles:
- Converting napari Labels layers to CTC format
- Creating temporary directory structures
- Preparing ground truth data (sparse masks, continuous tracks)
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
from .optim_pipeline import compute_scaling_factors

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


def _clean_directory(directory: Path, pattern: str = "*.tif"):
    """Remove any existing files matching pattern from a directory."""
    if not directory.exists():
        return
    for old_file in directory.glob(pattern):
        old_file.unlink()
    # Also clean man_track.txt if present
    txt_file = directory / 'man_track.txt'
    if txt_file.exists():
        txt_file.unlink()


def _clean_ctc_directory(directory: Path):
    """Remove CTC tracking files (man_track*.tif and man_track.txt) from a directory."""
    if not directory.exists():
        return
    for old_file in directory.glob("man_track*.tif"):
        old_file.unlink()
    txt_file = directory / 'man_track.txt'
    if txt_file.exists():
        txt_file.unlink()


def _build_label_timepoints(segmentation):
    """
    Build a dict mapping each label to the timepoints where it appears.
    
    Args:
        segmentation: Array with shape (T, Y, X) or (T, Z, Y, X)
        
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


def _build_track_mapping_split_at_gaps(segmentation):
    """
    Build mapping from original labels to CTC track segments, splitting at gaps.
    
    Works with both 3D (T, Y, X) and 4D (T, Z, Y, X) segmentation arrays.
    
    Args:
        segmentation: Array with shape (T, Y, X) or (T, Z, Y, X) with integer labels
        
    Returns:
        Array with same shape as input, with gaps filled
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
            
            # Get spatial coordinates of the label (works for 2D and 3D spatial)
            mask = (segmentation[prev_t] == label)
            if not np.any(mask):
                continue
            
            coords = np.argwhere(mask)  # Shape (N, spatial_ndim)
            centroid = np.round(np.mean(coords, axis=0)).astype(int)
            spatial_shape = mask.shape
            
            # Try to place placeholder at centroid or nearby
            placed = False
            search_radius = 10
            
            for radius in range(search_radius + 1):
                if placed:
                    break
                    
                if radius == 0:
                    # Try exact centroid
                    candidates = [tuple(centroid)]
                elif len(centroid) == 2:
                    # 2D spatial (Y, X): expand over perimeter of square
                    cy, cx = centroid
                    candidates = []
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if abs(dy) == radius or abs(dx) == radius:  # Only perimeter
                                y, x = cy + dy, cx + dx
                                if 0 <= y < spatial_shape[0] and 0 <= x < spatial_shape[1]:
                                    candidates.append((y, x))
                else:
                    # 3D spatial (Z, Y, X): expand over surface of cube
                    cz, cy, cx = centroid
                    candidates = []
                    for dz in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
                                if abs(dz) == radius or abs(dy) == radius or abs(dx) == radius:
                                    z, y, x = cz + dz, cy + dy, cx + dx
                                    if (0 <= z < spatial_shape[0] and
                                            0 <= y < spatial_shape[1] and
                                            0 <= x < spatial_shape[2]):
                                        candidates.append((z, y, x))
                
                # Try to place at first available position
                for spatial_pos in candidates:
                    full_idx = (gap_t,) + spatial_pos
                    if filled[full_idx] == 0:
                        filled[full_idx] = label
                        gap_details.append(f"Label {label}: t={gap_t} placeholder at {spatial_pos}")
                        placed = True
                        break
            
            if not placed:
                # Last resort: find ANY empty pixel in the frame
                empty_pixels = np.argwhere(filled[gap_t] == 0)
                if len(empty_pixels) > 0:
                    spatial_pos = tuple(empty_pixels[0])
                    full_idx = (gap_t,) + spatial_pos
                    filled[full_idx] = label
                    gap_details.append(f"Label {label}: t={gap_t} placeholder at {spatial_pos} [fallback]")
                    placed = True
            
            if placed:
                total_gaps_filled += 1
            else:
                # This is bad - no space to place placeholder
                print(f"  WARNING: Could not place placeholder for label {label} at t={gap_t}")
                print(f"    Thus, discarding this label for all timepoints")

                # Remove label from all timepoints
                for t in timepoints:
                    filled[t][filled[t] == label] = 0
                break  # No need to try further gaps for this label

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
    
    Example: label 1 present at t=0,1,3,4,9 becomes:
        segment 1: t=0-1, parent=0  (no parent)
        segment 2: t=3-4, parent=1  (edge expected from t=1 to t=3)
        segment 3: t=9-9, parent=2  (edge expected from t=4 to t=9)
    
    Args:
        segmentation: Array with shape (T, Y, X) or (T, Z, Y, X) with integer labels
            (gaps already filled)
        
    Returns:
        List of dicts: [{'new_id': 1, 'original_label': 5, 'start': 0, 
                         'end': 1, 'parent_id': 0}, ...]
    """
    n_timepoints = segmentation.shape[0]
    label_timepoints = _build_label_timepoints(segmentation)
    
    n_labels = len(label_timepoints)
    print(f"  Processing {n_labels} unique labels...")
    print(f"  Segmentation has {n_timepoints} timepoints (indices 0-{n_timepoints-1})")
    
    track_mapping = []
    new_track_id = 1
    total_gaps_bridged = 0
    
    for original_label, timepoints in sorted(label_timepoints.items()):
        timepoints_sorted = sorted(timepoints)
        
        # Split into continuous segments
        segments = []
        seg_start = timepoints_sorted[0]
        prev_t = timepoints_sorted[0]
        
        for t in timepoints_sorted[1:]:
            if t == prev_t + 1:
                # Continuous
                prev_t = t
            else:
                # Gap found — end current segment, start new one
                segments.append((seg_start, prev_t))
                seg_start = t
                prev_t = t
        # Don't forget the last segment
        segments.append((seg_start, prev_t))
        
        # Create track entries with parent linkage
        prev_segment_id = 0  # 0 means no parent in CTC format
        for start, end in segments:
            track_mapping.append({
                'new_id': new_track_id,
                'original_label': original_label,
                'start': start,
                'end': end,
                'parent_id': prev_segment_id
            })
            prev_segment_id = new_track_id
            new_track_id += 1
        
        if len(segments) > 1:
            total_gaps_bridged += len(segments) - 1
    
    n_segments = len(track_mapping)
    print(f"  Created {n_segments} track segments from {n_labels} labels")
    if total_gaps_bridged > 0:
        print(f"  {total_gaps_bridged} gap-bridging edges via parent linkage")
    
    return track_mapping


def _write_ctc_files(segmentation, track_mapping, tra_dir: Path):
    """
    Write man_track.txt and segmentation TIFFs for CTC ground truth.
    
    Each track segment only spans its actual continuous frames, so every
    timepoint listed in man_track.txt has a corresponding mask. Segments
    of the same original label are linked via parent IDs, creating edges
    across temporal gaps that the AOGM metric will evaluate.
    
    Args:
        segmentation: 3D array (T, Y, X) — the ORIGINAL segmentation (with gaps)
        track_mapping: list of track dicts from _build_track_mapping_split_at_gaps
        tra_dir: directory to write files into
    """
    # Write tracking file with parent linkage
    with open(tra_dir / 'man_track.txt', 'w') as f:
        for track in track_mapping:
            f.write(f"{track['new_id']} {track['start']} {track['end']} {track['parent_id']}\n")
    
    # Group tracks by timepoint for efficient relabelling
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
        
        if t in tracks_by_time:
            for track in tracks_by_time[t]:
                mask = (segmentation[t] == track['original_label'])
                relabeled[mask] = track['new_id']
        
        filename = tra_dir / f"man_track{t:04d}.tif"
        io.imsave(filename, relabeled.astype(np.uint16), check_contrast=False)
    
    # Count frames with actual data vs empty
    frames_with_data = sum(1 for t in range(n_timepoints) if np.any(segmentation[t] > 0))
    print(f"  Created {n_timepoints} TIF files ({frames_with_data} with masks, "
          f"{n_timepoints - frames_with_data} empty)")


def prepare_ground_truth_ctc(segmentation, output_dir):
    """
    Convert segmentation to CTC format ground truth.
    
    Splits each label into continuous track segments at temporal gaps,
    linked via parent IDs. This means:
    - Every track listed in man_track.txt has masks at all its timepoints
      (satisfies traccuracy's CTC validation)
    - Edges across gaps are expected via parent linkage, so the AOGM
      penalises btrack for failing to link across gaps
    - No placeholder pixels are inserted — btrack tracks on real data
    
    Args:
        segmentation: Array with shape (T, Y, X) or (T, Z, Y, X) with integer labels
        output_dir: Path to output directory
        
    Returns:
        Tuple of (tra_dir Path, segmentation unchanged)
    """
    output_dir = Path(output_dir)
    tra_dir = output_dir / '01_GT' / 'TRA'
    tra_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean any files from previous runs
    _clean_directory(tra_dir)
    
    # Build track segments split at gaps, linked via parent IDs
    print(f"  Building track segments from original segmentation...")
    track_mapping = _build_track_mapping_split_at_gaps(segmentation)
    
    # Write CTC files using original segmentation (no placeholders)
    _write_ctc_files(segmentation, track_mapping, tra_dir)
    
    return tra_dir, segmentation


def create_dataset_structure(segmentation: np.ndarray, output_dir: Path) -> Path:
    """
    Create dataset structure for btrack to track on.
    
    Uses the original segmentation as-is (with any temporal gaps).
    
    Args:
        segmentation: 3D array (T, Y, X) — original segmentation
        output_dir: Path to output directory
        
    Returns:
        Path to dataset root directory
    """
    output_dir = Path(output_dir)
    dataset_dir = output_dir / '01'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean any files from previous runs
    _clean_directory(dataset_dir)
    
    T = segmentation.shape[0]
    
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
    # Check dimensions: accept 3D (T, Y, X) and 4D (T, Z, Y, X)
    if segmentation.ndim not in (3, 4):
        return False, f"Segmentation must be 3D (T, Y, X) or 4D (T, Z, Y, X), got {segmentation.ndim}D"
    
    T = segmentation.shape[0]
    
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
    
    Both the dataset (what btrack tracks on) and the ground truth TIF masks
    use the original segmentation with gaps intact. The ground truth
    man_track.txt spans each track's full range so that edges across gaps
    are expected — this way the AOGM penalises broken links but not
    missing detections in gap frames.
    
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
    print(f"Shape: {segmentation.shape} ({'T, Z, Y, X' if segmentation.ndim == 4 else 'T, Y, X'})")
    print(f"Unique labels: {len(np.unique(segmentation)) - 1}")
    print(f"Voxel sizes: {voxel_sizes}")
    
    # Create ground truth in CTC format
    # - man_track.txt: tracks span full range (edges across gaps are expected)
    # - TIF masks: only where original segmentation has data (no placeholders)
    print("\nCreating ground truth (CTC format, sparse masks)...")
    gt_dir = work_dir / 'GT'
    tra_dir, _ = prepare_ground_truth_ctc(segmentation, gt_dir)
    
    # Create dataset structure (original segmentation for btrack to track on)
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