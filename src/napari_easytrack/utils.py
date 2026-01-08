"""
Utilities for btrack Tracking

Data Loading:
- Load multiple skeleton/segmentation files (time series)
- Load single 3D/4D stacks
- Auto-detection of data format (labeled vs binary)
- Flexible cropping and channel selection
- Support for 3D+T data (T, Z, Y, X)

Segmentation Processing:
- Clean segmentation (remove fragments)
- Reassign disconnected components
- Quality control operations

Usage:
    from utils import load_segmentation, clean_segmentation
    
    # Load any format automatically
    seg = load_segmentation('path/to/data.tif')
    
    # Clean segmentation
    cleaned = clean_segmentation(seg)
"""

import re
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage import io


# ============= DATA LOADING =============

def load_files_from_pattern(directory, pattern='*.tif', convert_to_labels=True, 
                            crop_edges=False, crop_pixels=2, keep_z=True):
    """
    Load multiple image files matching a pattern (e.g., time series).
    
    Args:
        directory: Path to directory containing files
        pattern: Glob pattern to match files (e.g., 'CorrSkel_t*.tif')
        convert_to_labels: If True, convert binary/skeleton to labeled regions
        crop_edges: If True, crop top and bottom edges
        crop_pixels: Number of pixels to crop from top/bottom
        keep_z: If True and files are 3D, keep Z dimension (output: T, Z, Y, X)
                If False and files are 3D, take max projection (output: T, Y, X)
    
    Returns:
        3D numpy array (T, Y, X) if files are 2D or keep_z=False
        4D numpy array (T, Z, Y, X) if files are 3D and keep_z=True
    """
    directory = Path(directory)
    
    # Try to sort by number in filename if present
    files = sorted(directory.glob(pattern))
    try:
        files = sorted(files, key=lambda x: int(re.search(r't(\d+)', x.name).group(1)))
    except (AttributeError, ValueError):
        # If no 't123' pattern, just use alphabetical
        pass
    
    if not files:
        raise ValueError(f"No files matching '{pattern}' found in {directory}")
    
    print(f"  Loading {len(files)} files from pattern: {pattern}")
    
    # Check first file to determine format
    first_frame = io.imread(files[0])
    print(f"  First file shape: {first_frame.shape}, dtype: {first_frame.dtype}")
    
    # Determine if this is 2D or 3D data per timepoint
    if first_frame.ndim == 2:
        # Simple 2D frames
        print(f"  Detected: 2D frames (will output T, Y, X)")
        is_3d = False
        channel_idx = None
        
    elif first_frame.ndim == 3:
        # Could be multi-channel 2D, or 3D volumes (Z-stacks)
        # Heuristic: if first dimension is small (<5), likely channels
        # Otherwise, likely Z-slices
        
        if first_frame.shape[0] < 5:
            # Likely channels
            print(f"  Detected: Multi-channel 2D with {first_frame.shape[0]} channels")
            channel_vars = [first_frame[i].var() for i in range(first_frame.shape[0])]
            channel_idx = np.argmax(channel_vars)
            print(f"  Using channel {channel_idx} (highest variance)")
            is_3d = False
        else:
            # Likely Z-stacks
            print(f"  Detected: 3D volumes with {first_frame.shape[0]} Z-slices")
            if keep_z:
                print(f"  Will output 4D: (T, Z, Y, X)")
            else:
                print(f"  Will take max Z-projection, output 3D: (T, Y, X)")
            is_3d = True
            channel_idx = None
    else:
        raise ValueError(f"Unsupported frame dimensionality: {first_frame.ndim}D")
    
    all_frames = []
    for file in files:
        frame = io.imread(file)
        
        # Extract channel if multi-channel 2D
        if channel_idx is not None:
            frame = frame[channel_idx]
        
        # Handle 3D volumes
        if is_3d:
            if not keep_z:
                # Take max projection across Z
                frame = frame.max(axis=0)
        
        # Crop edges if requested (applies to Y dimension)
        if crop_edges and crop_pixels > 0:
            if is_3d and keep_z:
                frame = frame[:, crop_pixels:-crop_pixels, :]
            else:
                frame = frame[crop_pixels:-crop_pixels, :]
        
        # Check if already labeled
        is_labeled = _is_already_labeled(frame)
        
        # Convert to labels if needed and not already labeled
        if convert_to_labels and not is_labeled:
            if is_3d and keep_z:
                # Convert each Z-slice
                labeled_frame = np.zeros(frame.shape, dtype=np.int32)
                for z in range(frame.shape[0]):
                    labeled_frame[z] = _convert_to_labels(frame[z])
                frame = labeled_frame
            else:
                frame = _convert_to_labels(frame)
        
        all_frames.append(frame)
    
    stacked = np.stack(all_frames, axis=0)
    
    # Ensure proper dtype
    if stacked.dtype == bool:
        stacked = stacked.astype(np.uint16)
    
    print(f"  Loaded shape: {stacked.shape}, dtype: {stacked.dtype}")
    if stacked.ndim == 3:
        print(f"  Labels per frame (first): {len(np.unique(stacked[0])) - 1}")
    elif stacked.ndim == 4:
        print(f"  Labels per frame (first Z-slice of first time): {len(np.unique(stacked[0, 0])) - 1}")
    
    return stacked


def load_single_stack(file_path, convert_to_labels=True, crop_edges=False, crop_pixels=2):
    """
    Load a single 3D/4D stack file.
    
    Handles various formats:
    - (T, Y, X) or (Z, Y, X) - simple 3D
    - (T, C, Y, X) - time with channels
    - (T, Z, Y, X) - time with Z-slices
    
    Args:
        file_path: Path to stack file
        convert_to_labels: If True, convert binary/skeleton to labeled regions
                          (automatically skipped if already labeled)
        crop_edges: If True, crop edges
        crop_pixels: Pixels to crop from top/bottom
        
    Returns:
        3D numpy array
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"  Loading single file: {file_path.name}")
    
    data = io.imread(file_path)
    print(f"  Original shape: {data.shape}, dtype: {data.dtype}")
    
    # Handle different dimensionalities
    if data.ndim == 3:
        # Simple (T/Z, Y, X)
        stack = data
        
    elif data.ndim == 4:
        # Could be (T, C, Y, X) or (T, Z, Y, X)
        if data.shape[1] < 4:
            # Likely channels (usually <4 channels)
            print(f"  Detected {data.shape[1]} channels")
            channel_vars = [data[:, i].var() for i in range(data.shape[1])]
            channel_idx = np.argmax(channel_vars)
            print(f"  Using channel {channel_idx} (highest variance)")
            stack = data[:, channel_idx]
        else:
            # Likely (T, Z, Y, X) - keep as 4D for 3D+T tracking
            print(f"  Detected Z-dimension: {data.shape[1]} slices")
            print(f"  Keeping as 4D for 3D+T tracking: (T, Z, Y, X)")
            stack = data
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    print(f"  Processed shape: {stack.shape}")
    
    # Crop if requested
    if crop_edges and crop_pixels > 0:
        if stack.ndim == 4:
            stack = stack[:, :, crop_pixels:-crop_pixels, :]
        else:
            stack = stack[:, crop_pixels:-crop_pixels, :]
        print(f"  After cropping: {stack.shape}")
    
    # Check if already labeled
    is_labeled = _is_already_labeled(stack)
    
    if is_labeled:
        print(f"  Data is already labeled")
        # Ensure proper dtype
        if stack.dtype == bool:
            stack = stack.astype(np.uint16)
        return stack
    
    # Convert to labels if needed
    if convert_to_labels:
        print(f"  Converting binary/skeleton to labels...")
        labeled_stack = np.zeros(stack.shape, dtype=np.int32)
        
        if stack.ndim == 4:
            # 4D: convert each Z-slice in each timepoint
            for t in range(stack.shape[0]):
                for z in range(stack.shape[1]):
                    labeled_stack[t, z] = _convert_to_labels(stack[t, z])
        else:
            # 3D: convert each timepoint/slice
            for i in range(stack.shape[0]):
                labeled_stack[i] = _convert_to_labels(stack[i])
        stack = labeled_stack
    
    print(f"  Final shape: {stack.shape}, dtype: {stack.dtype}")
    if stack.ndim == 3:
        print(f"  Labels per frame (first): {len(np.unique(stack[0])) - 1}")
    elif stack.ndim == 4:
        print(f"  Labels per frame (first Z-slice of first time): {len(np.unique(stack[0, 0])) - 1}")
    
    return stack


def load_segmentation(path, pattern='*.tif', convert_to_labels=True, 
                     crop_edges=False, crop_pixels=2, keep_z=True):
    """
    Smart loader that automatically determines format and loads appropriately.
    
    Args:
        path: Either a directory (for multiple files) or a file path (for single stack)
        pattern: Glob pattern (only used if path is a directory)
        convert_to_labels: Convert binary/skeleton to labeled regions
        crop_edges: Crop top/bottom edges
        crop_pixels: Amount to crop
        keep_z: For 3D data, keep Z dimension (True) or max project (False)
        
    Returns:
        3D numpy array (T/Z, Y, X) or 4D array (T, Z, Y, X) for 3D+T data
        
    Examples:
        # Load single stack
        seg = load_segmentation('data/stack.tif')
        
        # Load multiple files
        seg = load_segmentation('data/timelapse/', pattern='frame_*.tif')
        
        # Load 3D+T data
        seg = load_segmentation('data/3d_timelapse/', keep_z=True)
    """
    path = Path(path)
    
    if path.is_file():
        # Single file
        return load_single_stack(path, convert_to_labels, crop_edges, crop_pixels)
    elif path.is_dir():
        # Directory with multiple files
        return load_files_from_pattern(path, pattern, convert_to_labels, crop_edges, crop_pixels, keep_z)
    else:
        raise ValueError(f"Path does not exist: {path}")


def _is_already_labeled(data):
    """
    Determine if data is already labeled (vs binary/skeleton).
    
    Heuristic: if many unique integer values (>10), likely labeled.
    """
    unique_vals = np.unique(data)
    n_unique = len(unique_vals)
    
    is_integer_type = data.dtype in [np.uint8, np.uint16, np.uint32, 
                                     np.int8, np.int16, np.int32]
    
    return n_unique > 10 and is_integer_type


def _convert_to_labels(frame):
    """
    Convert a skeleton/boundary frame to labeled cell regions.
    Uses morphological dilation to assign boundaries to nearest cells.
    """
    if frame.dtype == bool:
        boundaries = frame
    else:
        boundaries = frame > 0
    
    # Invert and label the regions
    regions = ~boundaries
    labeled, n_labels = ndimage.label(regions)
    
    # Dilate each label into the boundary regions
    # This assigns boundary pixels to the nearest cell
    from scipy.ndimage import distance_transform_edt
    
    # For each boundary pixel, find the nearest labeled region
    distances, indices = distance_transform_edt(labeled == 0, return_indices=True)
    
    # Assign boundary pixels to their nearest cell
    labeled_filled = labeled.copy()
    boundary_mask = boundaries
    labeled_filled[boundary_mask] = labeled[tuple(indices[:, boundary_mask])]
    
    return labeled_filled


# ============= SEGMENTATION PROCESSING =============

def clean_segmentation(segmentation, verbose=True, min_size=5):
    """
    Clean segmentation by:
    1. Keeping only the largest connected component per label
    2. Reassigning smaller fragments and small labels to neighbors
    Args:
        segmentation: 3D array (T, Y, X) or 4D array (T, Z, Y, X) with integer labels
        verbose: If True, print progress information
        min_size: Minimum number of pixels/voxels for a label (default: 5)
    Returns:
        Cleaned segmentation with same shape and dtype
    """
    cleaned = segmentation.copy()
    is_3d = (segmentation.ndim == 4)
    n_timepoints = segmentation.shape[0]
    
    if verbose:
        print("\n" + "="*60)
        print("CLEANING SEGMENTATION")
        print(f"Segmentation is 3D: {is_3d}")
        print("="*60)
    
    total_reassigned = 0
    total_removed = 0
    
    for t in range(n_timepoints):
        # DEBUG: Only for frame 13
        if t == 13:
            structure = ndimage.generate_binary_structure(cleaned[t].ndim, connectivity=1)
            label_31_mask = (cleaned[t] == 31)
            if np.any(label_31_mask):
                labeled_31, num_31 = ndimage.label(label_31_mask, structure=structure)
                sizes_31 = ndimage.sum(label_31_mask, labeled_31, range(1, num_31 + 1)) if num_31 > 0 else []
                print(f"\nBEFORE Frame 13: Label 31 has {num_31} components, sizes: {sizes_31}")
        
        stats = _clean_frame(cleaned[t], min_size, is_3d)
        
        if t == 13:
            structure = ndimage.generate_binary_structure(cleaned[t].ndim, connectivity=1)
            label_31_mask = (cleaned[t] == 31)
            if np.any(label_31_mask):
                labeled_31, num_31 = ndimage.label(label_31_mask, structure=structure)
                sizes_31 = ndimage.sum(label_31_mask, labeled_31, range(1, num_31 + 1)) if num_31 > 0 else []
                print(f"AFTER Frame 13: Label 31 has {num_31} components, sizes: {sizes_31}\n")
        
        if verbose and (stats['reassigned'] > 0 or stats['removed'] > 0):
            timepoint_name = "Timepoint" if is_3d else "Frame"
            unit = "voxels" if is_3d else "pixels"
            print(f"  {timepoint_name} {t}: reassigned {stats['reassigned']} {unit}, "
                  f"removed {stats['removed']} small labels")
        
        total_reassigned += stats['reassigned']
        total_removed += stats['removed']
    
    if verbose:
        unit = "voxels" if is_3d else "pixels"
        print(f"\nTotal {unit} reassigned: {total_reassigned}")
        print(f"Total small labels removed: {total_removed}")
        print("="*60 + "\n")
    
    return cleaned


def _clean_frame(frame, min_size, is_3d):
    """
    Single-pass cleaning: Find and reassign disconnected fragments
    """
    structure = ndimage.generate_binary_structure(frame.ndim, connectivity=1)
    total_stats = {'reassigned': 0, 'removed': 0}
    
    # Get all unique labels
    unique_labels = np.unique(frame)
    unique_labels = unique_labels[unique_labels > 0]
    
    component_info = {}
    next_id = 0
    label_components = {}
    
    for label in unique_labels:
        label_mask = (frame == label)
        labeled, num_comps = ndimage.label(label_mask, structure=structure)
        
        if num_comps <= 1:
            continue  # Skip if only 1 component
        
        label_components[label] = []
        
        for comp_idx in range(1, num_comps + 1):
            comp_mask = (labeled == comp_idx)
            size = int(np.sum(comp_mask))
            
            component_info[next_id] = {
                'original_label': int(label),
                'size': size,
                'mask': comp_mask
            }
            label_components[label].append(next_id)
            next_id += 1
    
    components_to_reassign = []
    
    for label, comp_ids in label_components.items():
        sizes = [component_info[cid]['size'] for cid in comp_ids]
        largest_idx = np.argmax(sizes)
        largest_size = sizes[largest_idx]
        
        if largest_size < min_size:
            components_to_reassign.extend(comp_ids)
            total_stats['removed'] += 1
        else:
            for i, comp_id in enumerate(comp_ids):
                if i != largest_idx:
                    components_to_reassign.append(comp_id)
    
    for comp_id in components_to_reassign:
        info = component_info[comp_id]
        neighbor = _find_neighbor_label(info['mask'], frame, info['original_label'], structure)
        frame[info['mask']] = neighbor
        total_stats['reassigned'] += info['size']
    
    return total_stats


def _find_neighbor_label(mask, frame, current_label, structure):
    """
    Find most common neighboring label using dilation.
    """
    dilated = ndimage.binary_dilation(mask, structure=structure)
    neighbor_mask = dilated & ~mask
    
    neighbor_labels = frame[neighbor_mask]
    neighbor_labels = neighbor_labels[(neighbor_labels > 0) & (neighbor_labels != current_label)]
    
    if len(neighbor_labels) == 0:
        return 0
    
    unique, counts = np.unique(neighbor_labels, return_counts=True)
    return unique[np.argmax(counts)]


def _get_connectivity_structure(is_3d):
    """Get connectivity structure for ndimage.label."""
    if is_3d:
        return np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=bool)
    else:
        return np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=bool)

def get_cleaning_stats(original, cleaned):
    """
    Get statistics about what changed during cleaning.
    
    Args:
        original: Original segmentation
        cleaned: Cleaned segmentation
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'original_pixels': int(np.sum(original > 0)),
        'cleaned_pixels': int(np.sum(cleaned > 0)),
        'pixels_removed': int(np.sum(original > 0) - np.sum(cleaned > 0)),
        'original_labels': len(np.unique(original)) - 1,
        'cleaned_labels': len(np.unique(cleaned)) - 1,
    }
    return stats