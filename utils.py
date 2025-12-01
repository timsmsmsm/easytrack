"""
Utilities for Btrack Tracking

Data Loading:
- Load multiple skeleton/segmentation files (time series)
- Load single 3D/4D stacks
- Auto-detection of data format (labeled vs binary)
- Flexible cropping and channel selection

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
                            crop_edges=False, crop_pixels=2):
    """
    Load multiple image files matching a pattern (e.g., time series).
    
    Args:
        directory: Path to directory containing files
        pattern: Glob pattern to match files (e.g., 'CorrSkel_t*.tif')
        convert_to_labels: If True, convert binary/skeleton to labeled regions
        crop_edges: If True, crop top and bottom edges
        crop_pixels: Number of pixels to crop from top/bottom
    
    Returns:
        3D numpy array (T, Y, X)
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
    
    # Check first file for multi-channel
    first_frame = io.imread(files[0])
    if first_frame.ndim == 3:
        print(f"  Multi-channel image detected: {first_frame.shape}")
        channel_vars = [first_frame[i].var() for i in range(first_frame.shape[0])]
        channel_idx = np.argmax(channel_vars)
        print(f"  Using channel {channel_idx} (highest variance)")
    else:
        channel_idx = None
    
    all_frames = []
    for file in files:
        frame = io.imread(file)
        
        # Extract channel if multi-channel
        if channel_idx is not None:
            frame = frame[channel_idx]
        
        # Crop edges if requested
        if crop_edges and crop_pixels > 0:
            frame = frame[crop_pixels:-crop_pixels, :]
        
        # Check if already labeled
        is_labeled = _is_already_labeled(frame)
        
        # Convert to labels if needed and not already labeled
        if convert_to_labels and not is_labeled:
            frame = _convert_to_labels(frame)
        
        all_frames.append(frame)
    
    stacked = np.stack(all_frames, axis=0)
    
    # Ensure proper dtype
    if stacked.dtype == bool:
        stacked = stacked.astype(np.uint16)
    
    print(f"  Loaded shape: {stacked.shape}, dtype: {stacked.dtype}")
    print(f"  Labels per frame (first): {len(np.unique(stacked[0])) - 1}")
    
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
        if data.shape[1] < 10:
            # Likely channels (usually <10 channels)
            print(f"  Detected {data.shape[1]} channels")
            channel_vars = [data[:, i].var() for i in range(data.shape[1])]
            channel_idx = np.argmax(channel_vars)
            print(f"  Using channel {channel_idx} (highest variance)")
            stack = data[:, channel_idx]
        else:
            # Likely (T, Z, Y, X) - take max projection or middle slice
            print(f"  Detected Z-dimension: {data.shape[1]} slices")
            print(f"  Taking maximum projection across Z")
            stack = data.max(axis=1)
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    print(f"  Processed shape: {stack.shape}")
    
    # Crop if requested
    if crop_edges and crop_pixels > 0:
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
        for i in range(stack.shape[0]):
            labeled_stack[i] = _convert_to_labels(stack[i])
        stack = labeled_stack
    
    print(f"  Final shape: {stack.shape}, dtype: {stack.dtype}")
    print(f"  Labels per frame (first): {len(np.unique(stack[0])) - 1}")
    
    return stack


def load_segmentation(path, pattern='*.tif', convert_to_labels=True, 
                     crop_edges=False, crop_pixels=2):
    """
    Smart loader that automatically determines format and loads appropriately.
    
    Args:
        path: Either a directory (for multiple files) or a file path (for single stack)
        pattern: Glob pattern (only used if path is a directory)
        convert_to_labels: Convert binary/skeleton to labeled regions
        crop_edges: Crop top/bottom edges
        crop_pixels: Amount to crop
        
    Returns:
        3D numpy array (T/Z, Y, X)
        
    Examples:
        # Load single stack
        seg = load_segmentation('data/stack.tif')
        
        # Load multiple files
        seg = load_segmentation('data/timelapse/', pattern='frame_*.tif')
    """
    path = Path(path)
    
    if path.is_file():
        # Single file
        return load_single_stack(path, convert_to_labels, crop_edges, crop_pixels)
    elif path.is_dir():
        # Directory with multiple files
        return load_files_from_pattern(path, pattern, convert_to_labels, crop_edges, crop_pixels)
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
    Convert a binary/skeleton frame to labeled connected components.
    """
    if frame.dtype == bool:
        binary = frame
    else:
        # Threshold to binary
        max_val = frame.max()
        mean_val = frame.mean()
        # If mean is low (dark background), threshold high values
        # If mean is high (bright background), threshold low values
        binary = frame > (max_val / 2) if mean_val < (max_val / 2) else frame < (max_val / 2)
    
    # Label connected components
    labeled, n_labels = ndimage.label(binary)
    return labeled


# ============= SEGMENTATION PROCESSING =============

def clean_segmentation(segmentation, verbose=True):
    """
    Clean segmentation by keeping only the largest connected component per label.
    
    For each label in each frame, finds all disconnected fragments and:
    1. Keeps the largest fragment
    2. Reassigns smaller fragments to the neighboring label they touch most
    3. Removes isolated fragments (no neighbors) to background
    
    Uses 4-connectivity (no diagonals) for both component detection and neighbor finding.
    
    Args:
        segmentation: 3D array (T, Y, X) with integer labels
        verbose: If True, print progress information
        
    Returns:
        Cleaned segmentation with same shape and dtype
        
    Example:
        >>> cleaned = clean_segmentation(my_segmentation)
        >>> # Result: each label is a single connected component
    """
    cleaned = segmentation.copy()
    total_reassigned = 0
    
    if verbose:
        print("\n" + "="*60)
        print("CLEANING SEGMENTATION")
        print("="*60)
    
    for t in range(segmentation.shape[0]):
        frame = cleaned[t]
        unique_labels = np.unique(frame)
        unique_labels = unique_labels[unique_labels > 0]  # Skip background
        
        frame_reassigned = 0
        
        for label_id in unique_labels:
            # Get mask for this label
            mask = (frame == label_id)
            
            # Find connected components using 4-connectivity
            structure = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]], dtype=bool)
            labeled_components, num_components = ndimage.label(mask, structure=structure)
            
            if num_components > 1:
                # Multiple components - keep only the largest
                component_sizes = ndimage.sum(mask, labeled_components, range(1, num_components + 1))
                largest_component_idx = np.argmax(component_sizes) + 1
                
                # Process each smaller component
                for component_idx in range(1, num_components + 1):
                    if component_idx == largest_component_idx:
                        continue
                    
                    # Get mask for this small component
                    component_mask = (labeled_components == component_idx)
                    
                    # Find neighboring labels (4-connected)
                    neighbor_counts = {}
                    component_coords = np.argwhere(component_mask)
                    
                    for y, x in component_coords:
                        # Check 4 neighbors (up, down, left, right)
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            
                            # Check bounds
                            if 0 <= ny < frame.shape[0] and 0 <= nx < frame.shape[1]:
                                neighbor_label = frame[ny, nx]
                                
                                # Count neighbors that are different labels
                                if neighbor_label > 0 and neighbor_label != label_id:
                                    neighbor_counts[neighbor_label] = neighbor_counts.get(neighbor_label, 0) + 1
                    
                    # Reassign to most common neighbor, or remove if no neighbors
                    if neighbor_counts:
                        most_common_neighbor = max(neighbor_counts, key=neighbor_counts.get)
                        cleaned[t][component_mask] = most_common_neighbor
                        frame_reassigned += np.sum(component_mask)
                    else:
                        # No neighbors found, set to background
                        cleaned[t][component_mask] = 0
                        frame_reassigned += np.sum(component_mask)
        
        if verbose and frame_reassigned > 0:
            print(f"  Frame {t}: reassigned {frame_reassigned} pixels")
            total_reassigned += frame_reassigned
    
    if verbose:
        print(f"\nTotal pixels reassigned: {total_reassigned}")
        print("="*60 + "\n")
    
    return cleaned


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
