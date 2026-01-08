"""
Simple Example: Cell Tracking with Btrack

This example shows you how to:
1. Load your segmentation data (any format)
2. Open napari with the tracking widget
3. Select a preset and track!

Usage:
    1. Edit the settings below
    2. Run: python simple_example.py
    3. Click "Apply Tracking" in the widget
"""

import napari
from src.easytrack.utils import load_segmentation


# ============= YOUR SETTINGS =============

# Path to your data - can be either:
#   - A single file: 'path/to/stack.tif'
#   - A directory: 'path/to/folder/' (with pattern below)
#DATA_PATH = 'example_data/z_tracking_not_cleaned.tif'
#or
DATA_PATH = 'example_data/2d_time_example'

# Pattern (only needed if DATA_PATH is a directory)
PATTERN = '*.tif'

# Is your data already labeled, or does it need conversion?
# True = binary/skeleton that needs labeling
# False = already has unique labels per cell
CONVERT_TO_LABELS = True

# Do you want to crop edges?
CROP_EDGES = True
CROP_PIXELS = 4  # pixels to crop from edges (to prevent artifacts from skeletons not reaching edges)

# =========================================


if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("LOADING YOUR DATA")
    print("="*60)
    
    # Load data (automatically handles any format!)
    segmentation = load_segmentation(
        DATA_PATH,
        pattern=PATTERN,
        convert_to_labels=CONVERT_TO_LABELS,
        crop_edges=CROP_EDGES,
        crop_pixels=CROP_PIXELS
    )
    
    print(f"\n✅ Successfully loaded!")
    print(f"   Shape: {segmentation.shape}")
    print(f"   Data type: {segmentation.dtype}")
    
    # Open napari
    print("\n" + "="*60)
    print("OPENING NAPARI")
    print("="*60)
    
    viewer = napari.Viewer()
    
    # Add your segmentation
    layer_name = "Segmentation"
    seg_layer = viewer.add_labels(segmentation, name=layer_name)
    
    
    print("\n✅ Ready to track!")
    print("\n📋 Next steps:")
    print("   1. In the widget, select a preset")
    print("      → Try 'Epithelial Cells (Default)' first")
    print("   2. (Optional) Adjust parameters with sliders")
    print("   3. Click '🚀 Apply Tracking'")
    print("   4. Wait for tracking to complete")
    print("   5. A new 'Tracked' layer will appear!")
    print("\n💡 Tip: Cells with the same color = same track")
    print("="*60 + "\n")
    
    napari.run()
