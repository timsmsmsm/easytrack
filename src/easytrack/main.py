"""
Launch napari with the btrack preset selector widget.

This is the main entry point for the application.
"""

import sys
import multiprocessing as mp
import napari

from easytrack.widgets.widget import BtrackPresetWidget


def main():
    """Launch napari with the preset selector widget."""
    
    # Set multiprocessing start method
    if sys.platform == 'darwin':  # macOS
        mp.set_start_method('spawn', force=True)
    
    print("\n" + "="*60)
    print("BTRACK PARAMETER PRESET SELECTOR")
    print("="*60)
    print("\nLaunching napari with preset selector widget...")
    print("\nTo test:")
    print("  1. Load a segmentation: File -> Open -> your_segmentation.tif")
    print("  2. Select a preset from the dropdown")
    print("  3. (Optional) Tweak parameters with sliders")
    print("  4. Select your segmentation layer")
    print("  5. Click 'Apply Tracking'")
    print("  6. Use 'Cancel Tracking' button if needed")
    print("\nThe tracked result will appear as a new layer!")
    print("="*60)
    
    # Create viewer
    viewer = napari.Viewer()
    
    # Create and add widget
    widget = BtrackPresetWidget(viewer)
    viewer.window.add_dock_widget(
        widget.container,
        area='right',
        name='Btrack Presets'
    )
    
    # Add example data instructions
    print("\nTIP: If you don't have data loaded:")
    print("  You can test with: File -> Open Sample -> napari builtins -> cells3d")
    print("  (Use the 'nuclei' layer for testing)")
    
    # Run napari
    napari.run()


if __name__ == '__main__':
    main()
