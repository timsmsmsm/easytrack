"""Launch the btrack parameter optimization widget."""
import napari


def main():
    """Launch napari with the optimization widget."""
    viewer = napari.Viewer()
    
    from optim_widget import BtrackOptimizationWidget
    
    optim_widget = BtrackOptimizationWidget(viewer)
    
    viewer.window.add_dock_widget(
        optim_widget.container,
        name="btrack parameter tuning",
        area="right"
    )
    
    print("\n" + "="*60)
    print("BTRACK PARAMETER OPTIMIZATION")
    print("="*60)
    print("Upload a ground truth segmentation layer to optimize")
    print("btrack parameters for your specific dataset.")
    print("="*60 + "\n")
    
    napari.run()


if __name__ == "__main__":
    main()