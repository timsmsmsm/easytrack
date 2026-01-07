"""
Napari widget for btrack parameter preset selection.

Provides a GUI for:
- Selecting preset configurations
- Tweaking parameters with sliders
- Running tracking on segmentation layers
- Monitoring and cancelling tracking operations
- Cleaning segmentation (remove fragments)
- Visualizing tracks as napari tracks layer
- Saving current parameters to JSON config file
"""

from pathlib import Path
import traceback
import json

import napari

from napari import Viewer
import numpy as np
from magicgui.widgets import Container, Label, PushButton, ComboBox, create_widget, CheckBox, FileEdit
from qtpy.QtCore import QTimer

from easytrack.presets import get_presets, load_config_from_json, create_btrack_config_dict
from easytrack.analysis.tracking import TrackingManager
from easytrack.utils import clean_segmentation, get_cleaning_stats


# Parameter descriptions for tooltips
PARAM_DESCRIPTIONS = {
    'dist_thresh': (
        "Distance Threshold (dist_thresh)\n\n"
        "The maximum distance (in pixels or spatial units) that two cell positions "
        "can be apart for the algorithm to consider linking them as part of the same track.\n\n"
        "If two detections are farther apart than this, they won't be considered as "
        "candidates for connecting into a continuous path."
    ),
    'theta_dist': (
        "Border Distance Threshold (theta_dist)\n\n"
        "A distance from the edge of the field of view.\n\n"
        "This is used to decide if a cell track starting or ending near the border "
        "might be a cell entering/exiting the imaging area (rather than appearing/disappearing "
        "for other reasons like cell division or death).\n\n"
        "Tracks starting or ending within this distance from the edge are treated differently."
    ),
    'segmentation_miss_rate': (
        "Segmentation Miss Rate\n\n"
        "The probability that the segmentation algorithm fails to detect a real cell "
        "in any given frame.\n\n"
        "This helps the tracker account for the fact that sometimes real cells might "
        "be missed during segmentation."
    ),
    'lambda_link': (
        "Lambda Link\n\n"
        "A scaling factor that controls how much the distance between cell positions "
        "affects the probability of linking them.\n\n"
        "Smaller values make the algorithm more sensitive to distance (penalizing longer "
        "distances more heavily), while larger values make it more permissive about "
        "linking cells that are farther apart."
    ),
    'prob_not_assign': (
        "Probability Not Assign\n\n"
        "The probability that a detected/segmented object is NOT a real cell and should be ignored.\n\n"
        "This is the OPPOSITE of segmentation_miss_rate:\n"
        "  • segmentation_miss_rate: Real cells that were NOT detected (false negatives)\n"
        "  • prob_not_assign: Detected objects that are NOT real cells (false positives)\n\n"
        "Use this to handle noisy segmentation where some detections might be artifacts, "
        "debris, or segmentation errors.\n\n"
        "Lower values (0.01-0.05): Most detections are real cells.\n"
        "Higher values (0.2-0.5): More detections might be spurious artifacts."
    ),
    'div_hypothesis': (
        "Division Hypothesis\n\n"
        "Enable tracking of cell divisions (P_branch hypothesis).\n\n"
        "When enabled, the tracker will look for situations where one cell track "
        "splits into two daughter cell tracks, allowing reconstruction of cell lineages."
    )
}

class BtrackPresetWidget(Container):
    """Main widget for btrack parameter preset selection."""
    
    def __init__(self, viewer: napari.Viewer):
        super().__init__()  # Initialize Container
        self.viewer = viewer
        self.tracking_manager = TrackingManager()
        self.presets = get_presets()
        
        # Build UI
        self._build_widget()
    
    def _build_widget(self):
        """Build the magicgui widget."""
        
        # Header
        header = Label(value="<h2>Btrack Parameter Presets</h2>")
        
        # Preset selector
        preset_names = list(self.presets.keys())
        self.preset_selector = ComboBox(
            label="Select Preset:",
            choices=preset_names,
            value=preset_names[0]
        )
        self.preset_selector.changed.connect(self._on_preset_changed)
        
        # File picker for custom JSON (initially visible for "Custom JSON")
        self.json_file_picker = FileEdit(
            label="Config JSON File:",
            mode='r',
            filter='*.json'
        )
        self.json_file_picker.changed.connect(self._on_json_file_selected)
        self.json_file_picker.visible = (preset_names[0] == "Custom JSON")
        
        # Description display
        self.description_label = Label(
            value=self.presets[preset_names[0]]["description"]
        )
        
        # Divider
        divider1 = Label(value="<hr>")
        
        # Tweakable parameters section
        tweak_header = Label(value="<b>Fine-tune Parameters:</b>")
        
        # Get initial preset params
        initial_preset = self.presets[preset_names[0]]["config"]
        
        # Create sliders for key parameters with tooltips
        self.dist_thresh_slider = create_widget(
            value=initial_preset.get('dist_thresh', 40),
            annotation=int,
            label="dist_thresh",
            widget_type="FloatSlider",
            options={"min": 0.0, "max": 100.0, "tooltip": PARAM_DESCRIPTIONS['dist_thresh']}
        )
        
        self.theta_dist_slider = create_widget(
            value=initial_preset.get('theta_dist', 20.0),
            annotation=float,
            label="theta_dist",
            widget_type="FloatSlider",
            options={"min": 0.0, "max": 100.0, "tooltip": PARAM_DESCRIPTIONS['theta_dist']}
        )
        
        self.seg_miss_slider = create_widget(
            value=initial_preset.get('segmentation_miss_rate', 0.1),
            annotation=float,
            label="segmentation_miss_rate",
            widget_type="FloatSlider",
            options={"min": 0.0, "max": 1.0, "tooltip": PARAM_DESCRIPTIONS['segmentation_miss_rate']}
        )
        
        self.lambda_link_slider = create_widget(
            value=initial_preset.get('lambda_link', 10.0),
            annotation=int,
            label="lambda_link",
            widget_type="FloatSlider",
            options={"min": 0.0, "max": 100.0, "tooltip": PARAM_DESCRIPTIONS['lambda_link']}
        )
        
        self.prob_not_assign_slider = create_widget(
            value=initial_preset.get('prob_not_assign', 0.1),
            annotation=float,
            label="prob_not_assign",
            widget_type="FloatSlider",
            options={"min": 0.0, "max": 0.5, "tooltip": PARAM_DESCRIPTIONS['prob_not_assign']}
        )
        
        # Division hypothesis toggle (0 or 1)
        self.div_hypothesis_checkbox = CheckBox(
            value=bool(initial_preset.get('div_hypothesis', 1)),
            label="div_hypothesis",
            tooltip=PARAM_DESCRIPTIONS['div_hypothesis']
        )
        
        # Divider
        divider2 = Label(value="<hr>")
        
        # Layer selector
        layer_header = Label(value="<b>Select Segmentation Layer:</b>")
        self.layer_selector = create_widget(
            annotation=napari.layers.Labels,
            label="Segmentation"
        )
        
        # Action buttons
        self.track_button = PushButton(text="🚀 Apply Tracking")
        self.track_button.clicked.connect(self._on_track_clicked)
        
        self.cancel_button = PushButton(text="❌ Cancel Tracking")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.enabled = False  # Initially disabled
        self.cancel_button.tooltip = (
            "Cancel the current tracking operation.\n\n"
            "If a tracking run is taking longer than usual, it's best to cancel it. "
            "This typically means the parameters have been badly selected, and the "
            "result will likely be poor anyway."
        )
        
        self.clean_button = PushButton(text="🧹 Clean Segmentation")
        self.clean_button.clicked.connect(self._on_clean_clicked)
        self.clean_button.tooltip = (
            "Removes disconnected fragments from each label.\n\n"
            "Keeps the largest piece and reassigns smaller bits to "
            "neighboring labels. Useful to run before tracking as btrack "
            "cannot handle disconnected fragments well."
        )
        
        self.save_config_button = PushButton(text="💾 Save Config (JSON)")
        self.save_config_button.clicked.connect(self._on_save_config_clicked)
        self.save_config_button.tooltip = (
            "Save current parameters to a JSON config file.\n\n"
            "The saved file can be loaded later using the 'Custom JSON' preset option."
        )
        
        # Status label
        self.status_label = Label(value="Ready")
        
        # Progress info
        self.progress_label = Label(value="")
        
        # Assemble container
        self.extend([
            header,
            self.preset_selector,
            self.json_file_picker,
            self.description_label,
            divider1,
            tweak_header,
            self.dist_thresh_slider,
            self.theta_dist_slider,
            self.seg_miss_slider,
            self.lambda_link_slider,
            self.prob_not_assign_slider,
            self.div_hypothesis_checkbox,
            divider2,
            layer_header,
            self.layer_selector,
            self.track_button,
            self.cancel_button,
            self.clean_button,
            self.save_config_button,
            self.status_label,
            self.progress_label,
        ])
    
    def _on_preset_changed(self, preset_name: str):
        """Handle preset selection change."""
        preset = self.presets[preset_name]
        
        # Show/hide file picker based on preset selection
        if preset_name == "Custom JSON":
            self.json_file_picker.visible = True
        else:
            self.json_file_picker.visible = False
        
        # Update description
        self.description_label.value = preset["description"]
        
        # Update slider values
        params = preset["config"]
        self.dist_thresh_slider.value = params.get('dist_thresh', 40)
        self.theta_dist_slider.value = params.get('theta_dist', 20.0)
        self.seg_miss_slider.value = params.get('segmentation_miss_rate', 0.1)
        self.lambda_link_slider.value = params.get('lambda_link', 10.0)
        self.prob_not_assign_slider.value = params.get('prob_not_assign', 0.1)
        self.div_hypothesis_checkbox.value = bool(params.get('div_hypothesis', 1))
        
        self.status_label.value = f"Preset loaded: {preset_name}"
    
    def _on_json_file_selected(self, file_path):
        """Handle custom JSON file selection."""
        if not file_path or not Path(file_path).exists():
            return
        
        try:
            # Load parameters from the JSON file
            params = load_config_from_json(file_path)
            
            # Update the Custom JSON preset config
            self.presets["Custom JSON"]["config"] = params
            
            # Update UI with loaded parameters
            self.dist_thresh_slider.value = params.get('dist_thresh', 40)
            self.theta_dist_slider.value = params.get('theta_dist', 20.0)
            self.seg_miss_slider.value = params.get('segmentation_miss_rate', 0.1)
            self.lambda_link_slider.value = params.get('lambda_link', 10.0)
            self.prob_not_assign_slider.value = params.get('prob_not_assign', 0.1)
            self.div_hypothesis_checkbox.value = bool(params.get('div_hypothesis', 1))
            
            self.status_label.value = f"✅ Loaded config from: {Path(file_path).name}"
            
        except Exception as e:
            self.status_label.value = f"❌ Error loading JSON file"
            self.progress_label.value = f"Error: {str(e)}"
            print(f"Error loading JSON file: {e}")
            traceback.print_exc()
    
    def _get_current_params(self):
        """Get current parameter values from UI."""
        preset_name = self.preset_selector.value
        preset = self.presets[preset_name]
        
        # Start with preset config
        params = preset["config"].copy()
        
        # Override with slider values (user tweaks)
        params['dist_thresh'] = self.dist_thresh_slider.value
        params['theta_dist'] = self.theta_dist_slider.value
        params['segmentation_miss_rate'] = self.seg_miss_slider.value
        params['lambda_link'] = self.lambda_link_slider.value
        params['prob_not_assign'] = self.prob_not_assign_slider.value
        params['div_hypothesis'] = 1 if self.div_hypothesis_checkbox.value else 0
        
        return params
    
    def _on_save_config_clicked(self):
        """Handle save config button click."""
        from qtpy.QtWidgets import QFileDialog
        
        # Get current parameters
        params = self._get_current_params()
        
        # Create full btrack config dictionary
        config_dict = create_btrack_config_dict(params)
        
        # Open save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Config File",
            "my_config.json",
            "JSON Files (*.json)"
        )
        
        if not file_path:
            # User cancelled
            return
        
        # Ensure .json extension
        if not file_path.endswith('.json'):
            file_path += '.json'
        
        try:
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.status_label.value = f"✅ Config saved!"
            self.progress_label.value = f"Saved to: {Path(file_path).name}"
            print(f"\n✅ Configuration saved to: {file_path}")
            
        except Exception as e:
            self.status_label.value = "❌ Failed to save config"
            self.progress_label.value = f"Error: {str(e)}"
            print(f"Error saving config: {e}")
            traceback.print_exc()
    
    def _on_clean_clicked(self):
        """Handle clean button click."""
        # Get selected layer
        seg_layer = self.layer_selector.value
        
        if seg_layer is None:
            self.status_label.value = "❌ Please select a segmentation layer"
            return
        
        segmentation = seg_layer.data
        
        # Validate it's 3D or 4D
        if segmentation.ndim not in [3, 4]:
            self.status_label.value = "❌ Segmentation must be 3D (T, Y, X) or 4D (T, Z, Y, X)"
            return
        
        data_type = "2D+T" if segmentation.ndim == 3 else "3D+T"
        self.status_label.value = f"🔄 Cleaning {data_type} segmentation..."
        self.progress_label.value = "Removing fragments and reassigning pixels..."
        
        try:
            # Clean the segmentation (prints progress to console)
            cleaned_seg = clean_segmentation(segmentation, verbose=True)
            
            # Add as new layer
            layer_name = f"{seg_layer.name}_cleaned"
            self.viewer.add_labels(cleaned_seg, name=layer_name)
            
            # Get stats
            stats = get_cleaning_stats(segmentation, cleaned_seg)
            
            # Update status
            self.status_label.value = "✅ Cleaning complete!"
            self.progress_label.value = (
                f"Removed {stats['pixels_removed']} pixels, "
                f"reassigned disconnected fragments"
            )
            
            print(f"\n✅ Created cleaned layer: {layer_name}")
            print(f"   Original labeled pixels: {stats['original_pixels']}")
            print(f"   Cleaned labeled pixels: {stats['cleaned_pixels']}")
            print(f"   Pixels removed: {stats['pixels_removed']}")
            
        except Exception as e:
            self.status_label.value = "❌ Cleaning failed"
            self.progress_label.value = "See console for details"
            print(f"\n❌ Error during cleaning: {e}")
            traceback.print_exc()
    
    def _on_track_clicked(self):
        """Handle track button click."""
        
        # Get selected layer
        seg_layer = self.layer_selector.value
        
        if seg_layer is None:
            self.status_label.value = "❌ Please select a segmentation layer"
            return
        
        segmentation = seg_layer.data
        
        # Validate it's a time series (3D for 2D+T or 4D for 3D+T)
        if segmentation.ndim not in [3, 4]:
            self.status_label.value = "❌ Segmentation must be 3D (T, Y, X) or 4D (T, Z, Y, X)"
            return
        
        if segmentation.ndim == 3:
            print(f"Detected 2D+T data: {segmentation.shape} (T, Y, X)")
        elif segmentation.ndim == 4:
            print(f"Detected 3D+T data: {segmentation.shape} (T, Z, Y, X)")
        
        # Get parameters
        params = self._get_current_params()
        
        # Disable track button, enable cancel button during tracking
        self.track_button.enabled = False
        self.cancel_button.enabled = True
        self.status_label.value = "🔄 Tracking in progress..."
        self.progress_label.value = "Preparing..."
        
        # Start tracking
        self.tracking_manager.start_tracking(
            segmentation=segmentation,
            params=params,
            on_progress=self._on_tracking_progress,
            on_finished=self._on_tracking_finished,
            on_error=self._on_tracking_error
        )
    
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.status_label.value = "⚠️ Cancelling tracking..."
        self.progress_label.value = "Terminating process..."
        self.tracking_manager.cancel_current()
        # Give it a moment, then reset UI
        QTimer.singleShot(1000, self._reset_ui_after_cancel)
    
    def _reset_ui_after_cancel(self):
        """Reset UI after cancellation."""
        self.status_label.value = "❌ Tracking cancelled"
        self.progress_label.value = "Ready for new tracking"
        self.track_button.enabled = True
        self.cancel_button.enabled = False
    
    def _on_tracking_progress(self, message: str):
        """Handle progress updates from monitor."""
        self.progress_label.value = message
    
    def _on_tracking_finished(self, tracked_seg, track_info, napari_data, napari_properties, napari_graph):
        """Handle successful tracking completion."""
        # Add result as new labels layer
        preset_name = self.preset_selector.value
        labels_layer_name = f"Tracked Labels ({preset_name})"
        
        self.viewer.add_labels(tracked_seg, name=labels_layer_name)
        
        # Add tracks layer
        tracks_layer_name = f"Tracks ({preset_name})"
        try:
            print(f"\n{'='*60}")
            print(f"ADDING TRACKS LAYER")
            print(f"{'='*60}")
            print(f"Track data shape: {napari_data.shape}")
            print(f"Track data dtype: {napari_data.dtype}")
            print(f"Track data first 10 rows:")
            print(napari_data[:10])
            print(f"Unique track IDs: {len(np.unique(napari_data[:, 0]))} tracks")
            print(f"Track ID range: {napari_data[:, 0].min():.0f} to {napari_data[:, 0].max():.0f}")
            print(f"Tracked segmentation shape: {tracked_seg.shape}")
            print(f"Properties type: {type(napari_properties)}")
            print(f"Properties keys: {list(napari_properties.keys()) if isinstance(napari_properties, dict) else 'N/A'}")
            print(f"Graph type: {type(napari_graph)}")
            print(f"Graph dtype: {napari_graph.dtype if isinstance(napari_graph, np.ndarray) else 'N/A'}")
            print(f"Graph shape: {napari_graph.shape if isinstance(napari_graph, np.ndarray) else 'N/A'}")
            
            print(f"\nColumn statistics:")
            for i in range(napari_data.shape[1]):
                col_name = ['track_id', 'dim0', 'dim1', 'dim2', 'dim3', 'dim4'][i] if i < 6 else f'col{i}'
                print(f"  {col_name} (col {i}): min={napari_data[:, i].min():.2f}, max={napari_data[:, i].max():.2f}")
            print(f"{'='*60}")
            
            # Handle empty properties dict - napari expects None or a proper dict
            if napari_properties is not None:
                if isinstance(napari_properties, dict) and len(napari_properties) == 0:
                    print(f"Converting empty properties dict to None")
                    napari_properties = None
            
            # Handle graph - napari expects a dict {node_id: [parent_ids]} or None
            # Graph might be saved as 0-d array containing a dict (numpy object array)
            if napari_graph is not None:
                if isinstance(napari_graph, np.ndarray):
                    # Check if it's a 0-d array (scalar) containing an object
                    if napari_graph.shape == ():
                        print(f"Graph is 0-d array, extracting item...")
                        napari_graph = napari_graph.item()  # Extract the actual object
                        print(f"Extracted graph type: {type(napari_graph)}")
                    
                    # Now check what we have
                    if isinstance(napari_graph, np.ndarray):
                        if napari_graph.size == 0:
                            print(f"Graph is empty array, converting to None")
                            napari_graph = None
                        else:
                            # Should have been converted in tracking.py, but just in case
                            print(f"WARNING: Graph is still an array with shape {napari_graph.shape}, converting to dict")
                            graph_dict = {}
                            if napari_graph.ndim == 2 and napari_graph.shape[1] == 2:
                                for child, parent in napari_graph:
                                    child_id = int(child)
                                    parent_id = int(parent)
                                    if child_id not in graph_dict:
                                        graph_dict[child_id] = []
                                    graph_dict[child_id].append(parent_id)
                            napari_graph = graph_dict
                            print(f"Converted graph has {len(napari_graph)} nodes")
                
                # Now napari_graph should be either dict, None, or something else
                if isinstance(napari_graph, dict):
                    if len(napari_graph) == 0:
                        print(f"Converting empty graph dict to None")
                        napari_graph = None
                    else:
                        print(f"Graph is dict with {len(napari_graph)} nodes")
            
            print(f"Final graph type: {type(napari_graph)}")
            print(f"Final graph value: {napari_graph if napari_graph is None or (isinstance(napari_graph, dict) and len(napari_graph) < 5) else f'dict with {len(napari_graph)} entries'}")
            
            self.viewer.add_tracks(
                napari_data, 
                properties=napari_properties,
                graph=napari_graph,
                name=tracks_layer_name,
                tail_width=2,
                tail_length=10,
                head_length=0
            )
            print(f"\n✅ Created tracks layer: {tracks_layer_name}")
            print(f"   Napari interpreted dimensions: {self.viewer.layers[tracks_layer_name].ndim}")
            print(f"   Napari dims order: {self.viewer.dims.order}")
            print(f"   Napari dims range: {self.viewer.dims.range}")
            print(f"{'='*60}\n")
            print(f"   Track data shape: {napari_data.shape}")
            print(f"   Number of unique tracks: {len(np.unique(napari_data[:, 0]))}")
        except Exception as e:
            print(f"\n⚠️ Could not create tracks layer: {e}")
            traceback.print_exc()
        
        # Update status
        self.status_label.value = f"✅ Tracking complete!"
        self.progress_label.value = (
            f"Tracks: {track_info['total_tracks']} total, "
            f"{track_info['tracks_gt_5']} with >5 frames, "
            f"{track_info['tracks_gt_10']} with >10 frames"
        )
        
        # Re-enable track button, disable cancel button
        self.track_button.enabled = True
        self.cancel_button.enabled = False
        
        print("\n" + "="*60)
        print("TRACKING COMPLETE")
        print("="*60)
        print(f"Preset: {preset_name}")
        print(f"Total tracks: {track_info['total_tracks']}")
        print(f"Tracks > 1 frame: {track_info['tracks_gt_1']}")
        print(f"Tracks > 5 frames: {track_info['tracks_gt_5']}")
        print(f"Tracks > 10 frames: {track_info['tracks_gt_10']}")
        print(f"Created layers:")
        print(f"  - {labels_layer_name} (labels)")
        print(f"  - {tracks_layer_name} (tracks)")
        print("="*60)
    
    def _on_tracking_error(self, error_msg: str):
        """Handle tracking errors."""
        self.status_label.value = "❌ Tracking failed"
        self.progress_label.value = "See console for details"
        self.track_button.enabled = True
        self.cancel_button.enabled = False
        
        print("\n" + "="*60)
        print("TRACKING ERROR")
        print("="*60)
        print(error_msg)
        print("="*60)