"""
Napari widget for btrack parameter preset selection.

Provides a GUI for:
- Selecting preset configurations
- Tweaking parameters with sliders
- Running tracking on segmentation layers
- Monitoring and cancelling tracking operations
- Cleaning segmentation (remove fragments)
"""

from pathlib import Path
from typing import Optional
import traceback

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container, Label, PushButton, ComboBox, create_widget, CheckBox, FileEdit
from qtpy.QtCore import QTimer

from presets import get_presets, load_config_from_json
from tracking import TrackingManager
from utils import clean_segmentation, get_cleaning_stats


class BtrackPresetWidget:
    """Main widget for btrack parameter preset selection."""
    
    def __init__(self, viewer: napari.Viewer):
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
        
        # Create sliders for key parameters
        self.dist_thresh_slider = create_widget(
            value=initial_preset.get('dist_thresh', 40),
            annotation=int,
            label="Distance Threshold (dist_thresh)",
            widget_type="FloatSlider",
            options={"min": 0.0, "max": 100.0}
        )
        
        self.theta_dist_slider = create_widget(
            value=initial_preset.get('theta_dist', 20.0),
            annotation=float,
            label="Distance Threshold (theta_dist)",
            widget_type="FloatSlider",
            options={"min": 0.0, "max": 100.0}
        )
        
        self.seg_miss_slider = create_widget(
            value=initial_preset.get('segmentation_miss_rate', 0.1),
            annotation=float,
            label="Segmentation Miss Rate",
            widget_type="FloatSlider",
            options={"min": 0.0, "max": 1.0}
        )
        
        self.lambda_link_slider = create_widget(
            value=initial_preset.get('lambda_link', 10.0),
            annotation=int,
            label="Lambda Link",
            widget_type="FloatSlider",
            options={"min": 0.0, "max": 100.0}
        )
        
        self.prob_not_assign_slider = create_widget(
            value=initial_preset.get('prob_not_assign', 0.1),
            annotation=float,
            label="Prob Not Assign",
            widget_type="FloatSlider",
            options={"min": 0.0, "max": 0.5}
        )
        
        # Division hypothesis toggle (0 or 1)
        self.div_hypothesis_checkbox = CheckBox(
            value=bool(initial_preset.get('div_hypothesis', 1)),
            label="Enable Division Hypothesis",
            tooltip="Enable tracking of cell divisions (P_branch hypothesis)"
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
        
        self.clean_button = PushButton(text="🧹 Clean Segmentation")
        self.clean_button.clicked.connect(self._on_clean_clicked)
        
        # Clean button description
        self.clean_description = Label(
            value="<small>Removes disconnected fragments from each label.<br>"
                  "Keeps largest piece, reassigns smaller bits to<br>"
                  "neighboring labels. Useful before tracking.</small>"
        )
        
        # Status label
        self.status_label = Label(value="Ready")
        
        # Progress info
        self.progress_label = Label(value="")
        
        # Assemble container
        self.container = Container(widgets=[
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
            self.clean_description,
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
    
    def _on_clean_clicked(self):
        """Handle clean button click."""
        # Get selected layer
        seg_layer = self.layer_selector.value
        
        if seg_layer is None:
            self.status_label.value = "❌ Please select a segmentation layer"
            return
        
        segmentation = seg_layer.data
        
        # Validate it's 3D
        if segmentation.ndim != 3:
            self.status_label.value = "❌ Segmentation must be 3D (T, Y, X)"
            return
        
        self.status_label.value = "🔄 Cleaning segmentation..."
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
        
        # Validate it's a time series
        if segmentation.ndim != 3:
            self.status_label.value = "❌ Segmentation must be 3D (T, Y, X)"
            return
        
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
    
    def _on_tracking_finished(self, tracked_seg, track_info):
        """Handle successful tracking completion."""
        # Add result as new layer
        preset_name = self.preset_selector.value
        layer_name = f"Tracked ({preset_name})"
        
        self.viewer.add_labels(tracked_seg, name=layer_name)
        
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