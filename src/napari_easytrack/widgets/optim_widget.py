"""
Napari widget for btrack parameter tuning.

Provides a GUI for:
- Selecting ground truth Labels layer
- Configuring optimization parameters
- Running optimization in background
- Monitoring progress
- Applying optimized parameters to run tracking
- Cleaning segmentation
"""

import shutil
import subprocess
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path

import napari
import numpy as np
from magicgui.widgets import (
    Container, Label, PushButton, ComboBox,
    create_widget, CheckBox, FileEdit
)
from qtpy.QtCore import QTimer, Qt

from ..analysis.optim_backend import prepare_layer_for_optimization
from ..analysis.optim_manager import OptimizationManager
from ..analysis.tracking import run_tracking_with_params
from ..utils import clean_segmentation, get_cleaning_stats


class BtrackOptimizationWidget(Container):
    """Main widget for btrack parameter tuning."""
    
    def __init__(self, viewer: napari.Viewer):
        super().__init__()  # Initialize Container
        self.viewer = viewer
        self.optimization_manager = OptimizationManager()
        
        # State
        self.current_dataset = None
        self.current_gt_data = None
        self.current_work_dir = None
        self.best_trials = None
        self.progress_timer = None
        
        # Default output directory
        self.default_output_dir = Path.home() / 'easytrack_optimization'
        self.default_output_dir.mkdir(exist_ok=True)
        
        # Build UI
        self._build_widget()
        
        # Force initial validation check
        QTimer.singleShot(100, self._check_initial_layer)
    
    def _check_initial_layer(self):
        """Check if a layer is already selected on startup."""
        if self.layer_selector.value is not None:
            self._on_layer_changed(self.layer_selector.value)
    
    def _build_widget(self):
        """Build the magicgui widget."""
        
        # ============= INPUT SELECTION =============
        self.layer_selector = create_widget(
            annotation=napari.layers.Labels,
            label="Ground Truth Layer"
        )
        self.layer_selector.changed.connect(self._on_layer_changed)
        
        self.layer_stats_label = Label(value="<center><i>Select a layer</i></center>")
        
        # ============= OPTIMIZATION PARAMETERS =============
        self.study_name_input = create_widget(
            value=self._generate_study_name(),
            annotation=str,
            label="Study Name",
            options={"tooltip": "Name for this optimization study"}
        )
        
        self.n_trials_spinbox = create_widget(
            value=128,
            annotation=int,
            label="Trials",
            widget_type="SpinBox",
            options={
                "min": 1,
                "max": 500,
                "tooltip": "More trials = better optimization but longer runtime"
            }
        )
        
        self.timeout_spinbox = create_widget(
            value=60,
            annotation=int,
            label="Timeout (s)",
            widget_type="SpinBox",
            options={
                "min": 0,
                "max": 300,
                "tooltip": "Maximum time for each trial"
            }
        )
        
        self.timeout_penalty_input = create_widget(
            value="10000",
            annotation=str,
            label="Timeout Penalty",
            options={"tooltip": "Penalty value (AOGM) assigned to trials that timeout or fail. Higher = more discouraged. Set it to a value at least as high as your worst expected AOGM."}
        )
        
        self.sampler_combo = ComboBox(
            label="Sampler",
            choices=['tpe', 'random'],
            value='tpe',
            tooltip="TPE recommended for most cases"
        )
        
        self.parallel_checkbox = CheckBox(
            label="Parallel",
            value=True,
            tooltip="Run trials in parallel (faster)"
        )
        
        self.output_dir_picker = FileEdit(
            label="Output Dir",
            mode='d',
            value=str(self.default_output_dir),
            tooltip="Save results here"
        )
        
        # Voxel Size section toggle
        self.show_advanced_button = PushButton(text="▶ Voxel Size")
        self.show_advanced_button.clicked.connect(self._toggle_advanced)
        self.advanced_visible = False
        
        # Voxel Size section (hidden by default)
        self.voxel_t_spinbox = create_widget(
            value=1.0,
            annotation=float,
            label="Voxel T",
            widget_type="FloatSpinBox",
            options={"min": 0.1, "max": 10.0, "step": 0.1}
        )
        self.voxel_t_spinbox.visible = False
        
        self.voxel_y_spinbox = create_widget(
            value=1.0,
            annotation=float,
            label="Voxel Y",
            widget_type="FloatSpinBox",
            options={"min": 0.1, "max": 10.0, "step": 0.1}
        )
        self.voxel_y_spinbox.visible = False
        
        self.voxel_x_spinbox = create_widget(
            value=1.0,
            annotation=float,
            label="Voxel X",
            widget_type="FloatSpinBox",
            options={"min": 0.1, "max": 10.0, "step": 0.1}
        )
        self.voxel_x_spinbox.visible = False
        
        # ============= STATUS =============
        self.status_label = Label(value="Ready | Best: -- | Time: 0s")
        self.status_label.native.setAlignment(Qt.AlignCenter)
        
        # ============= RESULTS =============
        self.results_info_label = Label(value="")
        self.results_info_label.native.setAlignment(Qt.AlignCenter)
        
        self.best_trials_combo = ComboBox(
            label="Best Trial",
            choices=[],
            visible=False,
            tooltip="Select a trial to apply"
        )
        
        self.apply_tracking_button = PushButton(
            text="🚀 Apply Tracking",
            visible=False
        )
        self.apply_tracking_button.clicked.connect(self._on_apply_tracking_clicked)
        
        self.save_config_button = PushButton(
            text="💾 Save Config",
            visible=False
        )
        self.save_config_button.clicked.connect(self._on_save_config_clicked)
        
        # ============= CONTROL BUTTONS =============
        self.start_button = PushButton(text="Start Optimization")
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.enabled = False
        
        self.cancel_button = PushButton(text="Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.enabled = False
        self.cancel_button.tooltip = (
            "Cancel current optimization. "
            "Note: Cancellation is not fully reliable and may not always succeed."
        )
        self.clean_button = PushButton(text="🧹 Clean Segmentation")
        self.clean_button.clicked.connect(self._on_clean_clicked)
        self.clean_button.tooltip = (
            "Removes disconnected fragments from each label.\n\n"
            "Keeps the largest piece and reassigns smaller bits to "
            "neighboring labels. Useful to run before optimization as btrack "
            "cannot handle disconnected fragments well."
        )
        
        self.view_dashboard_button = PushButton(text="📊 View Dashboard")
        self.view_dashboard_button.clicked.connect(self._on_view_dashboard_clicked)
        self.view_dashboard_button.tooltip = "Open Optuna dashboard in browser"
        
        self.extend([
            self.layer_selector,
            self.layer_stats_label,
            self.study_name_input,
            self.n_trials_spinbox,
            self.timeout_spinbox,
            self.timeout_penalty_input,
            self.sampler_combo,
            self.parallel_checkbox,
            self.output_dir_picker,
            self.show_advanced_button,
            self.voxel_t_spinbox,
            self.voxel_y_spinbox,
            self.voxel_x_spinbox,
            self.status_label,
            self.results_info_label,
            self.best_trials_combo,
            self.apply_tracking_button,
            self.save_config_button,
            self.start_button,
            self.cancel_button,
            self.clean_button,
            self.view_dashboard_button,
        ])
    
    def _toggle_advanced(self):
        """Toggle visibility of advanced settings."""
        self.advanced_visible = not self.advanced_visible
        
        if self.advanced_visible:
            self.show_advanced_button.text = "▼ Voxel Size"
            self.voxel_t_spinbox.visible = True
            self.voxel_y_spinbox.visible = True
            self.voxel_x_spinbox.visible = True
        else:
            self.show_advanced_button.text = "▶ Voxel Size"
            self.voxel_t_spinbox.visible = False
            self.voxel_y_spinbox.visible = False
            self.voxel_x_spinbox.visible = False
    
    def _generate_study_name(self) -> str:
        """Generate a default study name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"optimization_{timestamp}"
    
    def _on_layer_changed(self, layer):
        """Handle layer selection change."""
        if layer is None:
            self.layer_stats_label.value = "<center><i>No layer selected</i></center>"
            self.start_button.enabled = False
            return
        
        try:
            # Get segmentation
            segmentation = layer.data
            
            # Validate
            if segmentation.ndim != 3:
                self.layer_stats_label.value = (
                    f"<center><font color='red'>❌ Must be 3D (T,Y,X), got {segmentation.ndim}D</font></center>"
                )
                self.start_button.enabled = False
                return
            
            T, Y, X = segmentation.shape
            unique_labels = np.unique(segmentation)
            num_labels = len(unique_labels[unique_labels > 0])
            
            if num_labels == 0:
                self.layer_stats_label.value = "<center><font color='red'>❌ No labels found</font></center>"
                self.start_button.enabled = False
                return
            
            if T < 2:
                self.layer_stats_label.value = (
                    f"<center><font color='red'>❌ Need ≥2 frames, got {T}</font></center>"
                )
                self.start_button.enabled = False
                return
            
            # Valid layer
            self.layer_stats_label.value = (
                f"<center><font color='green'>✓ {segmentation.shape} | {T} frames | {num_labels} labels</font></center>"
            )
            self.start_button.enabled = True
            
        except Exception as e:
            self.layer_stats_label.value = f"<center><font color='red'>❌ Error: {str(e)}</font></center>"
            self.start_button.enabled = False
    
    def _on_start_clicked(self):
        """Handle start optimization button click."""
        
        # Get selected layer
        layer = self.layer_selector.value
        if layer is None:
            self.status_label.value = "<font color='red'>❌ No layer selected</font>"
            return
        
        # Get parameters
        study_name = self.study_name_input.value
        n_trials = self.n_trials_spinbox.value
        timeout = self.timeout_spinbox.value
        
        # Validate and get timeout penalty
        try:
            timeout_penalty = float(self.timeout_penalty_input.value)
        except ValueError:
            self.status_label.value = "<font color='red'>❌ Invalid timeout penalty value</font>"
            return
        
        sampler = self.sampler_combo.value
        use_parallel = self.parallel_checkbox.value
        output_dir = Path(self.output_dir_picker.value)
        voxel_sizes = (
            self.voxel_t_spinbox.value,
            self.voxel_y_spinbox.value,
            self.voxel_x_spinbox.value
        )
        
        # Validate output directory
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.status_label.value = f"<font color='red'>❌ Cannot create output dir: {e}</font>"
            return
        
        # Update UI state
        self.start_button.enabled = False
        self.cancel_button.enabled = True
        self.layer_selector.enabled = False
        self.status_label.value = "🔄 Preparing data..."
        
        # Hide results from previous run
        self._hide_results_section()
        
        try:
            # Prepare data
            print("\n" + "="*60)
            print("PREPARING DATA")
            print("="*60)
            
            self.current_dataset, self.current_gt_data, self.current_work_dir = \
                prepare_layer_for_optimization(layer, output_dir, voxel_sizes)
            
            # Start optimization
            self.status_label.value = "🚀 Starting optimization..."
            
            self.optimization_manager.start_optimization(
                dataset=self.current_dataset,
                gt_data=self.current_gt_data,
                study_name=study_name,
                n_trials=n_trials,
                timeout=timeout,
                timeout_penalty=timeout_penalty,
                sampler=sampler,
                use_parallel_backend=use_parallel,
                on_progress=None,  # We'll poll instead
                on_finished=self._on_optimization_finished,
                on_error=self._on_optimization_error
            )
            
            # Start progress monitoring
            self._start_progress_monitoring()
            
        except Exception as e:
            self.status_label.value = f"<font color='red'>❌ Failed to start</font>"
            print(f"\n❌ Error starting optimization: {e}")
            traceback.print_exc()
            self._reset_ui_state()
    
    def _start_progress_monitoring(self):
        """Start timer to poll optimization progress."""
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self._update_progress)
        self.progress_timer.start(1000)  # Poll every 1 second
    
    def _update_progress(self):
        """Update progress display by polling optimization manager."""
        progress = self.optimization_manager.get_progress()
        
        if progress is not None:
            trial_num, best_aogm, elapsed = progress
            n_trials = self.n_trials_spinbox.value
            
            self.status_label.value = (
                f"🔄 Trial {trial_num}/{n_trials} | Best: {best_aogm:.2f} | Time: {elapsed}s"
            )
        
        # Check if complete
        if self.optimization_manager.is_complete():
            self.progress_timer.stop()
    
    def _on_optimization_finished(self, study):
        """Handle successful optimization completion."""
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        
        # Stop progress timer FIRST
        if self.progress_timer and self.progress_timer.isActive():
            self.progress_timer.stop()
            self.progress_timer = None
        
        # Get final progress
        progress = self.optimization_manager.get_progress()
        if progress:
            _, best_aogm, elapsed = progress
            self.status_label.value = f"<font color='green'>✅ Complete | Best: {best_aogm:.2f} | Time: {elapsed}s</font>"
        
        # Get best trials
        self.best_trials = self.optimization_manager.get_best_trials(max_trials=15)
        
        if self.best_trials:
            print(f"\nFound {len(self.best_trials)} best trial(s):")
            for trial in self.best_trials:
                print(f"  Trial {trial['number']}: AOGM = {trial['aogm']:.2f}")
            
            # Populate dropdown
            choices = []
            for trial in self.best_trials:
                label = f"Trial {trial['number']}: AOGM {trial['aogm']:.2f}"
                choices.append(label)
            
            self.best_trials_combo.choices = choices
            self.results_info_label.value = (
                f"<font color='green'><b>{len(self.best_trials)} trials completed</b></font>"
            )
            
            # Show results section
            self._show_results_section()
        else:
            self.results_info_label.value = (
                "<font color='orange'>⚠️ No completed trials</font>"
            )
        
        # Clean up temporary files
        self._cleanup_temp_files()
        
        # Reset UI
        self._reset_ui_state()
        
        print("="*60 + "\n")
    
    def _on_optimization_error(self, error_msg: str):
        """Handle optimization errors."""
        print(f"\n❌ Optimization error: {error_msg}")
        
        # Stop progress timer FIRST
        if self.progress_timer and self.progress_timer.isActive():
            self.progress_timer.stop()
            self.progress_timer = None
        
        self.status_label.value = "<font color='red'>❌ Failed</font>"
        self.results_info_label.value = (
            f"<font color='red'>{error_msg}</font><br>See console for details"
        )
        
        # Clean up
        self._cleanup_temp_files()
        self._reset_ui_state()
    
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.status_label.value = "⚠️ Cancelling..."
        self.optimization_manager.cancel_current()
        
        # Stop timer FIRST
        if self.progress_timer and self.progress_timer.isActive():
            self.progress_timer.stop()
            self.progress_timer = None
        
        # Clean up and reset
        self._cleanup_temp_files()
        
        # Give it a moment then reset UI
        QTimer.singleShot(1000, self._on_cancel_complete)
    
    def _on_cancel_complete(self):
        """Handle completion of cancellation."""
        self.status_label.value = "<font color='orange'>⚠️ Cancelled</font>"
        self.results_info_label.value = ""
        self._reset_ui_state()
    
    def _on_clean_clicked(self):
        """Handle clean button click."""
        # Get selected layer
        layer = self.layer_selector.value
        
        if layer is None:
            self.status_label.value = "❌ Please select a layer first"
            return
        
        segmentation = layer.data
        
        # Validate it's 3D
        if segmentation.ndim != 3:
            self.status_label.value = f"❌ Must be 3D (T,Y,X), got {segmentation.ndim}D"
            return
        
        self.status_label.value = "🔄 Cleaning segmentation..."
        
        try:
            # Clean the segmentation (prints progress to console)
            cleaned_seg = clean_segmentation(segmentation, verbose=True)
            
            # Add as new layer
            layer_name = f"{layer.name}_cleaned"
            self.viewer.add_labels(cleaned_seg, name=layer_name)
            
            # Get stats
            stats = get_cleaning_stats(segmentation, cleaned_seg)
            
            # Update status
            self.status_label.value = "✅ Cleaning complete!"
            self.results_info_label.value = (
                f"Removed {stats['pixels_removed']} pixels, "
                f"reassigned disconnected fragments"
            )
            
            print(f"\n✅ Created cleaned layer: {layer_name}")
            print(f"   Original labeled pixels: {stats['original_pixels']}")
            print(f"   Cleaned labeled pixels: {stats['cleaned_pixels']}")
            print(f"   Pixels removed: {stats['pixels_removed']}")
            
        except Exception as e:
            self.status_label.value = "❌ Cleaning failed"
            self.results_info_label.value = f"<font color='red'>Error: {str(e)}</font>"
            print(f"\n❌ Error during cleaning: {e}")
            traceback.print_exc()
    
    def _on_apply_tracking_clicked(self):
        """Handle apply tracking button click."""
        
        # Get selected trial
        if not self.best_trials_combo.value:
            n_trials = len(self.best_trials) if self.best_trials else 0
            self.results_info_label.value = (
                f"<font color='green'><b>{n_trials} trials available</b></font><br>"
                f"<font color='red'>❌ Select a trial first</font>"
            )
            return
        
        # Store current dropdown choices to restore later
        current_choices = list(self.best_trials_combo.choices)
        current_selection = self.best_trials_combo.value
        
        # Parse trial number
        trial_label = self.best_trials_combo.value
        trial_num = int(trial_label.split(":")[0].replace("Trial ", ""))
        
        # Find matching trial
        selected_trial = None
        for trial in self.best_trials:
            if trial['number'] == trial_num:
                selected_trial = trial
                break
        
        if selected_trial is None:
            n_trials = len(self.best_trials) if self.best_trials else 0
            self.results_info_label.value = (
                f"<font color='green'><b>{n_trials} trials available</b></font><br>"
                f"<font color='red'>❌ Trial not found</font>"
            )
            self.best_trials_combo.choices = current_choices
            return
        
        # Get original layer
        layer = self.layer_selector.value
        if layer is None:
            n_trials = len(self.best_trials) if self.best_trials else 0
            self.results_info_label.value = (
                f"<font color='green'><b>{n_trials} trials available</b></font><br>"
                f"<font color='red'>❌ Layer not found</font>"
            )
            self.best_trials_combo.choices = current_choices
            return
        
        # Update status
        self.status_label.value = "🔄 Running tracking..."
        self.apply_tracking_button.enabled = False
        
        try:
            # Get voxel sizes
            voxel_sizes = (
                self.voxel_t_spinbox.value,
                self.voxel_y_spinbox.value,
                self.voxel_x_spinbox.value
            )
            
            # Run tracking WITH napari data
            tracked_seg, tracks, stats, napari_data, napari_properties, napari_graph = run_tracking_with_params(
                segmentation=layer.data,
                params=selected_trial['params'],
                voxel_scale=voxel_sizes,
                return_napari=True
            )
            
            # Add tracked labels layer
            labels_layer_name = f"Tracked Labels (Trial {trial_num}, AOGM: {selected_trial['aogm']:.2f})"
            self.viewer.add_labels(tracked_seg, name=labels_layer_name)
            
            # Add tracks layer
            tracks_layer_name = f"Tracks (Trial {trial_num}, AOGM: {selected_trial['aogm']:.2f})"
            try:
                # Clean up empty dicts for napari
                if napari_properties is not None and isinstance(napari_properties, dict) and len(napari_properties) == 0:
                    napari_properties = None
                
                if napari_graph is not None and isinstance(napari_graph, dict) and len(napari_graph) == 0:
                    napari_graph = None
                
                self.viewer.add_tracks(
                    napari_data,
                    properties=napari_properties,
                    graph=napari_graph,
                    name=tracks_layer_name,
                    tail_width=2,
                    tail_length=10,
                    head_length=0
                )
                print(f"✅ Created tracks layer: {tracks_layer_name}")
            except Exception as e:
                print(f"⚠️ Could not create tracks layer: {e}")
                traceback.print_exc()
            
            # Update status
            self.status_label.value = "<font color='green'>✅ Tracking applied</font>"
            
            # Preserve trial count info
            n_trials = len(self.best_trials)
            self.results_info_label.value = (
                f"<font color='green'><b>{n_trials} trials available | Last applied: {labels_layer_name}</b></font><br>"
                f"Tracks: {stats['total_tracks']} | >5 frames: {stats['tracks_gt_5']} | >10 frames: {stats['tracks_gt_10']}"
            )
            
            print(f"\n✅ Tracking applied: {labels_layer_name}")
            print(f"✅ Tracks layer: {tracks_layer_name}")
            
        except Exception as e:
            self.status_label.value = "<font color='red'>❌ Tracking failed</font>"
            
            n_trials = len(self.best_trials) if self.best_trials else 0
            self.results_info_label.value = (
                f"<font color='green'><b>{n_trials} trials available</b></font><br>"
                f"<font color='red'>❌ Error: {str(e)}</font>"
            )
            print(f"\n❌ Error applying tracking: {e}")
            traceback.print_exc()
        
        finally:
            # Restore dropdown choices
            if current_choices:
                self.best_trials_combo.choices = current_choices
                try:
                    self.best_trials_combo.value = current_selection
                except:
                    pass
            
            self.apply_tracking_button.enabled = True

    def _on_save_config_clicked(self):
        """Handle save config button click."""
        from qtpy.QtWidgets import QFileDialog
        
        # Get selected trial
        if not self.best_trials_combo.value:
            n_trials = len(self.best_trials) if self.best_trials else 0
            self.results_info_label.value = (
                f"<font color='green'><b>{n_trials} trials available</b></font><br>"
                f"<font color='red'>❌ Select a trial first</font>"
            )
            return
        
        # Parse trial number
        trial_label = self.best_trials_combo.value
        trial_num = int(trial_label.split(":")[0].replace("Trial ", ""))
        
        # Find matching trial
        selected_trial = None
        for trial in self.best_trials:
            if trial['number'] == trial_num:
                selected_trial = trial
                break
        
        if selected_trial is None:
            return
        
        # Generate default filename
        study_name = self.study_name_input.value
        default_filename = f"{study_name}_trial{trial_num}_config.json"
        
        # Open save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Config File",
            default_filename,
            "JSON Files (*.json)"
        )
        
        if not file_path:
            # User cancelled
            return
        
        # Ensure .json extension
        if not file_path.endswith('.json'):
            file_path += '.json'
        
        try:
            # Import the write function
            from ..analysis.optim_pipeline import write_best_params_to_config
            
            # Write config
            write_best_params_to_config(selected_trial['params'], str(file_path))
            
            # Preserve trial count in success message
            n_trials = len(self.best_trials)
            self.results_info_label.value = (
                f"<font color='green'><b>{n_trials} trials available | Config saved: {Path(file_path).name}</b></font>"
            )
            
            print(f"\n✅ Config saved to: {file_path}")
            
        except Exception as e:
            n_trials = len(self.best_trials) if self.best_trials else 0
            self.results_info_label.value = (
                f"<font color='green'><b>{n_trials} trials available</b></font><br>"
                f"<font color='red'>❌ Error: {str(e)}</font>"
            )
            print(f"\n❌ Error saving config: {e}")
            traceback.print_exc()




    def _on_view_dashboard_clicked(self):
        """Launch Optuna dashboard in browser."""
        try:
            # Get the database path
            db_path = Path(self.optimization_manager.db_path).resolve()
            
            if not db_path.exists():
                self.status_label.value = "<font color='orange'>⚠️ No database found yet</font>"
                return
            
            # Start optuna-dashboard in a subprocess
            storage_url = f"sqlite:///{db_path}"
            
            print(f"\n📊 Launching Optuna dashboard...")
            print(f"   Database: {db_path}")
            print(f"   URL: http://127.0.0.1:8080")
            
            # Launch dashboard in background
            subprocess.Popen(
                ["optuna-dashboard", storage_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait a moment then open browser
            QTimer.singleShot(2000, lambda: webbrowser.open("http://127.0.0.1:8080"))
            
            self.status_label.value = "📊 Dashboard opening in browser..."
            
        except FileNotFoundError:
            self.status_label.value = "<font color='red'>❌ optuna-dashboard not installed</font>"
            print("\n❌ Error: optuna-dashboard not found")
            print("Install it with: pip install optuna-dashboard")
        except Exception as e:
            self.status_label.value = f"<font color='red'>❌ Failed to launch dashboard</font>"
            print(f"\n❌ Error launching dashboard: {e}")
    
    def _show_results_section(self):
        """Show the results section widgets."""
        self.best_trials_combo.visible = True
        self.apply_tracking_button.visible = True
        self.save_config_button.visible = True
    
    def _hide_results_section(self):
        """Hide the results section widgets."""
        self.best_trials_combo.visible = False
        self.apply_tracking_button.visible = False
        self.save_config_button.visible = False
        self.best_trials_combo.choices = []
        self.results_info_label.value = ""
    
    def _reset_ui_state(self):
        """Reset UI to ready state."""
        self.start_button.enabled = True
        self.cancel_button.enabled = False
        self.layer_selector.enabled = True
    
    def _cleanup_temp_files(self):
        """Clean up temporary working directory."""
        if self.current_work_dir and self.current_work_dir.exists():
            try:
                shutil.rmtree(self.current_work_dir)
                print(f"\n🧹 Cleaned up temporary files: {self.current_work_dir}")
            except Exception as e:
                print(f"\n⚠️  Could not clean up temp files: {e}")
        
        self.current_work_dir = None
        self.current_dataset = None
        self.current_gt_data = None