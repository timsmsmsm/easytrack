"""Tests for optim widget module."""
from __future__ import annotations

from collections.abc import Callable
import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from src.napari_easytrack.widgets.optim_widget import (
    BtrackOptimizationWidget,
)


def test_add_optimization_widget(
    make_napari_viewer: Callable,
):
    """Checks that the optimization widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    dw = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-easytrack",
        widget_name="Parameter Tuning",
    )

    # `add_plugin_dock_widget` may return either a single QDockWidget or a tuple
    # (QDockWidget, Container). Normalize to the dock widget before assertions.
    if isinstance(dw, tuple):
        dock_widget = dw[0]
    else:
        dock_widget = dw

    assert len(viewer.window._dock_widgets) == num_dw + 1

    # Explicitly remove the created dock widget to avoid Qt deletion errors.
    viewer.window.remove_dock_widget(dock_widget)


def test_optim_widget_with_labels(
    make_napari_viewer: Callable,
):
    """Test optimization widget with labels layer."""
    viewer = make_napari_viewer()
    
    # Create sample segmentation and ground truth
    seg_data = np.zeros((5, 50, 50), dtype=np.uint16)
    seg_data[0, 10:20, 10:20] = 1
    seg_data[1, 11:21, 11:21] = 1
    seg_data[2, 12:22, 12:22] = 1
    
    gt_data = seg_data.copy()
    
    # Add layers
    seg_layer = viewer.add_labels(seg_data, name="segmentation")
    gt_layer = viewer.add_labels(gt_data, name="ground_truth")
    
    # Add widget
    dw = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-easytrack",
        widget_name="Parameter Tuning",
    )
    
    if isinstance(dw, tuple):
        dock_widget = dw[0]
    else:
        dock_widget = dw
    
    # Verify widget was added
    assert len(viewer.window._dock_widgets) > 0
    
    # Clean up
    viewer.window.remove_dock_widget(dock_widget)


def test_optim_widget_initialization(
    make_napari_viewer: Callable,
):
    """Test that optimization widget initializes correctly."""
    viewer = make_napari_viewer()
    
    dw = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-easytrack",
        widget_name="Parameter Tuning",
    )
    
    if isinstance(dw, tuple):
        dock_widget, widget_instance = dw
    else:
        dock_widget = dw
        widget_instance = None
    
    # Check widget was created
    assert dock_widget is not None
    
    # Clean up
    viewer.window.remove_dock_widget(dock_widget)


def test_optim_widget_layer_selection(
    make_napari_viewer: Callable,
):
    """Test optimization widget layer selection."""
    viewer = make_napari_viewer()
    
    # Create layers
    seg_data = np.zeros((3, 30, 30), dtype=np.uint16)
    seg_data[:, 10:20, 10:20] = 1
    
    viewer.add_labels(seg_data, name="my_segmentation")
    viewer.add_labels(seg_data, name="my_ground_truth")
    
    # Add widget
    dw = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-easytrack",
        widget_name="Parameter Tuning",
    )
    
    if isinstance(dw, tuple):
        dock_widget, widget_instance = dw
    else:
        dock_widget = dw
        widget_instance = None
    
    # Verify layers are available
    assert len(viewer.layers) == 2
    
    # Clean up
    viewer.window.remove_dock_widget(dock_widget)
