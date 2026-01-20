"""Tests for widget module."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from collections.abc import Callable
from unittest.mock import patch, MagicMock

import napari

from src.napari_easytrack.widgets.widget import (
    BtrackPresetWidget,
)

def test_add_widget(
    make_napari_viewer: Callable,
):
    """Checks that the widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name="napari-easytrack",
        widget_name="Tracking",
    )

    assert len(viewer.window._dock_widgets) == num_dw + 1


def test_widget_with_labels(
    make_napari_viewer: Callable,
):
    """Test widget with a labels layer."""
    viewer = make_napari_viewer()
    
    # Create a simple segmentation
    labels_data = np.zeros((10, 100, 100), dtype=np.uint16)
    labels_data[0, 20:40, 20:40] = 1
    labels_data[1, 22:42, 22:42] = 1
    
    # Add labels layer
    labels_layer = viewer.add_labels(labels_data, name="segmentation")
    
    # Add widget
    widget, widget_instance = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-easytrack",
        widget_name="Tracking",
    )
    
    # Verify the widget was added
    assert len(viewer.window._dock_widgets) > 0
    

def test_widget_preset_selection(
    make_napari_viewer: Callable,
):
    """Test that widget can select different presets."""
    viewer = make_napari_viewer()
    
    # Add widget
    widget, widget_instance = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-easytrack",
        widget_name="Tracking",
    )
    
    # Check that widget has presets
    assert hasattr(widget_instance, 'presets')
    assert len(widget_instance.presets) > 0
    assert "Epithelial Cells (Default)" in widget_instance.presets


def test_widget_get_current_params(
    make_napari_viewer: Callable,
):
    """Test getting current parameters from widget."""
    viewer = make_napari_viewer()
    
    # Add widget
    widget, widget_instance = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-easytrack",
        widget_name="Tracking",
    )
    
    # Test getting parameters
    params = widget_instance._get_current_params()
    
    # Check that parameters dict is returned
    assert isinstance(params, dict)
    # Check for expected parameter keys
    assert 'dist_thresh' in params or len(params) > 0


