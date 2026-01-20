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