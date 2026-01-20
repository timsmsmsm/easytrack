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
    viewer.window.add_plugin_dock_widget(
        plugin_name="napari-easytrack",
        widget_name="Parameter Tuning",
    )

    assert len(viewer.window._dock_widgets) == num_dw + 1