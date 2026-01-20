"""Tests for widget module."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from collections.abc import Callable
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

import napari

from src.napari_easytrack.widgets.widget import (
    BtrackPresetWidget,
)

def test_add_segmentation_widget(
    make_napari_viewer: Callable,
):
    """Checks that the segmentation widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name="napari-easytrack",
        widget_name="Tracking",
    )

    assert len(viewer.window._dock_widgets) == num_dw + 1


