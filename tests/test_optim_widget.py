"""Tests for optim widget module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from src.napari_easytrack.widgets.optim_widget import (
    BtrackOptimizationWidget,
)

@pytest.fixture
def optim_widget():
    """Fixture for BtrackOptimizationWidget."""
    widget = BtrackOptimizationWidget()
    yield widget
    widget.close()