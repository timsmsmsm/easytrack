"""Pytest configuration for napari_easytrack tests."""

import sys
from pathlib import Path

import napari_easytrack

# Add the parent directory to the path so tests can import the modules
# This is done in conftest.py rather than individual test files
sys.path.insert(0, str(Path(__file__).parent.parent))
