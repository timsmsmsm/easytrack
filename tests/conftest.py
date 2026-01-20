"""Pytest configuration for napari_easytrack tests."""
from __future__ import annotations

import sys
from pathlib import Path

from collections.abc import Callable

import pytest

import napari

import napari_easytrack

# Add the parent directory to the path so tests can import the modules
# This is done in conftest.py rather than individual test files
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="function")
def viewer_with_labels(
    make_napari_viewer: Callable,
    labels: napari.layers.Labels,
) -> napari.Viewer:
    """Load a sample segmentation and the seeds use in generating the segmentation.

    Load sample cells and seeds and convert to Napari Points and Labels layers,
    respectively.

    Note, the Napari Labels will be 4D (TZYX) with a single frame in the time dimension
    and a single slice in Z.
    """

    viewer = make_napari_viewer()
    viewer.add_layer(labels)

    return viewer


@pytest.fixture(scope="function")
def viewer_with_image(
    make_napari_viewer: Callable,
    image: napari.layers.Image,
) -> napari.Viewer:
    """Create a Napari Viewer with a sample Image layer added to it."""

    viewer = make_napari_viewer()
    viewer.add_layer(image)

    return viewer