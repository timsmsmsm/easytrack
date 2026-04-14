"""Tests for GEFF export functionality."""

import numpy as np
import pytest
import os
from pathlib import Path

import networkx as nx

from src.napari_easytrack.geff_export import export_to_geff


class TestExportToGeff2D:
    """Tests for 2D+T (T,Y,X) GEFF export."""

    def _make_2d_data(self):
        """Create simple 2D+T napari tracking data: 2 tracks, 3 frames each."""
        # napari_data columns: [track_id, t, y, x]
        napari_data = np.array([
            [0, 0, 10.0, 20.0],
            [0, 1, 11.0, 21.0],
            [0, 2, 12.0, 22.0],
            [1, 1, 50.0, 60.0],
            [1, 2, 51.0, 61.0],
        ], dtype=float)
        napari_properties = {}
        napari_graph = {}
        return napari_data, napari_properties, napari_graph

    def test_creates_geff_file(self, tmp_path):
        """Test that a .geff directory/file is created."""
        napari_data, napari_properties, napari_graph = self._make_2d_data()
        out = str(tmp_path / "test.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        assert os.path.exists(out)

    def test_correct_node_count(self, tmp_path):
        """Test that the graph has the correct number of nodes (one per row)."""
        import geff
        napari_data, napari_properties, napari_graph = self._make_2d_data()
        out = str(tmp_path / "test.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        graph, _ = geff.read(out, backend="networkx")
        assert graph.number_of_nodes() == 5

    def test_node_attributes_2d(self, tmp_path):
        """Test that nodes have correct t, y, x attributes."""
        import geff
        napari_data, napari_properties, napari_graph = self._make_2d_data()
        out = str(tmp_path / "test.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        graph, _ = geff.read(out, backend="networkx")
        node_data = dict(graph.nodes(data=True))
        # Find node with t=0, y=10, x=20
        found = any(
            abs(d.get('t', -1) - 0) < 1e-6
            and abs(d.get('y', -1) - 10.0) < 1e-6
            and abs(d.get('x', -1) - 20.0) < 1e-6
            for d in node_data.values()
        )
        assert found, f"Expected node with t=0, y=10, x=20 but got: {list(node_data.values())}"

    def test_intratrack_edges(self, tmp_path):
        """Test that consecutive frames in the same track are connected."""
        import geff
        napari_data, napari_properties, napari_graph = self._make_2d_data()
        out = str(tmp_path / "test.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        graph, _ = geff.read(out, backend="networkx")
        # Track 0 has 3 frames, track 1 has 2 frames → 2+1 = 3 intra-track edges
        assert graph.number_of_edges() == 3

    def test_graph_is_directed(self, tmp_path):
        """Test that the exported graph is directed."""
        import geff
        napari_data, napari_properties, napari_graph = self._make_2d_data()
        out = str(tmp_path / "test.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        graph, _ = geff.read(out, backend="networkx")
        assert graph.is_directed()

    def test_metadata_axis_names(self, tmp_path):
        """Test that metadata contains correct axis names for 2D data."""
        import geff
        napari_data, napari_properties, napari_graph = self._make_2d_data()
        out = str(tmp_path / "test.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        _, metadata = geff.read(out, backend="networkx")
        axis_names = [ax.name for ax in metadata.axes]
        assert 't' in axis_names
        assert 'y' in axis_names
        assert 'x' in axis_names

    def test_track_id_attribute(self, tmp_path):
        """Test that nodes carry a track_id attribute."""
        import geff
        napari_data, napari_properties, napari_graph = self._make_2d_data()
        out = str(tmp_path / "test.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        graph, _ = geff.read(out, backend="networkx")
        for _, data in graph.nodes(data=True):
            assert 'track_id' in data


class TestExportToGeff3D:
    """Tests for 3D+T (T,Z,Y,X) GEFF export."""

    def _make_3d_data(self):
        """Create simple 3D+T napari tracking data: 1 track, 2 frames."""
        # napari_data columns: [track_id, t, z, y, x]
        napari_data = np.array([
            [0, 0, 5.0, 10.0, 20.0],
            [0, 1, 6.0, 11.0, 21.0],
        ], dtype=float)
        napari_properties = {}
        napari_graph = {}
        return napari_data, napari_properties, napari_graph

    def test_creates_geff_file_3d(self, tmp_path):
        """Test that a .geff file is created for 3D data."""
        napari_data, napari_properties, napari_graph = self._make_3d_data()
        out = str(tmp_path / "test3d.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        assert os.path.exists(out)

    def test_node_attributes_3d(self, tmp_path):
        """Test that nodes have t, z, y, x attributes for 3D data."""
        import geff
        napari_data, napari_properties, napari_graph = self._make_3d_data()
        out = str(tmp_path / "test3d.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        graph, _ = geff.read(out, backend="networkx")
        node_data = dict(graph.nodes(data=True))
        found = any(
            abs(d.get('z', -1) - 5.0) < 1e-6 and abs(d.get('t', -1) - 0) < 1e-6
            for d in node_data.values()
        )
        assert found, f"Expected node with z=5.0, t=0: {list(node_data.values())}"

    def test_metadata_axis_names_3d(self, tmp_path):
        """Test that metadata contains t, z, y, x axes for 3D data."""
        import geff
        napari_data, napari_properties, napari_graph = self._make_3d_data()
        out = str(tmp_path / "test3d.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        _, metadata = geff.read(out, backend="networkx")
        axis_names = [ax.name for ax in metadata.axes]
        assert 'z' in axis_names


class TestExportToGeffDivisions:
    """Tests for GEFF export with cell divisions (napari_graph)."""

    def _make_division_data(self):
        """Create data with a division: track 0 divides into tracks 1 and 2."""
        # Track 0: frames 0-2 (parent)
        # Track 1: frames 3-4 (child 1)
        # Track 2: frames 3-4 (child 2)
        napari_data = np.array([
            [0, 0, 10.0, 20.0],
            [0, 1, 11.0, 21.0],
            [0, 2, 12.0, 22.0],
            [1, 3, 13.0, 23.0],
            [1, 4, 14.0, 24.0],
            [2, 3, 15.0, 25.0],
            [2, 4, 16.0, 26.0],
        ], dtype=float)
        napari_properties = {}
        # Tracks 1 and 2 are children of track 0
        napari_graph = {1: [0], 2: [0]}
        return napari_data, napari_properties, napari_graph

    def test_division_edges_added(self, tmp_path):
        """Test that division edges are added from parent last point to child first point."""
        import geff
        napari_data, napari_properties, napari_graph = self._make_division_data()
        out = str(tmp_path / "div.geff")
        export_to_geff(napari_data, napari_properties, napari_graph, out)
        graph, _ = geff.read(out, backend="networkx")
        # Intra-track: track0=2, track1=1, track2=1 = 4
        # Division: 2 edges (parent->child1, parent->child2)
        # Total: 6
        assert graph.number_of_edges() == 6

    def test_single_child_division(self, tmp_path):
        """Test division with a single child (track 1 is child of track 0)."""
        import geff
        napari_data = np.array([
            [0, 0, 10.0, 20.0],
            [0, 1, 11.0, 21.0],
            [1, 2, 12.0, 22.0],
            [1, 3, 13.0, 23.0],
        ], dtype=float)
        napari_graph = {1: [0]}
        out = str(tmp_path / "single_child.geff")
        export_to_geff(napari_data, {}, napari_graph, out)
        graph, _ = geff.read(out, backend="networkx")
        # intra: track0=1, track1=1 = 2; division: 1 → total 3
        assert graph.number_of_edges() == 3


class TestExportToGeffEdgeCases:
    """Edge-case tests for GEFF export."""

    def test_single_node_track(self, tmp_path):
        """Test export with a single-frame track."""
        import geff
        napari_data = np.array([[0, 0, 5.0, 5.0]], dtype=float)
        out = str(tmp_path / "single.geff")
        export_to_geff(napari_data, {}, {}, out)
        graph, _ = geff.read(out, backend="networkx")
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0

    def test_overwrite_existing_file(self, tmp_path):
        """Test that overwrite=True allows writing to an existing path."""
        napari_data = np.array([[0, 0, 5.0, 5.0]], dtype=float)
        out = str(tmp_path / "overwrite.geff")
        export_to_geff(napari_data, {}, {}, out)
        # Should not raise even when file already exists if overwrite=True
        export_to_geff(napari_data, {}, {}, out, overwrite=True)

    def test_invalid_napari_data_shape_raises(self, tmp_path):
        """Test that invalid column count raises ValueError."""
        napari_data = np.array([[0, 0, 5.0]], dtype=float)  # Only 3 columns
        out = str(tmp_path / "bad.geff")
        with pytest.raises(ValueError):
            export_to_geff(napari_data, {}, {}, out)
