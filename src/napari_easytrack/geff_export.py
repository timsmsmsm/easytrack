"""GEFF export for napari-easytrack tracking results.

Converts napari tracking data (napari_data, napari_properties, napari_graph)
to the GEFF (Generic Event/Entity File Format) spatial graph format.

See https://github.com/live-image-tracking-tools/geff for the GEFF specification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import networkx as nx
import numpy as np

import geff


def export_to_geff(
    napari_data: np.ndarray,
    napari_properties: Dict[str, Any],
    napari_graph: Dict[int, list],
    output_path: str | Path,
    overwrite: bool = False,
) -> None:
    """Export napari tracking results to a GEFF file.

    Converts napari tracking data into a directed spatial graph and writes it
    using the GEFF format (zarr-backed).  Each row of *napari_data* becomes
    one node; intra-track sequential edges and inter-track division edges
    (from *napari_graph*) are added automatically.

    Args:
        napari_data: 2-D array of shape ``(N, 4)`` for 2-D+T data
            (columns: ``track_id``, ``t``, ``y``, ``x``) or ``(N, 5)``
            for 3-D+T data (columns: ``track_id``, ``t``, ``z``, ``y``,
            ``x``).
        napari_properties: Dictionary of per-row track properties (e.g.
            ``{"area": array(...)}``) produced by btrack.  These are stored
            as node attributes.
        napari_graph: Parent-to-child track mapping
            ``{child_track_id: [parent_track_id, ...]}``.  Used to add
            division edges between the last node of a parent track and the
            first node of each child track.
        output_path: Destination path for the ``.geff`` zarr store.
        overwrite: If ``True``, overwrite an existing file at *output_path*.
            Defaults to ``False``.

    Raises:
        ValueError: If *napari_data* does not have 4 or 5 columns.
    """
    napari_data = np.asarray(napari_data)
    n_cols = napari_data.shape[1] if napari_data.ndim == 2 else 0

    if n_cols == 4:
        is_3d = False
        col_names = ["track_id", "t", "y", "x"]
    elif n_cols == 5:
        is_3d = True
        col_names = ["track_id", "t", "z", "y", "x"]
    else:
        raise ValueError(
            f"napari_data must have 4 columns (track_id, t, y, x) for 2-D+T "
            f"data or 5 columns (track_id, t, z, y, x) for 3-D+T data, "
            f"but got {n_cols} columns."
        )

    # ------------------------------------------------------------------
    # Build the directed graph
    # ------------------------------------------------------------------
    graph = nx.DiGraph()

    # Map (track_id, t) → node index so we can add division edges later.
    # Each row in napari_data is assigned a unique integer node ID (row index).
    track_to_nodes: dict[int, list[int]] = {}  # track_id → ordered list of node ids

    for row_idx, row in enumerate(napari_data):
        track_id = int(row[0])

        # Build node attribute dict
        attrs: dict[str, Any] = {name: row[i] for i, name in enumerate(col_names)}
        attrs["track_id"] = track_id

        # Add any per-row properties
        for prop_name, prop_values in napari_properties.items():
            if len(prop_values) > row_idx:
                attrs[prop_name] = prop_values[row_idx]

        graph.add_node(row_idx, **attrs)

        track_to_nodes.setdefault(track_id, []).append(row_idx)

    # Sort nodes within each track by time so edges go forward in time
    for track_id in track_to_nodes:
        track_to_nodes[track_id].sort(
            key=lambda idx: napari_data[idx, 1]  # sort by t
        )

    # Intra-track sequential edges (parent → child within the same track)
    for node_ids in track_to_nodes.values():
        for src, dst in zip(node_ids[:-1], node_ids[1:]):
            graph.add_edge(src, dst)

    # Division / inter-track edges
    # napari_graph: {child_track_id: [parent_track_id, ...]}
    for child_track_id, parent_track_ids in napari_graph.items():
        child_nodes = track_to_nodes.get(child_track_id)
        if not child_nodes:
            continue
        child_first = child_nodes[0]  # first node of child track (earliest t)

        for parent_track_id in parent_track_ids:
            parent_nodes = track_to_nodes.get(parent_track_id)
            if not parent_nodes:
                continue
            parent_last = parent_nodes[-1]  # last node of parent track (latest t)
            graph.add_edge(parent_last, child_first)

    # ------------------------------------------------------------------
    # Write to GEFF
    # ------------------------------------------------------------------
    if is_3d:
        axis_names = ["t", "z", "y", "x"]
        axis_types = ["time", "space", "space", "space"]
    else:
        axis_names = ["t", "y", "x"]
        axis_types = ["time", "space", "space"]

    geff.write(
        graph,
        str(output_path),
        axis_names=axis_names,
        axis_types=axis_types,
        overwrite=overwrite,
    )
