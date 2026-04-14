"""
gt_filtering.py — Per-frame GT filtering for fair CTC evaluation
================================================================

Replaces the existing filter_man_track_to_segmentation() with a version
that clips each GT track to only the frames where the cell actually has
pixels in the segmentation, and splits tracks at temporal gaps.

This eliminates fn_nodes caused by GT-segmentation mismatches (the
dominant source of AOGM error in the Fluo-N2DH-GOWT1 benchmark).

Drop-in replacement: import and use filter_man_track_to_segmentation_v2
wherever the original was called.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile


def filter_man_track_to_segmentation_v2(
    tra_dir: Path,
    segmentation: np.ndarray,
) -> Path:
    """
    Create a filtered man_track.txt and TRA masks that only include
    (cell_id, frame) pairs actually present in the segmentation.

    Unlike the original filter, this:
    1. Clips each track's start/end to the frames where pixels exist
    2. Splits tracks at temporal gaps (frames where the cell vanishes)
       into separate sub-tracks, preserving parent linkage on the first
       sub-track only
    3. Removes tracks entirely if no frames have pixels
    4. Rebuilds TRA TIF masks directly from the segmentation (avoids
       all TRA-vs-SEG divergence issues)

    Returns path to a filtered TRA directory.
    """
    man_track_path = tra_dir / "man_track.txt"
    if not man_track_path.exists():
        return tra_dir

    # Build per-cell frame presence from segmentation
    cell_frames: dict[int, set[int]] = defaultdict(set)
    for t in range(segmentation.shape[0]):
        for label in np.unique(segmentation[t]):
            if label != 0:
                cell_frames[int(label)].add(t)

    # Read original tracks
    original_tracks = []
    with open(man_track_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            original_tracks.append({
                "id": int(parts[0]),
                "start": int(parts[1]),
                "end": int(parts[2]),
                "parent": int(parts[3]),
            })

    kept_ids = set()
    removed_ids = set()

    # Build new track list + a mapping from (frame, original_label) -> new_label
    # so we can rebuild TRA masks correctly
    new_tracks = []
    max_id = max((t["id"] for t in original_tracks), default=0)
    next_id = max_id + 1

    # frame_label_map: {frame: {original_cell_id: output_label}}
    # For the first contiguous run, output_label == original cell_id.
    # For subsequent split runs, output_label == next_id, next_id+1, ...
    frame_label_map: dict[int, dict[int, int]] = defaultdict(dict)

    for track in original_tracks:
        cell_id = track["id"]
        frames_present = sorted(cell_frames.get(cell_id, set()))

        if not frames_present:
            removed_ids.add(cell_id)
            continue

        # Find contiguous runs
        runs = _find_contiguous_runs(frames_present)

        # First run keeps the original cell ID and parent
        first_run = runs[0]
        new_tracks.append({
            "id": cell_id,
            "start": first_run[0],
            "end": first_run[-1],
            "parent": track["parent"],
        })
        kept_ids.add(cell_id)
        for f in first_run:
            frame_label_map[f][cell_id] = cell_id

        # Subsequent runs get new IDs
        for run in runs[1:]:
            new_id = next_id
            next_id += 1
            new_tracks.append({
                "id": new_id,
                "start": run[0],
                "end": run[-1],
                "parent": 0,
            })
            kept_ids.add(new_id)
            for f in run:
                frame_label_map[f][cell_id] = new_id

    # Clean up parent references to removed or split-away cells
    for track in new_tracks:
        if track["parent"] != 0 and track["parent"] not in kept_ids:
            track["parent"] = 0

    # Stats
    n_original = len(original_tracks)
    n_removed = len(removed_ids)
    n_split_new = len([t for t in new_tracks if t["id"] > max_id])
    n_clipped = 0
    for orig in original_tracks:
        if orig["id"] not in removed_ids:
            new = next((t for t in new_tracks if t["id"] == orig["id"]), None)
            if new and (new["start"] != orig["start"] or new["end"] != orig["end"]):
                n_clipped += 1

    print(f"    GT filtering v2: {n_original} original tracks → "
          f"{len(new_tracks)} output tracks")
    print(f"      Removed entirely: {n_removed}")
    print(f"      Clipped (start/end adjusted): {n_clipped}")
    print(f"      New sub-tracks from splits: {n_split_new}")

    if n_removed == 0 and n_clipped == 0 and n_split_new == 0:
        return tra_dir

    # Create filtered TRA directory (clean slate)
    filtered_dir = tra_dir.parent.parent / f"{tra_dir.parent.name}_filtered" / "TRA"
    if filtered_dir.exists():
        shutil.rmtree(filtered_dir)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    # Write new man_track.txt
    with open(filtered_dir / "man_track.txt", "w") as f:
        for track in new_tracks:
            f.write(f"{track['id']} {track['start']} {track['end']} "
                    f"{track['parent']}\n")

    # Rebuild TRA masks from segmentation
    #
    # Instead of patching the original TRA TIF masks (which can diverge
    # from the segmentation in subtle ways), we build fresh masks where
    # each pixel gets the output label from frame_label_map.  Pixels
    # belonging to removed cells or to frames outside a track's range
    # are simply absent (zero).
    n_frames = segmentation.shape[0]
    width = 3 if n_frames < 1000 else 4

    for t in range(n_frames):
        mask = np.zeros(segmentation.shape[1:], dtype=np.uint16)
        mapping = frame_label_map.get(t, {})

        if mapping:
            seg_frame = segmentation[t]
            for orig_label, out_label in mapping.items():
                mask[seg_frame == orig_label] = out_label

        tifffile.imwrite(
            str(filtered_dir / f"man_track{t:0{width}d}.tif"),
            mask,
        )

    return filtered_dir


def _find_contiguous_runs(frames: list[int]) -> list[list[int]]:
    """Split a sorted list of frame indices into contiguous runs."""
    if not frames:
        return []
    runs = [[frames[0]]]
    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1] + 1:
            runs[-1].append(frames[i])
        else:
            runs.append([frames[i]])
    return runs