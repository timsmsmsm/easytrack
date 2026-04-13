"""
prepare_wing_disc.py — Create CTC directory structure from wing disc data
=========================================================================

Prepares both wing disc datasets for use with benchmark_ctc.py:

  1. 2D+T wound healing (2d_wing_disc_wound_healing)
     Source: example_data/2d_time/2d_time_gt.tif
     A TYX segmentation stack with 2 known cell divisions.

  2. 3D cell shapes (3d_wing_disc)
     Source: example_data/z_tracking_example.tif
     A ZYX segmentation stack (tracking through z). No divisions.

For each dataset, this script:
  - Reads the single-stack GT TIF
  - Extracts per-cell track info (first/last frame for each label)
  - Writes man_track.txt with correct division encoding (CTC format)
  - Creates the CTC directory structure (01_GT/TRA/ and 01_GT/SEG/)

Usage
-----
    # Prepare both datasets:
    python prepare_wing_disc.py

    # Prepare only the 2D+T dataset:
    python prepare_wing_disc.py --datasets 2d

    # Prepare only the 3D dataset:
    python prepare_wing_disc.py --datasets 3d

    # Override paths:
    python prepare_wing_disc.py --data-dir ./ctc_data --force
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import tifffile


# =========================================================================
# Dataset definitions
# =========================================================================

WING_DISC_DATASETS = {
    "2d": {
        "name": "2d_wing_disc_wound_healing",
        "source": "../example_data/2d_time/2d_time_gt.tif",
        "description": "2D+T wound healing",
        "is_3d": False,
        "sequence": "01",
        # CTC convention: parent track must END before daughters START.
        # Since the parent label continues in the segmentation, we split
        # it into a pre-division track (new ID) and a post-division track
        # (keeps original label). Both daughters list the new ID as parent.
        # Format: (division_frame, parent_label, new_daughter_label)
        "divisions": [
            (21, 38, 60),
            (25, 77, 82),
        ],
    },
    "3d": {
        "name": "3d_wing_disc",
        "source": "../example_data/z_tracking_example.tif",
        "description": "3D cell shapes (tracking through z)",
        "is_3d": True,
        "sequence": "02",
        "divisions": [],
    },
}


# =========================================================================
# Track extraction
# =========================================================================

def extract_tracks_from_stack(seg: np.ndarray) -> dict:
    """
    Extract {label: (first_frame, last_frame)} from a segmentation stack.
    Works for both TYX (2D+T) and ZYX (3D) stacks — the first axis is
    treated as the "time" / "tracking" axis.
    """
    tracks = {}
    n_frames = seg.shape[0]

    for t in range(n_frames):
        labels = np.unique(seg[t])
        labels = labels[labels != 0]
        for label in labels:
            label = int(label)
            if label not in tracks:
                tracks[label] = [t, t]
            else:
                tracks[label][1] = t

    return {k: (v[0], v[1]) for k, v in tracks.items()}


# =========================================================================
# man_track.txt generation
# =========================================================================

def write_man_track(
    tracks: dict,
    divisions: list,
    output_path: Path,
) -> dict:
    """
    Write man_track.txt in CTC format: "label start end parent"

    CTC convention requires that parent tracks END before daughters START.
    For divisions where the parent label continues in the segmentation,
    we split the parent into a pre-division track (new ID) and a
    post-division track (keeps original label). Both daughters list the
    new pre-division ID as their parent.

    Returns relabel_map: {(frame, original_label): new_label} for
    pre-division frames that need relabelling in TRA masks.
    """
    if not divisions:
        # Simple case: no divisions, just write all tracks with parent=0
        lines = []
        for label in sorted(tracks.keys()):
            start, end = tracks[label]
            lines.append(f"{label} {start} {end} 0")

        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"  Wrote {len(lines)} tracks to {output_path}")
        return {}

    # Division case: need to split parent tracks
    max_id = max(tracks.keys())
    next_id = max_id + 1

    relabel_map = {}
    # parent_split: {parent_label: (pre_div_id, division_frame)}
    parent_split = {}
    # daughter -> pre_div_parent_id
    daughter_parent = {}

    for div_frame, parent_label, new_daughter_label in divisions:
        pre_div_id = next_id
        next_id += 1
        parent_split[parent_label] = (pre_div_id, div_frame)
        daughter_parent[new_daughter_label] = pre_div_id
        # The continuing parent_label also becomes a daughter
        daughter_parent[parent_label] = pre_div_id

    lines = []
    for label in sorted(tracks.keys()):
        start, end = tracks[label]

        if label in parent_split:
            # This label divides: create pre-division track with new ID
            pre_div_id, div_frame = parent_split[label]

            # Pre-division track: original start to div_frame - 1
            pre_start = start
            pre_end = div_frame - 1
            pre_parent = daughter_parent.get(label, 0)
            lines.append(f"{pre_div_id} {pre_start} {pre_end} {pre_parent}")

            # Post-division track: div_frame to end, parent = pre_div_id
            lines.append(f"{label} {div_frame} {end} {pre_div_id}")

            # Mark frames for relabelling in TRA masks
            for t in range(pre_start, pre_end + 1):
                relabel_map[(t, label)] = pre_div_id

        elif label in daughter_parent:
            # New daughter (not the continuing parent)
            parent_id = daughter_parent[label]
            lines.append(f"{label} {start} {end} {parent_id}")

        else:
            # Normal track, no division involvement
            lines.append(f"{label} {start} {end} 0")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Wrote {len(lines)} tracks to {output_path}")

    for div_frame, parent_label, new_daughter_label in divisions:
        pre_div_id, _ = parent_split[parent_label]
        p_start, p_end = tracks[parent_label]
        d_start, d_end = tracks[new_daughter_label]
        print(f"  Division at t={div_frame}: "
              f"pre-division ID {pre_div_id} (t={p_start}–{div_frame-1}) → "
              f"{parent_label} (t={div_frame}–{p_end}) + "
              f"{new_daughter_label} (t={d_start}–{d_end})")

    return relabel_map


# =========================================================================
# CTC directory structure
# =========================================================================

def create_ctc_structure(
    seg: np.ndarray,
    output_dir: Path,
    divisions: list,
    sequence: str = "01",
) -> None:
    """
    Create CTC directory structure:
      output_dir/
        01_GT/
          TRA/
            man_track.txt
            man_track000.tif, ...
          SEG/
            man_seg000.tif, ...
    """
    gt_dir = output_dir / f"{sequence}_GT"
    tra_dir = gt_dir / "TRA"
    seg_dir = gt_dir / "SEG"
    tra_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    n_frames = seg.shape[0]
    width = 3 if n_frames < 1000 else 4

    # Extract tracks and write man_track.txt first (need relabel_map)
    tracks = extract_tracks_from_stack(seg)
    print(f"  Found {len(tracks)} unique labels across {n_frames} frames")
    relabel_map = write_man_track(tracks, divisions, tra_dir / "man_track.txt")

    print(f"  Writing {n_frames} frames to CTC structure...")
    for t in range(n_frames):
        frame = seg[t].astype(np.uint16)

        # SEG masks keep original labels
        tifffile.imwrite(
            str(seg_dir / f"man_seg{t:0{width}d}.tif"),
            frame,
        )

        # TRA masks use relabelled IDs for pre-division parent frames
        tra_frame = frame.copy()
        for (rt, rlabel), new_label in relabel_map.items():
            if rt == t:
                tra_frame[tra_frame == rlabel] = new_label
        tifffile.imwrite(
            str(tra_dir / f"man_track{t:0{width}d}.tif"),
            tra_frame,
        )


# =========================================================================
# Per-dataset preparation
# =========================================================================

def prepare_dataset(
    key: str,
    data_dir: Path,
    force: bool = False,
) -> int:
    """Prepare one wing disc dataset. Returns 0 on success, 1 on error."""
    info = WING_DISC_DATASETS[key]
    name = info["name"]
    source = Path(info["source"]).resolve()
    divisions = info["divisions"]
    sequence = info["sequence"]

    print(f"\n{'─' * 60}")
    print(f"Preparing: {name} ({info['description']})")
    print(f"  Source: {source}")

    if not source.exists():
        print(f"  ERROR: Source file not found: {source}")
        return 1

    # New structure: wing_disc/{sequence}/01_GT/
    output_dir = data_dir / "wing_disc" / sequence
    gt_dir = output_dir / "01_GT"

    if gt_dir.exists() and not force:
        print(f"  CTC structure already exists at {gt_dir}")
        print(f"  Use --force to overwrite")
        return 0

    if gt_dir.exists():
        print(f"  Removing existing {gt_dir}")
        shutil.rmtree(gt_dir)

    # Remove stale cached dirs
    for stale in ["01_MERGED_SEG", "01_GT_filtered"]:
        stale_dir = output_dir / stale
        if stale_dir.exists():
            print(f"  Removing stale {stale_dir}")
            shutil.rmtree(stale_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    seg = tifffile.imread(str(source))
    print(f"  Loaded stack: shape={seg.shape}, dtype={seg.dtype}")

    # For 3D data (ZYX), the stack is used as-is — Z is the tracking axis
    # For 2D+T data (TYX), same thing — T is the tracking axis
    # Both are 3D arrays where axis 0 is the "frame" axis
    if seg.ndim != 3:
        print(f"  ERROR: Expected 3D stack (TYX or ZYX), got {seg.ndim}D")
        return 1

    create_ctc_structure(seg, output_dir, divisions, sequence="01")

    div_str = f" ({len(divisions)} divisions)" if divisions else " (no divisions)"
    print(f"  Done!{div_str}")
    return 0


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["2d", "3d"],
        choices=["2d", "3d"],
        help="Which datasets to prepare. Default: both.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("ctc_data"),
        help="CTC data directory (output). Default: ./ctc_data",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing CTC structure",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Preparing wing disc datasets for CTC benchmark")
    print("=" * 60)

    errors = 0
    for key in args.datasets:
        result = prepare_dataset(key, args.data_dir, args.force)
        errors += result

    print(f"\n{'=' * 60}")
    if errors:
        print(f"Completed with {errors} error(s)")
    else:
        print("All datasets prepared successfully!")
        print(f"\nRun the benchmark with:")
        names = " ".join(WING_DISC_DATASETS[k]["name"] for k in args.datasets)
        print(f"  python benchmark_ctc.py --datasets {names} --skip-optimisation --skip-download")
    print("=" * 60)

    return errors


if __name__ == "__main__":
    exit(main())
