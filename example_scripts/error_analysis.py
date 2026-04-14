"""
error_analysis.py — Detailed AOGM error attribution for easytrack benchmark
============================================================================

For each dataset/sequence, this script:
1. Loads GT and merged segmentation
2. Checks frame-level coverage mismatches (GT expects cell X at frame t,
   but segmentation doesn't have it)
3. Runs tracking (easytrack preset + btrack baseline)
4. Computes full AOGM breakdown by error type
5. Identifies which specific GT cells cause each error
6. Tests division detection (ns_nodes analysis)
7. Reports spatial/temporal error patterns

Usage:
    python error_analysis.py --data-dir ./ctc_data --datasets PhC-C2DH-U373
    python error_analysis.py --data-dir ./ctc_data  # all default datasets
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import tifffile
from skimage import io

# ── Imports from the benchmark / easytrack codebase ──
# These assume benchmark_ctc.py and the easytrack package are importable.
# Adjust sys.path if needed.

from benchmark_ctc import (
    DATASETS,
    DEFAULT_DATASETS,
    merge_gt_st_segmentation,
    load_ctc_segmentation,
    filter_man_track_to_segmentation,
    load_gt_tracking_graph,
    run_easytrack,
    run_btrack_baseline,
    _build_pred_graph,
    download_dataset,
    clean_segmentation,
)

from traccuracy import run_metrics, TrackingGraph
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics


# =====================================================================
# 1. GT vs segmentation coverage analysis
# =====================================================================

def parse_man_track(man_track_path: Path) -> List[dict]:
    """Parse man_track.txt → list of {id, start, end, parent}."""
    tracks = []
    with open(man_track_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            tracks.append({
                "id": int(parts[0]),
                "start": int(parts[1]),
                "end": int(parts[2]),
                "parent": int(parts[3]),
            })
    return tracks

def _parse_tra_frame_index(filename: str) -> int | None:
    """Extract frame index from man_track042.tif."""
    import re
    m = re.search(r'(\d+)\.tif', filename)
    return int(m.group(1)) if m else None


def check_gt_seg_coverage(
    man_track_path: Path,
    segmentation: np.ndarray,
) -> dict:
    """
    Check whether every (cell_id, frame) expected by the GT actually has
    pixels in the filtered TRA masks (which use the correct labels for
    split sub-tracks).

    Falls back to checking the segmentation directly if no TRA masks
    are found in the same directory as man_track.txt.

    Returns a dict with:
      - total_expected: total (cell, frame) pairs in GT
      - total_present: how many have pixels
      - total_missing: how many do NOT have pixels
      - missing_details: list of (cell_id, frame) that are missing
      - cells_with_missing: {cell_id: [list of missing frames]}
      - coverage_by_frame: {frame: {expected: N, present: N, missing: N}}
    """
    tracks = parse_man_track(man_track_path)
    tra_dir = man_track_path.parent

    # Try to use filtered TRA masks for label lookup (handles split IDs)
    tra_tifs = sorted(tra_dir.glob("man_track*.tif"))
    use_tra_masks = len(tra_tifs) > 0

    if use_tra_masks:
        labels_per_frame = {}
        for tif in tra_tifs:
            frame_idx = _parse_tra_frame_index(tif.name)
            if frame_idx is not None:
                mask = tifffile.imread(str(tif))
                labels = set(np.unique(mask).tolist())
                labels.discard(0)
                labels_per_frame[frame_idx] = labels
    else:
        # Fallback: use segmentation labels directly
        labels_per_frame = {}
        for t in range(segmentation.shape[0]):
            labels = set(np.unique(segmentation[t]).tolist())
            labels.discard(0)
            labels_per_frame[t] = labels

    total_expected = 0
    total_present = 0
    missing_details = []
    cells_with_missing = defaultdict(list)
    coverage_by_frame = defaultdict(lambda: {"expected": 0, "present": 0, "missing": 0})

    for track in tracks:
        cell_id = track["id"]
        for t in range(track["start"], track["end"] + 1):
            if t >= segmentation.shape[0]:
                continue
            total_expected += 1
            coverage_by_frame[t]["expected"] += 1
            if cell_id in labels_per_frame.get(t, set()):
                total_present += 1
                coverage_by_frame[t]["present"] += 1
            else:
                missing_details.append((cell_id, t))
                cells_with_missing[cell_id].append(t)
                coverage_by_frame[t]["missing"] += 1

    return {
        "total_expected": total_expected,
        "total_present": total_present,
        "total_missing": len(missing_details),
        "missing_details": missing_details,
        "cells_with_missing": dict(cells_with_missing),
        "coverage_by_frame": dict(coverage_by_frame),
    }




def print_coverage_report(coverage: dict, dataset: str, seq: str) -> None:
    """Print a readable coverage report."""
    print(f"\n  GT vs Segmentation Coverage ({dataset} seq {seq}):")
    print(f"    Total (cell, frame) pairs in GT: {coverage['total_expected']}")
    print(f"    Present in segmentation:         {coverage['total_present']}")
    print(f"    Missing from segmentation:       {coverage['total_missing']}")

    if coverage["total_missing"] == 0:
        print("    → Perfect coverage: every GT cell has pixels at every expected frame.")
        return

    pct = 100.0 * coverage["total_missing"] / max(coverage["total_expected"], 1)
    print(f"    → {pct:.1f}% of expected (cell, frame) pairs are missing")

    cells = coverage["cells_with_missing"]
    print(f"    Cells with missing frames: {len(cells)}")

    # Show worst offenders
    sorted_cells = sorted(cells.items(), key=lambda x: -len(x[1]))
    n_show = min(15, len(sorted_cells))
    print(f"    Top {n_show} cells by missing frames:")
    for cell_id, frames in sorted_cells[:n_show]:
        if len(frames) <= 10:
            frame_str = str(frames)
        else:
            frame_str = f"{frames[:5]}...{frames[-3:]} ({len(frames)} frames)"
        print(f"      Cell {cell_id}: missing at {frame_str}")

    # Frame distribution
    frames_with_missing = {
        t: v for t, v in coverage["coverage_by_frame"].items() if v["missing"] > 0
    }
    if frames_with_missing:
        worst_frames = sorted(frames_with_missing.items(), key=lambda x: -x[1]["missing"])[:10]
        print(f"    Frames with most missing cells:")
        for t, v in worst_frames:
            print(f"      Frame {t}: {v['missing']} missing / {v['expected']} expected")


# =====================================================================
# 2. Detailed AOGM error extraction
# =====================================================================

def compute_detailed_aogm(
    gt_data: TrackingGraph,
    tracked_seg: np.ndarray,
    lbep: np.ndarray,
    label: str = "",
) -> dict:
    """
    Compute CTC metrics and extract the detailed error breakdown,
    including which specific GT nodes/edges are affected.
    """
    pred_data = _build_pred_graph(tracked_seg, lbep)

    ctc_results, matched = run_metrics(
        gt_data=gt_data,
        pred_data=pred_data,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics()],
    )

    result = ctc_results[0]["results"]

    # Extract the individual error counts
    error_breakdown = {
        "TRA": result.get("TRA", None),
        "DET": result.get("DET", None),
        "AOGM": result.get("AOGM", None),
        "AOGM-D": result.get("AOGM-D", None),
        "fn_nodes": result.get("fn_nodes", 0),
        "fp_nodes": result.get("fp_nodes", 0),
        "ns_nodes": result.get("ns_nodes", 0),
        "fn_edges": result.get("fn_edges", 0),
        "fp_edges": result.get("fp_edges", 0),
        "ws_edges": result.get("ws_edges", 0),
    }

    # Weighted contributions
    weights = {
        "fn_nodes": 10,
        "fp_nodes": 1,
        "ns_nodes": 5,
        "fn_edges": 1.5,
        "fp_edges": 1,
        "ws_edges": 1,
    }
    error_breakdown["weighted"] = {}
    for key, w in weights.items():
        count = error_breakdown.get(key, 0)
        if count is None:
            count = 0
        error_breakdown["weighted"][key] = count * w

    return error_breakdown


def print_error_breakdown(errors: dict, label: str) -> None:
    """Print formatted error breakdown."""
    print(f"\n    {label}:")
    print(f"      TRA: {errors['TRA']:.4f}   DET: {errors['DET']:.4f}   AOGM: {errors['AOGM']:.2f}")
    print(f"      Error breakdown:")
    print(f"        fn_nodes (missed detections):    {errors['fn_nodes']:>5}  × 10   = {errors['weighted']['fn_nodes']:>8.1f}")
    print(f"        fp_nodes (spurious detections):  {errors['fp_nodes']:>5}  ×  1   = {errors['weighted']['fp_nodes']:>8.1f}")
    print(f"        ns_nodes (missed divisions):     {errors['ns_nodes']:>5}  ×  5   = {errors['weighted']['ns_nodes']:>8.1f}")
    print(f"        fn_edges (missed links):         {errors['fn_edges']:>5}  ×  1.5 = {errors['weighted']['fn_edges']:>8.1f}")
    print(f"        fp_edges (spurious links):       {errors['fp_edges']:>5}  ×  1   = {errors['weighted']['fp_edges']:>8.1f}")
    print(f"        ws_edges (wrong semantics):      {errors['ws_edges']:>5}  ×  1   = {errors['weighted']['ws_edges']:>8.1f}")

    total_weighted = sum(errors["weighted"].values())
    print(f"        {'─' * 50}")
    print(f"        Total weighted:                              {total_weighted:>8.1f}")

    # Dominant error type
    dominant = max(errors["weighted"].items(), key=lambda x: x[1])
    if dominant[1] > 0:
        pct = 100.0 * dominant[1] / max(total_weighted, 1)
        print(f"      → Dominant error: {dominant[0]} ({pct:.0f}% of AOGM)")


# =====================================================================
# 3. GT lineage / division analysis
# =====================================================================

def analyse_gt_divisions(man_track_path: Path) -> dict:
    """Analyse division events in the ground truth."""
    tracks = parse_man_track(man_track_path)

    # Build parent → children map
    children_of = defaultdict(list)
    for t in tracks:
        if t["parent"] != 0:
            children_of[t["parent"]].append(t["id"])

    divisions = {}
    for parent_id, child_ids in children_of.items():
        if len(child_ids) >= 2:
            # Find parent track
            parent_track = next((t for t in tracks if t["id"] == parent_id), None)
            divisions[parent_id] = {
                "parent_end": parent_track["end"] if parent_track else None,
                "children": child_ids,
                "child_starts": [
                    next((t["start"] for t in tracks if t["id"] == c), None)
                    for c in child_ids
                ],
            }

    # Track duration stats
    durations = [t["end"] - t["start"] + 1 for t in tracks]
    short_lived = [t for t in tracks if t["end"] - t["start"] + 1 <= 2]

    return {
        "n_tracks": len(tracks),
        "n_divisions": len(divisions),
        "divisions": divisions,
        "n_short_lived": len(short_lived),
        "short_lived_ids": [t["id"] for t in short_lived],
        "duration_stats": {
            "min": int(np.min(durations)) if durations else 0,
            "max": int(np.max(durations)) if durations else 0,
            "mean": float(np.mean(durations)) if durations else 0,
            "median": float(np.median(durations)) if durations else 0,
        },
    }


def print_division_report(div_info: dict, dataset: str, seq: str) -> None:
    """Print division analysis."""
    print(f"\n  GT Lineage Analysis ({dataset} seq {seq}):")
    print(f"    Total tracks: {div_info['n_tracks']}")
    print(f"    Division events: {div_info['n_divisions']}")
    print(f"    Short-lived tracks (≤2 frames): {div_info['n_short_lived']}")
    print(f"    Duration stats: min={div_info['duration_stats']['min']}, "
          f"max={div_info['duration_stats']['max']}, "
          f"mean={div_info['duration_stats']['mean']:.1f}, "
          f"median={div_info['duration_stats']['median']:.1f}")

    if div_info["n_divisions"] > 0:
        print(f"    Division details (first 10):")
        for i, (parent, info) in enumerate(list(div_info["divisions"].items())[:10]):
            children = info["children"]
            print(f"      Cell {parent} (ends t={info['parent_end']}) "
                  f"→ {children} (start at t={info['child_starts']})")
        if div_info["n_divisions"] > 10:
            print(f"      ... and {div_info['n_divisions'] - 10} more")


# =====================================================================
# 4. Tracking output analysis
# =====================================================================

def analyse_tracking_output(
    tracked_seg: np.ndarray,
    lbep: np.ndarray,
    label: str = "",
) -> dict:
    """Analyse what the tracker actually produced."""
    n_tracks = lbep.shape[0]
    durations = lbep[:, 2] - lbep[:, 1] + 1
    n_with_parent = int(np.sum(lbep[:, 3] != 0))

    # Check for duplicate labels in single frames
    dup_frames = 0
    for t in range(tracked_seg.shape[0]):
        labels = np.unique(tracked_seg[t])
        labels = labels[labels != 0]
        # Each label should appear once per frame (contiguous region)
        # This is more about checking the output is sane
        if len(labels) == 0:
            continue

    # Count cells per frame
    cells_per_frame = []
    for t in range(tracked_seg.shape[0]):
        labels = np.unique(tracked_seg[t])
        labels = labels[labels != 0]
        cells_per_frame.append(len(labels))

    # Untracked frames (frames where tracked_seg has labels but some
    # segmentation cells might be missing)
    return {
        "n_tracks": n_tracks,
        "n_with_parent": n_with_parent,
        "duration_min": int(np.min(durations)) if n_tracks > 0 else 0,
        "duration_max": int(np.max(durations)) if n_tracks > 0 else 0,
        "duration_mean": float(np.mean(durations)) if n_tracks > 0 else 0,
        "cells_per_frame_mean": float(np.mean(cells_per_frame)),
        "cells_per_frame_min": int(np.min(cells_per_frame)),
        "cells_per_frame_max": int(np.max(cells_per_frame)),
    }


def print_tracking_report(info: dict, label: str) -> None:
    """Print tracking output summary."""
    print(f"      {label}:")
    print(f"        Tracks: {info['n_tracks']}, with parent: {info['n_with_parent']}")
    print(f"        Duration: min={info['duration_min']}, max={info['duration_max']}, "
          f"mean={info['duration_mean']:.1f}")
    print(f"        Cells/frame: min={info['cells_per_frame_min']}, "
          f"max={info['cells_per_frame_max']}, mean={info['cells_per_frame_mean']:.1f}")


# =====================================================================
# 5. Per-cell error attribution (which GT cells cause fn_nodes?)
# =====================================================================

def attribute_fn_nodes(
    gt_man_track: Path,
    tracked_seg: np.ndarray,
) -> dict:
    """
    Identify which GT cells have false-negative nodes by checking
    whether each (cell_id, frame) in the GT has a matching detection
    in the tracked segmentation.

    Uses the filtered TRA masks (in the same directory as gt_man_track)
    for pixel lookup, so that split sub-track IDs (created by the v2
    filter) are resolved correctly.

    A GT node (cell_id, frame) is an fn_node if:
    - The cell exists in the TRA mask at that frame, AND
    - No tracked label covers the majority of that cell's pixels
      (i.e., the tracker failed to assign a track to those pixels)

    This is an approximation — the real CTC matcher uses IoU-based
    matching — but gives us per-cell attribution.
    """
    tracks = parse_man_track(gt_man_track)
    tra_dir = gt_man_track.parent

    # Load filtered TRA masks for pixel lookup (handles split sub-track IDs)
    tra_masks = {}
    for tif in sorted(tra_dir.glob("man_track*.tif")):
        frame_idx = _parse_tra_frame_index(tif.name)
        if frame_idx is not None:
            tra_masks[frame_idx] = tifffile.imread(str(tif))

    fn_nodes = []  # (cell_id, frame, reason)
    matched_nodes = []

    for track in tracks:
        cell_id = track["id"]
        for t in range(track["start"], track["end"] + 1):
            if t >= tracked_seg.shape[0]:
                continue

            # Get the cell's pixels from the TRA mask
            tra_mask = tra_masks.get(t)
            if tra_mask is None:
                fn_nodes.append((cell_id, t, "no_tra_mask"))
                continue

            cell_pixels = tra_mask == cell_id
            if not np.any(cell_pixels):
                fn_nodes.append((cell_id, t, "not_in_tra_mask"))
                continue

            # Does the tracked segmentation have any label covering
            # these pixels?
            tracked_labels_in_mask = tracked_seg[t][cell_pixels]
            tracked_labels_in_mask = tracked_labels_in_mask[tracked_labels_in_mask != 0]

            if len(tracked_labels_in_mask) == 0:
                fn_nodes.append((cell_id, t, "untracked"))
            else:
                # Check overlap quality (approximate IoU)
                dominant_label = np.bincount(tracked_labels_in_mask).argmax()
                overlap = np.sum(tracked_labels_in_mask == dominant_label)
                total_cell = np.sum(cell_pixels)
                total_tracked = np.sum(tracked_seg[t] == dominant_label)
                iou = overlap / (total_cell + total_tracked - overlap)
                if iou < 0.5:
                    fn_nodes.append((cell_id, t, f"low_iou={iou:.2f}"))
                else:
                    matched_nodes.append((cell_id, t))

    # Summarise by cell
    fn_by_cell = defaultdict(list)
    for cell_id, t, reason in fn_nodes:
        fn_by_cell[cell_id].append((t, reason))

    return {
        "total_fn_nodes": len(fn_nodes),
        "total_matched": len(matched_nodes),
        "fn_by_cell": dict(fn_by_cell),
        "fn_details": fn_nodes,
    }




def print_fn_node_report(fn_info: dict, label: str) -> None:
    """Print fn_node attribution."""
    print(f"\n      FN Node Attribution ({label}):")
    print(f"        Total fn_nodes (approx): {fn_info['total_fn_nodes']}")
    print(f"        Total matched:           {fn_info['total_matched']}")

    if fn_info["total_fn_nodes"] == 0:
        print("        → No false-negative nodes detected.")
        return

    # Categorise reasons
    reasons = defaultdict(int)
    for _, _, reason in fn_info["fn_details"]:
        if reason == "not_in_seg":
            reasons["not_in_segmentation"] += 1
        elif reason == "untracked":
            reasons["untracked_pixels"] += 1
        elif reason.startswith("low_iou"):
            reasons["low_iou_match"] += 1
    print(f"        Breakdown by reason:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"          {reason}: {count}")

    # Worst cells
    sorted_cells = sorted(fn_info["fn_by_cell"].items(), key=lambda x: -len(x[1]))
    n_show = min(15, len(sorted_cells))
    print(f"        Top {n_show} cells by fn_node count:")
    for cell_id, entries in sorted_cells[:n_show]:
        frames = [t for t, _ in entries]
        reasons_list = [r for _, r in entries]
        unique_reasons = set(reasons_list)
        if len(frames) <= 8:
            frame_str = str(frames)
        else:
            frame_str = f"[{frames[0]}..{frames[-1]}] ({len(frames)} frames)"
        print(f"          Cell {cell_id}: {len(entries)} fn_nodes at {frame_str} "
              f"({', '.join(unique_reasons)})")


# =====================================================================
# 6. Cleaning impact analysis
# =====================================================================

def analyse_cleaning_impact(
    seg_dir: Path,
    segmentation_cleaned: np.ndarray,
) -> dict:
    """Check how many cells/pixels were removed by clean_segmentation."""
    raw_seg = load_ctc_segmentation(seg_dir).astype(np.uint16)

    raw_labels = set()
    cleaned_labels = set()
    for t in range(raw_seg.shape[0]):
        raw_labels.update(np.unique(raw_seg[t]).tolist())
        cleaned_labels.update(np.unique(segmentation_cleaned[t]).tolist())
    raw_labels.discard(0)
    cleaned_labels.discard(0)

    removed = raw_labels - cleaned_labels
    raw_pixels = int(np.sum(raw_seg > 0))
    cleaned_pixels = int(np.sum(segmentation_cleaned > 0))

    return {
        "raw_labels": len(raw_labels),
        "cleaned_labels": len(cleaned_labels),
        "removed_labels": sorted(removed),
        "n_removed": len(removed),
        "raw_pixels": raw_pixels,
        "cleaned_pixels": cleaned_pixels,
        "pixels_removed": raw_pixels - cleaned_pixels,
    }


# =====================================================================
# Main analysis pipeline
# =====================================================================

def analyse_sequence(
    dataset_name: str,
    sequence: str,
    dataset_dir: Path,
    preset_name: str = "Epithelial Cells (Default)",
    gt_only: bool = False,
) -> dict:
    """Run full error analysis on a single dataset/sequence."""
    print(f"\n{'═' * 70}")
    print(f"  ANALYSIS: {dataset_name} / seq {sequence}")
    print(f"{'═' * 70}")

    tra_dir = dataset_dir / f"{sequence}_GT" / "TRA"
    man_track = tra_dir / "man_track.txt"
    if not man_track.exists():
        print(f"  [skip] No man_track.txt in {tra_dir}")
        return {"error": "no man_track.txt"}

    # ── Load segmentation ──
    if gt_only:
        seg_dir = dataset_dir / f"{sequence}_GT" / "SEG"
    else:
        merged_dir = merge_gt_st_segmentation(dataset_dir, sequence)
        seg_dir = merged_dir if merged_dir else dataset_dir / f"{sequence}_GT" / "SEG"

    if not seg_dir or not seg_dir.exists():
        print(f"  [skip] Segmentation not found")
        return {"error": "segmentation not found"}

    segmentation = load_ctc_segmentation(seg_dir).astype(np.uint16)
    segmentation_cleaned = clean_segmentation(segmentation, verbose=False).astype(np.uint16)

    # ── GT filtering ──
    filtered_tra_dir = filter_man_track_to_segmentation(tra_dir, segmentation_cleaned)
    filtered_man_track = filtered_tra_dir / "man_track.txt"
    if not filtered_man_track.exists():
        filtered_man_track = tra_dir / "man_track.txt"

    # ── 1. Coverage analysis ──
    coverage = check_gt_seg_coverage(filtered_man_track, segmentation_cleaned)
    print_coverage_report(coverage, dataset_name, sequence)

    # ── 2. Division / lineage analysis ──
    div_info = analyse_gt_divisions(filtered_man_track)
    print_division_report(div_info, dataset_name, sequence)

    # ── 3. Cleaning impact ──
    cleaning = analyse_cleaning_impact(seg_dir, segmentation_cleaned)
    print(f"\n  Cleaning Impact:")
    print(f"    Labels: {cleaning['raw_labels']} → {cleaning['cleaned_labels']} "
          f"(removed {cleaning['n_removed']})")
    if cleaning["n_removed"] > 0:
        print(f"    Removed label IDs: {cleaning['removed_labels'][:20]}"
              f"{'...' if cleaning['n_removed'] > 20 else ''}")
    print(f"    Pixels: {cleaning['raw_pixels']} → {cleaning['cleaned_pixels']} "
          f"(removed {cleaning['pixels_removed']})")

    # ── 4. Run tracking & evaluate ──
    gt_data_et = load_gt_tracking_graph(filtered_tra_dir)

    results = {}

    # easytrack preset
    print(f"\n  Running easytrack ({preset_name!r})...")
    try:
        tracked_et, lbep_et = run_easytrack(segmentation_cleaned, preset_name)
        errors_et = compute_detailed_aogm(gt_data_et, tracked_et, lbep_et, "easytrack")
        print_error_breakdown(errors_et, f"easytrack ({preset_name})")

        track_info_et = analyse_tracking_output(tracked_et, lbep_et, "easytrack")
        print_tracking_report(track_info_et, "easytrack output")

        fn_et = attribute_fn_nodes(filtered_man_track, tracked_et)
        print_fn_node_report(fn_et, "easytrack")

        results["easytrack"] = {
            "errors": errors_et,
            "tracking": track_info_et,
            "fn_attribution": fn_et,
        }
    except Exception as e:
        print(f"  easytrack ERROR: {e}")
        traceback.print_exc()

    # btrack baseline
    print(f"\n  Running btrack baseline...")
    gt_data_bt = load_gt_tracking_graph(filtered_tra_dir)
    try:
        tracked_bt, lbep_bt = run_btrack_baseline(segmentation_cleaned)
        errors_bt = compute_detailed_aogm(gt_data_bt, tracked_bt, lbep_bt, "btrack")
        print_error_breakdown(errors_bt, "btrack baseline")

        track_info_bt = analyse_tracking_output(tracked_bt, lbep_bt, "btrack")
        print_tracking_report(track_info_bt, "btrack output")

        fn_bt = attribute_fn_nodes(filtered_man_track, tracked_bt)
        print_fn_node_report(fn_bt, "btrack")

        results["btrack"] = {
            "errors": errors_bt,
            "tracking": track_info_bt,
            "fn_attribution": fn_bt,
        }
    except Exception as e:
        print(f"  btrack ERROR: {e}")
        traceback.print_exc()

    # ── 5. Comparison summary ──
    if "easytrack" in results and "btrack" in results:
        print(f"\n  {'─' * 60}")
        print(f"  COMPARISON SUMMARY ({dataset_name} seq {sequence}):")
        print(f"  {'─' * 60}")
        for err_type in ["fn_nodes", "fp_nodes", "ns_nodes", "fn_edges", "fp_edges", "ws_edges"]:
            et_val = results["easytrack"]["errors"].get(err_type, 0) or 0
            bt_val = results["btrack"]["errors"].get(err_type, 0) or 0
            diff = et_val - bt_val
            arrow = "←" if diff < 0 else ("→" if diff > 0 else "=")
            print(f"    {err_type:>10}: easytrack={et_val:>4}  btrack={bt_val:>4}  "
                  f"diff={diff:>+4} {arrow}")

    results["coverage"] = coverage
    results["divisions"] = div_info
    results["cleaning"] = cleaning
    return results


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ctc_data"))
    parser.add_argument("--datasets", nargs="+",
                        default=["PhC-C2DH-U373", "DIC-C2DH-HeLa", "Fluo-N2DH-GOWT1"])
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--preset", default="Epithelial Cells (Default)")
    parser.add_argument("--gt-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("easytrack — Detailed AOGM Error Analysis")
    print("=" * 70)

    data_dir = args.data_dir.resolve()

    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            print(f"  [skip] Unknown dataset: {dataset_name}")
            continue

        info = DATASETS[dataset_name]
        dataset_dir = data_dir / dataset_name

        if not dataset_dir.exists():
            if not args.skip_download:
                try:
                    download_dataset(dataset_name, data_dir)
                except Exception as e:
                    print(f"  [skip] Download failed for {dataset_name}: {e}")
                    continue
            else:
                print(f"  [skip] {dataset_dir} not found")
                continue

        sequences = args.sequences or info["sequences"]
        for seq in sequences:
            try:
                analyse_sequence(
                    dataset_name=dataset_name,
                    sequence=seq,
                    dataset_dir=dataset_dir,
                    preset_name=args.preset,
                    gt_only=args.gt_only,
                )
            except Exception as e:
                print(f"  ERROR analysing {dataset_name}/{seq}: {e}")
                traceback.print_exc()

    print(f"\n{'═' * 70}")
    print("Analysis complete.")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()