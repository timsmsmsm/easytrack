"""
benchmark_ctc.py — Benchmark easytrack on Cell Tracking Challenge datasets
==========================================================================

Downloads (optional) and evaluates easytrack on Cell Tracking Challenge
datasets, comparing easytrack presets against btrack defaults and
optionally running Optuna-based parameter optimisation.

By default the script merges gold-truth (GT) and silver-truth (ST)
segmentation masks before tracking, giving easytrack full frame coverage.
GT masks take priority; ST fills the gaps.  Each cell is relabelled with
its tracking (TRA) identity so labels are consistent over time.
Use ``--gt-only`` to skip the merge and use only GT SEG masks (original
behaviour).

Usage
-----
    # Download data automatically and benchmark both datasets:
    python benchmark_ctc.py

    # Use only GT segmentation (no ST merge):
    python benchmark_ctc.py --gt-only

    # Use already-downloaded data (skip the download step):
    python benchmark_ctc.py --skip-download --data-dir /path/to/ctc_data

    # Benchmark a single dataset:
    python benchmark_ctc.py --datasets Fluo-C2DL-Huh7

    # Choose a different preset:
    python benchmark_ctc.py --preset "Epithelial Cells (Default)"

    # Override the output directory for results:
    python benchmark_ctc.py --results-dir ./my_results

    # Run only the optimisation benchmark:
    python benchmark_ctc.py --only-optimisation

    # Skip optimisation (only preset-based tracking):
    python benchmark_ctc.py --skip-optimisation

    # Customise optimisation trials:
    python benchmark_ctc.py --n-random 32 --n-tpe 64 --optim-timeout 30

Data citation
-------------
If you use the CTC datasets in a publication, please cite:
  • Maška et al., Nature Methods 20, 1010–1020 (2023)
    https://doi.org/10.1038/s41592-023-01879-y
  • Ulman et al., Nature Methods 14, 1141–1152 (2017)
    https://doi.org/10.1038/nmeth.4473

And acknowledge the source: Cell Tracking Challenge
  https://celltrackingchallenge.net/

Conditions of use
-----------------
  1. Include the citations above in any publication using these datasets.
  2. CTC-related use does not require consent from challenge organisers.
  3. Public non-CTC scientific use requires explicit permission from organisers.
  4. Commercial use requires permission from organisers AND data providers.
  5. Cloning datasets or reference annotations is strictly forbidden.
     Use the official download links to share or reference them.
"""

from __future__ import annotations

import argparse
import csv
import ftplib
import glob
import json
import os
import re
import shutil
import ssl
import sys
import traceback
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pprint
import tifffile
from numpy import dtype, ndarray
from skimage import io
from traccuracy import run_metrics, TrackingGraph
from traccuracy.loaders import load_tiffs
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics

from napari_easytrack.analysis.optim_backend import _build_label_timepoints
from napari_easytrack.analysis.tracking import run_tracking_with_params
from napari_easytrack.presets import get_presets
from napari_easytrack.utils import clean_segmentation
from prepare_wing_disc import prepare_dataset as prepare_wing_disc_dataset, WING_DISC_DATASETS

import btrack

pp = pprint.PrettyPrinter(indent=4)


# ---------------------------------------------------------------------------
# SSL helper
# ---------------------------------------------------------------------------

def _make_ssl_context() -> ssl.SSLContext:
    """
    Return an SSL context that works on macOS even when the system
    certificate store is not wired up to Python (the common
    'certificate verify failed' error after a plain Python install).

    Strategy:
    1. Try the default context (works when certifi or system certs are fine).
    2. Fall back to certifi's bundle if it is installed.
    3. Last resort: unverified context (prints a warning).
    """
    try:
        ctx = ssl.create_default_context()
        import urllib.request as _ur
        _ur.urlopen("https://celltrackingchallenge.net/", context=ctx, timeout=5).close()
        return ctx
    except Exception:
        pass

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except ImportError:
        pass

    import warnings
    warnings.warn(
        "SSL certificate verification disabled. "
        "Install certifi (`pip install certifi`) or run "
        "`/Applications/Python\\ 3.x/Install\\ Certificates.command` "
        "for a proper fix.",
        stacklevel=2,
    )
    ctx = ssl._create_unverified_context()
    return ctx


# ---------------------------------------------------------------------------
# Dataset catalogue
# ---------------------------------------------------------------------------

DATASETS: Dict[str, Dict] = {
    "PhC-C2DH-U373": {
        "url": "https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip",
        "description": "2D+time: Glioblastoma-astrocytoma U373 cells on a polyacrylamide substrate",
        "is_3d": False,
        "has_st": True,
        "sequences": ["01", "02"],
        "downloadable": True,
    },
    "DIC-C2DH-HeLa": {
        "url": "https://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip",
        "description": "2D+time: HeLa cells on a flat glass (DIC microscopy)",
        "is_3d": False,
        "has_st": True,
        "sequences": ["01", "02"],
        "downloadable": True,
    },
    "Fluo-N2DH-GOWT1": {
        "url": "https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip",
        "description": "2D+time: GFP-GOWT1 mouse stem cells (fluorescence)",
        "is_3d": False,
        "has_st": True,
        "sequences": ["01", "02"],
        "downloadable": True,
    },
    "Fluo-C3DH-A549-SIM": {
        "url": "https://data.celltrackingchallenge.net/training-datasets/Fluo-C3DH-A549-SIM.zip",
        "description": "3D+time: Simulated GFP-actin-stained A549 Lung Cancer cells embedded in a Matrigel matrix",
        "is_3d": True,
        "has_st": True,
        "sequences": ["01", "02"],
        "downloadable": True,
    },
    "wing_disc": {
        "url": "../example_data/2d_time",
        "description": "2D+time: Drosophila wing disc epithelium wound healing (example data) and 3D: Individual 3D cell shapes of Drosophila Wing Disc",
        "is_3d": False,
        "has_st": False,
        "sequences": ["01", "02"],
        "downloadable": False,
    },
}

# Default datasets to benchmark (those with silver truth for full coverage)
DEFAULT_DATASETS = ["wing_disc", "PhC-C2DH-U373", "DIC-C2DH-HeLa", "Fluo-N2DH-GOWT1"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    """Simple download progress indicator."""
    downloaded = count * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * downloaded / total_size)
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%")
        sys.stdout.flush()


def download_ftp(target_dir: Path) -> None:
    """Download specified files from EBI FTP server."""

    # FTP server details
    HOST = "ftp.ebi.ac.uk"
    USER = "anonymous"

    # Remote directory path
    REMOTE_DIR = "biostudies/fire/S-BIAD/843/S-BIAD843/Files"

    # Files to download
    files_to_download = [
        "WD1.1_17-03_WT_MP.ome.zarr.zip",
        "WD3.2_21-03_WT_MP.ome.zarr.zip",
        "WD1_15-02_WT_confocalonly.ome.zarr.zip",
        "WD2.1_21-02_WT_confocalonly.ome.zarr.zip",
        "WD1_15-02_WT_confocalonly_segmented.ome.zarr.zip",
        "WD1.1_17-03_WT_MP_segmented.ome.zarr.zip",
        "WD2.1_21-02_WT_confocalonly_segmented.ome.zarr.zip",
        "WD3.2_21-03_WT_MP_segmented.ome.zarr.zip"
    ]

    try:
        print(f"Connecting to {HOST}...")
        ftp = ftplib.FTP(HOST)
        ftp.login(user=USER)
        print("Login successful")
        print(f"Changing to directory: {REMOTE_DIR}")
        ftp.cwd(REMOTE_DIR)
        ftp.voidcmd('TYPE I')
        print("Binary mode set")

        for filename in files_to_download:
            print(f"\nDownloading: {filename}")
            if Path(target_dir / filename).exists():
                print(f"  File {filename} already exists, skipping...")
                continue
            try:
                with open(target_dir / filename, 'wb') as local_file:
                    ftp.retrbinary(f'RETR {filename}', local_file.write)
                print(f"  Successfully downloaded: {filename}")
            except ftplib.error_perm as e:
                print(f"  Error downloading {filename}: {e}")
                continue
            print(f"  Extracting to {target_dir} …")
            zip_path = target_dir / filename
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
            zip_path.unlink()

        ftp.quit()
        print("\nAll downloads completed!")

    except ftplib.all_errors as e:
        print(f"FTP error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def download_dataset(name: str, data_dir: Path) -> Path:
    """Download and extract a CTC training dataset."""
    info = DATASETS[name]
    url = info["url"]
    target_dir = data_dir / name

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"  [skip] {name} already present at {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    if info["downloadable"]:
        if info["downloadable"] == "FTP":
            download_ftp(target_dir)
        else:
            download_ssl(data_dir, name, target_dir, url)
    else:
        source_path = Path(url)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path for {name} not found: {source_path}")
        if source_path.is_dir():
            if not any(target_dir.iterdir()):
                print(f"  Copying {name} from {source_path} to {target_dir} …")
                for item in source_path.iterdir():
                    dest = target_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            else:
                print(f"  [skip] Target directory {target_dir} already has content.")
        else:
            raise ValueError(f"URL for {name} is not a directory: {url}")

    return target_dir


def download_ssl(data_dir: Path, name: str, target_dir: Path, url):
    zip_path = data_dir / f"{name}.zip"
    print(f"  Downloading {name} from:\n    {url}")
    try:
        ssl_ctx = _make_ssl_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_ctx) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 8192
            count = 0
            with open(zip_path, "wb") as out_f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    count += 1
                    _progress_hook(count, block_size, total_size)
        print()
    except Exception as exc:
        print(f"\n  ERROR downloading {name}: {exc}")
        print("  Please download manually and re-run with --skip-download.")
        raise

    print(f"  Extracting to {target_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    zip_path.unlink()


# ---------------------------------------------------------------------------
# GT + ST segmentation merge helpers
# ---------------------------------------------------------------------------

def _parse_frame_index(filename: str) -> int:
    """Extract the zero-based frame index from a filename like man_seg042.tif."""
    m = re.search(r'(\d+)\.tif', os.path.basename(filename))
    if m is None:
        raise ValueError(f"Cannot parse frame index from {filename}")
    return int(m.group(1))


def _discover_seg_files(directory: str) -> dict[int, str]:
    """Return {frame_index: filepath} for all man_seg*.tif in a directory."""
    result = {}
    for fp in sorted(glob.glob(os.path.join(directory, "man_seg*.tif"))):
        result[_parse_frame_index(fp)] = fp
    return result


def _discover_tra_files(directory: str) -> dict[int, str]:
    """Return {frame_index: filepath} for all man_track*.tif in a directory."""
    result = {}
    for fp in sorted(glob.glob(os.path.join(directory, "man_track*.tif"))):
        result[_parse_frame_index(fp)] = fp
    return result


def _relabel_seg_with_tra(seg: np.ndarray, tra: np.ndarray) -> np.ndarray:
    """Relabel a segmentation mask using tracking marker labels."""
    out = np.zeros_like(seg, dtype=np.uint16)
    seg_labels = np.unique(seg)
    seg_labels = seg_labels[seg_labels != 0]

    for seg_label in seg_labels:
        seg_mask = seg == seg_label
        tra_in_seg = tra[seg_mask]
        tra_in_seg = tra_in_seg[tra_in_seg != 0]
        if len(tra_in_seg) == 0:
            continue
        labels, counts = np.unique(tra_in_seg, return_counts=True)
        out[seg_mask] = labels[np.argmax(counts)]

    return out


def _stack_folder_to_tif(folder: str) -> str | None:
    """Stack all mask*.tif frames in *folder* into a single ImageJ TIFF."""
    files = sorted(glob.glob(os.path.join(folder, "mask*.tif")))
    if not files:
        print(f"    [stack] No mask*.tif files in {folder} — skipping")
        return None

    frames = [tifffile.imread(f) for f in files]
    stack = np.stack(frames, axis=0).astype(np.uint16)

    ndim = stack.ndim
    if ndim == 3:
        axes = "TYX"
    elif ndim == 4:
        axes = "TZYX"
    else:
        axes = None

    out_path = os.path.join(folder, "merged_stack.tif")
    imagej_kwargs = {"imagej": True}
    if axes is not None:
        imagej_kwargs["metadata"] = {"axes": axes}

    tifffile.imwrite(out_path, stack, **imagej_kwargs)
    axes_str = axes if axes else f"{ndim}D"
    print(f"    [stack] {out_path}: {stack.shape} {stack.dtype} ({axes_str})")
    return out_path


def merge_gt_st_segmentation(
    dataset_dir: Path,
    sequence: str,
) -> Path | None:
    """
    Merge GT and ST segmentation masks for one CTC sequence, relabelling
    every cell with its TRA tracking identity.

    For frames where both GT and ST segmentation exist, GT masks take
    priority on a per-pixel basis: GT cells are used first, then ST cells
    fill in any remaining regions not covered by GT. This ensures full
    cell coverage even when GT only annotates a subset of cells.

    For frames with only GT or only ST, that source is used directly.
    """
    gt_seg_dir = str(dataset_dir / f"{sequence}_GT" / "SEG")
    st_seg_dir = str(dataset_dir / f"{sequence}_ST" / "SEG")
    tra_dir    = str(dataset_dir / f"{sequence}_GT" / "TRA")
    out_dir    = str(dataset_dir / f"{sequence}_MERGED_SEG")

    if os.path.exists(out_dir):
        return Path(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    gt_seg_files = _discover_seg_files(gt_seg_dir) if os.path.isdir(gt_seg_dir) else {}
    st_seg_files = _discover_seg_files(st_seg_dir) if os.path.isdir(st_seg_dir) else {}
    tra_files    = _discover_tra_files(tra_dir)     if os.path.isdir(tra_dir) else {}

    if not tra_files:
        print(f"    [WARNING] No TRA markers in {tra_dir} — cannot merge")
        return None

    all_frames = sorted(set(gt_seg_files.keys()) | set(st_seg_files.keys()))
    print(f"    Merge: {len(gt_seg_files)} GT + {len(st_seg_files)} ST frames, "
          f"{len(tra_files)} TRA frames → {len(all_frames)} union frames")

    n_gt_only = n_st_only = n_combined = n_skip = 0
    n_relabelled = 0

    for frame_idx in all_frames:
        if frame_idx not in tra_files:
            n_skip += 1
            continue

        tra = tifffile.imread(tra_files[frame_idx])
        has_gt = frame_idx in gt_seg_files
        has_st = frame_idx in st_seg_files

        if has_gt and has_st:
            gt_seg = tifffile.imread(gt_seg_files[frame_idx])
            st_seg = tifffile.imread(st_seg_files[frame_idx])
            if gt_seg.shape != tra.shape or st_seg.shape != tra.shape:
                print(f"    [WARNING] Frame {frame_idx:04d}: shape mismatch — skipping")
                continue
            st_relabelled = _relabel_seg_with_tra(st_seg, tra)
            gt_relabelled = _relabel_seg_with_tra(gt_seg, tra)
            combined = st_relabelled.copy()
            gt_mask = gt_relabelled > 0
            combined[gt_mask] = gt_relabelled[gt_mask]
            relabelled = combined
            n_combined += 1
        elif has_gt:
            gt_seg = tifffile.imread(gt_seg_files[frame_idx])
            if gt_seg.shape != tra.shape:
                print(f"    [WARNING] Frame {frame_idx:04d}: shape mismatch — skipping")
                continue
            relabelled = _relabel_seg_with_tra(gt_seg, tra)
            n_gt_only += 1
        else:
            st_seg = tifffile.imread(st_seg_files[frame_idx])
            if st_seg.shape != tra.shape:
                print(f"    [WARNING] Frame {frame_idx:04d}: shape mismatch — skipping")
                continue
            relabelled = _relabel_seg_with_tra(st_seg, tra)
            n_st_only += 1

        n_cells = len(np.unique(relabelled)) - (1 if 0 in relabelled else 0)
        n_relabelled += n_cells

        width = 3 if frame_idx < 1000 else 4
        tifffile.imwrite(
            os.path.join(out_dir, f"mask{frame_idx:0{width}d}.tif"),
            relabelled.astype(np.uint16),
        )

    tra_txt = os.path.join(tra_dir, "man_track.txt")
    if os.path.isfile(tra_txt):
        shutil.copy2(tra_txt, os.path.join(out_dir, "man_track.txt"))

    print(f"    Merge result: {n_combined} combined (GT+ST), "
          f"{n_gt_only} GT-only, {n_st_only} ST-only frames, "
          f"{n_skip} skipped (no TRA), "
          f"{n_relabelled} total cells relabelled")

    _stack_folder_to_tif(out_dir)
    return Path(out_dir)


# ---------------------------------------------------------------------------
# CTC data loading helpers
# ---------------------------------------------------------------------------

def load_ctc_segmentation(seg_dir: Path) -> np.ndarray:
    """Load CTC segmentation masks as a numpy array."""
    mask_files = sorted(seg_dir.glob("mask*.tif"))
    if not mask_files:
        mask_files = sorted(seg_dir.glob("man_seg*.tif"))
    if not mask_files:
        raise FileNotFoundError(
            f"No mask*.tif or man_seg*.tif files found in {seg_dir}"
        )
    frames = [io.imread(str(f)) for f in mask_files]
    segmentation = np.stack(frames, axis=0)
    print(f"    Loaded {len(frames)} frames → shape {segmentation.shape}, "
          f"dtype {segmentation.dtype}")
    return segmentation


def _fill_gaps_in_segmentation(segmentation):
    """Fill temporal gaps with placeholder pixels (for benchmark GT loading only)."""
    filled = segmentation.copy()
    label_timepoints = _build_label_timepoints(segmentation)
    
    for label, timepoints in label_timepoints.items():
        if len(timepoints) < 2:
            continue
        first_t, last_t = min(timepoints), max(timepoints)
        gaps = sorted(set(range(first_t, last_t + 1)) - set(timepoints))
        
        for gap_t in gaps:
            prev_t = max(t for t in timepoints if t < gap_t)
            mask = (segmentation[prev_t] == label)
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            if filled[gap_t, cy, cx] == 0:
                filled[gap_t, cy, cx] = label
    
    return filled


def load_gt_tracking_graph(tra_dir: Path) -> TrackingGraph:
    """Load CTC ground-truth as a traccuracy TrackingGraph."""
    from traccuracy.loaders import load_ctc_data

    track_paths = list(glob.glob(str(tra_dir / "man_track.txt")))
    if not track_paths:
        masks = load_tiffs(str(tra_dir))
        filled_segmentation = _fill_gaps_in_segmentation(masks)
        for t in range(filled_segmentation.shape[0]):
            frame_path = tra_dir / f"mask{t:03d}.tif"
            io.imsave(str(frame_path), filled_segmentation[t], check_contrast=False)
        lbep = _extract_lineage_from_tracked_seg(filled_segmentation)
        res_track_path = tra_dir / "man_track.txt"
        write_lbep_to_csv(lbep, res_track_path)

    gt_graph = load_ctc_data(
        str(tra_dir),
        str(tra_dir / "man_track.txt"),
        name="ground_truth",
    )
    print(f"    GT nodes: {len(gt_graph.nodes())}, GT edges: {len(gt_graph.edges())}")
    return gt_graph


def write_lbep_to_csv(lbep: ndarray[tuple[Any, ...], dtype[Any]], res_track_path: Path):
    with open(res_track_path, "w") as f:
        for idx in range(lbep.shape[0]):
            cell_id = int(lbep[idx, 0])
            start_frame = int(lbep[idx, 1])
            end_frame = int(lbep[idx, 2])
            parent_id = int(lbep[idx, 3])
            if parent_id == cell_id:
                parent_id = 0
            f.write(f"{cell_id} {start_frame} {end_frame} {parent_id}\n")


# ---------------------------------------------------------------------------
# Tracking runners
# ---------------------------------------------------------------------------

def _build_pred_graph(tracked_seg: np.ndarray, lbep: np.ndarray) -> TrackingGraph:
    """Build a traccuracy TrackingGraph from easytrack / btrack output."""
    from napari_easytrack.analysis.optim_pipeline import (
        ctc_to_graph,
        add_missing_attributes,
        _get_node_attributes,
    )
    import pandas as pd

    tracks_df = pd.DataFrame({
        "Cell_ID": lbep[:, 0],
        "Start": lbep[:, 1],
        "End": lbep[:, 2],
        "Parent_ID": [
            0 if lbep[idx, 3] == lbep[idx, 0] else lbep[idx, 3]
            for idx in range(lbep.shape[0])
        ],
    })
    detections_df = _get_node_attributes(tracked_seg)
    G = ctc_to_graph(tracks_df, detections_df)
    add_missing_attributes(G)
    return TrackingGraph(G, tracked_seg)


def _extract_lineage_from_tracked_seg(tracked_seg: np.ndarray) -> np.ndarray:
    """Extract lineage (LBEP) from tracked segmentation. Parent is always 0."""
    lbep_rows = []
    cell_ids = np.unique(tracked_seg)
    cell_ids = cell_ids[cell_ids != 0]
    axes = tuple(range(1, tracked_seg.ndim))
    
    for cell_id in cell_ids:
        frames = np.where(np.any(tracked_seg == cell_id, axis=axes))[0]
        if len(frames) == 0:
            continue
        lbep_rows.append([int(cell_id), int(frames[0]), int(frames[-1]), 0])

    return np.array(lbep_rows, dtype=np.int64)


def _save_ctc_results(
    tracked_seg: np.ndarray,
    lbep: np.ndarray,
    res_dir: Path,
) -> None:
    """Save tracking results in CTC format."""
    res_dir.mkdir(exist_ok=True)
    res_track_path = res_dir / "res_track.txt"
    write_lbep_to_csv(lbep, res_track_path)
    print(f"      Saved {res_track_path}")

    for t in range(tracked_seg.shape[0]):
        frame_path = res_dir / f"mask{t:03d}.tif"
        io.imsave(str(frame_path), tracked_seg[t], check_contrast=False)
    print(f"      Saved {tracked_seg.shape[0]} tracked frames to {res_dir}")
    _stack_folder_to_tif(str(res_dir))


def _evaluate_tracking(
    gt_data,
    tracked_seg: np.ndarray,
    lbep: np.ndarray,
) -> Dict:
    """Build prediction graph and compute CTC metrics."""
    pred_data = _build_pred_graph(tracked_seg, lbep)
    ctc_results, _ = run_metrics(
        gt_data=gt_data,
        pred_data=pred_data,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics()],
    )
    pp.pprint(ctc_results)
    return {
        "TRA": ctc_results[0]["results"]["TRA"],
        "DET": ctc_results[0]["results"]["DET"],
        "AOGM": ctc_results[0]["results"]["AOGM"],
    }


def run_easytrack(
    segmentation: np.ndarray,
    preset_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run easytrack on segmentation and return (tracked_seg, lbep)."""
    presets = get_presets()
    params = presets[preset_name]["config"]
    tracked_seg, tracks, stats, lbep = run_tracking_with_params(
        segmentation=segmentation,
        params=params,
        return_napari=False,
    )
    return tracked_seg, lbep


def run_btrack_baseline(
    segmentation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run btrack directly with default cell_config.json on segmentation."""
    ndim = segmentation.ndim
    if ndim == 3:
        T, Y, X = segmentation.shape
        volume = ((0, X), (0, Y))
    elif ndim == 4:
        T, Z, Y, X = segmentation.shape
        volume = ((0, X), (0, Y), (0, Z))
    else:
        raise ValueError(f"Unexpected segmentation ndim={ndim}")

    objects = btrack.utils.segmentation_to_objects(
        segmentation, properties=("area",)
    )
    from btrack import datasets
    btrack_config = datasets.cell_config()

    with btrack.BayesianTracker() as tracker:
        tracker.configure(btrack_config)
        tracker.append(objects)
        tracker.volume = volume
        tracker.track_interactive(step_size=100)
        tracker.optimize()
        btrack_tracks = tracker.tracks

    print(f"    [btrack] Found {len(btrack_tracks)} tracks")

    tracked_seg = np.zeros_like(segmentation, dtype=np.uint16)
    lbep_rows = []
    for track in btrack_tracks:
        track_id = track.ID
        start_frame = int(track.t[0])
        end_frame = int(track.t[-1])
        parent_id = int(track.parent) if track.parent is not None else 0
        lbep_rows.append([track_id, start_frame, end_frame, parent_id])

        for t_idx, t_frame in enumerate(track.t):
            t_frame = int(t_frame)
            if t_frame < 0 or t_frame >= segmentation.shape[0]:
                continue
            y = int(round(track.y[t_idx]))
            x = int(round(track.x[t_idx]))
            if ndim == 4:
                z = int(round(track.z[t_idx]))
                z = max(0, min(z, segmentation.shape[1] - 1))
                y = max(0, min(y, segmentation.shape[2] - 1))
                x = max(0, min(x, segmentation.shape[3] - 1))
                seg_label = segmentation[t_frame, z, y, x]
                if seg_label > 0:
                    tracked_seg[t_frame][segmentation[t_frame] == seg_label] = track_id
            else:
                y = max(0, min(y, segmentation.shape[1] - 1))
                x = max(0, min(x, segmentation.shape[2] - 1))
                seg_label = segmentation[t_frame, y, x]
                if seg_label > 0:
                    tracked_seg[t_frame][segmentation[t_frame] == seg_label] = track_id

    lbep = np.array(lbep_rows, dtype=np.int64)
    n_tracks = len(btrack_tracks)
    n_long = sum(1 for t in btrack_tracks if len(t.t) > 1)
    print(f"    [btrack] Tracks > 1 frame: {n_long} of {n_tracks}")
    return tracked_seg, lbep


# ---------------------------------------------------------------------------
# Per-sequence preset benchmark
# ---------------------------------------------------------------------------

from gt_filtering import filter_man_track_to_segmentation_v2 as filter_man_track_to_segmentation

def _ensure_wing_disc_prepared(dataset_name: str, data_dir: Path) -> None:
    """Auto-run prepare_wing_disc if GT structure doesn't exist yet."""
    wing_disc_keys = {v["name"]: k for k, v in WING_DISC_DATASETS.items()}
    if dataset_name not in wing_disc_keys:
        return
    gt_dir = data_dir / dataset_name / "01_GT"
    if gt_dir.exists():
        return
    key = wing_disc_keys[dataset_name]
    print(f"    Wing disc GT not found — running preparation automatically...")
    prepare_wing_disc_dataset(key, data_dir)

def benchmark_sequence(
    dataset_name: str,
    sequence: str,
    dataset_dir: Path,
    preset_name: str,
    gt_only: bool = False,
    skip_btrack: bool = False,
) -> List[Dict]:
    """Benchmark easytrack (and optionally btrack baseline) on a single CTC sequence."""
    tra_dir = dataset_dir / f"{sequence}_GT" / "TRA"
    man_track = tra_dir / "man_track.txt"
    if not man_track.exists():
        print(f"  [skip] man_track.txt not found in {tra_dir}")

    print(f"\n  ── Sequence {sequence} ──")
    print(f"    GT dir: {tra_dir}")

    results = []
    res_dir_et = dataset_dir / f"{sequence}_RES_easytrack"
    res_dir_bt = dataset_dir / f"{sequence}_RES_btrack"

    try:
        _ensure_wing_disc_prepared(dataset_name, dataset_dir.parent)

        if gt_only:
            seg_dir = dataset_dir / f"{sequence}_GT" / "SEG"
            seg_source = "GT only"
            if not seg_dir.exists():
                print(f"  [skip] GT SEG directory not found: {seg_dir}")
                return [{"dataset": dataset_name, "sequence": sequence, "error": "GT SEG not found"}]
        else:
            info = DATASETS.get(dataset_name, {})
            has_st = info.get("has_st", True)
            if has_st:
                print("    Merging GT + ST segmentation …")
                merged_dir = merge_gt_st_segmentation(dataset_dir, sequence)
                if merged_dir is None:
                    print("    Falling back to GT SEG only")
                    seg_dir = dataset_dir / f"{sequence}_GT" / "SEG"
                    seg_source = "GT only (merge failed)"
                    if not seg_dir.exists():
                        return [{"dataset": dataset_name, "sequence": sequence, "error": "GT SEG not found and merge failed"}]
                else:
                    seg_dir = merged_dir
                    seg_source = "GT + ST merged"
            else:
                print("    No ST segmentation available, using GT SEG …")
                seg_dir = dataset_dir / f"{sequence}_GT" / "SEG"
                seg_source = "GT only"
                if not seg_dir.exists():
                    return [{"dataset": dataset_name, "sequence": sequence, "error": "GT SEG not found"}]

        print(f"    Loading segmentation ({seg_source}) …")
        segmentation = load_ctc_segmentation(seg_dir)
        segmentation = segmentation.astype(np.uint16)

        print("    Cleaning segmentation …")
        segmentation = clean_segmentation(segmentation, verbose=False)
        segmentation = segmentation.astype(np.uint16)

        print("    Filtering GT to match segmentation coverage …")
        filtered_tra_dir = filter_man_track_to_segmentation(tra_dir, segmentation)

        print("    Loading GT tracking graph …")
        gt_data = load_gt_tracking_graph(filtered_tra_dir)

        # easytrack
        print(f"\n    ▸ easytrack (preset: {preset_name!r})")
        try:
            tracked_seg_et, lbep_et = run_easytrack(segmentation, preset_name)
            print("    Building prediction graph …")
            et_metrics = _evaluate_tracking(gt_data, tracked_seg_et, lbep_et)
            print("    Saving results …")
            _save_ctc_results(tracked_seg_et, lbep_et, res_dir_et)
            results.append({
                "dataset": dataset_name, "sequence": sequence, "method": "easytrack",
                "preset": preset_name, "seg_source": seg_source,
                "n_frames": segmentation.shape[0], "input_shape": str(segmentation.shape),
                **et_metrics,
            })
        except Exception as exc:
            print(f"    easytrack ERROR: {exc}")
            traceback.print_exc()
            results.append({"dataset": dataset_name, "sequence": sequence,
                            "method": "easytrack", "preset": preset_name, "error": str(exc)})

        # btrack baseline
        if not skip_btrack:
            print(f"\n    ▸ btrack baseline (default cell_config)")
            try:
                gt_data = load_gt_tracking_graph(filtered_tra_dir)
                tracked_seg_bt, lbep_bt = run_btrack_baseline(segmentation)
                print("    Building prediction graph …")
                bt_metrics = _evaluate_tracking(gt_data, tracked_seg_bt, lbep_bt)
                print("    Saving results …")
                _save_ctc_results(tracked_seg_bt, lbep_bt, res_dir_bt)
                results.append({
                    "dataset": dataset_name, "sequence": sequence, "method": "btrack",
                    "preset": "default cell_config", "seg_source": seg_source,
                    "n_frames": segmentation.shape[0], "input_shape": str(segmentation.shape),
                    **bt_metrics,
                })
            except Exception as exc:
                print(f"    btrack ERROR: {exc}")
                traceback.print_exc()
                results.append({"dataset": dataset_name, "sequence": sequence,
                                "method": "btrack", "error": str(exc)})

        return results

    except Exception as exc:
        print(f"    ERROR: {exc}")
        traceback.print_exc()
        return [{"dataset": dataset_name, "sequence": sequence, "error": str(exc)}]


# ---------------------------------------------------------------------------
# Optimisation benchmark helpers
# ---------------------------------------------------------------------------

def _prepare_optimisation_data(
    dataset_dir: Path,
    sequence: str,
    seg_source_dir: Path,
) -> tuple:
    """
    Prepare dataset and GT data for optimisation from CTC directory structure.
    Uses the real man_track.txt (with division info) for GT.
    """
    from napari_easytrack.analysis.optim_backend import (
        CellTrackingChallengeDataset,
        create_dataset_structure,
    )
    
    _ensure_wing_disc_prepared(dataset_dir.name, dataset_dir.parent)

    segmentation = load_ctc_segmentation(seg_source_dir)
    segmentation = segmentation.astype(np.uint16)
    segmentation = clean_segmentation(segmentation, verbose=False)
    segmentation = segmentation.astype(np.uint16)
    
    work_dir = dataset_dir / f"{sequence}_optim_temp"
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = create_dataset_structure(segmentation, work_dir / 'dataset')
    
    dataset = CellTrackingChallengeDataset(
        name=f"{dataset_dir.name}_{sequence}",
        path=dataset_root,
        experiment='01',
        scale=(1.0, 1.0, 1.0),
    )
    

    tra_dir = dataset_dir / f"{sequence}_GT" / "TRA"
    filtered_tra_dir = filter_man_track_to_segmentation(tra_dir, segmentation)
    gt_data = load_gt_tracking_graph(filtered_tra_dir)
    
    return dataset, gt_data, segmentation, work_dir


def run_optimisation_benchmark(
    dataset_name: str,
    train_seq: str,
    dataset_dir: Path,
    train_seg_dir: Path,
    db_path: str = "btrack_benchmark.db",
    n_random: int = 64,
    n_tpe: int = 128,
    timeout: int = 25,
    timeout_penalty: float = 10000,
) -> tuple[dict, str]:
    """
    Run optimisation on a single sequence.
    Does n_random trials with random sampler, then n_tpe trials with TPE.
    """
    from napari_easytrack.analysis.optim_pipeline import optimize_dataset_with_timeout
    
    today = date.today().isoformat()  # e.g. "2026-04-10"
    
    print(f"\n    {'='*50}")
    print(f"    OPTIMISATION: {dataset_name} seq {train_seq}")
    print(f"    {'='*50}")
    
    print(f"    Preparing optimisation data...")
    dataset, gt_data, segmentation, work_dir = _prepare_optimisation_data(
        dataset_dir, train_seq, train_seg_dir
    )
    
    study_name = f"{today}_optim_{dataset_name}_{train_seq}"
    
    # Phase 1: Random search
    print(f"\n    Phase 1: {n_random} random trials...")
    try:
        study = optimize_dataset_with_timeout(
            dataset=dataset,
            gt_data=gt_data,
            objectives='1obj',
            study_name=study_name,
            n_trials=n_random,
            timeout=timeout,
            timeout_penalty=timeout_penalty,
            use_parallel_backend=True,
            sampler='random',
        )
        best_so_far = study.best_trial.value
        print(f"    Random phase complete. Best AOGM: {best_so_far:.2f}")
    except Exception as e:
        print(f"    Random phase failed: {e}")
        traceback.print_exc()
    
    # Phase 2: TPE
    print(f"\n    Phase 2: {n_tpe} TPE trials...")
    try:
        study = optimize_dataset_with_timeout(
            dataset=dataset,
            gt_data=gt_data,
            objectives='1obj',
            study_name=study_name,
            n_trials=n_tpe,
            timeout=timeout,
            timeout_penalty=timeout_penalty,
            use_parallel_backend=True,
            sampler='tpe',
        )
    except Exception as e:
        print(f"    TPE phase failed: {e}")
        traceback.print_exc()
    
    best_trial = study.best_trial
    best_params = best_trial.params
    best_aogm = best_trial.value
    
    print(f"\n    Best trial: #{best_trial.number}")
    print(f"    Best AOGM: {best_aogm:.2f}")
    print(f"    {'='*50}")
    
    # Clean up temp files
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass
    
    return best_params, study_name


def evaluate_with_params(
    params: dict,
    segmentation: np.ndarray,
    gt_data,
    label: str,
) -> dict:
    """Run tracking with given params and evaluate against GT."""
    print(f"      Evaluating on {label}...")
    tracked_seg, tracks, stats, lbep = run_tracking_with_params(
        segmentation=segmentation,
        params=params,
        return_napari=False,
    )
    metrics = _evaluate_tracking(gt_data, tracked_seg, lbep)
    print(f"      {label}: TRA={metrics['TRA']:.4f}, DET={metrics['DET']:.4f}, AOGM={metrics['AOGM']:.2f}")
    return metrics


def benchmark_optimisation(
    dataset_name: str,
    train_seq: str,
    test_seq: str,
    train_dataset_dir: Path,
    test_dataset_dir: Path,
    train_seg_dir: Path,
    test_seg_dir: Path,
    results_dir: Path,
    db_path: str = "btrack_benchmark.db",
    n_random: int = 64,
    n_tpe: int = 128,
    timeout: int = 25,
) -> list[dict]:
    """
    Run optimisation on train_seq, evaluate on both train and test sequences.
    """
    results = []
    
    try:
        best_params, study_name = run_optimisation_benchmark(
            dataset_name=dataset_name,
            train_seq=train_seq,
            dataset_dir=train_dataset_dir,
            train_seg_dir=train_seg_dir,
            db_path=db_path,
            n_random=n_random,
            n_tpe=n_tpe,
            timeout=timeout,
        )
        
        # Save best params as JSON
        params_dir = results_dir / "optimised_configs"
        params_dir.mkdir(parents=True, exist_ok=True)
        
        from napari_easytrack.analysis.optim_pipeline import write_best_params_to_config
        config_file = params_dir / f"{study_name}_best_config.json"
        write_best_params_to_config(best_params, str(config_file))
        print(f"    Saved btrack config to: {config_file}")
        
        raw_file = params_dir / f"{study_name}_raw_params.json"
        with open(raw_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"    Saved raw params to: {raw_file}")
        
        # Evaluate on training sequence
        print(f"\n    Evaluating on training data ({train_seq})...")
        train_seg = load_ctc_segmentation(train_seg_dir).astype(np.uint16)
        train_seg = clean_segmentation(train_seg, verbose=False).astype(np.uint16)
        train_tra_dir = train_dataset_dir / f"{train_seq}_GT" / "TRA"
        filtered_train_tra_dir = filter_man_track_to_segmentation(train_tra_dir, train_seg)
        train_gt = load_gt_tracking_graph(filtered_train_tra_dir)
        
        train_metrics = evaluate_with_params(
            best_params, train_seg, train_gt, label=f"train_{train_seq}"
        )
        results.append({
            "dataset": dataset_name, "sequence": train_seq,
            "method": f"optimised (train={train_seq})", "eval_type": "train",
            "study_name": study_name, "n_trials": f"{n_random}R+{n_tpe}T",
            "seg_source": "GT + ST merged",
            "n_frames": train_seg.shape[0], "input_shape": str(train_seg.shape),
            **train_metrics,
        })
        
        # Evaluate on test sequence
        print(f"\n    Evaluating on test data ({test_seq})...")
        test_seg = load_ctc_segmentation(test_seg_dir).astype(np.uint16)
        test_seg = clean_segmentation(test_seg, verbose=False).astype(np.uint16)
        test_tra_dir = test_dataset_dir / f"{test_seq}_GT" / "TRA"
        filtered_test_tra_dir = filter_man_track_to_segmentation(test_tra_dir, test_seg)
        test_gt = load_gt_tracking_graph(filtered_test_tra_dir)
        
        test_metrics = evaluate_with_params(
            best_params, test_seg, test_gt, label=f"test_{test_seq}"
        )
        results.append({
            "dataset": dataset_name, "sequence": test_seq,
            "method": f"optimised (train={train_seq})", "eval_type": "test",
            "study_name": study_name, "n_trials": f"{n_random}R+{n_tpe}T",
            "seg_source": "GT + ST merged",
            "n_frames": test_seg.shape[0], "input_shape": str(test_seg.shape),
            **test_metrics,
        })
        
    except Exception as exc:
        print(f"    Optimisation ERROR: {exc}")
        traceback.print_exc()
        results.append({
            "dataset": dataset_name, "sequence": f"{train_seq}->{test_seq}",
            "method": "optimised", "error": str(exc),
        })
    
    return results


# ---------------------------------------------------------------------------
# Cross-validation pairing and orchestration
# ---------------------------------------------------------------------------

CROSS_VALIDATION_PAIRS = {
    "PhC-C2DH-U373": [("01", "02"), ("02", "01")],
    "DIC-C2DH-HeLa": [("01", "02"), ("02", "01")],
    "Fluo-N2DH-GOWT1": [("01", "02"), ("02", "01")],
    "Fluo-C2DL-Huh7": [("01", "02"), ("02", "01")],
    "Fluo-C3DH-A549-SIM": [("01", "02"), ("02", "01")],
    "wing_disc": [("01", "02"), ("02", "01")],
}

def _get_seg_dir(dataset_dir: Path, sequence: str, gt_only: bool, dataset_name: str = "") -> Path:
    """Get the segmentation directory, merging GT+ST if needed."""
    if gt_only:
        seg_dir = dataset_dir / f"{sequence}_GT" / "SEG"
        if not seg_dir.exists():
            raise FileNotFoundError(f"GT SEG not found: {seg_dir}")
        return seg_dir

    info = DATASETS.get(dataset_name, {})
    has_st = info.get("has_st", True)
    if has_st:
        merged_dir = merge_gt_st_segmentation(dataset_dir, sequence)
        if merged_dir is not None:
            return merged_dir

    seg_dir = dataset_dir / f"{sequence}_GT" / "SEG"
    if seg_dir.exists():
        return seg_dir
    raise FileNotFoundError(f"No segmentation found for {dataset_dir}/{sequence}")


def run_optimisation_benchmarks(
    datasets_to_run: list[str],
    data_dir: Path,
    results_dir: Path,
    gt_only: bool = False,
    db_path: str = "btrack_benchmark.db",
    n_random: int = 64,
    n_tpe: int = 128,
    timeout: int = 25,
) -> list[dict]:
    """Run optimisation benchmarks with cross-validation for all requested datasets."""
    all_results = []
    standard_datasets = [d for d in datasets_to_run]
    
    # Standard datasets: cross-validate within dataset
    for dataset_name in standard_datasets:
        if dataset_name not in CROSS_VALIDATION_PAIRS:
            print(f"  [skip optimisation] No cross-validation pairs for {dataset_name}")
            continue
        
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            print(f"  [skip] {dataset_dir} does not exist")
            continue
        
        for train_seq, test_seq in CROSS_VALIDATION_PAIRS[dataset_name]:
            print(f"\n{'─'*70}")
            print(f"OPTIMISATION: {dataset_name} — train on {train_seq}, test on {test_seq}")
            print(f"{'─'*70}")
            
            try:
                train_seg_dir = _get_seg_dir(dataset_dir, train_seq, gt_only, dataset_name)
                test_seg_dir = _get_seg_dir(dataset_dir, test_seq, gt_only, dataset_name)  
                
                pair_results = benchmark_optimisation(
                    dataset_name=dataset_name,
                    train_seq=train_seq, test_seq=test_seq,
                    train_dataset_dir=dataset_dir, test_dataset_dir=dataset_dir,
                    train_seg_dir=train_seg_dir, test_seg_dir=test_seg_dir,
                    results_dir=results_dir, db_path=db_path,
                    n_random=n_random, n_tpe=n_tpe, timeout=timeout,
                )
                all_results.extend(pair_results)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                traceback.print_exc()
                all_results.append({
                    "dataset": dataset_name, "sequence": f"{train_seq}->{test_seq}",
                    "method": "optimised", "error": str(exc),
                })
    
    return all_results


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(results: List[Dict], output_dir: Path) -> None:
    """Save preset benchmark results as CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ctc_benchmark_results.json"
    csv_path = output_dir / "ctc_benchmark_results.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results (JSON): {json_path}")

    if results:
        fieldnames = sorted(
            {k for r in results for k in r.keys()},
            key=lambda k: (k not in ("dataset", "sequence", "method", "preset"), k),
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"  Results (CSV):  {csv_path}")


def save_optimisation_results(results: list[dict], output_dir: Path) -> None:
    """Save optimisation benchmark results separately."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "optimisation_results.json"
    csv_path = output_dir / "optimisation_results.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Optimisation results (JSON): {json_path}")

    if results:
        fieldnames = sorted(
            {k for r in results for k in r.keys()},
            key=lambda k: (k not in ("dataset", "sequence", "method", "eval_type"), k),
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"  Optimisation results (CSV):  {csv_path}")


def print_summary(results: List[Dict]) -> None:
    """Print a human-readable summary table for preset benchmarks."""
    print("\n" + "=" * 80)
    print("PRESET BENCHMARK SUMMARY")
    print("=" * 80)
    fmt = "{:<22} {:<5} {:<12} {:>7} {:>7} {:>10}"
    print(fmt.format("Dataset", "Seq", "Method", "TRA", "DET", "AOGM"))
    print("-" * 80)
    for r in results:
        if "error" in r:
            method = r.get("method", "?")
            print(f"  {r['dataset']:<20} {r['sequence']:<5} {method:<12} "
                  f"ERROR: {r['error']}")
        else:
            tra = r.get("TRA", float("nan"))
            det = r.get("DET", float("nan"))
            aogm = r.get("AOGM", float("nan"))
            method = r.get("method", "?")
            print(fmt.format(
                r["dataset"], r["sequence"], method,
                f"{tra:.4f}" if isinstance(tra, float) else str(tra),
                f"{det:.4f}" if isinstance(det, float) else str(det),
                f"{aogm:.2f}" if isinstance(aogm, float) else str(aogm),
            ))
    print("=" * 80)
    print(
        "\nMetric definitions (Cell Tracking Challenge):\n"
        "  TRA  – Tracking accuracy  (higher is better, max 1.0)\n"
        "  DET  – Detection accuracy (higher is better, max 1.0)\n"
        "  AOGM – Acyclic Oriented Graph Matching error (lower is better)\n"
    )


def print_optimisation_summary(results: list[dict]) -> None:
    """Print a summary table for optimisation results."""
    print("\n" + "=" * 95)
    print("OPTIMISATION BENCHMARK SUMMARY")
    print("=" * 95)
    fmt = "{:<22} {:<15} {:<28} {:<6} {:>7} {:>7} {:>10}"
    print(fmt.format("Dataset", "Sequence", "Method", "Type", "TRA", "DET", "AOGM"))
    print("-" * 95)
    for r in results:
        if "error" in r:
            method = r.get("method", "?")
            print(f"  {r['dataset']:<20} {r.get('sequence','?'):<15} {method:<28} "
                  f"ERROR: {r['error']}")
        else:
            tra = r.get("TRA", float("nan"))
            det = r.get("DET", float("nan"))
            aogm = r.get("AOGM", float("nan"))
            method = r.get("method", "?")
            eval_type = r.get("eval_type", "?")
            print(fmt.format(
                r["dataset"], r["sequence"], method, eval_type,
                f"{tra:.4f}" if isinstance(tra, float) else str(tra),
                f"{det:.4f}" if isinstance(det, float) else str(det),
                f"{aogm:.2f}" if isinstance(aogm, float) else str(aogm),
            ))
    print("=" * 95)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        choices=list(DATASETS.keys()), metavar="DATASET",
        help=f"Datasets to benchmark. Choices: {list(DATASETS.keys())}. Default: {DEFAULT_DATASETS}.",
    )
    parser.add_argument(
        "--sequences", nargs="+", default=None, metavar="SEQ",
        help="Sequence IDs to evaluate (e.g. 01 02). Default: all sequences.",
    )
    parser.add_argument(
        "--preset", default="Epithelial Cells (Default)",
        help="easytrack preset name. Default: 'Epithelial Cells (Default)'.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("ctc_data"),
        help="Directory to store / load CTC datasets. Default: ./ctc_data",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("ctc_results"),
        help="Directory to write result files. Default: ./ctc_results",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Do not attempt to download datasets.",
    )
    parser.add_argument(
        "--gt-only", action="store_true",
        help="Use only GT segmentation masks (skip GT + ST merge).",
    )
    parser.add_argument(
        "--skip-btrack", action="store_true",
        help="Skip the btrack baseline comparison.",
    )
    parser.add_argument(
        "--skip-optimisation", action="store_true",
        help="Skip the optimisation benchmark.",
    )
    parser.add_argument(
        "--only-optimisation", action="store_true",
        help="Only run the optimisation benchmark (skip preset-based tracking).",
    )
    parser.add_argument(
        "--n-random", type=int, default=64,
        help="Number of random search trials for optimisation. Default: 64.",
    )
    parser.add_argument(
        "--n-tpe", type=int, default=128,
        help="Number of TPE trials for optimisation. Default: 128.",
    )
    parser.add_argument(
        "--optim-timeout", type=int, default=25,
        help="Per-trial timeout in seconds for optimisation. Default: 25.",
    )
    parser.add_argument(
        "--db-path", type=str, default="btrack_benchmark.db",
        help="SQLite database path for Optuna studies. Default: btrack_benchmark.db",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 80)
    print("easytrack — Cell Tracking Challenge Benchmark")
    print("=" * 80)
    print(
        "\nData citation:\n"
        "  Maška et al., Nature Methods 20, 1010–1020 (2023)\n"
        "  https://doi.org/10.1038/s41592-023-01879-y\n"
        "  Ulman et al., Nature Methods 14, 1141–1152 (2017)\n"
        "  https://doi.org/10.1038/nmeth.4473\n"
        "  Data source: Cell Tracking Challenge "
        "(https://celltrackingchallenge.net/)\n"
    )
    if args.gt_only:
        print("  Segmentation source: GT only (--gt-only)")
    else:
        print("  Segmentation source: GT + ST merged (use --gt-only to disable)")

    if args.only_optimisation:
        print("  Methods: optimisation only (--only-optimisation)")
    else:
        if args.skip_btrack:
            print("  Methods: easytrack only (--skip-btrack)")
        else:
            print("  Methods: easytrack vs btrack baseline")
        if not args.skip_optimisation:
            print(f"  Optimisation: {args.n_random} random + {args.n_tpe} TPE trials, "
                  f"{args.optim_timeout}s timeout")
        else:
            print("  Optimisation: skipped (--skip-optimisation)")
    print()

    data_dir: Path = args.data_dir.resolve()
    results_dir: Path = args.results_dir.resolve()
    results: List[Dict] = []
    optim_results: List[Dict] = []

    # Download all datasets first
    for dataset_name in args.datasets:
        if not args.skip_download:
            try:
                download_dataset(dataset_name, data_dir)
            except Exception:
                print(f"  Skipping {dataset_name} due to download failure.")

    # Preset-based benchmarks (easytrack + btrack)
    if not args.only_optimisation:
        for dataset_name in args.datasets:
            info = DATASETS[dataset_name]
            dataset_dir = data_dir / dataset_name

            if not dataset_dir.exists():
                if args.skip_download:
                    print(f"  ERROR: --skip-download but {dataset_dir} missing.")
                continue

            print(f"\n{'─' * 70}")
            print(f"Dataset: {dataset_name}")
            print(f"  {info['description']}")

            sequences = args.sequences if args.sequences else info["sequences"]
            for seq in sequences:
                seq_results = benchmark_sequence(
                    dataset_name=dataset_name, sequence=seq,
                    dataset_dir=dataset_dir, preset_name=args.preset,
                    gt_only=args.gt_only, skip_btrack=args.skip_btrack,
                )
                results.extend(seq_results)

    # Optimisation benchmarks with cross-validation
    if not args.skip_optimisation:
        print("\n" + "=" * 80)
        print("OPTIMISATION BENCHMARKS")
        print("=" * 80)

        optim_results = run_optimisation_benchmarks(
            datasets_to_run=args.datasets,
            data_dir=data_dir,
            results_dir=results_dir,
            gt_only=args.gt_only,
            db_path=args.db_path,
            n_random=args.n_random,
            n_tpe=args.n_tpe,
            timeout=args.optim_timeout,
        )

    # Save & summarise
    if results:
        save_results(results, results_dir)
        print_summary(results)

    if optim_results:
        save_optimisation_results(optim_results, results_dir)
        print_optimisation_summary(optim_results)

    if not results and not optim_results:
        print("\nNo results to report.")

    return 0


if __name__ == "__main__":
    sys.exit(main())