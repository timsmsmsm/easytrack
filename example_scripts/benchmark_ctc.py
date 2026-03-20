"""
benchmark_ctc.py — Benchmark easytrack on Cell Tracking Challenge datasets
==========================================================================

Downloads (optional) and evaluates easytrack on:
  • Fluo-C2DL-Huh7   – 2D + time (hepatocellular carcinoma cells)
  • Fluo-C3DL-MDA231 – 3D + time (breast-carcinoma cells in collagen)

Usage
-----
    # Download data automatically and benchmark both datasets:
    python benchmark_ctc.py

    # Use already-downloaded data (skip the download step):
    python benchmark_ctc.py --skip-download --data-dir /path/to/ctc_data

    # Benchmark a single dataset:
    python benchmark_ctc.py --datasets Fluo-C2DL-Huh7

    # Choose a different preset:
    python benchmark_ctc.py --preset "Epithelial Cells (Default)"

    # Override the output directory for results:
    python benchmark_ctc.py --results-dir ./my_results

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
  2. CTC-related use does not require consent from challenge organizers.
  3. Public non-CTC scientific use requires explicit permission from organizers.
  4. Commercial use requires permission from organizers AND data providers.
  5. Cloning datasets or reference annotations is strictly forbidden.
     Use the official download links to share or reference them.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import ssl

import numpy as np
from skimage import io


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
        # Quick probe – if this raises, certs are not available
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
    "Fluo-C2DL-Huh7": {
        "url": "https://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip",
        "description": "2D+time: hepatocellular carcinoma cells (Huh7)",
        "is_3d": False,
        "sequences": ["01", "02"],
    },
    "Fluo-C3DL-MDA231": {
        "url": "https://data.celltrackingchallenge.net/training-datasets/Fluo-C3DL-MDA231.zip",
        "description": "3D+time: breast carcinoma cells (MDA231) in collagen",
        "is_3d": True,
        "sequences": ["01", "02"],
    },
}

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


def download_dataset(name: str, data_dir: Path) -> Path:
    """
    Download and extract a CTC training dataset.

    Parameters
    ----------
    name:
        Dataset name (key in DATASETS).
    data_dir:
        Root directory to download into.  The zip is extracted to
        ``data_dir / name /``.

    Returns
    -------
    Path
        Path to the extracted dataset directory.
    """
    info = DATASETS[name]
    url = info["url"]
    target_dir = data_dir / name

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"  [skip] {name} already present at {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
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
        print()  # newline after progress bar
    except Exception as exc:
        print(f"\n  ERROR downloading {name}: {exc}")
        print("  Please download manually and re-run with --skip-download.")
        raise

    print(f"  Extracting to {target_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    zip_path.unlink()
    return target_dir


# ---------------------------------------------------------------------------
# CTC data loading helpers
# ---------------------------------------------------------------------------

def load_ctc_segmentation(tra_dir: Path) -> np.ndarray:
    """
    Load CTC ground-truth tracking masks (man_track*.tif) as a numpy array.

    Parameters
    ----------
    tra_dir:
        Path to a ``*_GT/TRA`` directory that contains ``man_track*.tif``
        files and a ``man_track.txt`` file.

    Returns
    -------
    np.ndarray
        Array of shape ``(T, Y, X)`` or ``(T, Z, Y, X)`` with integer labels.
    """
    mask_files = sorted(tra_dir.glob("man_track*.tif"))
    if not mask_files:
        raise FileNotFoundError(f"No man_track*.tif files found in {tra_dir}")

    frames = [io.imread(str(f)) for f in mask_files]
    segmentation = np.stack(frames, axis=0)
    print(f"    Loaded {len(frames)} frames → shape {segmentation.shape}, dtype {segmentation.dtype}")
    return segmentation


def load_gt_tracking_graph(tra_dir: Path):
    """
    Load CTC ground-truth as a traccuracy TrackingGraph.

    Parameters
    ----------
    tra_dir:
        Path to a ``*_GT/TRA`` directory.

    Returns
    -------
    traccuracy.TrackingGraph
    """
    from traccuracy.loaders import load_ctc_data

    gt_graph = load_ctc_data(
        str(tra_dir),
        str(tra_dir / "man_track.txt"),
        name="ground_truth",
    )
    print(f"    GT nodes: {len(gt_graph.nodes())}, GT edges: {len(gt_graph.edges())}")
    return gt_graph


# ---------------------------------------------------------------------------
# Tracking helpers
# ---------------------------------------------------------------------------

def run_tracking_on_segmentation(
    segmentation: np.ndarray,
    preset_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run easytrack on a segmentation array using a named preset.

    Parameters
    ----------
    segmentation:
        Integer-labeled segmentation, shape ``(T, Y, X)`` or ``(T, Z, Y, X)``.
    preset_name:
        Name of the easytrack preset to use (see ``get_presets()``).

    Returns
    -------
    tracked_segmentation:
        Updated segmentation with consistent track IDs across time.
    lbep:
        Numpy array of shape ``(N, 4)`` with columns
        ``[track_id, start_frame, end_frame, parent_id]``.
    """
    import btrack
    from btrack import utils, config as btrack_config

    from src.napari_easytrack.analysis.tracking import (
        get_default_config_path,
        scale_matrix,
    )
    from src.napari_easytrack.presets import get_presets

    presets = get_presets()
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {available}"
        )

    params = presets[preset_name]["config"]
    if not params:
        raise ValueError(
            f"Preset '{preset_name}' has no config. "
            "Please choose a different preset or provide a custom JSON config."
        )

    # Load base config and apply preset parameters
    conf = btrack_config.load_config(get_default_config_path())

    motion_attrs = {
        "P": scale_matrix(conf.motion_model.P, 150.0, params.get("p_sigma", 150.0)),
        "G": scale_matrix(conf.motion_model.G, 15.0, params.get("g_sigma", 15.0)),
        "R": scale_matrix(conf.motion_model.R, 5.0, params.get("r_sigma", 5.0)),
        "accuracy": params.get("accuracy", 7.5),
        "max_lost": params.get("max_lost", 5),
        "prob_not_assign": params.get("prob_not_assign", 0.1),
    }
    hyp_attrs = {
        k: params[k]
        for k in [
            "theta_dist", "lambda_time", "lambda_dist", "lambda_link",
            "lambda_branch", "theta_time", "dist_thresh", "time_thresh",
            "apop_thresh", "segmentation_miss_rate",
        ]
        if k in params
    }

    for k, v in motion_attrs.items():
        setattr(conf.motion_model, k, v)
    for k, v in hyp_attrs.items():
        setattr(conf.hypothesis_model, k, v)

    if params.get("div_hypothesis", 1) == 1:
        conf.hypothesis_model.hypotheses = [
            "P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"
        ]
    else:
        conf.hypothesis_model.hypotheses = [
            "P_FP", "P_init", "P_term", "P_link", "P_dead"
        ]
    conf.enable_optimisation = True

    max_search_radius = params.get("max_search_radius", 100)

    # Spatial volume (no time axis)
    ndim = segmentation.ndim - 1  # subtract time axis
    spatial_shape = segmentation.shape[1:]
    if ndim == 2:
        Y, X = spatial_shape
        volume = ((0, X), (0, Y))
    else:
        Z, Y, X = spatial_shape
        volume = ((0, X), (0, Y), (0, Z))

    objects = utils.segmentation_to_objects(segmentation, properties=("area",))
    print(f"    Extracted {len(objects)} cell objects")

    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(conf)
        tracker.volume = volume
        tracker.max_search_radius = max_search_radius
        tracker.append(objects)
        tracker.track(step_size=100)
        tracker.optimize()

        tracks = tracker.tracks
        lbep = tracker.LBEP
        tracked_seg = utils.update_segmentation(
            np.asarray(segmentation), tracks
        )
        print(f"    Found {len(tracks)} tracks")

    return tracked_seg, lbep


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    lbep: np.ndarray,
    tracked_segmentation: np.ndarray,
    gt_data,
) -> Dict[str, float]:
    """
    Compute CTC and Division metrics against ground truth.

    Parameters
    ----------
    lbep:
        LBEP array from btrack (shape ``(N, 4)``).
    tracked_segmentation:
        Segmentation updated with track IDs.
    gt_data:
        Ground-truth TrackingGraph (from :func:`load_gt_tracking_graph`).

    Returns
    -------
    dict
        Metric name → value mapping.
    """
    from src.napari_easytrack.analysis.optim_pipeline import calculate_accuracy

    results = calculate_accuracy(lbep, tracked_segmentation, gt_data)
    return results


# ---------------------------------------------------------------------------
# Per-sequence benchmark
# ---------------------------------------------------------------------------

def benchmark_sequence(
    dataset_name: str,
    sequence: str,
    dataset_dir: Path,
    preset_name: str,
) -> Dict:
    """
    Benchmark easytrack on a single CTC sequence.

    Parameters
    ----------
    dataset_name:
        e.g. ``"Fluo-C2DL-Huh7"``.
    sequence:
        Sequence ID, e.g. ``"01"`` or ``"02"``.
    dataset_dir:
        Root directory of the extracted dataset.
    preset_name:
        easytrack preset name.

    Returns
    -------
    dict
        Result dict including all metrics and metadata.
    """
    tra_dir = dataset_dir / f"{sequence}_GT" / "TRA"
    if not tra_dir.exists():
        print(f"  [skip] GT directory not found: {tra_dir}")
        return {"dataset": dataset_name, "sequence": sequence, "error": "GT not found"}

    man_track = tra_dir / "man_track.txt"
    if not man_track.exists():
        print(f"  [skip] man_track.txt not found in {tra_dir}")
        return {"dataset": dataset_name, "sequence": sequence, "error": "man_track.txt missing"}

    print(f"\n  ── Sequence {sequence} ──")
    print(f"    GT dir: {tra_dir}")

    try:
        # 1. Load GT segmentation as tracking input
        print("    Loading segmentation …")
        segmentation = load_ctc_segmentation(tra_dir)
        segmentation = segmentation.astype(np.uint16)

        # 2. Load GT tracking graph for evaluation
        print("    Loading GT tracking graph …")
        gt_data = load_gt_tracking_graph(tra_dir)

        # 3. Run tracking
        print(f"    Running tracking (preset: {preset_name!r}) …")
        tracked_seg, lbep = run_tracking_on_segmentation(segmentation, preset_name)

        # 4. Compute metrics
        print("    Computing metrics …")
        metrics = compute_metrics(lbep, tracked_seg, gt_data)

        result = {
            "dataset": dataset_name,
            "sequence": sequence,
            "preset": preset_name,
            "n_frames": segmentation.shape[0],
            "input_shape": str(segmentation.shape),
            **metrics,
        }
        print(f"    ✓ TRA={metrics.get('TRA', 'N/A'):.4f}  "
              f"DET={metrics.get('DET', 'N/A'):.4f}  "
              f"AOGM={metrics.get('AOGM', 'N/A'):.2f}")
        return result

    except Exception as exc:
        print(f"    ERROR: {exc}")
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "sequence": sequence,
            "preset": preset_name,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(results: List[Dict], output_dir: Path) -> None:
    """Save benchmark results as both CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ctc_benchmark_results.json"
    csv_path = output_dir / "ctc_benchmark_results.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results (JSON): {json_path}")

    if results:
        fieldnames = sorted(
            {k for r in results for k in r.keys()},
            key=lambda k: (k not in ("dataset", "sequence", "preset"), k),
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"  Results (CSV):  {csv_path}")


def print_summary(results: List[Dict]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    fmt = "{:<22} {:<5} {:>7} {:>7} {:>10}"
    print(fmt.format("Dataset", "Seq", "TRA", "DET", "AOGM"))
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['dataset']:<20} {r['sequence']:<5} ERROR: {r['error']}")
        else:
            tra = r.get("TRA", float("nan"))
            det = r.get("DET", float("nan"))
            aogm = r.get("AOGM", float("nan"))
            print(fmt.format(
                r["dataset"], r["sequence"],
                f"{tra:.4f}" if isinstance(tra, float) else str(tra),
                f"{det:.4f}" if isinstance(det, float) else str(det),
                f"{aogm:.2f}" if isinstance(aogm, float) else str(aogm),
            ))
    print("=" * 70)
    print(
        "\nMetric definitions (Cell Tracking Challenge):\n"
        "  TRA  – Tracking accuracy  (higher is better, max 1.0)\n"
        "  DET  – Detection accuracy (higher is better, max 1.0)\n"
        "  AOGM – Acyclic Oriented Graph Matching error (lower is better)\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        metavar="DATASET",
        help=(
            "Datasets to benchmark. "
            f"Choices: {list(DATASETS.keys())}. "
            "Default: all datasets."
        ),
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        metavar="SEQ",
        help="Sequence IDs to evaluate (e.g. 01 02). Default: all sequences.",
    )
    parser.add_argument(
        "--preset",
        default="Epithelial Cells (Default)",
        help="easytrack preset name. Default: 'Epithelial Cells (Default)'.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ctc_data"),
        help="Directory to store / load CTC datasets. Default: ./ctc_data",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("ctc_results"),
        help="Directory to write result files. Default: ./ctc_results",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not attempt to download datasets; assume they are already in --data-dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("easytrack — Cell Tracking Challenge Benchmark")
    print("=" * 70)
    print(
        "\nData citation:\n"
        "  Maška et al., Nature Methods 20, 1010–1020 (2023)\n"
        "  https://doi.org/10.1038/s41592-023-01879-y\n"
        "  Ulman et al., Nature Methods 14, 1141–1152 (2017)\n"
        "  https://doi.org/10.1038/nmeth.4473\n"
        "  Data source: Cell Tracking Challenge (https://celltrackingchallenge.net/)\n"
    )

    data_dir: Path = args.data_dir.resolve()
    results: List[Dict] = []

    for dataset_name in args.datasets:
        info = DATASETS[dataset_name]
        print(f"\n{'─' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"  {info['description']}")

        # --- Download -------------------------------------------------------
        if not args.skip_download:
            try:
                dataset_dir = download_dataset(dataset_name, data_dir)
            except Exception:
                print(f"  Skipping {dataset_name} due to download failure.")
                continue
        else:
            dataset_dir = data_dir / dataset_name
            if not dataset_dir.exists():
                print(
                    f"  ERROR: --skip-download specified but {dataset_dir} does not exist."
                )
                continue
            print(f"  Using data from {dataset_dir}")

        # --- Benchmark sequences --------------------------------------------
        sequences = args.sequences if args.sequences else info["sequences"]
        for seq in sequences:
            result = benchmark_sequence(
                dataset_name=dataset_name,
                sequence=seq,
                dataset_dir=dataset_dir,
                preset_name=args.preset,
            )
            results.append(result)

    # --- Save & summarise ---------------------------------------------------
    if results:
        save_results(results, args.results_dir.resolve())
        print_summary(results)
    else:
        print("\nNo results to report.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
