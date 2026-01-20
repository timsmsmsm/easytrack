[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18200898.svg)](https://doi.org/10.5281/zenodo.18200898)
[![Licence](https://img.shields.io/pypi/l/napari-easytrack.svg?color=green)](https://raw.githubusercontent.com/timsmsmsm/easytrack/main/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/napari-easytrack.svg?color=green)](https://pypi.org/project/napari-easytrack)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-easytrack.svg?color=green)](https://python.org)
[![Documentation](https://readthedocs.org/projects/napari-easytrack/badge/?version=latest)](https://napari-easytrack.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/timsmsmsm/easytrack/actions/workflows/tests.yml/badge.svg)](https://github.com/timsmsmsm/easytrack/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/timsmsmsm/easytrack/badge.svg?branch=main)](https://coveralls.io/github/timsmsmsm/easytrack?branch=main)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-easytrack)](https://napari-hub.org/plugins/napari-easytrack)

# napari-easytrack

`napari-easytrack` is a napari plugin for automated parameter tuning in cell tracking. It optimises
[`btrack`](https://github.com/quantumjot/btrack) obtaining a set of optimal tracking parameters for a dataset. Using
that optimal set of parameters, `napari-easytrack` can then track the cells in the dataset,
improving tracking accuracy and reducing manual correction time.

`napari-easytrack` provides two widgets in napari:

1. An **optimization widget** that optimizes tracking parameters based on a small subset of manually annotated
   ground-truth data.
2. A **tracking widget** that uses the optimized parameters to track the entire dataset. Here, we provide different
   tracking presets so users can choose the one that best fits their data without optimization. If no preset fits the
   data,
   users should try to optimize the parameters first with the **optimization widget**.

## Installation

Create a venv environment with Python 3.11 (recommended) or Python 3.10.

```sh
 python -m venv napari_easytrack-env
 ```

First, install [napari](https://napari.org/index.html#installation).

Then, install easytrack via pip:

```sh 
python -m pip install napari-easytrack
```

To install the latest development version of `EpiTools` clone this repository
and run

```sh
python -m pip install -e .
```

## Usage

To use `napari-easytrack`, first launch napari:

```sh
napari
```

Once in napari, click on the "Plugins" menu, then select "napari-easytrack" and click "Tracking" to open the tracking
widget. We recommend starting with the `Tracking` widget to test the plugin with the provided presets.

### Tracking Widget

Once in the `Tracking` widget, you can select one of the presets from the dropdown menu:

- `Epithelial cells`: for tracking epithelial cells in 2D+time datasets.
- `Epithelial cells (Z-tracking)`: for tracking epithelial cells in 3D (space) datasets.
- `Custom JSON`: if none of the presets fit your data, you can provide a custom JSON file with tracking parameters
  optimised for your dataset. You can obtain this JSON file by first using the `Parameter tuning` widget.

Once you have selected your presets, select the "Segmentation Layer" to apply the tracking to and click "Apply
Tracking".
We also provide, in case it is needed, a "Clean Segmentation" and "Remove Small Objects" to improve the provided
segmentation. In addition, you can also save your own configuration of parameters as a JSON file for future use by
clicking on "Save Config (JSON)".

### Parameter tuning Widget

To optimise your own tracking parameters specific to your dataset, you require to provide some ground-truth data with
cells segmented and tracked. You will select this dataset as "Ground Truth Layer" in the `Parameter tuning` widget. As a
first trial, we recommend using a small subset of your data (e.g., 10-20 frames) with a few cells tracked (e.g., 5-10
cells).
With all the default parameters, click on "Start Optimization" to begin the optimisation process. You can cancel the
process at any time by clicking on "Stop Optimization". Once the optimisation is finished, you can save the optimal
parameters as a JSON file by clicking on "Save Config". You can then use this JSON file in the `Tracking` widget to
track
your entire dataset, selecting "Custom JSON" in the presets dropdown menu.

## Issues

If you encounter any problems, please
[file an issue](https://github.com/timsmsmsm/easytrack/issues) along with a
detailed description.

## Citation

If you use `napari-easytrack` in your research, please cite the following paper:

```bibtex
@software{Huygelen_napari-easytrack,
    author = {Huygelen, Tim and Lowe, Alan and Mao, Yanlan and Vicente-Munuera, Pablo},
    license = {MIT},
    title = {{napari-easytrack}},
    url = {https://github.com/timsmsmsm/easytrack},
    year = {2026},
    doi = {10.5281/zenodo.18200898},
}
```