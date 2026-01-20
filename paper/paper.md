---
title: 'easytrack: A napari plugin for automated parameter tuning in cell tracking'
tags:
  - Python
  - napari
  - cell biology
  - image analysis
  - computational biology
  - bioimaging
  - Bayesian optimisation
  - parameter tuning
authors:
  - name: Tim Huygelen
    orcid: 0009-0007-0026-4531
    affiliation: "1, 2"
  - name: Alan Lowe
    orcid: 0000-0002-0558-3597
    affiliation: 3
  - name: Yanlan Mao
    corresponding: true
    orcid: 0000-0002-8722-4992
    affiliation: "1, 2"
  - name: Pablo Vicente-Munuera
    corresponding: true
    orcid: 0000-0001-5402-7637
    affiliation: "1, 2"
affiliations:
  - name: Laboratory for Molecular Cell Biology, University College London, London, United Kingdom
    index: 1
  - name: Institute for the Physics of Living Systems, University College London, London, United Kingdom
    index: 2
  - name: Chan Zuckerberg Initiative, Redwood City, CA, USA
    index: 3
date: 9 Jan 2026
bibliography: paper.bib
---

# Summary

Life is in constant movement, even at the microscopic scale of cells. Quantifying cellular dynamics, or how cells move, is a significant challenge, and manually annotating microscopy time-lapses is extremely time-consuming. In response, the bioimage community has developed algorithms to automatically track cells over time and space. However, cell tracking software remains imperfect and difficult to use, often requiring significant manual parameter tuning, as is the case for btrack [@btrack:2017; @btrack:2021]. easytrack aims to democratise the use of tracking software by providing an easy-to-use graphical interface for tracking and automating the tedious parameter tuning step in btrack.

# Statement of need

Cell tracking, the process of following individual cells through time and space, is essential for quantifying dynamic
cellular behaviours such as migration, division, and morphological changes. These measurements underpin research into
tissue healing [@Tetley:2019], developmental biology [@Valon:2021], and cancer progression [@Hong:2016]. Despite
advances in computational methods [@Ulman:2017; @Maska:2023], many tracking algorithms require careful tuning of
multiple parameters to achieve accurate results [@Loffler:2021; @Chenouard:2014]. 

The lack of accessible parameter tuning tools creates a barrier between technological capability and practical
application. Researchers often resort to default parameters or limited manual exploration of the parameter space,
potentially missing optimal configurations for their specific datasets. While hyperparameter optimisation frameworks
like Optuna [@Optuna:2019] have proven effective in machine learning contexts, their application to cell tracking
software remains limited, with no existing tools providing an accessible interface for biologists.

`easytrack` addresses this gap by implementing automated parameter tuning within the napari ecosystem [@napari:2019],
leveraging Bayesian optimisation techniques including Tree-structured Parzen Estimators (TPE) [@Ozaki:2022]. By
automating what was previously a manual, iterative process, `easytrack` makes high-accuracy cell tracking accessible to
researchers without computational expertise whilst reducing the time required from hours to minutes.

# State of the field

Cellular dynamics (how cells move, divide, and interact) are fundamental to understanding biological processes from
tissue development to disease progression. Time-lapse microscopy enables researchers to capture these processes, but
extracting quantitative information requires accurately tracking individual cells across image sequences. Current cell
tracking software falls into two categories: general-use tools with modest accuracy, or high-performance algorithms like
btrack [@btrack:2017; @btrack:2021] that require extensive manual parameter tuning. This manual tuning is
labour-intensive, requires expertise, and often needs repetition for different datasets or experimental conditions. 
For instance, for btrack, a Bayesian cell tracking algorithm, this involves configuring 18 parameters across its motion 
and hypothesis models. 
`easytrack` democratises access to high-accuracy cell tracking by automating the parameter optimisation process through
a user-friendly napari plugin interface.

# Software Design

`easytrack` is implemented as `napari-easytrack`, a plugin for the napari multi-dimensional image viewer, integrating
seamlessly with the growing napari bioimage analysis ecosystem. The design philosophy emphasises composability over
reinvention, building upon established open-source tools: btrack for tracking [@btrack:2017; @btrack:2021], Optuna for
optimisation [@Optuna:2019], and traccuracy for evaluation [@Traccuracy:2023].

## Architecture

The plugin provides two complementary widgets (Figure \ref{fig:workflow}):

**Parameter tuning widget:** Automates the process of finding optimal btrack parameters for given data using Bayesian optimisation. Users
provide ground truth annotations (cells that have been segmented and tracked), and the widget optimises btrack's 18
parameters by minimising the difference between predicted and ground truth tracking. The optimisation can be performed
using TPE (Tree-structured Parzen Estimator) and random search and evaluates tracking quality using the AOGM metric from
the Cell Tracking Challenge [@Maska:2014]. Optimised parameters are saved as JSON configuration files for subsequent
use.

**Tracking widget:** Applies previously optimised or manually created parameter configurations to new segmented
time-lapse or 3D images. The widget parses JSON configuration files, executes btrack with the specified parameters, and
displays tracked cells with their trajectories overlaid for visual inspection and manual correction if needed.

Both widgets include segmentation preprocessing functionality to remove small artefacts that can compromise tracking
accuracy. 

![Figure 1. Workflow of the napari plugin
`easytrack`. The parameter tuning widget (left) uses ground truth annotations to optimise btrack parameters via Bayesian optimisation, whilst the tracking widget (right) applies these optimised parameters to new datasets.\label{fig:workflow}](Fig_1_Workflow.png)

# Research impact statement

Sophisticated cell tracking algorithms often remain underutilised due to their complexity [@Soelistyo:2023]. `easytrack`
addresses this usability barrier by automating parameter tuning, enabling:

1. **Reproducible research:** Parameter configurations can be saved, shared, and reused, reducing variability across
   studies
2. **Efficiency gains:** Optimisation completes in tens of minutes compared to hours or days of manual tuning
3. **Accessibility:** Non-experts can leverage state-of-the-art tracking without deep algorithmic knowledge

In the age of artificial intelligence, there are many tools obtaining segmentation from microscopy
images [@Cellpose:2021]. However, for complex 3D segmentations a tracking step is required to get an accurate
result [@Paci:2025]. The plugin has already been integrated into napari-EpiTools [@EpiTools:2025], a tool for analysing
packed epithelial tissues that was previously missing a tracking algorithm with a simplified graphical user interface.
This integration allows EpiTools' user base to benefit from automated parameter tuning for accurate 3D tracking. While
btrack is one of the most popular and robust plugins in napari, it is not specialised for packed epithelial tissues.
Therefore, even though `easytrack` is in its early stages, it fills a gap in the napari community for easy-to-use
tracking tools, particularly for epithelial tissues and 3D stitching, whilst its implementation as a standalone plugin
ensures broader accessibility across the napari community.

# AI usage policy

GitHub Copilot and Claude were used to assist in writing portions of the code for this project. All code was reviewed,
tested, and edited by the authors.

# Acknowledgements

We thank the Mao laboratory for providing valuable feedback on the graphical user interface and thoroughly testing
`easytrack` during development. We also acknowledge the napari and btrack communities for creating the foundational
tools upon which this work builds.

# References