---
title: 'EasyTrack: A napari plugin for automated parameter tuning in cell tracking'
tags:
  - Python
  - napari
  - cell biology
  - image analysis
  - computational biology
  - bioimaging
authors:
  - name: Tim Huygelen
    orcid: 0009-0007-0026-4531
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Alan Lowe
    orcid: 0000-0002-0558-3597
    affiliation: 3
  - name: Yanlan Mao
    corresponding: true # (This is how to denote the corresponding author)
    orcid: 0000-0002-8722-4992
    affiliation: "1, 2"
  - name: Pablo Vicente-Munuera
    corresponding: true # (This is how to denote the corresponding author)
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

Life is in constant movement, even microscopic bodies like cells. To understand in a quantitative way, cellular dynamics, 
or how cells move around, we analyse images taken with microscopes where we can highlight cellular structures like their
membranes or nuclei. For that, the bioimage community has developed a series of algorithms to correctly track cells in
different environments, but they are complex. `EasyTrack` aims at democratising the use of tracking algorithms by 
reducing the complexity of one of the algorithms providing an easy-to-use graphical interface. 

# Statement of need

Cell tracking is the task of following individual cells over time and space, essential for understanding dynamic
processes in cell biology. For instance, cell tracking can reveal how tissues heal themselves [@Tetley:2019], or how 
tissues grow and develop [@Valon:2021]. Despite significant advances in computational tools for automated cell tracking 
in time-lapse microscopy images [@Btrack:2017], current tracking software often requires time-consuming manual parameter
tuning to achieve accurate results. Here, we developed `EasyTrack` to automatically obtain the best Btrack parameters 
allowing the user to obtain truthful tracked images in a faster and more reliable way.

# Software design

`EasyTrack` is implemented as a plugin for napari [@napari:2019] called napari-EasyTrack given its growing bioimage community. 
The design strategy behind `EasyTrack` is not to reinvent the wheel but to taking advantage of broadly-used open source software to create a more efficient 
approach to cell tracking.

The plugin consists of two widgets: the parameter tuning widget and the tracking widget (Fig. \ref{fig:workflow}). The parameter tuning widget aims
at obtaining the best sets of parameters of Btrack [@Btrack:2017] using Optuna [@Optuna:2019]. Btrack has X parameters
which needs to be selected to perform a good cell tracking. Using our Ground Truth (GT) dataset with cells already 
segmented and tracked, we can obtain the quality of the tracking with traccuracy [@Traccuracy:2023]. Then, we use Optuna
to minimise the difference between our GT tracking accuracy and Btrack's prediction with a set of parameters. By 
minimising that difference, we get the optima configuration as a JSON file to track with Btrack a segmented image in time or space.
`EasyTrack`'s second widget (tracking) uses a configuration file (JSON) to predict a tracking given an input segmented time-lapse 
or 3D image. We parse the JSON file and use its parameters to run Btrack obtaining a stack of images with their cells tracked 
alongside their trajectory, which can used to correct them. In both widgets, we provide a button to correct the segmentation
by removing tiny barely visible objects that can create tracking errors.

![Figure 1. Pipeline of the napari plugin `EasyTrack`.\label{fig:workflow}](Fig_1_Workflow.png "Workflow")

# Research impact statement

In the age of artificial intelligence, there are many tools obtaining segmentation from microscopy images [@Cellpose:2021].
However, for complex 3D segmentations a tracking step is required to get an accurate result [@Paci:2025]. We developed
napari-EpiTools [@EpiTools:2025] a few years ago, and it was missing a tracking algorithm with a simplified graphical user 
interface. EpiTools has already integrated `EasyTrack` allowing EpiTools' user base to use it. In addition, 
Btrack is one of the most popular and robust plugins in napari, but it is not specialised for packed tissues, like EpiTools is. 
Therefore, even though `EasyTrack` is in its early stages, it fills
a gap in the napari-community for easy to use tracking tools, particularly for packed tissues and 3D structures.

# AI Usage Policy

GitHub Copilot was used to assist in writing some of the code for this project.
All code was reviewed and edited by the authors.

# Acknowledgements

We thank the Mao lab for giving useful feedback of the graphical user interface and testing `EasyTrack`.

# References
