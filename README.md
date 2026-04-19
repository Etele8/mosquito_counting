# Mosquito Attraction Image Analysis

> Subtitle: Classical computer-vision pipeline for counting mosquitoes in light-attraction experiments.

This repository contains a practical image-analysis workflow for mosquito attraction experiments. It combines manual preprocessing, classical computer vision, and lightweight reporting so experiment folders can be converted into inspectable counts, masks, and summary plots.

For a recruiter or engineering manager, the value is not just "image processing" in the abstract. The project shows the kind of engineering judgment that matters in real applied work: dealing with inconsistent raw data, building repeatable local tooling, exposing outputs that can be checked visually, and keeping the workflow simple enough for a lab or collaborator to rerun.

## Hero

The project processes batches of experimental chamber photos and estimates mosquito counts from each image. Raw images are manually cropped into a consistent region of interest, then a connected-component pipeline detects candidate insects, exports inspection masks and visualizations, and writes CSV summaries that can be plotted or analyzed further.

Why it matters: this is the kind of messy, real-world workflow where the engineering challenge is as much about robustness and inspectability as it is about the counting algorithm itself.

## Why This Is Interesting

- It solves a real applied problem rather than a benchmark-only exercise.
- It shows practical computer-vision tradeoffs: manual ROI selection, thresholding, morphology, connected-component filtering, and output inspection.
- It keeps the analysis auditable by exporting masks, per-image counts, and summary artifacts instead of only producing a final number.
- It demonstrates an engineering mindset: make the workflow understandable, rerunnable, and easy to review.

## Features

- Batch counting from folders of cropped experiment images
- Per-image and per-folder CSV exports
- Binary mask and connected-component visual exports for manual validation
- Support for both standard experiment batches and a gradient-style five-section workflow
- Optional R plotting script for summary visualizations

## Repository Structure

```text
.
|-- counter.py
|-- gradient_counter.py
|-- requirements.txt
|-- requirements-full.txt
|-- plots/
|-- utils/
|-- scripts/
|-- data/
`-- docs/
```

## Tech Stack

- Python 3.11+ for the core processing scripts
- OpenCV for image preprocessing and connected-component analysis
- NumPy and pandas for numerical work and tabular exports
- R (`readr`, `dplyr`, `ggplot2`) for optional exploratory plotting
- PowerShell helper scripts for Windows-friendly setup and demo commands

## Method / Approach

1. Raw photos are cropped to a consistent chamber region using the interactive helper in [utils/cropping.py](/d:/intezet/gabor/utils/cropping.py:1).
2. The main pipeline in [counter.py](/d:/intezet/gabor/counter.py:1) converts images to grayscale, blurs them, applies adaptive thresholding, and uses morphological cleanup.
3. Connected components are filtered by area to approximate candidate mosquitoes.
4. For each image, the script writes a binary mask, a colorized connected-component visualization, and per-image count rows.
5. Per-folder and aggregated CSV outputs are then used for downstream analysis and optional plotting in [utils/vis.r](/d:/intezet/gabor/utils/vis.r:1).

## Results

The repository currently includes two lightweight example plot exports:

- [plots/boxplot_0813.pdf](/d:/intezet/gabor/plots/boxplot_0813.pdf)
- [plots/boxplot_0816.pdf](/d:/intezet/gabor/plots/boxplot_0816.pdf)

## Quickstart

If you only want to verify the repository wiring on Windows:

```powershell
.\scripts\setup.ps1
.\scripts\smoke_test.ps1
.\scripts\run_demo.ps1
```

`run_demo.ps1` uses the existing local sample path `data\images\20250816\cropped` by default. If that dataset is not present on the machine, the script will fail with a clear message instead of silently doing the wrong thing.

## How To Run

### Windows / PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python counter.py --input-root "data\images\20250816\cropped" --output-root "data\outputs\portfolio_demo"
```

### Optional gradient experiment workflow

```powershell
python gradient_counter.py --folder "data\images\20250702\mixed\20perc_23.45" --output-folder "data\outputs\gradient_demo"
```

### Optional plotting step

Open R or RStudio and adjust `csv_path` in [utils/vis.r](/d:/intezet/gabor/utils/vis.r:1), then run the script to export a PDF summary plot.

## Reproducibility / Limitations

- The full raw dataset is not public in this repository because it is large and not curated for public release.
- The counting pipeline depends on image quality and chamber consistency; thresholds and area filters may need tuning per capture setup.
- Cropping is currently semi-manual, which improves control but limits full automation.
- The repo is reproducible as a code workflow, but not yet fully reproducible as a public data package.
- `requirements.txt` is intentionally minimal for the main pipeline; [requirements-full.txt](/d:/intezet/gabor/requirements-full.txt) preserves the broader exploratory environment snapshot.
