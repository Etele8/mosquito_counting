Insect Counting

Manual cropping and OpenCV-based counting pipeline for insect images.
Includes a recursive 4-corner cropper, two counting modes (homogeneous background and gradient background), and an R script for overlapped boxplots with median trend lines.

Contents

Manual 4-Corner Cropper (recursive; draggable handles; resume-friendly)

Counting – Homogeneous Background (processes all subfolders; produces per-image masks/visuals and a project-level long CSV)

Counting – Gradient Background (per-image sectioning; per-image annotated visuals and CSV with section counts/colors)

R Plotting (overlapped boxplots per timepoint and experiment + median trends)

Folder Layout (convention)
project/
  data/
    images/
      <DATE>/
        cropped/                 # output of the cropper (mirrors input tree)
        01_5perc/                # subfolder naming convention: <experimentId>_<timepoint>perc
          img_0001.jpg
          ...
        01_30perc/
        02_5perc/
        ...
    outputs/
      mosquito_outputs_<DATE>/
        <experimentId>_<timepoint>perc/
          masks/
          visuals/
          mosquito_counts.csv    # per-subfolder
        all_images_long.csv      # project-level long CSV (one row per image)
  utils/
    cropping.py
  scripts/
    count_all.py                 # homogeneous background
    count_sections.py            # gradient background (5 vertical sections)
  R/
    plot_box_overlapped.R


Image types: .jpg, .jpeg, .png are accepted. Unreadable files are skipped.
Naming convention: subfolders are parsed as experimentId_timepointPerc (e.g., 01_30perc → experiment_id=01, timepoint_min=30).

Requirements
Python

Python: “latest stable” (e.g., 3.11/3.12) compatible with the libraries below.

Create requirements.txt:

opencv-python>=4.9,<5
numpy>=1.26
pandas>=2.2
openpyxl>=3.1


Install on Windows (PowerShell):

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

R

Packages used by the plotting script:

install.packages(c("readr", "dplyr", "ggplot2"))

1) Manual 4-Corner Cropper

Script: utils/cropping.py
Purpose: Recursively walk data/images/<DATE>, display each image fullscreen (downscaled to your screen), let you click four corners (TL → TR → BR → BL), drag corners to fine-tune, and save a perspective-warped crop to a mirrored output tree under data/images/<DATE>/cropped.

Controls

Left-click: add a corner (until you have 4).

Drag: click near a corner to grab it; drag to adjust; release to drop.

n: accept current box and save the crop.

r: reset (clear corners) for the current image.

q or Esc: quit immediately. You can rerun later; already-saved files are skipped.

Configuration (top of cropping.py)

NEW_FOLDER  = "<DATE>"                               # e.g., "20250816"
INPUT_ROOT  = root / "data" / "images" / NEW_FOLDER  # discovered recursively
OUTPUT_ROOT = root / "data" / "images" / NEW_FOLDER / "cropped"


Run:

python utils/cropping.py

2) Counting – Homogeneous Background (all subfolders)

Script: scripts/count_all.py
Purpose: For every subfolder under data/images/<DATE>/cropped, count insects per image using adaptive threshold + morphology + connected components. Saves per-image masks and visuals, a per-subfolder mosquito_counts.csv, and a project-level all_images_long.csv (one row per image, with experiment_id, timepoint_min, filename, mosquito_count, subfolder).

How it works (per image)

Convert to grayscale, Gaussian blur.

Adaptive threshold (binary inverse) to isolate dark objects.

Morphological open + erode to clean noise.

connectedComponentsWithStats → keep components with area in [25, 1500].

Count = number of valid components, plus +1 for each “extra large” component (area > 800) to offset small merges.

Save _mask.png (binary) and _vis.png (colored components).

Configuration (top of count_all.py)

INPUT_ROOT  = root / "data" / "images" / "<DATE>" / "cropped"
OUTPUT_ROOT = root / "data" / "outputs" / "mosquito_outputs_<DATE>"


Run:

python scripts/count_all.py


Outputs

data/outputs/mosquito_outputs_<DATE>/<experimentId>_<timepoint>perc/masks/*.png

data/outputs/mosquito_outputs_<DATE>/<experimentId>_<timepoint>perc/visuals/*.png

data/outputs/mosquito_outputs_<DATE>/<experimentId>_<timepoint>perc/mosquito_counts.csv

data/outputs/mosquito_outputs_<DATE>/all_images_long.csv
(Columns: experiment_id, timepoint_min, filename, mosquito_count, subfolder)

3) Counting – Gradient Background (5 vertical sections)

Script: scripts/count_sections.py
Purpose: When the background has a horizontal gradient, split each image into 5 vertical sections, run the same detection pipeline per section, and write:

a combined mask,

an annotated original with section counts and section dividers,

a per-folder CSV with total and section counts plus the section mean colors (RGB).

Configuration (top of count_sections.py)

FOLDER        = "data/images/<DATE>/<some_subfolder>"   # one input folder
OUTPUT_FOLDER = "data/outputs/mosquito_outputs_<DATE>/<target>"


Run:

python scripts/count_sections.py


Per-row CSV schema (examples)

filename

total_mosquitoes

section_1 … section_5

section_1_color … section_5_color (e.g., rgb(210, 180, 140))

Saved images

masks/<base>_mask.png (stitched section masks)

visuals/<base>_annotated.png (original + vertical dividers + counts)

4) Plotting (R)

Script: R/plot_box_overlapped.R
Input: data/outputs/mosquito_outputs_<DATE>/all_images_long.csv (from the homogeneous pipeline)

Generates overlapped boxplots per timepoint and experiment, with per-experiment median trend lines, saved to a PDF.

library(readr)
library(dplyr)
library(ggplot2)

csv_path <- "data/outputs/mosquito_outputs_<DATE>/all_images_long.csv"
out_pdf  <- "plots/boxplots_<DATE>.pdf"

df <- read_csv(csv_path, show_col_types = FALSE) %>%
  mutate(
    experiment_id  = as.factor(experiment_id),
    timepoint_min  = as.integer(timepoint_min),
    mosquito_count = as.numeric(mosquito_count)
  ) %>%
  filter(!is.na(timepoint_min), !is.na(mosquito_count))

time_levels <- sort(unique(df$timepoint_min))
df <- df %>% mutate(timepoint_min_f = factor(timepoint_min, levels = time_levels))

med_df <- df %>%
  group_by(experiment_id, timepoint_min, timepoint_min_f) %>%
  summarise(median_count = median(mosquito_count, na.rm = TRUE), .groups = "drop")

p <- ggplot(df, aes(x = timepoint_min_f, y = mosquito_count)) +
  geom_boxplot(
    aes(group = interaction(timepoint_min_f, experiment_id),
        fill  = experiment_id,
        color = experiment_id),
    position = position_identity(),
    width = 0.7, alpha = 0.35, outlier.alpha = 0.4
  ) +
  geom_line(
    data = med_df,
    aes(x = timepoint_min_f, y = median_count, color = experiment_id, group = experiment_id),
    linewidth = 0.9
  ) +
  geom_point(
    data = med_df,
    aes(x = timepoint_min_f, y = median_count, color = experiment_id),
    size = 2
  ) +
  labs(
    title = "Mosquito counts by time (overlapped boxes per experiment)",
    x = "Time (minutes)",
    y = "Mosquito count",
    fill = "Experiment",
    color = "Experiment"
  ) +
  scale_x_discrete(labels = as.character(time_levels)) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank())

ggsave(out_pdf, p, width = 11, height = 6.5, dpi = 150)

Troubleshooting

Too many/few detections: adjust adaptive threshold parameters (block size and C), morphology kernel sizes, and area thresholds (min_area, max_area, extra size).

Very large images: processing is CPU-bound; consider downsampling the input (if accuracy allows) or limiting per-image concurrency to manage memory.

Cropping window not fullscreen: some Windows setups ignore the fullscreen hint; you can resize the window manually. The script never upscales for display.

CSV opens with odd encoding in Excel: pandas writes UTF-8 by default. If needed, open via Excel’s “From Text/CSV” and choose UTF-8.
