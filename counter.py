"""Batch-count mosquitoes from cropped experiment images.

This script is the main public entry point for the repository. It walks a
directory of cropped experiment folders, runs a classical computer-vision
pipeline, and writes both per-folder CSV summaries and inspection artifacts
(binary masks and colorized connected-component views).
"""

import argparse
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "images" / "20250816" / "cropped"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "outputs" / "mosquito_outputs_0816"

# ---------- DETECTION (same as before) ----------
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 6)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    eroded = cv2.erode(opened, np.ones((2, 2), np.uint8))
    return eroded

def filter_components(mask, min_area=25, max_area=1500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    final_mask = np.zeros_like(mask)
    valid_labels = []
    extra = 0
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            final_mask[labels == i] = 255
            valid_labels.append(i)
            if area > 800:
                extra += 1
    return final_mask, labels, valid_labels, extra

def visualize_components(labels, valid_labels):
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if labels.max() < 1:
        return out
    rng = np.random.default_rng(seed=42)
    color_map = rng.integers(0, 255, size=(labels.max() + 1, 3), dtype=np.uint8)
    for label in valid_labels:
        out[labels == label] = color_map[label]
    return out

def parse_folder_name(folder_name: str):
    """
    Parse 'experimentid_timepointperc' → ('experimentid', minutes).
    If only '60perc...' exists, exp_id = full name, minutes parsed.
    """
    name = folder_name.strip()
    m = re.match(r'^(?P<exp>[^_]+)_(?P<min>\d+)\s*perc', name, flags=re.IGNORECASE)
    if m:
        return m.group('exp'), int(m.group('min'))
    m2 = re.match(r'^(?P<min>\d+)\s*perc', name, flags=re.IGNORECASE)
    if m2:
        return re.sub(r'\s+', '', name), int(m2.group('min'))
    return name, -1

def process_folder(folder: Path, output_root: Path, min_area: int, max_area: int):
    exp_id, minutes = parse_folder_name(folder.name)
    print(f"\nProcessing: {folder.name} -> experiment_id={exp_id}, time={minutes} min")

    out_dir = output_root / folder.name
    masks_dir = out_dir / "masks"
    visuals_dir = out_dir / "visuals"
    masks_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    file_list = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    sub_results = []
    image_rows = []

    for index, img_path in enumerate(file_list, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip unreadable] {img_path.name}")
            continue

        processed = preprocess_image(img)
        filtered_mask, labels, valid_labels, extra = filter_components(
            processed, min_area=min_area, max_area=max_area
        )
        colored = visualize_components(labels, valid_labels)

        base = img_path.stem
        cv2.imwrite(str(masks_dir / f"{base}_mask.png"), filtered_mask)
        cv2.imwrite(str(visuals_dir / f"{base}_vis.png"), colored)

        count = len(valid_labels) + extra
        sub_results.append({"filename": img_path.name, "mosquito_count": count})
        image_rows.append(
            {
                "experiment_id": exp_id,
                "timepoint_min": minutes,
                "filename": img_path.name,
                "mosquito_count": count,
                "subfolder": folder.name,
            }
        )
        print(f"  {index}/{len(file_list)} {img_path.name}: {count}")

    pd.DataFrame(sub_results).to_csv(out_dir / "mosquito_counts.csv", index=False)
    print(f"  CSV saved -> {out_dir / 'mosquito_counts.csv'}")
    return image_rows


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Count mosquitoes in cropped experiment images and export CSV summaries."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Folder containing cropped experiment subfolders. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Folder for generated CSVs, masks, and visualizations. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=25,
        help="Minimum connected-component area kept as a candidate mosquito.",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=1500,
        help="Maximum connected-component area kept as a candidate mosquito.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_root}. "
            "See data/README.md for notes on omitted local datasets."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    image_rows = []
    for folder in sorted(input_root.iterdir()):
        if folder.is_dir():
            image_rows.extend(
                process_folder(
                    folder,
                    output_root=output_root,
                    min_area=args.min_area,
                    max_area=args.max_area,
                )
            )

    images_long_csv = output_root / "all_images_long.csv"
    all_images = pd.DataFrame(image_rows)
    if not all_images.empty:
        all_images = all_images.sort_values(["timepoint_min", "experiment_id", "filename"])
    all_images.to_csv(images_long_csv, index=False)
    print(f"\nPer-image long CSV saved: {images_long_csv}")


if __name__ == "__main__":
    main()
