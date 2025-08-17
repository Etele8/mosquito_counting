import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
root = Path(__file__).resolve().parent
INPUT_ROOT  = root / "data" / "images" / "20250816/cropped"
OUTPUT_ROOT = root / "data" / "outputs" / "mosquito_outputs_0816"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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
    color_map = np.random.randint(0, 255, size=(labels.max() + 1, 3), dtype=np.uint8)
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

# ---------- MAIN ----------
image_rows = []   # <-- per-image long rows

for folder in sorted(INPUT_ROOT.iterdir()):
    if not folder.is_dir():
        continue

    exp_id, minutes = parse_folder_name(folder.name)
    print(f"\nProcessing: {folder.name}  ->  exp_id={exp_id}, time={minutes} min")

    out_dir   = OUTPUT_ROOT / folder.name
    masks_dir = out_dir / "masks"
    vis_dir   = out_dir / "visuals"
    masks_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    file_list = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    sub_results = []
    sub_total = 0

    for i, img_path in enumerate(file_list, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip unreadable] {img_path.name}")
            continue

        processed = preprocess_image(img)
        filtered_mask, labels, valid_labels, extra = filter_components(processed)
        colored = visualize_components(labels, valid_labels)

        base = img_path.stem
        cv2.imwrite(str(masks_dir / f"{base}_mask.png"), filtered_mask)
        cv2.imwrite(str(vis_dir   / f"{base}_vis.png"),  colored)

        count = len(valid_labels) + extra
        sub_total += count
        sub_results.append({"filename": img_path.name, "mosquito_count": count})

        # -------- per-image LONG row --------
        image_rows.append({
            "experiment_id": exp_id,
            "timepoint_min": minutes,
            "filename": img_path.name,
            "mosquito_count": count,
            "subfolder": folder.name
        })

        print(f"  {i}/{len(file_list)}  {img_path.name}: {count}")

    # per-subfolder CSV
    pd.DataFrame(sub_results).to_csv(out_dir / "mosquito_counts.csv", index=False)
    print(f"  CSV saved → {out_dir / 'mosquito_counts.csv'}")

# ---------- SAVE BIG CSVs ----------
# Per-image long CSV
images_long_csv = OUTPUT_ROOT / "all_images_long.csv"
pd.DataFrame(image_rows).sort_values(["timepoint_min", "experiment_id", "filename"]).to_csv(images_long_csv, index=False)
print(f"\n✅ Per-image long CSV saved: {images_long_csv}")