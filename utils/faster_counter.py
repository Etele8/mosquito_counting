import os
import cv2
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
FOLDER = "D:\\intezet\\gabor\\data\\images\\20250716\\60perc_cropped"
OUTPUT_FOLDER = "data/outputs/mosquito_outputs_07016/60"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "masks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "visuals"), exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_FOLDER, "mosquito_counts.csv")

# ---------- FUNCTIONS ----------

def preprocess_image(image):
    """Convert to grayscale, blur, and threshold to isolate dark objects."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 6
    )
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    eroded = cv2.erode(opened, np.ones((2, 2), np.uint8))
    return eroded

def filter_components(mask, min_area=25, max_area=1500):
    """Keep only connected components in a given area range."""
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
    """Color each component for visual feedback."""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    color_map = np.random.randint(0, 255, size=(labels.max() + 1, 3), dtype=np.uint8)
    for label in valid_labels:
        out[labels == label] = color_map[label]
    return out

# ---------- MAIN LOOP ----------

results = []
file_list = [f for f in os.listdir(FOLDER) if f.lower().endswith((".jpg", ".png"))]

for i, filename in enumerate(file_list, 1):
    path = os.path.join(FOLDER, filename)
    img = cv2.imread(path)
    if img is None:
        continue

    processed = preprocess_image(img)
    filtered_mask, labels, valid_labels, extra = filter_components(processed)
    colored = visualize_components(labels, valid_labels)

    base = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "masks", f"{base}_mask.png"), filtered_mask)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "visuals", f"{base}_vis.png"), colored)

    results.append({
        "filename": filename,
        "mosquito_count": len(valid_labels) + extra
    })

    print(f"{i}/{len(file_list)} {filename}: {len(valid_labels) + extra} mosquitoes detected")

# ---------- SAVE CSV ----------

df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False)
print(f"\nCSV saved to: {CSV_PATH}")
