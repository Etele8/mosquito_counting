import os
import cv2
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
FOLDER = "D:/intezet/gabor/data/images/20250702/mixed/20perc_23.45"
OUTPUT_FOLDER = "data/outputs/mosquito_outputs_0702/mixed/20"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "masks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "visuals"), exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_FOLDER, "mosquito_counts.csv")

# ---------- PROCESSING FUNCTIONS ----------

def preprocess_image(image):
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
    eroded = cv2.morphologyEx(opened, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
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

def annotate_original(image, section_counts):
    annotated = image.copy()
    h, w = image.shape[:2]
    section_width = w // 5

    for i, count in enumerate(section_counts):
        x_start = i * section_width
        # Draw vertical line
        if i > 0:
            cv2.line(annotated, (x_start, 0), (x_start, h), (0, 0, 0), 4)

        # Write count
        cv2.putText(
            annotated,
            f"{count}db",
            (x_start + 160, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    return annotated

def compute_section_mean_color(image_section):
    mean_color = image_section.mean(axis=(0, 1))  # BGR
    return int(mean_color[2]), int(mean_color[1]), int(mean_color[0])  # Convert to RGB

# ---------- MAIN ----------

results = []
file_list = [f for f in os.listdir(FOLDER) if f.lower().endswith((".jpg", ".png"))]

for i, filename in enumerate(file_list, 1):
    path = os.path.join(FOLDER, filename)
    img = cv2.imread(path)
    if img is None:
        continue

    h, w = img.shape[:2]
    section_width = w // 5
    total_count = 0
    section_counts = []
    section_colors = []

    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for s in range(5):
        x_start = s * section_width
        x_end = (s + 1) * section_width if s < 4 else w
        section = img[:, x_start:x_end]

        # Color detection
        r, g, b = compute_section_mean_color(section)
        section_colors.append((r, g, b))

        # Mosquito detection
        proc = preprocess_image(section)
        filtered_mask, labels, valid_labels, extra = filter_components(proc)

        combined_mask[:, x_start:x_end] = filtered_mask

        section_total = len(valid_labels) + extra
        section_counts.append(section_total)
        total_count += section_total

    annotated_img = annotate_original(img, section_counts)

    # Save outputs
    base = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "masks", f"{base}_mask.png"), combined_mask)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "visuals", f"{base}_annotated.png"), annotated_img)

    row = {
        "filename": filename,
        "total_mosquitoes": total_count,
    }

    # Add counts and colors
    for s in range(5):
        row[f"section_{s+1}"] = section_counts[s]
        row[f"section_{s+1}_color"] = f"rgb({section_colors[s][0]}, {section_colors[s][1]}, {section_colors[s][2]})"

    results.append(row)

    print(f"{i}/{len(file_list)} {filename}: {total_count} mosquitoes detected")

# Save CSV
df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False)
print(f"\nCSV saved to: {CSV_PATH}")
