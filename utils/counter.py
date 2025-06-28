import os
import cv2
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
IMAGE_FOLDER = "D:/intezet/gabor/data/feher_1"
OUTPUT_FOLDER = "mosquito_outputs"
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
    return opened

def filter_components(mask, min_area=150, max_area=1500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    final_mask = np.zeros_like(mask)
    valid_labels = []

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            final_mask[labels == i] = 255
            valid_labels.append(i)
    
    return final_mask, labels, valid_labels

def visualize_components(labels, valid_labels):
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    color_map = np.random.randint(0, 255, size=(labels.max() + 1, 3), dtype=np.uint8)
    
    for label in valid_labels:
        out[labels == label] = color_map[label]
    
    return out

def resize_for_display(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def crop_to_white_paper(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image  # fallback: return original

    # Find the largest contour (assumed to be the paper)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    cropped = image[y:y+h, x:x+w]
    return cropped


# ---------- MAIN ----------

results = []
i = 0
for filename in os.listdir(IMAGE_FOLDER):
    i += 1
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    path = os.path.join(IMAGE_FOLDER, filename)
    img = cv2.imread(path)

    if img is None:
        continue

    processed = preprocess_image(img)
    filtered_mask, labels, valid_labels = filter_components(processed)
    colored = visualize_components(labels, valid_labels)
    cropped = crop_to_white_paper(img)

    # Save outputs
    base = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "masks", f"{base}_mask.png"), filtered_mask)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "visuals", f"{base}_vis.png"), colored)
    # cv2.imwrite(os.path.join(OUTPUT_FOLDER, "cropped", f"{base}_original.png"), cropped)

    # Log results
    results.append({
        "filename": filename,
        "mosquito_count": len(valid_labels)
    })

    print(f" {i}/{len(os.listdir(IMAGE_FOLDER))} {filename}: {len(valid_labels)} mosquitoes")

# Save CSV
df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False)
print(f"\nCSV saved to: {CSV_PATH}")