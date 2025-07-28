import os
import cv2
import numpy as np

FOLDER = r'D:\intezet\gabor\data\images\20250716\60'
OUTPUT = r'D:\intezet\gabor\data\images\20250716\60perc_cropped'
os.makedirs(OUTPUT, exist_ok=True)

files = [f for f in os.listdir(FOLDER) if f.lower().endswith(('.jpg', '.png'))]
files.sort()
print("Loaded", len(files), "images.")

def order_points(pts):
    pts = np.array(pts)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_and_warp(image, pts):
    rect = order_points(pts)
    widthA = np.linalg.norm(rect[2] - rect[3])
    widthB = np.linalg.norm(rect[1] - rect[0])
    heightA = np.linalg.norm(rect[1] - rect[2])
    heightB = np.linalg.norm(rect[0] - rect[3])
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0,0], [maxW-1,0], [maxW-1,maxH-1], [0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped

# Get screen resolution
def get_screen_resolution():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except:
        return 1920, 1080

screen_w, screen_h = get_screen_resolution()

for i, fname in enumerate(files):
    img = cv2.imread(os.path.join(FOLDER, fname))
    orig_h, orig_w = img.shape[:2]
    window_name = 'Click 4 corners: r=reset, q=quit'

    # Compute scaling factor to fit fullscreen, but not enlarge
    scale = min(screen_w / orig_w, screen_h / orig_h, 1.0)
    disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
    img_disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    points_disp = []
    img_show = img_disp.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img_show)

    def click_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_disp) < 4:
            points_disp.append([x, y])
            cv2.circle(img_show, (x, y), 8, (0,255,255), -1)
            if len(points_disp) > 1:
                cv2.line(img_show, tuple(points_disp[-2]), (x, y), (255, 0, 255), 2)
            cv2.imshow(window_name, img_show)
            if len(points_disp) == 4:
                cv2.line(img_show, tuple(points_disp[3]), tuple(points_disp[0]), (255, 0, 255), 2)
                cv2.imshow(window_name, img_show)
                cv2.waitKey(400)
                # Map display points back to original image coordinates
                points_orig = [[int(x/scale), int(y/scale)] for x, y in points_disp]
                crop = crop_and_warp(img, points_orig)
                out_path = os.path.join(OUTPUT, fname)
                cv2.imwrite(out_path, crop)
                print(f"[{i+1}/{len(files)}] {fname} -> SAVED to {out_path}")
                cv2.destroyWindow(window_name)
    cv2.setMouseCallback(window_name, click_points)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('r'):
            points_disp = []
            img_show = img_disp.copy()
            cv2.imshow(window_name, img_show)
        elif key == ord('q'):
            print("Quitting.")
            exit()
        if len(points_disp) == 4:
            break

print("\nDone! All cropped images saved to", OUTPUT)
