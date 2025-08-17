import os
import cv2
import numpy as np
from pathlib import Path

NEW_FOLDER = "20250816"  # root folder containing many subfolders

root = Path(__file__).resolve().parent.parent
INPUT_ROOT  = root / "data" / "images" / NEW_FOLDER
OUTPUT_ROOT = root / "data" / "images" / NEW_FOLDER / "cropped2"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- geometry ----------
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = pts[np.argmin(s)]    # TL
    rect[2] = pts[np.argmax(s)]    # BR
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def crop_and_warp(image, pts):
    rect = order_points(pts)
    widthA  = np.linalg.norm(rect[2] - rect[3])
    widthB  = np.linalg.norm(rect[1] - rect[0])
    heightA = np.linalg.norm(rect[1] - rect[2])
    heightB = np.linalg.norm(rect[0] - rect[3])
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    maxW = max(maxW, 10); maxH = max(maxH, 10)
    dst = np.array([[0,0], [maxW-1,0], [maxW-1,maxH-1], [0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))

# ---------- display helpers ----------
def get_screen_resolution():
    try:
        import tkinter as tk
        _root = tk.Tk(); _root.withdraw()
        w, h = _root.winfo_screenwidth(), _root.winfo_screenheight()
        _root.destroy()
        return w, h
    except:
        return 1920, 1080

def draw_overlay(img_base, pts, handle_r=4):
    """Return a fresh overlay image with polygon + handles drawn."""
    vis = img_base.copy()
    if len(pts) > 0:
        # lines
        for i in range(1, len(pts)):
            cv2.line(vis, tuple(pts[i-1]), tuple(pts[i]), (255, 0, 255), 1)
        if len(pts) == 4:
            cv2.line(vis, tuple(pts[3]), tuple(pts[0]), (255, 0, 255), 1)
        # handles
        for (x, y) in pts:
            cv2.circle(vis, (x, y), handle_r, (0, 255, 255), -1)
            # cv2.circle(vis, (x, y), handle_r+2, (0, 0, 0), 1)
    return vis

def nearest_handle(pts, x, y, tol=18):
    """Return index of nearest point within tol pixels; else None."""
    if not pts:
        return None
    d2 = [(px-x)**2 + (py-y)**2 for (px, py) in pts]
    j = int(np.argmin(d2))
    return j if d2[j] <= tol*tol else None

# ---------- main ----------
screen_w, screen_h = get_screen_resolution()

# Collect all images recursively
img_paths = [p for p in INPUT_ROOT.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
img_paths.sort()
print(f"Loaded {len(img_paths)} images from {INPUT_ROOT}")

# Remember last rectangle in NORMALIZED coords (x/W, y/H)
state = {"last_pts_norm": None}

for idx, src in enumerate(img_paths, start=1):
    rel = src.relative_to(INPUT_ROOT)
    dst = OUTPUT_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"[{idx}/{len(img_paths)}] {rel} -> already exists, skipping")
        continue

    img = cv2.imread(str(src))
    if img is None:
        print(f"[{idx}/{len(img_paths)}] {rel} -> failed to read, skipping")
        continue

    H, W = img.shape[:2]
    win = "Corners (TL,TR,BR,BL) | n=accept, r=reset, q=quit | drag handles to adjust"

    # Scale to fit screen (no upscaling)
    scale = min(screen_w / W, screen_h / H, 1.0)
    disp_w, disp_h = int(W * scale), int(H * scale)
    img_disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # UI context for this image
    ctx = {
        "points_disp": [],          # list[[x,y], ...] in display coords
        "img_base": img_disp,       # immutable base
        "img_show": img_disp.copy(),
        "drag_idx": None,           # index of point being dragged
        "done": False
    }

    # Pre-draw last rectangle if available
    if state["last_pts_norm"] and len(state["last_pts_norm"]) == 4:
        prev_orig = [(int(xn * W), int(yn * H)) for (xn, yn) in state["last_pts_norm"]]
        prev_disp = [(int(px * scale), int(py * scale)) for (px, py) in prev_orig]
        ctx["points_disp"] = [list(p) for p in prev_disp]
        ctx["img_show"] = draw_overlay(ctx["img_base"], ctx["points_disp"])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, ctx["img_show"])

    def on_mouse(event, x, y, flags, param):
        c = param
        pts = c["points_disp"]

        if event == cv2.EVENT_LBUTTONDOWN:
            # grab existing handle or add a new point
            hit = nearest_handle(pts, x, y)
            if hit is not None:
                c["drag_idx"] = hit
            elif len(pts) < 4:
                pts.append([x, y])
                c["img_show"] = draw_overlay(c["img_base"], pts)
                cv2.imshow(win, c["img_show"])

        elif event == cv2.EVENT_MOUSEMOVE:
            if c["drag_idx"] is not None:
                # drag selected handle, clamp to window
                j = c["drag_idx"]
                pts[j][0] = max(0, min(x, c["img_base"].shape[1]-1))
                pts[j][1] = max(0, min(y, c["img_base"].shape[0]-1))
                c["img_show"] = draw_overlay(c["img_base"], pts)
                cv2.imshow(win, c["img_show"])

        elif event == cv2.EVENT_LBUTTONUP:
            c["drag_idx"] = None

    cv2.setMouseCallback(win, on_mouse, ctx)

    def accept_current_box():
        if len(ctx["points_disp"]) != 4:
            return
        # map display -> original
        pts_orig = [(int(px/scale), int(py/scale)) for (px, py) in ctx["points_disp"]]
        crop = crop_and_warp(img, pts_orig)
        cv2.imwrite(str(dst), crop)
        print(f"[{idx}/{len(img_paths)}] {rel} -> SAVED to {dst}")
        # remember normalized for next image
        state["last_pts_norm"] = [(px / W, py / H) for (px, py) in pts_orig]
        ctx["done"] = True
        cv2.destroyWindow(win)

    # key loop
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('n'):
            accept_current_box()
        elif key == ord('r'):
            ctx["points_disp"].clear()
            ctx["img_show"] = ctx["img_base"].copy()
            cv2.imshow(win, ctx["img_show"])
        elif key == ord('q') or key == 27:  # ESC
            print("Quitting.")
            cv2.destroyAllWindows()
            raise SystemExit

        if ctx["done"]:
            break

print(f"\nDone! All cropped images saved under: {OUTPUT_ROOT}")