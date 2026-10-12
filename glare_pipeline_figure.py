"""
Generate a 4-stage pipeline strip for the presentation:

  [ 1. RGB night frame ] -> [ 2. Semantic seg ] -> [ 3. Headlight mask ] -> [ 4. Glare applied ]

Uses REAL CARLA snapshots if available (run seg_snapshot.py first):
    seg_snapshot_rgb.png  +  seg_snapshot_labels.png / seg_snapshot_color.png
Otherwise falls back to a synthetic night scene with two cars so the figure
still works without CARLA.

Output: glare_pipeline_figure.png
Run: python3 glare_pipeline_figure.py
"""
import os
import cv2
import numpy as np
from glare_mask import extracted_glare_mask, glare, VEHICLE_TAG

PANEL_W, PANEL_H = 480, 300


def load_real():
    """Return (rgb_bgr, seg_labels, seg_color_or_None) from seg_snapshot.py output, or None."""
    rgb_p = "seg_snapshot_rgb.png"
    lab_p = "seg_snapshot_labels.png"      # vehicle-highlighted label view
    col_p = "seg_snapshot_color.png"       # cityscapes palette
    if not (os.path.exists(rgb_p)):
        return None
    rgb = cv2.imread(rgb_p)
    if rgb is None:
        return None
    # For the mask we need the true label channel. seg_snapshot_labels.png has
    # vehicles set to 255; reconstruct a label array where 255 -> VEHICLE_TAG.
    seg_color = cv2.imread(col_p) if os.path.exists(col_p) else None
    labels = None
    if os.path.exists(lab_p):
        lab_img = cv2.imread(lab_p, cv2.IMREAD_GRAYSCALE)
        if lab_img is not None:
            labels = np.where(lab_img >= 250, VEHICLE_TAG, 0).astype(np.uint8)
    return rgb, labels, seg_color


def make_synthetic():
    """Dark night scene with two cars (bright headlights) + a seg label map."""
    H, W = 600, 960
    rgb = np.full((H, W, 3), 12, dtype=np.uint8)
    cv2.rectangle(rgb, (0, 400), (W, H), (38, 38, 44), -1)  # road
    seg = np.zeros((H, W), dtype=np.uint8)

    def car(cx, cy, cw, ch, body):
        x0, y0 = cx - cw // 2, cy - ch // 2
        cv2.rectangle(rgb, (x0, y0), (x0 + cw, y0 + ch), (body,) * 3, -1)
        seg[y0:y0 + ch, x0:x0 + cw] = VEHICLE_TAG
        for hx in (x0 + int(cw * 0.15), x0 + int(cw * 0.85)):
            hy = y0 + int(ch * 0.72)
            cv2.circle(rgb, (hx, hy), max(4, cw // 22), (255, 255, 255), -1)

    car(330, 330, 240, 130, 60)   # near car
    car(600, 300, 130, 75, 55)    # farther car
    # colorized seg for display: road grey, vehicles blue-ish
    seg_color = np.zeros((H, W, 3), dtype=np.uint8)
    seg_color[400:, :] = (80, 60, 60)
    seg_color[seg == VEHICLE_TAG] = (142, 0, 0)  # CARLA-ish vehicle color (BGR)
    return rgb, seg, seg_color


def fit(img):
    return cv2.resize(img, (PANEL_W, PANEL_H))


def label(img, title, subtitle):
    out = fit(img).copy()
    cv2.rectangle(out, (0, 0), (PANEL_W, 46), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, subtitle, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (PANEL_W - 1, PANEL_H - 1), (70, 70, 70), 1)
    return out


def arrow_strip(h):
    """A thin vertical arrow separator column."""
    col = np.zeros((h, 46, 3), dtype=np.uint8)
    cy = h // 2
    cv2.arrowedLine(col, (6, cy), (40, cy), (255, 255, 255), 2, tipLength=0.4)
    return col


# --- Load data ---
real = load_real()
if real is not None and real[1] is not None:
    rgb, labels, seg_color = real
    source = "real CARLA frame"
else:
    rgb, labels, seg_color = make_synthetic()
    source = "synthetic scene (run seg_snapshot.py for real frames)"

# --- Stage outputs ---
mask, rgb_frame = extracted_glare_mask(rgb, labels, brightness_thresh=110)
glared = glare(mask, rgb_frame, intensity=3.5)

# Mask as a viewable 3-channel image
mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Segmentation panel: use color palette if available, else the label highlight
if seg_color is not None:
    seg_panel = seg_color
else:
    seg_panel = cv2.cvtColor(np.where(labels == VEHICLE_TAG, 255, 40).astype(np.uint8),
                             cv2.COLOR_GRAY2BGR)

p1 = label(rgb_frame, "1. RGB (night)", "Raw camera frame")
p2 = label(seg_panel, "2. SEGMENTATION", "Vehicles = class tag 14")
p3 = label(mask_vis, "3. HEADLIGHT MASK", "Front-corner bright pixels")
p4 = label(glared, "4. GLARE APPLIED", "Camera occlusion effect")

h = PANEL_H
sep = arrow_strip(h)
strip = np.hstack([p1, sep, p2, sep, p3, sep, p4])

# Header
header_h = 58
canvas = np.zeros((strip.shape[0] + header_h, strip.shape[1], 3), dtype=np.uint8)
canvas[header_h:, :] = strip
cv2.putText(canvas, "Glare Pipeline:  RGB  ->  Segmentation  ->  Headlight Mask  ->  Glare",
            (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imwrite("glare_pipeline_figure.png", canvas)
print("Wrote glare_pipeline_figure.png (%dx%d) using %s" %
      (canvas.shape[1], canvas.shape[0], source))
