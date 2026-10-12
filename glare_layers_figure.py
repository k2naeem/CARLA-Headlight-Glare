"""
Generate an annotated presentation figure that breaks the headlight glare into
its three optical layers, using the ACTUAL functions from glare_mask.py so the
figure matches the real pipeline.

Output: glare_layers_figure.png
  A 2x2 panel:
    [ Core (blown-out white) ]  [ Radial halo (lens scatter) ]
    [ Star flares (aperture) ]  [ Combined -> "reads like a real camera" ]

Run: python3 glare_layers_figure.py   (no CARLA needed)
"""
import cv2
import numpy as np
from glare_mask import _radial_halo, _draw_star_flares, _headlight_centers

# --- Build one synthetic headlight on a dark night background ---
H, W = 420, 560
bg = np.full((H, W, 3), 10, dtype=np.uint8)
cv2.rectangle(bg, (0, 300), (W, H), (35, 35, 40), -1)  # faint ground

# One bright headlight blob
mask = np.zeros((H, W), dtype=np.uint8)
cx, cy, r = W // 2, H // 2, 10
cv2.circle(mask, (cx, cy), r, 255, -1)

centers = _headlight_centers(mask)

# --- Layer 1: pure-white blown-out core (spread a little, like glare()) ---
white_seed = np.zeros((H, W, 3), dtype=np.float32)
white_seed[mask > 0] = (255.0, 255.0, 255.0)
core = cv2.GaussianBlur(np.clip(white_seed, 0, 255), (0, 0), sigmaX=4.0)
core_vis = np.clip(bg.astype(np.float32) + core, 0, 255).astype(np.uint8)

# --- Layer 2: radial halo (soft lens-scatter glow) ---
halo = _radial_halo((H, W), centers)
halo_vis = np.clip(bg.astype(np.float32) + halo * 1.2, 0, 255).astype(np.uint8)

# --- Layer 3: star flares (aperture diffraction spikes) ---
# Boosted brightness for the isolated panel so the spikes project clearly.
flare = _draw_star_flares((H, W), centers)
flare_vis = np.clip(bg.astype(np.float32) + flare * 5.0, 0, 255).astype(np.uint8)

# --- Combined (the same additive + screen composite glare() uses) ---
base = bg.astype(np.float32)
lit = base + halo * 1.4 + flare * 0.7
b = np.clip(lit / 255.0, 0.0, 1.0)
c = np.clip(core / 255.0, 0.0, 1.0)
combined = (1.0 - (1.0 - b) * (1.0 - c))
combined_vis = np.clip(combined * 255.0, 0, 255).astype(np.uint8)


def label(img, title, subtitle):
    """Add a title + subtitle banner to a panel."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (W, 54), (0, 0, 0), -1)
    cv2.putText(out, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, subtitle, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W - 1, H - 1), (70, 70, 70), 1)
    return out


p1 = label(core_vis, "1. CORE", "Sensor saturates -> pure white")
p2 = label(halo_vis, "2. HALO", "Lens scatter -> soft glow disc")
p3 = label(flare_vis, "3. STAR FLARES", "Aperture diffraction spikes")
p4 = label(combined_vis, "COMBINED", "Reads like a real night camera")

top = np.hstack([p1, p2])
bottom = np.hstack([p3, p4])
grid = np.vstack([top, bottom])

# Header strip
header_h = 60
canvas = np.zeros((grid.shape[0] + header_h, grid.shape[1], 3), dtype=np.uint8)
canvas[header_h:, :] = grid
cv2.putText(canvas, "Headlight Glare = 3 Optical Layers", (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imwrite("glare_layers_figure.png", canvas)
print("Wrote glare_layers_figure.png  (%dx%d)" % (canvas.shape[1], canvas.shape[0]))
