"""
Generate presentation figures for the glare slide deck. All figures are built
from the ACTUAL glare_mask.py functions so they match the real pipeline.

Outputs (in ./slide_figures/):
  fig1_problem.png        - a car with headlights + white plate + road pool
  fig2_geometry.png       - projected headlight points (from 3D bbox) on the car
  fig3_three_way_and.png  - geometry region | vehicle seg | result mask
  fig4_glare_layers.png   - core / halo / starburst / veil / combined
  fig5_pipeline.png       - RGB -> mask -> glare, one strip

Run: python3 make_slide_figures.py   (no CARLA needed)
"""
import os
import cv2
import numpy as np
import glare_mask as gm
from glare_mask import (
    geometry_headlight_mask, glare, VEHICLE_TAG,
    _draw_star_flares, _radial_halo, _veiling_glare, _build_spectral_star,
)

OUT = "slide_figures"
os.makedirs(OUT, exist_ok=True)
H, W = 540, 960


def base_scene():
    """A dark night scene: one car with headlights, a bright white license
    plate in the center, and a bright headlight pool on the road below it."""
    frame = np.full((H, W, 3), 14, dtype=np.uint8)
    cv2.rectangle(frame, (0, 330), (W, H), (46, 46, 52), -1)          # road
    seg = np.zeros((H, W), dtype=np.uint8)                            # labels

    # vehicle body
    vx, vy, vw, vh = 360, 210, 240, 130
    cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (60, 60, 60), -1)
    seg[vy:vy + vh, vx:vx + vw] = VEHICLE_TAG

    # bright headlight pool ON THE ROAD in front (NOT tagged vehicle)
    cv2.ellipse(frame, (vx + vw // 2, 400), (150, 45), 0, 0, 360, (150, 150, 150), -1)

    # WHITE license plate in the CENTER-front (bright, must be rejected)
    px, py = vx + vw // 2, vy + int(vh * 0.8)
    cv2.rectangle(frame, (px - 26, py - 10), (px + 26, py + 10), (255, 255, 255), -1)

    # two round headlights at the front corners (bright, must be kept)
    hlpx = []
    for hx in (vx + int(vw * 0.15), vx + int(vw * 0.85)):
        hy = vy + int(vh * 0.7)
        cv2.circle(frame, (hx, hy), 11, (255, 255, 255), -1)
        hlpx.append((hx, hy, 9.0))     # (px, py, depth_m)
    return frame, seg, (vx, vy, vw, vh)


def label(img, text, color=(0, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (70, 70, 70), 1)
    return out


# ---- Figure 1: the problem ----
frame, seg, (vx, vy, vw, vh) = base_scene()
cv2.imwrite(os.path.join(OUT, "fig1_problem.png"),
            label(frame, "Problem: plate + road pool are ALSO bright"))

# ---- Figure 2: geometry projection ----
# Headlights come from the 3D bbox corners -> projected pixels (here we use the
# synthetic corner positions as stand-ins for the projected points).
hl = [(vx + int(vw * 0.15), vy + int(vh * 0.7), 9.0),
      (vx + int(vw * 0.85), vy + int(vh * 0.7), 9.0)]
geo = frame.copy()
# show the vehicle bbox and the two geometry-derived headlight anchors
cv2.rectangle(geo, (vx, vy), (vx + vw, vy + vh), (0, 200, 0), 2)
for (px, py, d) in hl:
    cv2.circle(geo, (px, py), 16, (0, 0, 255), 2)
cv2.imwrite(os.path.join(OUT, "fig2_geometry.png"),
            label(geo, "Geometry: 3D bbox -> projected headlight points (red)"))

# ---- Figure 3: the three-way AND ----
region = geometry_headlight_mask((W, H), hl, base_radius_px=10, max_radius=16) > 0
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
bright = gray >= 110
vehicle = seg == VEHICLE_TAG
result = region & vehicle  # our current gate (brightness optional)

def as_bgr(mask_bool):
    return cv2.cvtColor((mask_bool.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)

panel_region = label(as_bgr(region), "1. GEOMETRY region (where HLs can be)")
panel_veh = label(as_bgr(vehicle), "2. VEHICLE segmentation (on the car)")
panel_res = label(as_bgr(result), "3. AND -> headlight seed (plate/road gone)")
strip = np.hstack([panel_region, panel_veh, panel_res])
cv2.imwrite(os.path.join(OUT, "fig3_three_way_and.png"), strip)

# ---- Figure 4: glare layers ----
mask = (result.astype(np.uint8)) * 255
# individual layers on the dark scene
flare_pts = hl
core_seed = np.zeros_like(frame, dtype=np.float32)
core_seed[mask > 0] = 255.0
core_vis = np.clip(frame.astype(np.float32) + cv2.GaussianBlur(core_seed, (0, 0), 4.0), 0, 255).astype(np.uint8)
halo_vis = np.clip(frame.astype(np.float32) + _radial_halo(frame.shape, flare_pts) * 1.2, 0, 255).astype(np.uint8)
flare_vis = np.clip(frame.astype(np.float32) + _draw_star_flares(frame.shape, flare_pts) * 2.0, 0, 255).astype(np.uint8)
veil_vis = np.clip(frame.astype(np.float32) + _veiling_glare(frame.shape, flare_pts) * 1.5, 0, 255).astype(np.uint8)
combined = glare(mask, frame, intensity=3.5, headlight_pixels=hl)

top = np.hstack([label(core_vis, "CORE (saturation)"), label(halo_vis, "HALO (scatter)")])
mid = np.hstack([label(flare_vis, "STARBURST (FFT of aperture)"), label(veil_vis, "VEILING GLARE (occlusion)")])
bot = np.hstack([label(combined, "COMBINED"), label(frame, "ORIGINAL (no glare)")])
grid = np.vstack([top, mid, bot])
cv2.imwrite(os.path.join(OUT, "fig4_glare_layers.png"), grid)

# ---- Figure 5: pipeline strip ----
p_rgb = label(frame, "1. RGB frame")
p_mask = label(as_bgr(result), "2. Headlight-only mask")
p_out = label(combined, "3. Glare applied")
sep = np.zeros((H, 24, 3), dtype=np.uint8)
cv2.arrowedLine(sep, (4, H // 2), (20, H // 2), (255, 255, 255), 2, tipLength=0.5)
strip5 = np.hstack([p_rgb, sep, p_mask, sep, p_out])
cv2.imwrite(os.path.join(OUT, "fig5_pipeline.png"), strip5)

# ---- Bonus: the FFT aperture -> starburst derivation ----
canon = _build_spectral_star()
star_img = np.clip(canon / (canon.max() + 1e-9) * 255.0, 0, 255).astype(np.uint8)
star_img = cv2.resize(star_img, (H, H))
cv2.imwrite(os.path.join(OUT, "fig6_fft_star.png"),
            label(star_img, "Starburst = |FFT(hex aperture)|^2, 5 wavelengths"))

print("Wrote figures to ./%s/" % OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
