"""Physically-motivated headlight glare for CARLA night scenes.

Oncoming headlights are localized from vehicle bounding-box geometry, gated
against depth and semantic segmentation so that only vehicles the ego camera
genuinely resolves can emit glare, then rendered as four optical layers:

    core    saturated white at the source
    bloom   multi-scale soft halo
    star    far-field diffraction pattern of the aperture, sampled across five
            wavelengths so the rays show chromatic dispersion
    veil    broad low-amplitude haze that lifts the local black level

The starburst follows Hullin et al. (2011), "Physically-Based Real-Time Lens
Flare Rendering", sec. 7.1.2: the far-field pattern of a point source seen
through an aperture is the power spectrum of the aperture shape. The veiling
layer is what actually occludes road users, washing out contrast around the
light rather than covering pixels.

NOTE: the underscore-prefixed layer functions (_draw_star_flares, _radial_halo,
_veiling_glare, _build_spectral_star, _headlight_centers) are imported directly
by the figure scripts in this repo. Treat their signatures as public.
"""
import cv2
import numpy as np


# Cityscapes-aligned CARLA palette (0.9.14/0.9.15). Verify for your build with
# np.unique() over a frame containing a visible car. A cyclist occupies both 13
# and 19 — rider and bicycle carry separate labels, so anything measuring a
# cyclist must count both.
VEHICLE_TAG = 14
PEDESTRIAN_TAG = 12
RIDER_TAG = 13
BICYCLE_TAG = 19


def _to_bgr(image_or_array):
    """Accept either a CARLA image object or a numpy array and return a BGR frame."""
    if isinstance(image_or_array, np.ndarray):
        arr = image_or_array
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return arr  # already BGR
    # CARLA image object
    arr = np.frombuffer(image_or_array.raw_data, dtype=np.uint8)
    arr = np.reshape(arr, (image_or_array.height, image_or_array.width, 4))
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)


def _to_labels(seg_or_array):
    """Accept either a CARLA seg image or numpy array and return the label channel."""
    if isinstance(seg_or_array, np.ndarray):
        arr = seg_or_array
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, 2]  # CARLA encodes class id in the red channel
        return arr  # already single-channel labels
    arr = np.frombuffer(seg_or_array.raw_data, dtype=np.uint8)
    arr = np.reshape(arr, (seg_or_array.height, seg_or_array.width, 4))
    return arr[:, :, 2]


# ---------------------------------------------------------------------------
# Geometry-based headlight localization
# ---------------------------------------------------------------------------

# Headlight position as a fraction of the vehicle's bounding-box extent. The
# forward push is deliberately modest so the lights stay attached to the body
# when a car is close and angled.
FORWARD_FRAC = 0.35
LATERAL_FRAC = 0.62
HEIGHT_FRAC = -0.10   # near bbox-centre height, not down at the bumper


def build_camera_intrinsics(width, height, fov_deg):
    """Standard pinhole intrinsic matrix K for a CARLA RGB camera."""
    f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    K = np.array([
        [f, 0.0, width / 2.0],
        [0.0, f, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


def headlight_points_local(extent, bb_center=(0.0, 0.0, 0.0)):
    """Return the two headlight positions in the vehicle's LOCAL frame."""
    ex, ey, ez = extent
    cx, cy, cz = bb_center
    x = cx + FORWARD_FRAC * ex
    z = cz + HEIGHT_FRAC * ez
    y_off = LATERAL_FRAC * ey
    return [
        (x, cy + y_off, z),
        (x, cy - y_off, z),
    ]


def _world_to_camera_matrix(cam_matrix_world):
    """Invert the camera's world matrix to get world->camera."""
    return np.linalg.inv(np.asarray(cam_matrix_world, dtype=np.float64))


def project_points_to_image(world_points, cam_matrix_world, K):
    """Project 3D world points into image pixels."""
    if len(world_points) == 0:
        return []

    w2c = _world_to_camera_matrix(cam_matrix_world)
    pts = np.asarray(world_points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homog = np.hstack([pts, ones]).T

    cam = w2c @ homog

    x_ue, y_ue, z_ue = cam[0, :], cam[1, :], cam[2, :]
    pinhole = np.vstack([y_ue, -z_ue, x_ue])

    depth = pinhole[2, :]
    out = []
    for i in range(pinhole.shape[1]):
        d = depth[i]
        if d <= 0.01:
            out.append((None, None, d))
            continue
        uvw = K @ pinhole[:, i]
        px = uvw[0] / uvw[2]
        py = uvw[1] / uvw[2]
        out.append((float(px), float(py), float(d)))
    return out


def headlight_pixels_for_vehicles(vehicle_records, cam_matrix_world, K,
                                  image_size, cam_location=None,
                                  max_distance=None, edge_margin=0):
    """Project every vehicle's headlights to (px, py, depth_m) image points.

    max_distance caps the range in metres: a car at 100 m is a few pixels wide
    but would still get a full-size starburst stamped on it. edge_margin keeps
    projections away from the frame border, where they are least trustworthy.

    Facing is NOT checked here — camera_detected_headlights() gates on it first.
    """
    width, height = image_size
    world_pts = []

    for rec in vehicle_records:
        local_pts = headlight_points_local(rec['extent'], rec.get('bb_center', (0.0, 0.0, 0.0)))
        M = np.asarray(rec['matrix'], dtype=np.float64)
        for (lx, ly, lz) in local_pts:
            p = M @ np.array([lx, ly, lz, 1.0], dtype=np.float64)
            world_pts.append((p[0], p[1], p[2]))

    projected = project_points_to_image(world_pts, cam_matrix_world, K)

    result = []
    for (px, py, d) in projected:
        if px is None:
            continue
        if max_distance is not None and d > max_distance:
            continue
        m = edge_margin
        if px < m or px >= width - m or py < m or py >= height - m:
            continue
        result.append((px, py, d))
    return result


def visible_headlight_pixels(headlight_pixels, depth_map, image_size,
                             tol_m=2.0, tol_frac=0.08, win=1):
    """Drop headlights occluded by nearer geometry, per the depth buffer.

    A headlight is kept unless the buffer reports something closer than the
    headlight's own camera-space depth. Background behind the car's silhouette
    reads as farther and is therefore kept — the projection is a point, the
    silhouette is not. `depth_map` may be downscaled and is indexed
    proportionally; the farthest depth in a small window is used so sub-pixel
    error at a car's edge does not cause false drops. Tolerance grows with
    range because depth precision and vehicle length both matter more far away.
    """
    if depth_map is None or not headlight_pixels:
        return list(headlight_pixels)

    depth_h, depth_w = depth_map.shape[:2]
    width, height = image_size
    scale_x = depth_w / float(width)
    scale_y = depth_h / float(height)

    kept = []
    for (px, py, d) in headlight_pixels:
        ix = int(round(px * scale_x))
        iy = int(round(py * scale_y))
        x0, x1 = max(0, ix - win), min(depth_w, ix + win + 1)
        y0, y1 = max(0, iy - win), min(depth_h, iy + win + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        buffered = float(np.max(depth_map[y0:y1, x0:x1].astype(np.float32)))
        if buffered >= d - max(tol_m, tol_frac * d):
            kept.append((px, py, d))
    return kept


# ---------------------------------------------------------------------------
# Per-vehicle detection gate
# ---------------------------------------------------------------------------


def vehicle_bbox_corners_local(extent, bb_center=(0.0, 0.0, 0.0)):
    """The 8 corners of a vehicle's bounding box in its LOCAL frame."""
    ex, ey, ez = extent
    cx, cy, cz = bb_center
    return [
        (cx + sx * ex, cy + sy * ey, cz + sz * ez)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def project_vehicle_footprint(rec, cam_matrix_world, K, image_size):
    """Pixel rect covered by a vehicle's 3D bbox, plus its camera depth span.

    Returns (x0, y0, x1, y1, depth_near, depth_far) clipped to the image, or None
    when the vehicle is behind the camera or entirely outside the frame.
    """
    corners_local = vehicle_bbox_corners_local(
        rec['extent'], rec.get('bb_center', (0.0, 0.0, 0.0))
    )
    M = np.asarray(rec['matrix'], dtype=np.float64)
    world = [tuple((M @ np.array([lx, ly, lz, 1.0], dtype=np.float64))[:3])
             for (lx, ly, lz) in corners_local]
    projected = project_points_to_image(world, cam_matrix_world, K)

    xs, ys, ds = [], [], []
    for (px, py, d) in projected:
        if px is None:
            continue  # corner is behind the camera plane
        xs.append(px)
        ys.append(py)
        ds.append(d)
    if len(xs) < 4:
        return None  # mostly or entirely behind the camera

    width, height = image_size
    x0 = int(np.floor(max(0.0, min(xs))))
    y0 = int(np.floor(max(0.0, min(ys))))
    x1 = int(np.ceil(min(float(width), max(xs))))
    y1 = int(np.ceil(min(float(height), max(ys))))
    if x0 >= x1 or y0 >= y1:
        return None  # clipped away entirely -> not in the camera's view at all

    return (x0, y0, x1, y1, float(min(ds)), float(max(ds)))


def count_detected_vehicle_pixels(footprint, depth_map, seg_labels, image_size,
                                  depth_tol_m=2.5, depth_tol_frac=0.10):
    """How many pixels the camera actually attributes to THIS vehicle.

    A pixel counts only when segmentation labels it a vehicle AND the depth
    buffer there agrees with this vehicle's own bbox depth span. The depth term
    is what makes the count instance-specific rather than "some car is here".
    """
    x0, y0, x1, y1, d_near, d_far = footprint

    if seg_labels is not None:
        vehicle_px = (seg_labels[y0:y1, x0:x1] == VEHICLE_TAG)
    else:
        vehicle_px = np.ones((y1 - y0, x1 - x0), dtype=bool)
    if not vehicle_px.any():
        return 0
    if depth_map is None:
        return int(np.count_nonzero(vehicle_px))

    # Depth is stored downscaled; crop the matching region and blow it back up
    # to the footprint size. INTER_NEAREST throughout — averaging depth across a
    # silhouette edge invents a surface that is neither car nor background.
    width, height = image_size
    depth_h, depth_w = depth_map.shape[:2]
    scale_x = depth_w / float(width)
    scale_y = depth_h / float(height)

    dx0 = max(0, int(np.floor(x0 * scale_x)))
    dy0 = max(0, int(np.floor(y0 * scale_y)))
    dx1 = min(depth_w, max(int(np.ceil(x1 * scale_x)), dx0 + 1))
    dy1 = min(depth_h, max(int(np.ceil(y1 * scale_y)), dy0 + 1))
    if dx0 >= dx1 or dy0 >= dy1:
        return 0

    depth_crop = depth_map[dy0:dy1, dx0:dx1].astype(np.float32)
    depth_crop = cv2.resize(depth_crop, (vehicle_px.shape[1], vehicle_px.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    tol = max(depth_tol_m, depth_tol_frac * d_near)
    on_this_vehicle = (depth_crop >= d_near - tol) & (depth_crop <= d_far + tol)
    return int(np.count_nonzero(vehicle_px & on_this_vehicle))


def camera_detected_headlights(vehicle_records, cam_matrix_world, K, image_size,
                               depth_map=None, seg_labels=None, cam_location=None,
                               require_facing=True, facing_thresh=0.6,
                               max_distance=70.0, min_detected_px=150,
                               depth_tol_m=2.5, depth_tol_frac=0.10,
                               edge_margin=2, return_reports=False):
    """Headlight pixels for ONLY the vehicles the ego camera actually detects.

    A vehicle must clear every one of these to contribute glare:
      1. it faces the camera enough for its headlights to be pointed our way
      2. its bbox projects to a rect that overlaps the frame at all
      3. it is within max_distance
      4. at least `min_detected_px` pixels in that rect are labelled vehicle AND
         carry depth matching this vehicle's own range -> the camera sees it
      5. the individual headlight is not itself occluded (a car can be detected
         while one of its two lights hides behind a pole)

    Set return_reports=True to also get a per-vehicle verdict, which is what the
    diagnostic overlay uses to show why a given car was or wasn't allowed.
    """
    width, height = image_size
    kept = []
    reports = []

    for rec in vehicle_records:
        report = {'location': rec.get('location'), 'footprint': None,
                  'distance': None, 'detected_px': 0, 'headlights': [],
                  'status': None}

        # 1. Facing: headlights are directional, a car pointed away can't glare.
        if require_facing and cam_location is not None:
            forward = np.asarray(rec['forward'], dtype=np.float64)
            vehicle_loc = np.asarray(rec['location'], dtype=np.float64)
            camera_loc = np.asarray(cam_location, dtype=np.float64)
            to_cam = camera_loc - vehicle_loc
            norm = np.linalg.norm(to_cam)
            if norm < 1e-6:
                report['status'] = 'degenerate'
                reports.append(report)
                continue
            if float(np.dot(forward, to_cam / norm)) < facing_thresh:
                report['status'] = 'facing_away'
                reports.append(report)
                continue

        # 2. Is any part of the vehicle inside the camera's view?
        footprint = project_vehicle_footprint(rec, cam_matrix_world, K, image_size)
        if footprint is None:
            report['status'] = 'off_camera'
            reports.append(report)
            continue
        report['footprint'] = footprint[:4]
        report['distance'] = footprint[4]

        # 3. Range cap.
        if max_distance is not None and footprint[4] > max_distance:
            report['status'] = 'too_far'
            reports.append(report)
            continue

        # 4. The Gate: does the camera actually resolve this vehicle?
        detected_px = count_detected_vehicle_pixels(
            footprint, depth_map, seg_labels, image_size,
            depth_tol_m=depth_tol_m, depth_tol_frac=depth_tol_frac,
        )
        report['detected_px'] = detected_px
        if detected_px < min_detected_px:
            # Hidden behind geometry, or too small on screen to count as seen.
            report['status'] = 'not_detected'
            reports.append(report)
            continue

        # 5. Per-headlight occlusion, for the one-light-behind-a-pole case.
        #    Facing was already checked in step 1, so it is not rechecked here.
        headlights = headlight_pixels_for_vehicles(
            [rec], cam_matrix_world, K, image_size,
            cam_location=cam_location,
            max_distance=max_distance, edge_margin=edge_margin,
        )
        headlights = visible_headlight_pixels(headlights, depth_map, image_size)
        report['headlights'] = headlights
        report['status'] = 'detected' if headlights else 'headlights_hidden'
        kept.extend(headlights)
        reports.append(report)

    if return_reports:
        return kept, reports
    return kept


def geometry_headlight_mask(image_size, headlight_pixels,
                            base_radius_px=10.0, ref_distance_m=15.0,
                            min_radius=3, max_radius=16):
    """Build the ALLOWED-region seed from projected headlight pixel positions."""
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for (px, py, depth) in headlight_pixels:
        r = base_radius_px * (ref_distance_m / max(depth, 1e-3))
        r = int(np.clip(r, min_radius, max_radius))
        cv2.circle(mask, (int(round(px)), int(round(py))), r, 255, -1)
    return mask


def geometry_glare_mask(rgb_image, seg_image, headlight_pixels,
                        brightness_thresh=110, require_bright=True,
                        require_vehicle=True, **region_kwargs):
    """Geometry region AND (optionally) vehicle segmentation AND (optionally) bright."""
    rgb_frame = _to_bgr(rgb_image)
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    region = geometry_headlight_mask((w, h), headlight_pixels, **region_kwargs) > 0

    mask_bool = region
    if require_vehicle:
        labels = _to_labels(seg_image)
        mask_bool = mask_bool & (labels == VEHICLE_TAG)
    if require_bright:
        mask_bool = mask_bool & (gray >= brightness_thresh)

    mask = mask_bool.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, rgb_frame


def extracted_glare_mask(rgb_image, seg_image, brightness_thresh=180):
    """Isolate headlight pixels by detecting bright compact blobs on vehicles."""
    rgb_frame = _to_bgr(rgb_image)
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    semantic_labels = _to_labels(seg_image)
    vehicle_mask = (semantic_labels == VEHICLE_TAG).astype(np.uint8)
    vehicle_bool = vehicle_mask.astype(bool)

    kx, ky = 7, 9
    veh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    vehicle_dil = cv2.dilate(vehicle_mask, veh_kernel, anchor=(kx // 2, ky - 1), iterations=1)

    _, bright = cv2.threshold(gray, brightness_thresh, 255, cv2.THRESH_BINARY)
    bright_on_vehicle = cv2.bitwise_and(bright, bright, mask=vehicle_dil)

    headlight_mask = np.zeros_like(gray, dtype=np.uint8)
    max_blob_area = (h * w) * 0.015

    num, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(bright_on_vehicle, 8)
    for i in range(1, num):
        barea = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if barea < 2 or barea > max_blob_area:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 2.8:
            continue
        fill = barea / float(max(1, bw * bh))
        if fill > 0.9 and aspect > 1.6:
            continue
        blob = (comp_labels == i)
        overlap = np.count_nonzero(blob & vehicle_bool) / float(barea)
        if overlap < 0.5:
            continue
        headlight_mask[blob] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    headlight_mask = cv2.dilate(headlight_mask, kernel, iterations=1)

    return headlight_mask, rgb_frame


def _headlight_centers(headlight_mask, min_area=4):
    """Return (cx, cy, area) for each headlight blob in the mask."""
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(headlight_mask, 8)
    centers = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids[i]
            centers.append((int(cx), int(cy), int(area)))
    return centers


# ---------------------------------------------------------------------------
# Diffraction starburst: |FFT|^2 of the aperture shape, sampled across
# wavelengths. See the module docstring for the physical basis.
# ---------------------------------------------------------------------------

STAR_SPIKES = 6          # aperture blades -> number of primary star points
_FFT_STAR_CACHE = {}

# Sampled wavelengths (nm) and their approximate linear-sRGB (B, G, R) weights.
# Short wavelengths (blue) diffract to a SMALLER radius, long (red) to a LARGER
# radius — this radial-by-wavelength scaling is what produces the color fringe.

_WAVELENGTHS = [
    (440.0, (1.00, 0.10, 0.20)),   # blue
    (490.0, (0.70, 0.70, 0.05)),   # cyan
    (540.0, (0.10, 1.00, 0.10)),   # green
    (590.0, (0.05, 0.70, 0.80)),   # amber
    (640.0, (0.10, 0.15, 1.00)),   # red
]
_REF_WL = 540.0          # green as the reference wavelength for scaling


def _aperture_polygon(size, blades=STAR_SPIKES, supersample=2):
    """Filled regular-polygon aperture mask (float 0..1), centered.

    Rendered at `supersample`x resolution and downscaled for anti-aliasing,
    which suppresses the FFT sampling-grid (checkerboard) artifact.
    """
    ss = size * supersample
    ap = np.zeros((ss, ss), dtype=np.uint8)
    cxy = ss / 2.0
    R = ss * 0.06                                # small aperture -> long spikes
    pts = []
    for k in range(blades):
        ang = 2 * np.pi * k / blades + np.pi / 2.0
        pts.append([int(cxy + R * np.cos(ang)), int(cxy + R * np.sin(ang))])
    cv2.fillConvexPoly(ap, np.array(pts, dtype=np.int32), 255)
    ap = cv2.resize(ap, (size, size), interpolation=cv2.INTER_AREA)
    return ap.astype(np.float32) / 255.0


def _build_fft_star(size, blades=STAR_SPIKES):
    """Diffraction starburst = power spectrum of the aperture (grayscale 0..1)."""
    key = (size, blades)
    if key in _FFT_STAR_CACHE:
        return _FFT_STAR_CACHE[key]

    aperture = _aperture_polygon(size, blades)
    F = np.fft.fftshift(np.fft.fft2(aperture))
    power = np.abs(F) ** 2                       # far-field intensity pattern
    # Log-compress the enormous dynamic range so the faint rays stay visible
    star = np.log1p(power)
    star /= (star.max() + 1e-9)
    # Light blur to soften the discrete-FFT sampling grid
    star = cv2.GaussianBlur(star, (0, 0), sigmaX=size * 0.0015 + 0.6)

    # The raw FFT carries an off-ray checkerboard haze from the discrete grid.

    cxy = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = xx - cxy
    dy = yy - cxy
    r_norm = np.sqrt(dx * dx + dy * dy) / (size / 2.0) + 1e-6
    theta = np.arctan2(dy, dx)
    radial = np.exp(-r_norm * 3.0) + 0.12 * np.exp(-r_norm * 1.0)
    ang_spikes = np.clip(np.cos(theta * blades), 0, 1) ** 8
    envelope = radial * (0.30 + 0.70 * ang_spikes)   # smooth, no grid

    # Away from the rays the envelope is ~0, so the grid cannot show through.
    star = star * envelope
    star = star + 0.45 * envelope                # crisp bright rays

    # Fade the inner region so the aperture's hexagonal signature lives in the rays
    core_suppress = np.clip((r_norm - 0.05) / 0.15, 0.0, 1.0) ** 0.8
    star = star * (0.20 + 0.80 * core_suppress)

    star /= (star.max() + 1e-9)
    star = np.clip(star, 0.0, 1.0) ** 1.4
    _FFT_STAR_CACHE[key] = star
    return star


# The expensive FFT + 5-wavelength warp is done ONCE at this canonical size,
# cached, and just resized per-headlight at runtime (resize is ~free vs. FFT).
_CANON_STAR_SIZE = 513


def _build_spectral_star(size=_CANON_STAR_SIZE, blades=STAR_SPIKES):
    """BGR starburst: the FFT star summed across wavelengths, cached.

    Call with the default size and resize the result. A per-frame varying size
    defeats the cache and re-runs the FFT on every call.
    """
    key = ('spectral', size, blades)
    if key in _FFT_STAR_CACHE:
        return _FFT_STAR_CACHE[key]

    base = _build_fft_star(size, blades)         # grayscale diffraction pattern
    cxy = (size - 1) / 2.0
    acc = np.zeros((size, size, 3), dtype=np.float32)

    for wl, (wb, wg, wr) in _WAVELENGTHS:
        scale = wl / _REF_WL                     # >1 red spreads, <1 blue tightens
        M = cv2.getRotationMatrix2D((cxy, cxy), 0.0, scale)
        warped = cv2.warpAffine(base, M, (size, size),
                                flags=cv2.INTER_LINEAR, borderValue=0.0)
        acc[:, :, 0] += warped * wb              # B
        acc[:, :, 1] += warped * wg              # G
        acc[:, :, 2] += warped * wr              # R

    acc /= (acc.max() + 1e-9)

    # Radial edge fade so the square sprite has no hard rectangular seam
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r_norm = np.sqrt((xx - cxy) ** 2 + (yy - cxy) ** 2) / (size / 2.0)
    edge_fade = np.clip(1.0 - r_norm, 0.0, 1.0) ** 1.5
    acc *= edge_fade[:, :, None]

    _FFT_STAR_CACHE[key] = acc
    return acc


def depth_intensity(depth_m, ref_m=12.0, min_scale=0.35, max_scale=1.8):
    """Range-based brightness factor for a headlight at distance `depth_m`.

    Softened inverse-distance rather than inverse-square, which is too
    aggressive at these ranges, clamped so far lights stay visible and close
    ones don't blow out the frame.
    """
    if depth_m is None or depth_m <= 1e-3:
        return max_scale
    scale = (ref_m / depth_m) ** 0.6
    return float(np.clip(scale, min_scale, max_scale))


def depth_sprite_radius(depth_m, ref_m=12.0, base_r=150.0, min_r=34, max_r=300):
    """Star sprite radius from distance: closer light -> bigger star.

    min_r is kept small so a distant car, only a few pixels wide, does not wear
    a star several times its own width.
    """
    if depth_m is None or depth_m <= 1e-3:
        return max_r
    # 0.75 rather than 0.5: the gentler falloff put a ~124 px star on a car at
    # 70 m that is only ~28 px wide at this FOV.
    r = base_r * (ref_m / depth_m) ** 0.75
    return int(np.clip(r, min_r, max_r))


# Pre-built starburst variants: the canonical FFT star at different rotations
# with randomized per-spike brightness.

_N_STAR_VARIANTS = 6
_STAR_VARIANT_CACHE = {}


def _build_star_variants(size, spikes=STAR_SPIKES):
    """Return a list of varied starburst sprites (rotation + per-spike jitter)."""
    key = ('variants', size, spikes)
    if key in _STAR_VARIANT_CACHE:
        return _STAR_VARIANT_CACHE[key]

    canon = _build_spectral_star(size, spikes)          # BGR float, base star
    cxy = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    theta = np.arctan2(yy - cxy, xx - cxy)

    rng = np.random.default_rng(1234)                   # deterministic set
    variants = []
    for vi in range(_N_STAR_VARIANTS):
        # 1) random rotation so the star axis differs per headlight
        rot = float(rng.uniform(0, 360.0 / spikes))
        M = cv2.getRotationMatrix2D((cxy, cxy), rot, 1.0)
        s = cv2.warpAffine(canon, M, (size, size), flags=cv2.INTER_LINEAR)

        # 2) per-spike brightness jitter: multiply by a smooth angular mask that
        #    randomly strengthens/weakens each of the `spikes` rays -> uneven,
        #    natural-looking streaks instead of a perfectly symmetric star.
        gains = rng.uniform(0.6, 1.35, size=spikes).astype(np.float32)
        ang_gain = np.zeros_like(theta)
        for k in range(spikes):
            centre = 2 * np.pi * k / spikes
            d = np.cos(theta - centre)
            ang_gain += gains[k] * np.clip(d, 0, 1) ** 6
        ang_gain = 0.55 + 0.45 * (ang_gain / (ang_gain.max() + 1e-9))
        s = s * ang_gain[:, :, None]

        variants.append(np.clip(s, 0.0, 1.0).astype(np.float32))

    _STAR_VARIANT_CACHE[key] = variants
    return variants


def _draw_star_flares(shape, flare_points, spikes=STAR_SPIKES):
    """Stamp an FFT-aperture diffraction starburst (with chromatic dispersion)
    at each headlight. `flare_points` is a list of (cx, cy, depth_m); the sprite
    size and per-flare intensity both scale with distance. Returns float32 BGR.

    Each headlight is assigned one of several pre-built VARIANTS (different
    rotation + uneven spike brightness) so the streaks vary per light and don't
    look like an identical repeated stamp. Variants are built once and cached —
    per-frame cost is just a resize.
    """
    h, w = shape[:2]
    flare = np.zeros((h, w, 3), dtype=np.float32)
    if not flare_points:
        return flare

    variants = _build_star_variants(_CANON_STAR_SIZE, spikes)

    for (cx, cy, depth) in flare_points:
        sprite_r = depth_sprite_radius(depth)
        size = sprite_r * 2 + 1
        inten = depth_intensity(depth)
        # Deterministic variant pick. Quantize position to a coarse grid so a
        # moving car keeps the same star for many frames while neighboring lights 
        # still differ from each other.
        qx, qy = int(cx) // 80, int(cy) // 80
        vi = (qx * 73856093 ^ qy * 19349663) % len(variants)
        canon = variants[vi]
        sprite_bgr = cv2.resize(canon, (size, size),
                                interpolation=cv2.INTER_LINEAR) * 255.0 * inten

        x0, y0 = int(cx) - sprite_r, int(cy) - sprite_r
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(w, x0 + size), min(h, y0 + size)
        if fx0 >= fx1 or fy0 >= fy1:
            continue
        sx0, sy0 = fx0 - x0, fy0 - y0
        flare[fy0:fy1, fx0:fx1] += sprite_bgr[sy0:sy0 + (fy1 - fy0),
                                               sx0:sx0 + (fx1 - fx0)]

    return flare


def _radial_halo(shape, flare_points):
    """Soft bright halo around each headlight, sized+brightened by distance.

    `flare_points` is a list of (cx, cy, depth_m).
    """
    h, w = shape[:2]
    halo = np.zeros((h, w, 3), dtype=np.float32)
    if not flare_points:
        return halo
    for (cx, cy, depth) in flare_points:
        # Closer -> bigger, brighter halo (but kept contained)
        radius = int(np.clip(24.0 * (12.0 / max(depth, 1e-3)) ** 0.5, 14, 40))
        val = 255.0 * depth_intensity(depth)
        cv2.circle(halo, (int(cx), int(cy)), radius, (val, val, val), -1)
    # Blur at 1/4 scale: visually identical to a full-res wide blur for a soft
    # halo, and far cheaper at 1080p.
    sigma_full = max(w / 200.0, 6)
    f = 4
    small = cv2.resize(halo, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_LINEAR)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, sigma_full / f))
    halo = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return halo


def _veiling_glare(shape, flare_points):
    """Broad, soft luminance haze around each headlight (float32 BGR).

    Stray light scattered across the lens lifts the black level over a wide
    region, washing out nearby detail
    """
    h, w = shape[:2]
    veil = np.zeros((h, w, 3), dtype=np.float32)
    if not flare_points:
        return veil

    for (cx, cy, depth) in flare_points:
        # Seed a bright dot; the wide blur below turns it into a broad wash.
        # Proximity scales how strong the veil is.
        amp = 255.0 * depth_intensity(depth, ref_m=12.0, min_scale=0.25, max_scale=1.5)
        cv2.circle(veil, (int(cx), int(cy)), max(6, int(w * 0.006)),
                   (amp, amp, amp), -1)

    # Two-scale wide blur at 1/8 size: ~64x cheaper than full resolution and
    # visually identical for a broad soft haze.
    def _cheap_wide_blur(img, sigma_full):
        sh, sw = img.shape[:2]
        f = 8
        small = cv2.resize(img, (max(1, sw // f), max(1, sh // f)),
                           interpolation=cv2.INTER_LINEAR)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, sigma_full / f))
        return cv2.resize(small, (sw, sh), interpolation=cv2.INTER_LINEAR)

    medium = _cheap_wide_blur(veil, w * 0.05)
    broad = _cheap_wide_blur(veil, w * 0.14)
    veil = medium * 0.6 + broad * 0.4
    return veil


def glare(headlight_mask, rgb_frame, intensity=3.5, headlight_pixels=None,
          veil_strength=0.22):
    """Composite headlight glare onto a frame: core, bloom, starburst, veil.
    """
    h, w = rgb_frame.shape[:2]

    # 1. Saturated white core
    white_seed = np.zeros_like(rgb_frame, dtype=np.float32)
    white_seed[headlight_mask > 0] = (255.0, 255.0, 255.0)

    # 2. Multi-scale bloom, tight layers weighted heavier. The broadest passes
    #    are omitted because the veiling layer already covers that falloff.
    bloom = np.zeros_like(white_seed)
    layers = [
        (2, 31, 1.6),
        (4, 31, 1.2),
        (8, 31, 0.8),
        (16, 31, 0.45),
    ]
    for factor, ksize, weight in layers:
        small = cv2.resize(white_seed, (max(1, w // factor), max(1, h // factor)),
                           interpolation=cv2.INTER_LINEAR)
        small = cv2.GaussianBlur(small, (ksize, ksize), 0)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        bloom += up * weight
    total_weight = sum(l[2] for l in layers)
    bloom = (bloom / total_weight) * intensity

    # 3. Starburst + halo. Prefer depth-carrying headlight_pixels; fall back to
    #    depthless mask centers (assume the reference distance) if not provided.

    # Only flare a headlight the mask confirms is visible:

    if headlight_pixels:
        mask_bool = headlight_mask > 0
        mh, mw = mask_bool.shape[:2]
        flare_points = []
        for (px, py, d) in headlight_pixels:
            ix, iy = int(round(px)), int(round(py))
            win = int(np.clip(10.0 * (15.0 / max(d, 1e-3)), 4, 16)) + 6
            x0, y0 = max(0, ix - win), max(0, iy - win)
            x1, y1 = min(mw, ix + win + 1), min(mh, iy + win + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            if np.any(mask_bool[y0:y1, x0:x1]):
                flare_points.append((px, py, d))
    else:
        flare_points = [(cx, cy, 12.0) for (cx, cy, _a) in _headlight_centers(headlight_mask)]

    halo = _radial_halo(rgb_frame.shape, flare_points)
    flare = _draw_star_flares(rgb_frame.shape, flare_points)
    veil = _veiling_glare(rgb_frame.shape, flare_points)

    # 4. Composite. Work in normalized 0 -> 1 

    inv255 = np.float32(1.0 / 255.0)
    b = rgb_frame.astype(np.float32) * inv255
    b += (bloom + halo * (intensity * 0.35) + flare * (intensity * 0.45)) * inv255
    np.clip(b, 0.0, 1.0, out=b)

    # Veiling glare: screen-blend to lift the black level (contrast loss). This
    # is the occlusion effect. 1-(1-b)(1-v) == b + v - b*v.
    v = veil * (inv255 * veil_strength)
    np.clip(v, 0.0, 1.0, out=v)
    b += v - b * v          # screen blend, in-place
    np.clip(b, 0.0, 1.0, out=b)

    # Screen-blend the blown-out white cores on top.

    core1 = white_seed[:, :, 0] * inv255
    core1 = cv2.GaussianBlur(core1, (0, 0), sigmaX=4.0)
    core = core1[:, :, None]
    b += core - b * core    # screen blend, in-place (broadcasts over channels)

    np.clip(b, 0.0, 1.0, out=b)
    return (b * 255.0).astype(np.uint8)
