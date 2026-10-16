"""Live headlight glare for any CARLA agent, one camera frame at a time.

glare_mask_closed_loop.py knows how to draw glare on a frame once it has the
frame's segmentation, depth, camera pose and the positions of every car. This
file supplies those inputs during a live run, so the same glare reaches the
images an agent drives on.

    GlareInjector   spawns a semantic and a depth camera matched to each glared
                    RGB camera, waits for their images from the same frame,
                    reads vehicles from the world and runs the glare.
                    It knows nothing about any agent, so a test script and an
                    agent call the same apply() and run the same code.

    GlareMixin      adds glare to any agent whose tick() returns the camera
                    images in a dict (Bench2Drive agents use tick_data['imgs']).
                    Put it first in the class list:

                        class GlareQwenAgent(GlareMixin, QwenAgent):
                            pass

                    Inside Bench2Drive it builds its own injector on the first
                    tick. A script that owns the world and ego builds one itself
                    and hands it over with agent.use_glare_injector(injector).

Severity comes from the GLARE_SEVERITY environment variable (default 1.0), so
each run can choose a level without a code change. 0 turns glare off and the
frames reach the agent unmodified. GLARE_STRICT=1 stops the run if a frame
would reach the agent without glare, instead of only logging it.
"""
import json
import logging
import os
import threading
import time

import cv2
import numpy as np

import carla

from glare_mask_closed_loop import (
    build_camera_intrinsics, camera_detected_headlights, focal_from_K,
    geometry_glare_mask, glare, _build_star_variants, _CANON_STAR_SIZE,
)

logger = logging.getLogger(__name__)

# The three front cameras get glare. Oncoming headlights are in front of the
# ego; the rear cameras would only see following traffic.
GLARE_CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')

# Headlight bits of carla.VehicleLightState. A car glares only with one set.
_LOW_BEAM = int(carla.VehicleLightState.LowBeam)
_HIGH_BEAM = int(carla.VehicleLightState.HighBeam)

# How many frames of seg/depth to keep while waiting. Older ones are dropped.
_MAX_BUFFERED_FRAMES = 10


def spec_transform(spec):
    """carla.Transform for a leaderboard-style sensor spec (metres, degrees)."""
    return carla.Transform(
        carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
        carla.Rotation(pitch=spec['pitch'], yaw=spec['yaw'], roll=spec['roll']),
    )


def decode_depth_m(image, downscale=4):
    """CARLA depth image -> metres, downscaled with nearest-neighbour, float16.

    CARLA packs depth into the B, G and R channels as one 24-bit integer
    normalised to a 1000 m far plane. Nearest-neighbour, not averaging: an
    average across a car's edge invents a surface that is neither the car nor
    what is behind it.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (image.height, image.width, 4))
    b = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    r = arr[:, :, 2].astype(np.float32)
    depth_m = (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0) * 1000.0
    small = cv2.resize(depth_m, (image.width // downscale, image.height // downscale),
                       interpolation=cv2.INTER_NEAREST)
    return small.astype(np.float16)


def decode_labels(image):
    """CARLA semantic image -> per-pixel class ids (stored in the red channel)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (image.height, image.width, 4))
    return arr[:, :, 2].copy()


class _GlareCamera:
    """One glared RGB camera: its intrinsics, and the seg/depth pair matched to it."""

    def __init__(self, spec):
        self.id = spec['id']
        self.spec = spec
        self.width = int(spec['width'])
        self.height = int(spec['height'])
        self.fov = float(spec['fov'])
        self.K = build_camera_intrinsics(self.width, self.height, self.fov)
        self.focal_px = focal_from_K(self.K)
        self.sensors = []
        # frame -> labels / depth / (cam matrix, cam location)
        self.seg = {}
        self.depth = {}
        self.pose = {}


class GlareInjector:
    """Adds glare to live camera frames. Agent-independent; see the module docstring.

    world, hero     the CARLA world and the ego vehicle the cameras ride on
    sensor_specs    the agent's sensors() list (camera id, pose, size, FOV)
    severity        glare level; 0 = off, 1 = the tuned look
    strict          if a frame's seg/depth never arrives, raise instead of
                    passing the frame through without glare

    Every apply() call is logged in self.log, with the fields the run record
    needs: frame, camera, severity, glaring headlights, gate verdicts, and the
    time spent waiting for seg/depth (wait_ms) and drawing glare (glare_ms).
    """

    def __init__(self, world, hero, sensor_specs, cameras=GLARE_CAMERAS,
                 severity=1.0, depth_downscale=4, max_distance=70.0,
                 min_detected_px=150, intensity=3.5, veil_strength=0.22,
                 frame_timeout_s=2.0, strict=False):
        self.world = world
        self.hero = hero
        self.severity = float(severity)
        self.depth_downscale = depth_downscale
        self.max_distance = max_distance
        self.min_detected_px = min_detected_px
        self.intensity = intensity
        self.veil_strength = veil_strength
        self.frame_timeout_s = frame_timeout_s
        self.strict = strict

        self.log = []
        self.missing_aux_frames = 0
        self._cond = threading.Condition()
        self._records_frame = None
        self._records = []
        self._static = {}            # actor id -> (bbox extent, bbox centre, 4+ wheels)

        self.cameras = {}
        rgb_specs = {s['id']: s for s in sensor_specs
                     if s.get('type') == 'sensor.camera.rgb'}
        for cam_id in cameras:
            if cam_id not in rgb_specs:
                logger.warning("[GLARE] %s is not in the sensor list; no glare on it", cam_id)
                continue
            cam = _GlareCamera(rgb_specs[cam_id])
            # With glare off there is nothing to compute, so skip the extra cameras.
            if self.severity > 0.0:
                self._spawn_aux(cam)
            self.cameras[cam_id] = cam

        # Build the cached FFT starbursts now, so the first frame of the run is
        # not a few hundred ms slower than the rest.
        if self.severity > 0.0:
            _build_star_variants(_CANON_STAR_SIZE)

        logger.info("[GLARE] injector ready: severity %.2f on %s",
                    self.severity, ", ".join(self.cameras) or "no cameras")

    # --- extra cameras -----------------------------------------------------

    def _spawn_aux(self, cam):
        """Spawn a semantic and a depth camera at exactly this RGB camera's pose.

        Same position, angle, resolution and FOV, so every seg/depth pixel lines
        up with the RGB pixel it describes. Both stay attached to the ego.
        """
        library = self.world.get_blueprint_library()
        tf = spec_transform(cam.spec)
        for kind in ('sensor.camera.semantic_segmentation', 'sensor.camera.depth'):
            bp = library.find(kind)
            bp.set_attribute('image_size_x', str(cam.width))
            bp.set_attribute('image_size_y', str(cam.height))
            bp.set_attribute('fov', str(cam.fov))
            sensor = self.world.spawn_actor(bp, tf, attach_to=self.hero)
            if kind.endswith('semantic_segmentation'):
                sensor.listen(lambda image, c=cam: self._on_seg(c, image))
            else:
                sensor.listen(lambda image, c=cam: self._on_depth(c, image))
            cam.sensors.append(sensor)

    def _on_seg(self, cam, image):
        labels = decode_labels(image)
        tf = image.transform              # the camera's world pose at capture
        pose = (tf.get_matrix(), (tf.location.x, tf.location.y, tf.location.z))
        with self._cond:
            cam.seg[image.frame] = labels
            cam.pose[image.frame] = pose
            self._prune(cam)
            self._cond.notify_all()

    def _on_depth(self, cam, image):
        depth = decode_depth_m(image, self.depth_downscale)
        with self._cond:
            cam.depth[image.frame] = depth
            self._prune(cam)
            self._cond.notify_all()

    @staticmethod
    def _prune(cam):
        for buf in (cam.seg, cam.depth, cam.pose):
            while len(buf) > _MAX_BUFFERED_FRAMES:
                del buf[min(buf)]

    def _wait_for(self, cam, frame):
        """Seg, depth and pose for `frame`, waiting for them if they are late.

        The leaderboard only waits for the agent's own sensors. These extra
        cameras are not registered with it, so their images for the current
        frame can still be in flight when tick() runs.
        """
        deadline = time.monotonic() + self.frame_timeout_s
        with self._cond:
            while True:
                if frame in cam.seg and frame in cam.depth and frame in cam.pose:
                    return cam.seg[frame], cam.depth[frame], cam.pose[frame]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    # --- vehicles ----------------------------------------------------------

    @staticmethod
    def _lights_on(actor):
        try:
            return bool(int(actor.get_light_state()) & (_LOW_BEAM | _HIGH_BEAM))
        except (RuntimeError, TypeError, ValueError):
            return False

    def _vehicle_records(self, frame):
        """Every car that could glare, read once per frame and shared by all cameras.

        Skips the ego, anything under four wheels (bicycles, motorcycles), and
        any car whose headlights are off. In synchronous mode the world does not
        move until the next tick, so these positions match the frame's images.
        """
        if self._records_frame == frame:
            return self._records

        records = []
        for actor in self.world.get_actors().filter('vehicle.*'):
            try:
                if actor.id == self.hero.id or not actor.is_alive:
                    continue
                if actor.id not in self._static:
                    bb = actor.bounding_box
                    wheels = int(actor.attributes.get('number_of_wheels', 4))
                    self._static[actor.id] = (
                        (bb.extent.x, bb.extent.y, bb.extent.z),
                        (bb.location.x, bb.location.y, bb.location.z),
                        wheels >= 4,
                    )
                extent, bb_center, four_wheels = self._static[actor.id]
                if not four_wheels or not self._lights_on(actor):
                    continue
                tf = actor.get_transform()
                fwd = tf.get_forward_vector()
                records.append({
                    'id': actor.id,
                    'matrix': tf.get_matrix(),
                    'extent': extent,
                    'bb_center': bb_center,
                    'forward': (fwd.x, fwd.y, fwd.z),
                    'location': (tf.location.x, tf.location.y, tf.location.z),
                })
            except RuntimeError:
                continue          # actor destroyed mid-read

        self._records_frame = frame
        self._records = records
        return records

    # --- the one call agents and scripts make ------------------------------

    def apply(self, cam_id, bgr, frame):
        """Return `bgr` (H x W x 3, BGR) with glare added for this frame.

        Returns the input unchanged when severity is 0, the camera is not
        glared, or no car is allowed to glare. If the frame's seg/depth never
        arrives, the frame is passed through without glare and counted in
        self.missing_aux_frames (or raises, with strict=True).
        """
        cam = self.cameras.get(cam_id)
        if cam is None or self.severity <= 0.0:
            return bgr

        entry = {'frame': int(frame), 'camera': cam_id, 'severity': self.severity,
                 'glaring_headlights': 0, 'verdicts': {}, 'aux_missing': False}

        t_wait = time.perf_counter()
        aux = self._wait_for(cam, frame)
        entry['wait_ms'] = round((time.perf_counter() - t_wait) * 1000.0, 1)
        t0 = time.perf_counter()
        if aux is None:
            self.missing_aux_frames += 1
            entry['aux_missing'] = True
            self.log.append(entry)
            msg = ("[GLARE] no seg/depth for %s frame %d after %.1f s; "
                   "frame passed through WITHOUT glare" % (cam_id, frame, self.frame_timeout_s))
            if self.strict:
                raise RuntimeError(msg)
            logger.warning(msg)
            return bgr
        seg, depth, (cam_matrix, cam_loc) = aux

        # Bench2Drive hands the agent a [:, :, :3] view of a BGRA buffer. OpenCV
        # cannot read that strided view, so copy it into a contiguous array.
        rgb = np.ascontiguousarray(bgr)

        records = self._vehicle_records(frame)
        headlights, reports = camera_detected_headlights(
            records, cam_matrix, cam.K, (cam.width, cam.height),
            depth_map=depth, seg_labels=seg, cam_location=cam_loc,
            require_facing=True, max_distance=self.max_distance,
            min_detected_px=self.min_detected_px, return_reports=True,
        )
        for rep in reports:
            entry['verdicts'][rep['status']] = entry['verdicts'].get(rep['status'], 0) + 1
        entry['glaring_headlights'] = len(headlights)

        if headlights:
            mask, rgb = geometry_glare_mask(rgb, seg, headlights, require_bright=False,
                                            require_vehicle=True, focal_px=cam.focal_px)
            out = glare(mask, rgb, intensity=self.intensity, headlight_pixels=headlights,
                        veil_strength=self.veil_strength, severity=self.severity,
                        focal_px=cam.focal_px)
        else:
            out = bgr

        entry['glare_ms'] = round((time.perf_counter() - t0) * 1000.0, 1)
        self.log.append(entry)
        return out

    # --- run record and cleanup -------------------------------------------

    def write_log(self, path):
        """One JSON line per apply() call: the glare fields for the run record."""
        with open(path, 'w', encoding='utf-8') as fh:
            for entry in self.log:
                fh.write(json.dumps(entry) + '\n')

    def destroy(self):
        for cam in self.cameras.values():
            for sensor in cam.sensors:
                try:
                    sensor.stop()
                    if sensor.is_alive:
                        sensor.destroy()
                except RuntimeError:
                    pass
            cam.sensors = []
        if self.missing_aux_frames:
            logger.warning("[GLARE] %d frame(s) went to the agent without glare "
                           "because their seg/depth was missing", self.missing_aux_frames)


class GlareMixin:
    """Adds glare to an agent's camera images before the agent uses them.

    Works with any agent whose tick() returns a dict holding the camera images
    under `glare_images_key`, keyed by sensor id, and whose input_data entries
    are (frame, data) pairs, as in Bench2Drive. The glare is applied inside
    tick(), so anything the agent does with the images afterwards (DeepSight
    writes them to disk and reads them back as its input and history) already
    sees the glared version.
    """

    glare_cameras = GLARE_CAMERAS
    glare_images_key = 'imgs'

    def use_glare_injector(self, injector):
        """Use an injector built by the caller (scripts that own the world and ego)."""
        self._glare = injector

    def tick(self, input_data):
        tick_data = super().tick(input_data)
        injector = self._glare_injector()
        if injector is None:
            return tick_data
        images = tick_data.get(self.glare_images_key, {})
        for cam_id in self.glare_cameras:
            if cam_id in images and cam_id in input_data:
                frame = input_data[cam_id][0]
                images[cam_id] = injector.apply(cam_id, images[cam_id], frame)
        return tick_data

    def _glare_injector(self):
        """Create the injector on the first tick.

        The ego does not exist yet when setup() runs, so this cannot happen
        there. Imported here so the module loads outside the leaderboard.
        """
        injector = getattr(self, '_glare', None)
        if injector is None:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            hero = CarlaDataProvider.get_hero_actor()
            if hero is None:
                return None
            severity = float(os.environ.get('GLARE_SEVERITY', '1.0'))
            strict = os.environ.get('GLARE_STRICT', '0').lower() in ('1', 'true', 'yes')
            injector = GlareInjector(CarlaDataProvider.get_world(), hero, self.sensors(),
                                     cameras=self.glare_cameras, severity=severity,
                                     strict=strict)
            self._glare = injector
        return injector

    def destroy(self):
        injector = getattr(self, '_glare', None)
        if injector is not None:
            # The agent's own output folder if it has one (DeepSight sets
            # self.save_path per route), else the leaderboard's SAVE_PATH.
            save_path = getattr(self, 'save_path', None) or os.environ.get('SAVE_PATH')
            if save_path:
                injector.write_log(os.path.join(save_path, 'glare_log.jsonl'))
            injector.destroy()
            self._glare = None
        super().destroy()
