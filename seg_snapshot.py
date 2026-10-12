"""
Quick semantic-segmentation snapshot for the presentation.

Spawns the ego vehicle + a couple of NPCs on the same Town07 night scene,
attaches a semantic segmentation camera (same pose as the first-person cam),
ticks a few frames, and saves:

  seg_snapshot_color.png   -> CityScapes-palette colorized view (nice for slides)
  seg_snapshot_labels.png  -> raw label channel (what the glare code reads)
  seg_snapshot_rgb.png     -> matching RGB frame for side-by-side

Run the CARLA server first, then: python3 seg_snapshot.py
"""
import carla
import logging
import time
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

client = carla.Client('localhost', 2000)
client.set_timeout(60.0)
client.load_world('Town07')
world = client.get_world()
time.sleep(2.0)

# Night, no fog (matches the main scenario)
weather = carla.WeatherParameters(
    cloudiness=80.0, precipitation=0.0, sun_altitude_angle=-90.0,
    fog_density=0.0, wetness=0.0,
)
world.set_weather(weather)

# Synchronous so we can grab a deterministic frame
settings = world.get_settings()
original = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

ego_bp = bp_lib.find('vehicle.tesla.model3')
ego = world.try_spawn_actor(ego_bp, spawn_points[19])
actors = [ego]

# A couple of vehicles ahead with headlights on, so the seg clearly shows cars
carla_map = world.get_map()
ego_wp = carla_map.get_waypoint(spawn_points[19].location)
hl = carla.VehicleLightState(
    carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam |
    carla.VehicleLightState.HighBeam
)
for dist in (18, 35):
    wps = ego_wp.next(dist)
    if wps:
        t = wps[0].transform
        t.location.z += 0.5
        npc = world.try_spawn_actor(bp_lib.find('vehicle.lincoln.mkz_2017'), t)
        if npc:
            npc.set_light_state(hl)
            actors.append(npc)

# --- Cameras: RGB + semantic segmentation at the same pose ---
cam_tf = carla.Transform(carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=0.0))
W, H = 1920, 1080

rgb_bp = bp_lib.find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', str(W))
rgb_bp.set_attribute('image_size_y', str(H))
rgb_bp.set_attribute('fov', '100')
rgb_cam = world.spawn_actor(rgb_bp, cam_tf, attach_to=ego)
actors.append(rgb_cam)

seg_bp = bp_lib.find('sensor.camera.semantic_segmentation')
seg_bp.set_attribute('image_size_x', str(W))
seg_bp.set_attribute('image_size_y', str(H))
seg_bp.set_attribute('fov', '100')
seg_cam = world.spawn_actor(seg_bp, cam_tf, attach_to=ego)
actors.append(seg_cam)

latest = {}
rgb_cam.listen(lambda img: latest.__setitem__('rgb', img))
seg_cam.listen(lambda img: latest.__setitem__('seg', img))

# Let the scene settle and grab a frame
for _ in range(20):
    world.tick()
    time.sleep(0.02)

if 'seg' in latest and 'rgb' in latest:
    seg_img = latest['seg']
    rgb_img = latest['rgb']

    # 1. Raw label channel FIRST (this is what glare_mask.py reads).
    #    CARLA encodes the class id in the red channel of the BGRA buffer.
    seg_raw = np.reshape(np.frombuffer(seg_img.raw_data, dtype=np.uint8), (H, W, 4))
    labels = seg_raw[:, :, 2].copy()
    # Highlight the vehicle class (tag 14) in white for an easy-to-read slide
    vehicle_highlight = np.where(labels == 14, 255, labels * 3).astype(np.uint8)
    cv2.imwrite('seg_snapshot_labels.png', vehicle_highlight)

    # 2. Colorized CityScapes palette (nice visual for slides).
    #    convert() mutates raw_data in place, so do this AFTER reading labels.
    seg_img.convert(carla.ColorConverter.CityScapesPalette)
    seg_color = np.reshape(np.frombuffer(seg_img.raw_data, dtype=np.uint8), (H, W, 4))[:, :, :3]
    cv2.imwrite('seg_snapshot_color.png', seg_color)

    # 3. Matching RGB for side-by-side
    rgb = np.reshape(np.frombuffer(rgb_img.raw_data, dtype=np.uint8), (H, W, 4))[:, :, :3]
    cv2.imwrite('seg_snapshot_rgb.png', rgb)

    logger.info("Saved seg_snapshot_color.png, seg_snapshot_labels.png, seg_snapshot_rgb.png")
else:
    logger.warning("Did not receive both frames")

# Cleanup
for a in actors:
    try:
        if a.is_alive:
            a.destroy()
    except RuntimeError:
        pass
world.apply_settings(original)
logger.info("Done")
