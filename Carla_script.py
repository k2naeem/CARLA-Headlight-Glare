import carla
import logging
import math
import random
import time
import os
import numpy as np
import cv2
from glare_mask import (
    glare, VEHICLE_TAG, PEDESTRIAN_TAG, RIDER_TAG, BICYCLE_TAG,
    build_camera_intrinsics, geometry_glare_mask,
    camera_detected_headlights, project_points_to_image,
    headlight_points_local,
)


logging.basicConfig(
    level=logging.INFO,

    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Diagnostic outputs. False (default)

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


# Run timing (single source of truth) 
FIXED_DELTA = 0.05                  # seconds per tick 
DURATION_S = 45.0                   # length of the run
FPS = 1.0 / FIXED_DELTA             # video framerate, matches the tick rate



logger.info("Connecting to CARLA at localhost:2000")
client = carla.Client('localhost', 2000)
client.set_timeout(60.0)

# Towns shipped with a standard CARLA install. TOWN_INDEX selects one.
town_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05',
             'Town06', 'Town07', 'Town08', 'Town09', 'Town10',
             'Town11', 'Town12']
TOWN_INDEX = 6

logger.info("Loading world: %s", town_list[TOWN_INDEX])
client.load_world(town_list[TOWN_INDEX])

world = client.get_world()
logger.info("World loaded")


time.sleep(2.0)

env_objects = world.get_environment_objects(carla.CityObjectLabel.Poles)
env_object_ids = [obj.id for obj in env_objects]
if env_object_ids:
    world.enable_environment_objects(set(env_object_ids), False)
    logger.info("Disabled %d pole/street light objects", len(env_object_ids))

# Turn off any traffic lights (they also emit light)
for tl in world.get_actors().filter('traffic.light*'):
    tl.set_state(carla.TrafficLightState.Off)
    tl.freeze(True)
logger.info("All traffic/street lights disabled")


# --- Synchronous mode: fixed timestep, deterministic sensor delivery ---

settings = world.get_settings()
original_settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = FIXED_DELTA
world.apply_settings(settings)

# Configure traffic manager for synchronous mode.
# IMPORTANT: do NOT fall back to another port if 8000 is bound. 

TM_PORT = 8000
try:
    traffic_manager = client.get_trafficmanager(TM_PORT)
except RuntimeError as e:
    raise RuntimeError(
        "Traffic manager port %d is already bound (%s).\n"
        "A previous run is probably still alive and controlling this world.\n"
        "Fix it before rerunning:\n"
        "  1) pkill -f %s\n"
        "  2) lsof -i :%d   (should print nothing)\n"
        "  3) restart the CARLA server if the port stays bound\n"
        "Do NOT just switch ports - two clients on one world will crash the server."
        % (TM_PORT, e, os.path.basename(__file__), TM_PORT)
    )

logger.info("Traffic manager connected on port %d", TM_PORT)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_global_distance_to_leading_vehicle(2.5)
logger.info("Synchronous mode enabled (fixed 20 FPS)")

weather = carla.WeatherParameters(
    cloudiness=80.0,             # Overcast, dark sky (no moonlight bleed)
    precipitation=0.0,           # No rain
    precipitation_deposits=0.0,  # Dry roads
    sun_altitude_angle=-90.0,    # Sun fully below horizon - pitch dark night
    wetness=0.0,                 # Dry
    fog_density=0.0,             # No fog - clear dark night
    fog_distance=0.0,
    fog_falloff=0.0,
    wind_intensity=0.0,
    scattering_intensity=0.0,
)

world.set_weather(weather)
logger.info("Weather applied - clear, pitch-dark night (no fog, no rain)")



# --- NPC traffic ------------------------------------------------------------

# Only 4-wheeled vehicles: bikes and motorcycles have different headlight
# geometry, and the projection offsets below assume a car.
all_vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
vehicle_blueprints = [bp for bp in all_vehicle_blueprints
                      if int(bp.get_attribute('number_of_wheels').as_int()) >= 4]
logger.info("Found %d 4-wheeled vehicle blueprints (excluded bikes)", len(vehicle_blueprints))

# Run spawn_points.py to visualise these indices in the simulator.
spawn_points = world.get_map().get_spawn_points()
logger.info("Found %d spawn points", len(spawn_points))

carla_map = world.get_map()

# --- Ego route: drive from spawn point 88 toward spawn point 82 ---
EGO_SPAWN_IDX = 88
DEST_SPAWN_IDX = 82

ego_spawn_point = spawn_points[EGO_SPAWN_IDX]
destination_loc = spawn_points[DEST_SPAWN_IDX].location
logger.info("Ego route: spawn %d (%.1f, %.1f) -> spawn %d (%.1f, %.1f)",
            EGO_SPAWN_IDX, ego_spawn_point.location.x, ego_spawn_point.location.y,
            DEST_SPAWN_IDX, destination_loc.x, destination_loc.y)


LIGHTS_OFF = carla.VehicleLightState(carla.VehicleLightState.NONE)

# Ego is spawned first so every NPC can be placed relative to it.
ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)

if ego_vehicle is None:
    # Try another point if first fails
    for sp in spawn_points:
        ego_vehicle = world.try_spawn_actor(ego_bp, sp)
        if ego_vehicle is not None:
            ego_spawn_point = sp
            break

if ego_vehicle is None:
    raise RuntimeError("Failed to spawn ego vehicle.")

logger.info("Spawned ego vehicle: actor_id=%s", ego_vehicle.id)

ego_waypoint = carla_map.get_waypoint(ego_spawn_point.location)

npc_list = []

# Single NPC model so the headlight geometry offsets are exact for every instance
NPC_MODEL = 'vehicle.lincoln.mkz_2017'
npc_vehicle_list = [world.get_blueprint_library().find(NPC_MODEL)]

HEADLIGHTS_ON = carla.VehicleLightState(
    carla.VehicleLightState.Position |
    carla.VehicleLightState.LowBeam |
    carla.VehicleLightState.HighBeam
)

# The ego drives on low beams, as any real car would at night.

EGO_LIGHTS = carla.VehicleLightState(
    carla.VehicleLightState.Position |
    carla.VehicleLightState.LowBeam
)

opposite_npc_ids = set()   # NPCs whose headlights must stay ON (oncoming)
opposite_npcs = []         # actor refs so we can reapply headlights each second


# --- Build the ego's route: walk waypoints from 88 toward 82 -----------------
ROUTE_STEP = 2.0           # metres between route waypoints


def build_route(start_wp, dest_location, step=ROUTE_STEP, max_len=1500.0):
    """Greedy route: walk waypoints forward; at branches take the successor
    closest to the destination. Returns a list of waypoints (index * step = dist).
    """
    route = [start_wp]
    wp = start_wp
    travelled = 0.0
    while travelled < max_len:
        nxts = wp.next(step)
        if not nxts:
            break
        wp = (nxts[0] if len(nxts) == 1
              else min(nxts, key=lambda w: w.transform.location.distance(dest_location)))
        travelled += step
        route.append(wp)
        if wp.transform.location.distance(dest_location) < 6.0:
            break
    return route


def oncoming_lane_wp(wp):
    """Waypoint in the ONCOMING lane (opposite lane_id sign), or None."""
    for cand in (wp.get_left_lane(), wp.get_right_lane()):
        if cand is None or cand.lane_type != carla.LaneType.Driving:
            continue
        if cand.lane_id * wp.lane_id < 0:
            return cand
    return None


ego_route = build_route(ego_waypoint, destination_loc)
route_len_m = (len(ego_route) - 1) * ROUTE_STEP
logger.info("Built ego route: %d waypoints, ~%.0f m toward spawn %d",
            len(ego_route), route_len_m, DEST_SPAWN_IDX)


def route_wp_at(dist_m):
    """Route waypoint at `dist_m` along the route (None if beyond the route)."""
    idx = int(round(dist_m / ROUTE_STEP))
    if 0 <= idx < len(ego_route):
        return ego_route[idx]
    return None


# --- Oncoming traffic in BATCHES along the route ----------------------------
# Each batch is a tight stack of vehicles so the lead car's glare occludes the
# ones behind it.
BATCH_SIZE = 4             # vehicles per batch (3-4)
IN_BATCH_SPACING = 7.0     # metres between cars WITHIN a batch (tight stack)
BATCH_INTERVAL = 70.0      # metres between batches along the route
FIRST_BATCH_AT = 45.0      # metres ahead of ego for the first batch

batch_anchor = FIRST_BATCH_AT
batch_num = 0
while batch_anchor < route_len_m - 20.0:
    anchor_wp = route_wp_at(batch_anchor)
    if anchor_wp is None:
        break
    onc_wp = oncoming_lane_wp(anchor_wp)
    if onc_wp is None:
        batch_anchor += BATCH_INTERVAL      # no oncoming lane here, skip ahead
        continue

    batch_num += 1
    spawned_in_batch = 0
    for k in range(BATCH_SIZE):
        # Step along the ONCOMING lane's own direction to stack the batch
        offs = k * IN_BATCH_SPACING
        if offs == 0:
            wp_k = onc_wp
        else:
            nxt = onc_wp.next(offs)
            if not nxt:
                continue
            wp_k = nxt[0]

        tf = wp_k.transform
        tf.location.z += 0.5
        npc = world.try_spawn_actor(random.choice(npc_vehicle_list), tf)
        if npc is not None:
            npc.set_light_state(HEADLIGHTS_ON)   # oncoming -> headlights on
            npc_list.append(npc)
            opposite_npc_ids.add(npc.id)
            opposite_npcs.append(npc)
            spawned_in_batch += 1

    logger.info("Oncoming batch %d at %.0fm: spawned %d/%d vehicles",
                batch_num, batch_anchor, spawned_in_batch, BATCH_SIZE)
    batch_anchor += BATCH_INTERVAL

logger.info("Spawned %d oncoming batches (%d vehicles with headlights)",
            batch_num, len(opposite_npcs))


# --- Same-lane vehicles AHEAD of the ego (the leading stack) -----------------
# These sit well beyond the obstacle's spawn range

LEAD_STACK_DISTANCES = (60.0, 74.0, 88.0)
lead_npcs = []
for dist in LEAD_STACK_DISTANCES:
    wp = route_wp_at(dist)
    if wp is None:
        continue
    tf = wp.transform
    tf.location.z += 0.5
    npc = world.try_spawn_actor(random.choice(npc_vehicle_list), tf)
    if npc is not None:
        npc.set_light_state(LIGHTS_OFF)
        npc_list.append(npc)
        lead_npcs.append(npc)
        logger.info("Spawned NPC %.0fm ahead (same lane): actor_id=%s", dist, npc.id)

# --- Spawn NPC vehicles BEHIND the ego (same lane) ---
behind_distances = [15, 30, 50]  # Meters behind ego - close together

for dist in behind_distances:
    # .previous() goes backward along the road
    waypoints_behind = ego_waypoint.previous(dist)
    if not waypoints_behind:
        continue
    wp = waypoints_behind[0]
    spawn_transform = wp.transform
    spawn_transform.location.z += 0.5

    npc = world.try_spawn_actor(random.choice(npc_vehicle_list), spawn_transform)
    if npc is not None:
        npc.set_light_state(LIGHTS_OFF)
        npc_list.append(npc)
        logger.info("Spawned NPC %dm behind (same lane): actor_id=%s", dist, npc.id)
    else:
        logger.warning("Failed to spawn NPC %dm behind", dist)

logger.info("Spawned %d total NPC vehicles", len(npc_list))
if npc_list:
    _bb = npc_list[0].bounding_box
    logger.info("[GEOM] NPC bbox extent=(%.2f, %.2f, %.2f)  bb_center=(%.2f, %.2f, %.2f)",
                _bb.extent.x, _bb.extent.y, _bb.extent.z,
                _bb.location.x, _bb.location.y, _bb.location.z)


# --- Occlusion scenario: a road user the camera loses in the glare -------------
# A vulnerable road user that the ego CAMERA cannot resolve because it sits
# inside the oncoming headlight glare, while the LIDAR returns it plainly.

# The governing geometry: at ~30 m the projection scale is ~800 px/rad, so the
# 3.5 m between lane centres spans only ~95 px, while the starburst alone has a
# ~70 px radius and the veiling glare spans several hundred. Anything near the
# lane line at roughly the oncoming car's range therefore falls inside the glare.
OBSTACLE_KIND = 'cyclist'      # 'cyclist' or 'pedestrian'

OBSTACLE_SPAWN_AT_S = 2.0      # appears at THIS moment, seconds into the run
OBSTACLE_SPAWN_RETRY_S = 3.0   # retried only if that spot is physically blocked

# Default is a fixed distance ahead, the result is clamped to the MIN/MAX range.
OBSTACLE_AHEAD_M = 20.0
OBSTACLE_ANCHOR_WINDOW_M = 24.0  # snap behind the front oncoming car if within this
OBSTACLE_BEHIND_NPC_M = 3.0      # how far BEHIND that car (further from the ego)
OBSTACLE_MIN_AHEAD_M = 10.0      # the ego needs room to react
OBSTACLE_MAX_AHEAD_M = 26.0

# 'near' starts the object ON the lane line: it sits in the ~0.69 m of clear road
# between the oncoming cars' flank and the line

OBSTACLE_LATERAL_EDGE = 'near'

# The cyclist moves ~1.75 m sideways into the ego's lane, so it arrives at about
# OBSTACLE_SPAWN_AT_S + 1.75 / OBSTACLE_SPEED seconds. Slower = longer in the glare.
OBSTACLE_SPEED = 2.0           # m/s
OBSTACLE_HOLD_IN_LANE = True   # stop in the ego's lane so the ego reaches it

# LIDAR detection test for the cyclist

OBSTACLE_LIDAR_RADIUS_M = 1.5
LIDAR_GROUND_CLEARANCE_M = 0.3      # returns this far above local ground = an object
EGO_HALF_LENGTH_M = 2.6             # self-return exclusion box around the sensor
EGO_HALF_WIDTH_M = 1.2

CYCLIST_MODEL = 'vehicle.diamondback.century'
CYCLIST_COLOR = '255,255,255'   # bike frame colour, "R,G,B"; light so it reads at night
# Rider appearance. CARLA 2-wheelers expose a `driver_id` attribute on most
# builds; its valid values are logged at startup. None keeps the default rider.
CYCLIST_DRIVER_ID = None

if OBSTACLE_KIND == 'cyclist':
    obstacle_bp = world.get_blueprint_library().find(CYCLIST_MODEL)
    # A cyclist is labelled 19 (bicycle) + 13 (rider) and never 14 (car), so the
    # glare detection gate can never mistake it for a headlight source.
    OBSTACLE_TAGS = (BICYCLE_TAG, RIDER_TAG)
    if obstacle_bp is not None:
        if obstacle_bp.has_attribute('color'):
            obstacle_bp.set_attribute('color', CYCLIST_COLOR)
            logger.info("Cyclist frame colour set to %s", CYCLIST_COLOR)
        else:
            logger.info("%s has no 'color' attribute; frame keeps its default",
                        CYCLIST_MODEL)
        if obstacle_bp.has_attribute('driver_id'):
            logger.info("Cyclist rider options (driver_id): %s",
                        obstacle_bp.get_attribute('driver_id').recommended_values)
            if CYCLIST_DRIVER_ID is not None:
                obstacle_bp.set_attribute('driver_id', str(CYCLIST_DRIVER_ID))
                logger.info("Cyclist rider set to driver_id=%s", CYCLIST_DRIVER_ID)
        else:
            logger.info("%s has no 'driver_id' attribute; rider is fixed on this "
                        "build", CYCLIST_MODEL)
else:
    _walker_bps = world.get_blueprint_library().filter('walker.pedestrian.*')
    obstacle_bp = random.choice(_walker_bps) if _walker_bps else None
    if obstacle_bp is not None:
        try:
            obstacle_bp.set_attribute('is_invincible', 'false')
        except (IndexError, RuntimeError):
            pass  # attribute absent on some builds
    OBSTACLE_TAGS = (PEDESTRIAN_TAG,)

if obstacle_bp is None:
    logger.warning("No blueprint for OBSTACLE_KIND=%r - occlusion scenario disabled",
                   OBSTACLE_KIND)
else:
    logger.info("Occlusion obstacle: %s (kind=%s, semantic tags %s)",
                obstacle_bp.id, OBSTACLE_KIND, OBSTACLE_TAGS)

# NOTE: the obstacle is deliberately NOT added to npc_list. Everything in that list
# is fed to the glare pipeline as a candidate headlight source; a cyclist in there
# would start emitting starbursts of its own.
obstacle = None
obstacle_control = None        # WalkerControl for a pedestrian, None for a cyclist
obstacle_start_tick = None
obstacle_run_duration_s = 0.0
obstacle_lidar_hits = []       # above-ground returns on it, one entry per scan in range


def _front_oncoming_car():
    """Nearest headlights-ON oncoming NPC still in front of the ego.

    Returns (actor, distance_m) or (None, None). This is the glare source the
    obstacle gets positioned against.
    """
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    best, best_d = None, None
    for npc in opposite_npcs:
        try:
            if not npc.is_alive:
                continue
            loc = npc.get_location()
            dx, dy = loc.x - ego_loc.x, loc.y - ego_loc.y
            if dx * fwd.x + dy * fwd.y <= 0.0:
                continue                    # already gone past the ego
            dist = (dx * dx + dy * dy) ** 0.5
            if best_d is None or dist < best_d:
                best, best_d = npc, dist
        except RuntimeError:
            continue                        # actor reaped mid-iteration
    return best, best_d


def _try_spawn_occlusion_obstacle():
    """Put the obstacle just behind the front oncoming car, inside its glare.
    Returns (actor, control, run_duration_s); (None, None, 0.0) if no oncoming car
    is close enough yet or the spot is blocked
    """
    if obstacle_bp is None:
        return None, None, 0.0

    ego_tf = ego_vehicle.get_transform()
    ego_wp = carla_map.get_waypoint(ego_tf.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return None, None, 0.0

    # Where to put it. The default is a fixed short distance ahead; if the front
    # oncoming car is close enough to be the glare source right now, sit just BEHIND
    # it instead (further from the ego along the route), so from the camera the
    # object is right at that car's headlights.
    front_npc, npc_dist = _front_oncoming_car()
    if front_npc is not None and npc_dist <= OBSTACLE_ANCHOR_WINDOW_M:
        cross_dist = npc_dist + OBSTACLE_BEHIND_NPC_M
        anchor = "%.1f m behind oncoming car %s (%.1f m out)" % (
            OBSTACLE_BEHIND_NPC_M, front_npc.id, npc_dist)
    else:
        cross_dist = OBSTACLE_AHEAD_M
        anchor = ("fixed %.0f m (nearest oncoming car %s)" % (
            OBSTACLE_AHEAD_M,
            "at %.1f m, outside the %.0f m anchor window" % (
                npc_dist, OBSTACLE_ANCHOR_WINDOW_M)
            if npc_dist is not None else "none ahead"))

    # Clamp regardless of which path was taken, so it is never stranded far away.
    cross_dist = max(OBSTACLE_MIN_AHEAD_M, min(cross_dist, OBSTACLE_MAX_AHEAD_M))

    ahead = ego_wp.next(cross_dist)
    if not ahead:
        return None, None, 0.0
    cross_wp = ahead[0]                     # the ego's lane at the crossing point
    lane_right = cross_wp.transform.get_right_vector()

    # Which side the oncoming traffic - and therefore the glare - is on. 
    onc_wp = oncoming_lane_wp(cross_wp)
    if onc_wp is not None:
        to_onc_x = onc_wp.transform.location.x - cross_wp.transform.location.x
        to_onc_y = onc_wp.transform.location.y - cross_wp.transform.location.y
        onc_side = 1.0 if (to_onc_x * lane_right.x
                           + to_onc_y * lane_right.y) > 0.0 else -1.0
    else:
        onc_side = -1.0          # no opposing lane here; assume oncoming on the left

    # 'near' puts it ON the lane line (the ego lane's far edge), which is as close to
    # the glare as it can get while staying in the clear strip beside the oncoming cars.
    if OBSTACLE_LATERAL_EDGE == 'far':
        lateral = cross_wp.lane_width * 1.5
    else:
        lateral = cross_wp.lane_width / 2.0
    spawn_loc = carla.Location(
        x=cross_wp.transform.location.x + lane_right.x * onc_side * lateral,
        y=cross_wp.transform.location.y + lane_right.y * onc_side * lateral,
        z=cross_wp.transform.location.z + (0.4 if OBSTACLE_KIND == 'cyclist' else 1.0),
    )

    # Move out into the ego's lane, i.e. away from the oncoming side.
    move_x, move_y = -onc_side * lane_right.x, -onc_side * lane_right.y
    norm = (move_x * move_x + move_y * move_y) ** 0.5
    if norm < 1e-6:
        return None, None, 0.0
    move_x, move_y = move_x / norm, move_y / norm

    actor = world.try_spawn_actor(
        obstacle_bp,
        carla.Transform(spawn_loc,
                        carla.Rotation(yaw=math.degrees(math.atan2(move_y, move_x)))),
    )
    if actor is None:
        return None, None, 0.0              # spot blocked this tick; retry later

    travel = lateral if OBSTACLE_HOLD_IN_LANE else lateral * 2.0
    duration = travel / OBSTACLE_SPEED
    control = None

    if OBSTACLE_KIND == 'cyclist':

        try:
            actor.enable_constant_velocity(
                carla.Vector3D(x=OBSTACLE_SPEED, y=0.0, z=0.0))
        except (AttributeError, RuntimeError) as exc:
            logger.warning("[OBST] Constant-velocity mode unavailable (%s); the "
                           "cyclist will sit still instead of riding out", exc)
    else:
        # Direct WalkerControl, NOT the AI walker controller: the AI navigates
       
        control = carla.WalkerControl(
            direction=carla.Vector3D(x=move_x, y=move_y, z=0.0),
            speed=OBSTACLE_SPEED, jump=False)
        actor.apply_control(control)

    # Let the ego drive on instead of braking. collision_detection targets this ONE
    # actor, so the ego still brakes normally for everything else
    try:
        traffic_manager.collision_detection(ego_vehicle, actor, False)
    except (AttributeError, RuntimeError) as exc:
        logger.warning("[OBST] Could not disable ego collision detection for %s "
                       "(%s) - the ego may brake for it", actor.id, exc)

    spawn_dist = ego_tf.location.distance(spawn_loc)
    logger.info("[OBST] %s %s spawned at t=%.1f s, %.1f m ahead of the ego "
                "[anchor: %s], on the '%s' edge at (%.1f, %.1f); crossing %.2f m "
                "into the lane at %.1f m/s (%.2f s)",
                OBSTACLE_KIND, actor.id, OBSTACLE_SPAWN_AT_S, spawn_dist, anchor,
                OBSTACLE_LATERAL_EDGE, spawn_loc.x, spawn_loc.y, travel,
                OBSTACLE_SPEED, duration)
    return actor, control, duration


#Adding sensors to vehicle

#Creating the transform needed to add LIDAR to the top of the vehicle

lidar_init_trans = carla.Transform(carla.Location(z=1.5))

#Retrieves the properties of the LIDAR sensor

lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

# Configure LIDAR for denser point cloud (better BEV visualization)
lidar_bp.set_attribute('channels', '128')
lidar_bp.set_attribute('points_per_second', '2400000')
lidar_bp.set_attribute('range', '50.0')
lidar_bp.set_attribute('rotation_frequency', '20')      # Match FPS for 1 full scan per frame
lidar_bp.set_attribute('upper_fov', '10.0')
lidar_bp.set_attribute('lower_fov', '-30.0')

# --- Sensor model matched to the clear, dry night being simulated ------------

lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.004')  # CARLA clean-air default
lidar_bp.set_attribute('dropoff_general_rate', '0.10')          # ordinary sensor loss
lidar_bp.set_attribute('dropoff_zero_intensity', '0.4')         # weak returns, stock rate
lidar_bp.set_attribute('dropoff_intensity_limit', '0.8')
lidar_bp.set_attribute('noise_stddev', '0.01')                  # 1 cm range jitter

lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to = ego_vehicle)

logger.info("Spawned lidar sensor: actor_id=%s", lidar.id)

# --- LIDAR BEV rendering settings ---
BEV_SIZE = 800           # Higher resolution for crisper output
BEV_RANGE = 50.0         # How many meters around the ego vehicle to visualize
lidar_bev_frames = []    # Collect BEV frames for video encoding


def lidar_to_bev(point_cloud):
    """Convert LIDAR point cloud to a Bird's Eye View image matching reference style."""
    # Get point cloud as numpy array (N x 4: x, y, z, intensity)
    data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
    data = data.reshape((-1, 4))

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    intensity = data[:, 3]

    # Filter points within BEV range
    mask = (np.abs(x) < BEV_RANGE) & (np.abs(y) < BEV_RANGE)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    intensity = intensity[mask]

    # Convert world coordinates to pixel coordinates
    px = ((x + BEV_RANGE) / (2 * BEV_RANGE) * BEV_SIZE).astype(np.int32)
    py = ((y + BEV_RANGE) / (2 * BEV_RANGE) * BEV_SIZE).astype(np.int32)

    # Clamp to image bounds
    px = np.clip(px, 0, BEV_SIZE - 1)
    py = np.clip(py, 0, BEV_SIZE - 1)

    # Create BEV image - black background
    bev = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.uint8)

    # Height-based HSV colormap: hue runs magenta (ground) -> red (highest)
    z_min, z_max = -2.0, 4.0
    z_norm = np.clip((z - z_min) / (z_max - z_min), 0, 1)
    hue = ((1.0 - z_norm) * 150).astype(np.uint8)
    sat = np.full_like(hue, 255, dtype=np.uint8)
    val = np.clip((0.5 + z_norm * 0.5 + intensity * 0.3) * 255, 80, 255).astype(np.uint8)

    hsv_pixels = np.stack([hue, sat, val], axis=-1).reshape(-1, 1, 3)
    bgr_pixels = cv2.cvtColor(hsv_pixels.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)

    # Paint points onto BEV
    bev[py, px] = bgr_pixels

    # --- Draw elevated objects (vehicles, structures) with thicker points ---
    # Objects above ground level (z > -0.5) get 2x2 pixels for visibility
    elevated_mask = z > -0.5
    if np.any(elevated_mask):
        epx = px[elevated_mask]
        epy = py[elevated_mask]
        ecolors = bgr_pixels[elevated_mask]

        epx2 = np.clip(epx + 1, 0, BEV_SIZE - 1)
        epy2 = np.clip(epy + 1, 0, BEV_SIZE - 1)
        bev[epy2, epx] = ecolors
        bev[epy, epx2] = ecolors
        bev[epy2, epx2] = ecolors

        # Extra bright for very tall objects (vehicles, walls) - 3x3 block
        tall_mask = z[elevated_mask] > 0.5
        if np.any(tall_mask):
            tpx = epx[tall_mask]
            tpy = epy[tall_mask]
            tcolors = ecolors[tall_mask]
            tpx3 = np.clip(tpx - 1, 0, BEV_SIZE - 1)
            tpy3 = np.clip(tpy - 1, 0, BEV_SIZE - 1)
            bev[tpy3, tpx] = tcolors
            bev[tpy, tpx3] = tcolors
            bev[tpy3, tpx3] = tcolors

    # --- Mark the occlusion obstacle ---

    if obstacle is not None:
        try:
            obs_loc = obstacle.get_location()
            # World -> sensor frame, using the LIDAR's own pose at capture time.
            world_to_sensor = np.linalg.inv(
                np.asarray(point_cloud.transform.get_matrix()))
            local = world_to_sensor @ np.array([obs_loc.x, obs_loc.y, obs_loc.z, 1.0])
            ox, oy = float(local[0]), float(local[1])
            if abs(ox) < BEV_RANGE and abs(oy) < BEV_RANGE:
                near = ((x - ox) ** 2 + (y - oy) ** 2) < OBSTACLE_LIDAR_RADIUS_M ** 2
                not_ego = ~((np.abs(x) < EGO_HALF_LENGTH_M) & (np.abs(y) < EGO_HALF_WIDTH_M))
                cand = near & not_ego
                if np.any(cand):
                    # Local ground = the lowest returns around the cyclist.
                    ground_z = float(np.percentile(z[cand], 5))
                    hits = int(np.count_nonzero(
                        cand & (z > ground_z + LIDAR_GROUND_CLEARANCE_M)))
                else:
                    hits = 0
                obstacle_lidar_hits.append(hits)
                opx = int(np.clip((ox + BEV_RANGE) / (2 * BEV_RANGE) * BEV_SIZE,
                                  0, BEV_SIZE - 1))
                opy = int(np.clip((oy + BEV_RANGE) / (2 * BEV_RANGE) * BEV_SIZE,
                                  0, BEV_SIZE - 1))
                cv2.circle(bev, (opx, opy), 18, (0, 255, 255), 2)
                cv2.putText(bev, "CYCLIST: LIDAR DETECTED" if hits else "CYCLIST",
                            (opx + 22, opy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                            cv2.LINE_AA)
        except RuntimeError:
            pass          # obstacle already destroyed

    # Add label
    cv2.putText(bev, "LIDAR BEV", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    lidar_bev_frames.append(bev.copy())

    if len(lidar_bev_frames) % 30 == 0:
        logger.info("Captured %d LIDAR BEV frames", len(lidar_bev_frames))


lidar.listen(lidar_to_bev)

# --- 1st person (driver POV) camera -----------------------------------------
# The RGB, segmentation and depth cameras below MUST share this resolution,
# FOV and transform, or the masks will not line up with the RGB frame.
FP_WIDTH, FP_HEIGHT, FP_FOV = 1920, 1080, 100.0
FP_K = build_camera_intrinsics(FP_WIDTH, FP_HEIGHT, FP_FOV)

first_person_transform = carla.Transform(
    carla.Location(x=1.5, z=1.4),  # roughly driver eye position
    carla.Rotation(pitch=0.0)
)

first_person_bp = world.get_blueprint_library().find('sensor.camera.rgb')
first_person_bp.set_attribute('image_size_x', str(FP_WIDTH))
first_person_bp.set_attribute('image_size_y', str(FP_HEIGHT))
first_person_bp.set_attribute('fov', f'{FP_FOV:g}')
# Dark surroundings, but let headlights blow out so the glare mask has a signal.
first_person_bp.set_attribute('exposure_mode', 'manual')
first_person_bp.set_attribute('exposure_compensation', '-0.5')
first_person_bp.set_attribute('bloom_intensity', '3.0')
first_person_bp.set_attribute('lens_flare_intensity', '1.0')
first_person_bp.set_attribute('motion_blur_intensity', '0.5')
first_person_bp.set_attribute('motion_blur_max_distortion', '0.5')

first_person_camera = world.spawn_actor(
    first_person_bp, first_person_transform, attach_to=ego_vehicle
)
logger.info("Spawned 1st person camera: actor_id=%s", first_person_camera.id)

# Collect first-person + semantic frames keyed by frame ID so we can pair them
first_person_frames = []          # ordered list for the plain (non-glare) video
fp_rgb_by_frame = {}              # frame_id -> BGR array
fp_seg_by_frame = {}              # frame_id -> semantic label array
fp_cam_by_frame = {}              # frame_id -> (cam 4x4 world matrix, cam location)
fp_vehicles_by_frame = {}         # frame_id -> list of vehicle geometry records


def _vehicle_records():
    """Snapshot every NPC's world transform + bbox extent for headlight projection."""
    records = []
    for npc in npc_list:
        try:
            if not npc.is_alive:
                continue
            tf = npc.get_transform()
            bb = npc.bounding_box
            ext = bb.extent
            fwd = tf.get_forward_vector()
            records.append({
                'matrix': tf.get_matrix(),
                'extent': (ext.x, ext.y, ext.z),
                # CARLA vehicle bounding boxes have a local center offset from the actor origin 
                'bb_center': (bb.location.x, bb.location.y, bb.location.z),
                'forward': (fwd.x, fwd.y, fwd.z),
                'location': (tf.location.x, tf.location.y, tf.location.z),
            })
        except RuntimeError:
            continue
    return records


def save_first_person(image):
    """Callback: store RGB frame + the ground-truth geometry needed for projection."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((FP_HEIGHT, FP_WIDTH, 4))
    frame = array[:, :, :3].copy()  # Drop alpha -> BGR
    first_person_frames.append(frame)
    fp_rgb_by_frame[image.frame] = frame

    # image.transform is the camera's world transform at capture time
    cam_tf = image.transform
    fp_cam_by_frame[image.frame] = (
        cam_tf.get_matrix(),
        (cam_tf.location.x, cam_tf.location.y, cam_tf.location.z),
    )
    fp_vehicles_by_frame[image.frame] = _vehicle_records()

    if len(first_person_frames) % 30 == 0:
        logger.info("Captured %d first-person frames so far", len(first_person_frames))


first_person_camera.listen(save_first_person)


# --- Semantic segmentation camera matched to the first-person camera ---

seg_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
seg_bp.set_attribute('image_size_x', str(FP_WIDTH))
seg_bp.set_attribute('image_size_y', str(FP_HEIGHT))
seg_bp.set_attribute('fov', f'{FP_FOV:g}')

seg_camera = world.spawn_actor(
    seg_bp, first_person_transform, attach_to=ego_vehicle
)
logger.info("Spawned semantic segmentation camera: actor_id=%s", seg_camera.id)


def save_seg(image):
    """Callback: store the semantic label channel keyed by frame ID."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((FP_HEIGHT, FP_WIDTH, 4))
    labels = array[:, :, 2].copy()  # CARLA encodes class id in the red channel
    fp_seg_by_frame[image.frame] = labels


seg_camera.listen(save_seg)


# --- Depth camera matched to the first-person camera -----

depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
depth_bp.set_attribute('image_size_x', str(FP_WIDTH))
depth_bp.set_attribute('image_size_y', str(FP_HEIGHT))
depth_bp.set_attribute('fov', f'{FP_FOV:g}')

depth_camera = world.spawn_actor(
    depth_bp, first_person_transform, attach_to=ego_vehicle
)
logger.info("Spawned depth camera: actor_id=%s", depth_camera.id)

# Full-res float32 depth would be ~8 MB per frame. The detection gate only needs
# coarse depth, so store it downscaled and as float16 (~260 KB/frame).
DEPTH_DOWNSCALE = 4
DEPTH_W = FP_WIDTH // DEPTH_DOWNSCALE
DEPTH_H = FP_HEIGHT // DEPTH_DOWNSCALE
fp_depth_by_frame = {}            # frame_id -> downscaled depth in METRES


def save_depth(image):
    """Callback: decode CARLA's packed depth to metres and store it downscaled."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((FP_HEIGHT, FP_WIDTH, 4))
    # CARLA packs depth across the BGR channels as a 24-bit integer.
    b = array[:, :, 0].astype(np.float32)
    g = array[:, :, 1].astype(np.float32)
    r = array[:, :, 2].astype(np.float32)
    normalized = (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0)
    depth_m = normalized * 1000.0   # CARLA's far plane is 1000 m
    # INTER_NEAREST, not area-averaging: averaging across a depth discontinuity

    small = cv2.resize(depth_m, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_NEAREST)
    fp_depth_by_frame[image.frame] = small.astype(np.float16)


depth_camera.listen(save_depth)


# --- 3rd person chase camera (context view for the exported video) ----------

EXPORT_DIR = 'exports'
os.makedirs(EXPORT_DIR, exist_ok=True)

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# 6 m behind, 4 m above, angled slightly down.
third_person_transform = carla.Transform(
    carla.Location(x=-6.0, z=4.0),
    carla.Rotation(pitch=-15.0)
)

third_person_bp = world.get_blueprint_library().find('sensor.camera.rgb')
third_person_bp.set_attribute('image_size_x', str(VIDEO_WIDTH))
third_person_bp.set_attribute('image_size_y', str(VIDEO_HEIGHT))
third_person_bp.set_attribute('fov', '90')

# Spawn and attach to ego vehicle
third_person_camera = world.spawn_actor(
    third_person_bp, third_person_transform, attach_to=ego_vehicle
)
logger.info("Spawned 3rd person camera: actor_id=%s", third_person_camera.id)

# List to collect frames in memory for video encoding
video_frames = []


def save_third_person(image):
    """Callback: convert CARLA image to numpy array and store for video."""
    # Convert raw data to numpy BGRA array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 4))
    # Drop alpha channel -> BGR (OpenCV format)
    frame = array[:, :, :3]
    video_frames.append(frame.copy())
    if len(video_frames) % 30 == 0:
        logger.info("Captured %d video frames so far", len(video_frames))


third_person_camera.listen(save_third_person)


# Enable autopilot for NPC vehicles only (they drive normally)
for npc in npc_list:
    npc.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.update_vehicle_lights(npc, False)  # TM won't manage lights
    if npc.id in opposite_npc_ids:
        npc.set_light_state(HEADLIGHTS_ON)    # oncoming: the glare source
    else:
        npc.set_light_state(LIGHTS_OFF)       # same-direction NPCs stay dark

logger.info("Enabled autopilot for %d NPC vehicles", len(npc_list))

# Let the lead stack run slightly quicker than the ego so the gap opened up by
# LEAD_STACK_DISTANCES doesn't collapse before the obstacle appears

for npc in lead_npcs:
    try:
        traffic_manager.vehicle_percentage_speed_difference(npc, -12.0)  # ~12% faster
    except (AttributeError, RuntimeError):
        pass  # non-essential tuning; TM API varies by version

# Enable autopilot for the ego vehicle as well
ego_vehicle.set_autopilot(True, traffic_manager.get_port())
traffic_manager.update_vehicle_lights(ego_vehicle, False)  # TM won't manage lights
ego_vehicle.set_light_state(EGO_LIGHTS)
logger.info("Ego vehicle autopilot enabled, low beams on")

# Steer the ego along the planned route toward spawn 82. set_path() exists in
# CARLA 0.9.13+; if unavailable we fall back to plain autopilot 
try:
    route_locations = [wp.transform.location for wp in ego_route[::10]]  # every ~20 m
    traffic_manager.set_path(ego_vehicle, route_locations)
    logger.info("Ego route path set (%d waypoints toward spawn %d)",
                len(route_locations), DEST_SPAWN_IDX)
except Exception as e:
    logger.warning("traffic_manager.set_path unavailable (%s) - using plain autopilot", e)

TOTAL_TICKS = int(DURATION_S / FIXED_DELTA)
logger.info("Running for %.0f seconds (%d ticks)", DURATION_S, TOTAL_TICKS)

# Reapply weather right before the run and confirm it took (guards against stale state)
world.set_weather(weather)
applied = world.get_weather()
logger.info("Confirmed weather: sun_altitude=%.1f, fog_density=%.1f, precip=%.1f",
            applied.sun_altitude_angle, applied.fog_density, applied.precipitation)

# Make sure opposing NPC headlights are on before the first frame is captured
for npc in opposite_npcs:
    if npc.is_alive:
        npc.set_light_state(HEADLIGHTS_ON)

OBSTACLE_SPAWN_TICK = int(OBSTACLE_SPAWN_AT_S / FIXED_DELTA)
OBSTACLE_RETRY_TICKS = int(OBSTACLE_SPAWN_RETRY_S / FIXED_DELTA)
logger.info("[OBST] %s appears at tick %d (t=%.1f s), %.0f m ahead - or %.1f m "
            "behind the front oncoming car if one is within %.0f m. Clamped to "
            "%.0f-%.0f m. Retrying until tick %d only if that spot is blocked.",
            OBSTACLE_KIND, OBSTACLE_SPAWN_TICK, OBSTACLE_SPAWN_AT_S,
            OBSTACLE_AHEAD_M, OBSTACLE_BEHIND_NPC_M, OBSTACLE_ANCHOR_WINDOW_M,
            OBSTACLE_MIN_AHEAD_M, OBSTACLE_MAX_AHEAD_M,
            OBSTACLE_SPAWN_TICK + OBSTACLE_RETRY_TICKS)

for tick in range(TOTAL_TICKS):
    world.tick()

    # --- Occlusion obstacle moves out of the glare into the ego's lane ---
    if obstacle is None:
        if OBSTACLE_SPAWN_TICK <= tick <= OBSTACLE_SPAWN_TICK + OBSTACLE_RETRY_TICKS:
            (obstacle, obstacle_control,
             obstacle_run_duration_s) = _try_spawn_occlusion_obstacle()
            if obstacle is not None:
                obstacle_start_tick = tick
    else:
        obs_elapsed = (tick - obstacle_start_tick) * FIXED_DELTA
        try:
            if obs_elapsed <= obstacle_run_duration_s:
                if obstacle_control is not None:
                    # Reassert: a walker keeps its last command

                    obstacle.apply_control(obstacle_control)
            elif obs_elapsed <= obstacle_run_duration_s + 0.1:
                # Reached the ego's lane - hold there, in the ego's path.
                if obstacle_control is not None:
                    obstacle.apply_control(carla.WalkerControl(speed=0.0))
                elif OBSTACLE_HOLD_IN_LANE:
                    obstacle.disable_constant_velocity()
        except RuntimeError:
            pass  # obstacle was destroyed - nothing to steer

    # Reapply lights so they stay on through the run
    if tick % 20 == 0:
        for npc in opposite_npcs:
            if npc.is_alive:
                npc.set_light_state(HEADLIGHTS_ON)
        ego_vehicle.set_light_state(EGO_LIGHTS)
    if tick % 100 == 0:
        elapsed = tick * FIXED_DELTA
        logger.info("%.1f seconds elapsed (%d/%d ticks)", elapsed, tick, TOTAL_TICKS)

if obstacle is None:
    logger.warning("[OBST] Obstacle never spawned. Every attempt between %.1f s and "
                   "%.1f s found the spot blocked, or the route ran out ahead of the "
                   "ego. Try a different OBSTACLE_AHEAD_M (now %.0f m).",
                   OBSTACLE_SPAWN_AT_S, OBSTACLE_SPAWN_AT_S + OBSTACLE_SPAWN_RETRY_S,
                   OBSTACLE_AHEAD_M)
else:
    logger.info("[OBST] Spawned at %.1f s (tick %d); in the ego's lane from %.1f s",
                obstacle_start_tick * FIXED_DELTA, obstacle_start_tick,
                obstacle_start_tick * FIXED_DELTA + obstacle_run_duration_s)

# The LIDAR half of the comparison: was the cyclist detected, scan by scan?
if obstacle_lidar_hits:
    scans_with_hits = sum(1 for h in obstacle_lidar_hits if h > 0)
    logger.info("[OBST] LIDAR detected the cyclist in %d of %d scans while it was "
                "in LIDAR range. Compare against the camera contrast figures below.",
                scans_with_hits, len(obstacle_lidar_hits))
    if scans_with_hits == 0:
        logger.warning("[OBST] Zero LIDAR returns on the obstacle across the whole "
                       "run. Check the LIDAR range (%s m) against how far ahead it "
                       "spawned (up to %.0f m).",
                       lidar_bp.get_attribute('range'), OBSTACLE_MAX_AHEAD_M)

logger.info("Finished run - stopping sensors")

# Stop listening before encoding video
first_person_camera.stop()
seg_camera.stop()
depth_camera.stop()
third_person_camera.stop()
lidar.stop()

def encode_video(frames, path, size, label):
    """Write `frames` to `path` as mp4v at FPS. Warns and skips if empty."""
    if not frames:
        logger.warning("No %s frames captured - %s not created", label,
                       os.path.basename(path))
        return
    logger.info("Encoding %d %s frames to %s at %.1f FPS",
                len(frames), label, path, FPS)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, size)
    for frame in frames:
        writer.write(frame)
    writer.release()
    logger.info("%s video saved to %s", label, path)


VIDEO_OUTPUT_PATH = os.path.join(EXPORT_DIR, 'ego_third_person.mp4')
FPV_OUTPUT_PATH = os.path.join(EXPORT_DIR, 'ego_first_person.mp4')

encode_video(video_frames, VIDEO_OUTPUT_PATH,
             (VIDEO_WIDTH, VIDEO_HEIGHT), "3rd person")
encode_video(first_person_frames, FPV_OUTPUT_PATH,
             (FP_WIDTH, FP_HEIGHT), "1st person")

# --- Encode 1st Person video WITH glare mask applied ---
# Pair RGB + semantic frames by frame ID, apply smooth headlight glare.
GLARE_OUTPUT_PATH = os.path.join(EXPORT_DIR, 'ego_first_person_glare.mp4')

# Frames need RGB + segmentation + camera pose + vehicle geometry to be usable
_base_frame_ids = (
    set(fp_rgb_by_frame.keys())
    & set(fp_seg_by_frame.keys())
    & set(fp_cam_by_frame.keys())
)
# Depth is also required for the detection gate.

paired_frame_ids = sorted(_base_frame_ids & set(fp_depth_by_frame.keys()))
if not paired_frame_ids and _base_frame_ids:
    logger.warning("[GATE] No frames carry depth - the detection gate is running "
                   "SEGMENTATION-ONLY, which cannot tell one car from another. "
                   "Expect ghost glare to come back. Check the depth camera.")
    paired_frame_ids = sorted(_base_frame_ids)
elif len(paired_frame_ids) < len(_base_frame_ids):
    logger.warning("Dropped %d frame(s) with no matching depth image",
                   len(_base_frame_ids) - len(paired_frame_ids))

# --- Detection gate thresholds ------------------------------------------------
# A vehicle only glares once the camera genuinely resolves it. MIN_DETECTED_PX is
# the number of full-res pixels that must be BOTH labelled vehicle AND at that
# vehicle's own depth.

MAX_GLARE_DISTANCE_M = 70.0
MIN_DETECTED_PX = 150

if paired_frame_ids:
    logger.info("Applying glare to %d paired frames -> %s", len(paired_frame_ids), GLARE_OUTPUT_PATH)

    # --- Diagnostics (DEBUG only) -----
    if DEBUG:
      # Inspect an EARLY frame (oncoming cars still in view); mid-run frames are
      # after the oncoming traffic has passed.
      diag_fid = paired_frame_ids[min(20, len(paired_frame_ids) - 1)]
      diag_rgb = fp_rgb_by_frame[diag_fid]
      diag_seg = fp_seg_by_frame[diag_fid]
      diag_gray = cv2.cvtColor(diag_rgb, cv2.COLOR_BGR2GRAY)
      logger.info("[DIAG] RGB brightness: max=%d, mean=%.1f, pixels>110=%d, pixels>140=%d",
                  int(diag_gray.max()), float(diag_gray.mean()),
                  int(np.count_nonzero(diag_gray > 110)), int(np.count_nonzero(diag_gray > 140)))
      labels_present, counts = np.unique(diag_seg, return_counts=True)
      logger.info("[DIAG] Semantic labels present: %s",
                  ", ".join("%d:%d" % (l, c) for l, c in zip(labels_present, counts)))
      veh_px = int(np.count_nonzero(diag_seg == VEHICLE_TAG))
      logger.info("[DIAG] Vehicle pixels (tag=%d): %d", VEHICLE_TAG, veh_px)
      cv2.imwrite(os.path.join(EXPORT_DIR, 'diag_rgb.png'), diag_rgb)
      cv2.imwrite(os.path.join(EXPORT_DIR, 'diag_seg_labels.png'), diag_seg)

      # Colour per gate verdict. GREEN = allowed to glare, everything else = blocked
      GATE_COLORS = {
          'detected':          (0, 255, 0),    # green   - camera resolves it, glare ON
          'not_detected':      (0, 0, 255),    # red     - hidden / too small to see
          'off_camera':        (255, 0, 255),  # magenta - outside the frame entirely
          'too_far':           (0, 140, 255),  # orange  - beyond the glare range
          'facing_away':       (0, 255, 255),  # yellow  - headlights point elsewhere
          'headlights_hidden': (255, 255, 0),  # cyan    - seen, but lights occluded
          'degenerate':        (128, 128, 128),
      }

      def _draw_projection_overlay(fid):
          """Annotate frame `fid` with the detection gate's verdict per vehicle.
          """
          rgb = fp_rgb_by_frame[fid]
          cam_matrix, cam_loc = fp_cam_by_frame[fid]
          vehicles = fp_vehicles_by_frame.get(fid, [])
          hl, reports = camera_detected_headlights(
              vehicles, cam_matrix, FP_K, (FP_WIDTH, FP_HEIGHT),
              depth_map=fp_depth_by_frame.get(fid),
              seg_labels=fp_seg_by_frame.get(fid),
              cam_location=cam_loc, require_facing=True,
              max_distance=MAX_GLARE_DISTANCE_M, min_detected_px=MIN_DETECTED_PX,
              return_reports=True,
          )
          vis = rgb.copy()

          for rep in reports:
              color = GATE_COLORS.get(rep['status'], (200, 200, 200))
              rect = rep['footprint']
              if rect is not None:
                  x0, y0, x1, y1 = rect
                  cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
                  label = "%s  %.0fm  %dpx" % (rep['status'], rep['distance'] or 0.0,
                                               rep['detected_px'])
                  cv2.putText(vis, label, (x0, max(14, y0 - 6)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
              else:
                  # No rect means it never projected into the frame

                  cen = project_points_to_image([rep['location']], cam_matrix, FP_K)
                  if cen and cen[0][0] is not None:
                      cpx, cpy, cd = cen[0]
                      if 0 <= cpx < FP_WIDTH and 0 <= cpy < FP_HEIGHT:
                          cv2.drawMarker(vis, (int(cpx), int(cpy)), color,
                                         cv2.MARKER_TILTED_CROSS, 18, 2)
                          cv2.putText(vis, "%s %.0fm" % (rep['status'], cd),
                                      (int(cpx) + 10, int(cpy)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

          for (px, py, d) in hl:
              cv2.circle(vis, (int(px), int(py)), 18, (0, 255, 0), 2)
              cv2.putText(vis, "GLARE %.0fm" % d, (int(px) + 20, int(py)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

          cv2.putText(vis, "green=glare allowed   red=not detected   "
                           "magenta=off camera   orange=too far   yellow=facing away",
                      (20, FP_HEIGHT - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                      (255, 255, 255), 1, cv2.LINE_AA)
          return vis, len(hl)

      # Per-vehicle projection detail: why headlight points collapse or pile up.
      _dv = fp_vehicles_by_frame.get(diag_fid, [])
      _cm, _cl = fp_cam_by_frame[diag_fid]
      logger.info("[DIAG] --- per-vehicle projection (frame %s) ---", diag_fid)
      for vi, rec in enumerate(_dv[:6]):
          lpts = headlight_points_local(rec['extent'], rec.get('bb_center', (0, 0, 0)))
          M = np.asarray(rec['matrix'], dtype=np.float64)
          wpts = [tuple((M @ np.array([lx, ly, lz, 1.0]))[:3]) for (lx, ly, lz) in lpts]
          proj = project_points_to_image(wpts, _cm, FP_K)
          logger.info("[DIAG] veh%d loc=(%.1f,%.1f,%.1f) ext=%s bbc=%s -> HLpx=%s",
                      vi, rec['location'][0], rec['location'][1], rec['location'][2],
                      tuple(round(e, 2) for e in rec['extent']),
                      tuple(round(c, 2) for c in rec.get('bb_center', (0, 0, 0))),
                      [(None if p[0] is None else (round(p[0]), round(p[1]))) for p in proj])

      # Dump several EARLY frames so oncoming cars are still in view
      for tag, idx in [("early05", 5), ("early10", 10), ("early20", 20), ("early40", 40)]:
          if idx < len(paired_frame_ids):
              fid = paired_frame_ids[idx]
              vis, nhl = _draw_projection_overlay(fid)
              path = os.path.join(EXPORT_DIR, 'diag_projection_%s.png' % tag)
              cv2.imwrite(path, vis)
              logger.info("[DIAG] %s: %d headlight points -> %s", tag, nhl, path)

      proj_vis, _ = _draw_projection_overlay(diag_fid)
      cv2.imwrite(os.path.join(EXPORT_DIR, 'diag_projection.png'), proj_vis)

    # Side-by-side comparison output: [ no glare | with glare ], each 960 wide
    COMPARE_OUTPUT_PATH = os.path.join(EXPORT_DIR, 'ego_first_person_compare.mp4')
    half_w = 960
    half_h = 540  # keep 16:9 per panel; final frame is 1920x540

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    glare_writer = cv2.VideoWriter(GLARE_OUTPUT_PATH, fourcc, FPS,
                                   (FP_WIDTH, FP_HEIGHT))
    compare_writer = cv2.VideoWriter(COMPARE_OUTPUT_PATH, fourcc, FPS, (half_w * 2, half_h))

    # --- Obstacle occlusion measurement --------------------------------------
    # The scenario uses glare hides the obstacle. This measures it rather than
    # asserting it: on every frame where the object is visible, compare the contrast
    # between its pixels and the surrounding background, with and without glare.
    # Glare lifts the local black level, so the silhouette loses separation from its
    # background - that contrast drop IS the occlusion, as a number. Read it beside
    # the [OBST] LIDAR point counts for the two halves of the argument.
    OBS_MIN_PX = 60           # below this the object is a few specks; skip contrast
    obs_frames = 0            # frames big enough to measure contrast on
    obs_any_frames = 0        # frames the object appears in AT ALL
    obs_peak_px = 0
    obs_first_i = None
    obs_last_i = None
    obs_contrast_plain_sum = 0.0
    obs_contrast_glare_sum = 0.0
    worst_obs = None          # (drop, fid, n_px, c_plain, c_glare, crops)

    def _obstacle_contrast(seg_labels, plain_bgr, glare_bgr):
        """(n_px, contrast_plain, contrast_glare, crop_box) for the obstacle.

        n_px is always returned so the caller can tell "never on screen" apart from
        "on screen but too small to measure"; the contrasts are None in the latter
        case. Contrast = |mean luminance of the object - mean luminance of a ring of
        background around it|, computed on a crop so this stays cheap per frame.

        """
        obs = np.isin(seg_labels, OBSTACLE_TAGS)
        n_px = int(np.count_nonzero(obs))
        if n_px == 0:
            return 0, None, None, None
        if n_px < OBS_MIN_PX:
            return n_px, None, None, None

        ys, xs = np.nonzero(obs)
        pad = 60
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(seg_labels.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(seg_labels.shape[1], int(xs.max()) + pad + 1)

        obs_crop = obs[y0:y1, x0:x1]
        ring = cv2.dilate(obs_crop.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0
        ring &= ~obs_crop
        if not ring.any():
            return n_px, None, None, None

        def contrast(bgr):
            gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            return abs(float(gray[obs_crop].mean()) - float(gray[ring].mean()))

        return n_px, contrast(plain_bgr), contrast(glare_bgr), (x0, y0, x1, y1)

    total_mask_px = 0
    total_hl_points = 0
    gate_counts = {}          # status -> how many vehicle-frames landed there
    for i, fid in enumerate(paired_frame_ids):
        rgb = fp_rgb_by_frame[fid]
        seg = fp_seg_by_frame[fid]
        cam_matrix, cam_loc = fp_cam_by_frame[fid]
        vehicles = fp_vehicles_by_frame.get(fid, [])

        # 1. DETECTION GATE: a vehicle contributes headlights only once the ego
        #    camera actually detects it
        hl_pixels, gate_reports = camera_detected_headlights(
            vehicles, cam_matrix, FP_K, (FP_WIDTH, FP_HEIGHT),
            depth_map=fp_depth_by_frame.get(fid), seg_labels=seg,
            cam_location=cam_loc, require_facing=True,
            max_distance=MAX_GLARE_DISTANCE_M, min_detected_px=MIN_DETECTED_PX,
            return_reports=True,
        )
        total_hl_points += len(hl_pixels)
        for rep in gate_reports:
            gate_counts[rep['status']] = gate_counts.get(rep['status'], 0) + 1

        # 2. Geometry region AND vehicle segmentation.
        mask, rgb_frame = geometry_glare_mask(
            rgb, seg, hl_pixels,
            require_bright=False, require_vehicle=True,
        )
        total_mask_px += int(np.count_nonzero(mask))
        # Pass hl_pixels so glare size/intensity scale with each vehicle's RANGE
        glared = glare(mask, rgb_frame, intensity=3.5, headlight_pixels=hl_pixels)
        glare_writer.write(glared)

        # 3. How well can the obstacle still be seen once glare is applied?
        n_px, c_plain, c_glare, obs_box = _obstacle_contrast(seg, rgb_frame, glared)
        if n_px:
            obs_any_frames += 1
            obs_peak_px = max(obs_peak_px, n_px)
            if obs_first_i is None:
                obs_first_i = i
            obs_last_i = i
        if c_plain is not None:
            px0, py0, px1, py1 = obs_box
            obs_frames += 1
            obs_contrast_plain_sum += c_plain
            obs_contrast_glare_sum += c_glare
            drop = c_plain - c_glare
            if worst_obs is None or drop > worst_obs[0]:
                worst_obs = (drop, fid, n_px, c_plain, c_glare,
                             rgb_frame[py0:py1, px0:px1].copy(),
                             glared[py0:py1, px0:px1].copy())

        # --- Build labeled side-by-side frame ---
        left = cv2.resize(rgb_frame, (half_w, half_h))   # original (no glare)
        right = cv2.resize(glared, (half_w, half_h))     # with glare
        cv2.putText(left, "NO GLARE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, "WITH GLARE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        combined = np.hstack([left, right])
        # thin divider line between the two panels
        cv2.line(combined, (half_w, 0), (half_w, half_h), (60, 60, 60), 2)
        compare_writer.write(combined)

        if (i + 1) % 30 == 0:
            logger.info("Glare-processed %d/%d frames", i + 1, len(paired_frame_ids))

    glare_writer.release()
    compare_writer.release()
    logger.info("Glare first-person video saved to %s (total headlight mask px across run=%d)",
                GLARE_OUTPUT_PATH, total_mask_px)
    logger.info("Side-by-side comparison video saved to %s", COMPARE_OUTPUT_PATH)
    logger.info("[DIAG] Glaring headlight points across run: %d (avg %.1f/frame)",
                total_hl_points, total_hl_points / max(1, len(paired_frame_ids)))
    logger.info("[GATE] Vehicle-frame verdicts: %s",
                ", ".join("%s=%d" % (k, v) for k, v in sorted(gate_counts.items())))

    # --- Obstacle occlusion result -------------------------------------------
    # Report WHEN the object was first on screen

    if obs_any_frames:
        logger.info("[OBST] On screen in %d frames: first at %.2f s (frame index %d), "
                    "last at %.2f s (index %d), peak size %d px. Look there in %s.",
                    obs_any_frames, obs_first_i / FPS, obs_first_i,
                    obs_last_i / FPS, obs_last_i, obs_peak_px,
                    os.path.basename(COMPARE_OUTPUT_PATH))
    elif obstacle is not None:
        logger.warning("[OBST] Obstacle %s existed but NEVER appeared in any frame. "
                       "Either it spawned outside the camera's view, or OBSTACLE_TAGS "
                       "%s are wrong for this build. Check the '[DIAG] Semantic "
                       "labels present' line for the real tags.",
                       obstacle.id, OBSTACLE_TAGS)

    if obs_frames:
        mean_plain = obs_contrast_plain_sum / obs_frames
        mean_glare = obs_contrast_glare_sum / obs_frames
        logger.info("[OBST] Measurable in %d frames. Mean obstacle/background "
                    "contrast: %.1f without glare -> %.1f with glare (%.0f%% lost)",
                    obs_frames, mean_plain, mean_glare,
                    100.0 * (mean_plain - mean_glare) / max(mean_plain, 1e-6))
        drop, fid, n_px, c_plain, c_glare, plain_crop, glare_crop = worst_obs
        logger.info("[OBST] Worst frame %s: %d px, contrast %.1f -> %.1f "
                    "(drop %.1f) - the moment the object is most hidden",
                    fid, n_px, c_plain, c_glare, drop)
        # Side-by-side crop of that moment, scaled up for a slide.
        h_c = max(plain_crop.shape[0], glare_crop.shape[0])
        scale = max(1, int(320 / max(1, h_c)))
        pc = cv2.resize(plain_crop, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST)
        gc = cv2.resize(glare_crop, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST)
        cv2.putText(pc, "NO GLARE", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(gc, "WITH GLARE", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(EXPORT_DIR, 'diag_obstacle_occlusion.png'),
                    np.hstack([pc, gc]))
        logger.info("[OBST] Wrote %s/diag_obstacle_occlusion.png", EXPORT_DIR)
    elif obs_any_frames:
        logger.info("[OBST] Object was on screen but never larger than %d px, too "
                    "small to measure contrast on. Lower OBSTACLE_AHEAD_M (now "
                    "%.0f m) so it appears closer to the camera.",
                    OBS_MIN_PX, OBSTACLE_AHEAD_M)
    if total_mask_px == 0:
        logger.warning("[DIAG] Headlight mask EMPTY for every frame. Check: "
                       "(a) projected points > 0? if 0, geometry/facing test is rejecting all; "
                       "(b) vehicle pixels > 0? if 0, VEHICLE_TAG is wrong; "
                       "(c) bright pixels > 0? if 0, lower brightness_thresh. "
                       "Also try increasing base_radius_px if projection is slightly off.")
else:
    logger.warning("No paired RGB+seg frames - glare video not created")

BEV_OUTPUT_PATH = os.path.join(EXPORT_DIR, 'lidar_bev.mp4')
encode_video(lidar_bev_frames, BEV_OUTPUT_PATH, (BEV_SIZE, BEV_SIZE), "LIDAR BEV")

logger.info("Exports complete: %s", EXPORT_DIR)

# Cleanup actors
logger.info("Destroying actors")
for actor in [first_person_camera, seg_camera, depth_camera,
              third_person_camera, lidar, ego_vehicle, obstacle] + npc_list:
    if actor is None:
        continue
    try:
        if actor.is_alive:
            actor.destroy()
    except RuntimeError:
        pass  # already destroyed by the server


try:
    world.apply_settings(original_settings)
    traffic_manager.set_synchronous_mode(False)
    world.tick()
    time.sleep(0.5)
except RuntimeError:
    pass

logger.info("Done")
