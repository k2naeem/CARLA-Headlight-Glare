"""Night scenario with live glare on DeepSight's cameras, fed to a stand-in agent.

Same Town07 night scenario as Carla_script.py: oncoming traffic with high beams,
a cyclist crossing inside their glare. What changes is when the glare happens.
Carla_script.py records clean frames and draws glare after the run. Here, on
every tick:

    CARLA renders DeepSight's six cameras (1600x900, default camera settings)
      -> GlareInjector adds glare to the three front views
      -> DeepSightInputAgent saves the views to disk at JPEG quality 20 and
         builds the 10-image input DeepSight would read (4 history + 6 current)

The ego still drives on Traffic Manager: the agent only reproduces DeepSight's
input, it does not drive. Usage:

    python3 carla_script_closed_loop.py --severity 1.0
    python3 carla_script_closed_loop.py --severity 0      # clean run, same seed

Outputs go to exports_closed_loop/severity_<s>/ (see the end of this file).
"""
import argparse
import carla
import json
import logging
import math
import random
import threading
import time
import os
import numpy as np
import cv2
from glare_mask_closed_loop import (
    PEDESTRIAN_TAG, RIDER_TAG, BICYCLE_TAG, SEVERITY_LEVELS,
)
from glare_injector import GlareInjector, GLARE_CAMERAS, spec_transform
from deepsight_input_agent import (
    JPEG_QUALITY, HISTORY_STEPS, SURROUND_CAMERAS, make_glare_agent_class,
)


logging.basicConfig(
    level=logging.INFO,

    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--host', default='localhost')
parser.add_argument('--port', type=int, default=2000)
parser.add_argument('--severity', type=float,
                    default=float(os.environ.get('GLARE_SEVERITY', '1.0')),
                    help='glare level; sweep levels are %s (default 1.0, or $GLARE_SEVERITY)'
                         % (SEVERITY_LEVELS,))
parser.add_argument('--duration', type=float, default=45.0, help='seconds of simulation')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed; keep it fixed so runs at different severities match')
parser.add_argument('--front-only', action='store_true',
                    help='spawn only the three front cameras (faster); the rear '
                         'slots of the model input are filled with black')
parser.add_argument('--mixed-traffic', action='store_true',
                    help='oncoming vans and trucks as well as cars, to test that '
                         'the gate counts them')
parser.add_argument('--chase-cam', action='store_true',
                    help='also record a 3rd-person video for context')
parser.add_argument('--strict', action='store_true',
                    help='stop the run if any frame would reach the agent without glare')
parser.add_argument('--model-view-every', type=int, default=20,
                    help='save the tiled 10-image model input every N steps (0 = never)')
parser.add_argument('--out', default=None, help='output folder')
args = parser.parse_args()

random.seed(args.seed)

# Run timing (single source of truth)
FIXED_DELTA = 0.05                  # seconds per tick
DURATION_S = args.duration          # length of the run
FPS = 1.0 / FIXED_DELTA             # video framerate, matches the tick rate



logger.info("Connecting to CARLA at %s:%d", args.host, args.port)
client = carla.Client(args.host, args.port)
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
        "Do NOT just switch ports — two clients on one world will crash the server."
        % (TM_PORT, e, os.path.basename(__file__), TM_PORT)
    )

logger.info("Traffic manager connected on port %d", TM_PORT)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_random_device_seed(args.seed)   # same traffic behaviour per seed
traffic_manager.set_global_distance_to_leading_vehicle(2.5)
logger.info("Synchronous mode enabled (fixed 20 FPS)")

weather = carla.WeatherParameters(
    cloudiness=80.0,             # Overcast, dark sky (no moonlight bleed)
    precipitation=0.0,           # No rain
    precipitation_deposits=0.0,  # Dry roads
    sun_altitude_angle=-90.0,    # Sun fully below horizon — pitch dark night
    wetness=0.0,                 # Dry
    fog_density=0.0,             # No fog — clear dark night
    fog_distance=0.0,
    fog_falloff=0.0,
    wind_intensity=0.0,
    scattering_intensity=0.0,
)

world.set_weather(weather)
logger.info("Weather applied — clear, pitch-dark night (no fog, no rain)")



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

# One NPC model by default, matching the open-loop results. --mixed-traffic adds
# a van and a truck: they segment as 15/16, not 14, which tests that the
# detection gate counts them (Bench2Drive traffic includes both).
NPC_MODEL = 'vehicle.lincoln.mkz_2017'
MIXED_TRAFFIC_MODELS = (NPC_MODEL, 'vehicle.mercedes.sprinter',
                        'vehicle.carlamotors.carlacola', 'vehicle.ford.ambulance')
npc_vehicle_list = [world.get_blueprint_library().find(NPC_MODEL)]
if args.mixed_traffic:
    npc_vehicle_list = []
    for _name in MIXED_TRAFFIC_MODELS:
        try:
            npc_vehicle_list.append(world.get_blueprint_library().find(_name))
        except (IndexError, RuntimeError):
            logger.warning("Blueprint %s not in this CARLA build; skipped", _name)
    logger.info("Mixed traffic: %s", ", ".join(bp.id for bp in npc_vehicle_list))

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
behind_distances = [15, 30, 50]  # Meters behind ego — close together

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

# The governing geometry, on DeepSight's CAM_FRONT (focal ~1142 px): at ~30 m
# the cyclist on the lane line is ~41 px from the inner headlight, inside a
# starburst of ~107 px radius. Glare sizes scale with focal length, so this
# ratio is the same as on the open-loop camera.
OBSTACLE_KIND = 'cyclist'      # 'cyclist' or 'pedestrian'

OBSTACLE_SPAWN_AT_S = 2.0      # appears at this moment, seconds into the run
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

# No LIDAR here: this script tests the camera input path. The LIDAR comparison
# lives in Carla_script.py.

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
    logger.warning("No blueprint for OBSTACLE_KIND=%r — occlusion scenario disabled",
                   OBSTACLE_KIND)
else:
    logger.info("Occlusion obstacle: %s (kind=%s, semantic tags %s)",
                obstacle_bp.id, OBSTACLE_KIND, OBSTACLE_TAGS)

# NOTE: the obstacle is not added to npc_list, which drives the light and
# autopilot setup below. It cannot glare either way: the injector skips anything
# with fewer than four wheels.
obstacle = None
obstacle_control = None        # WalkerControl for a pedestrian, None for a cyclist
obstacle_start_tick = None
obstacle_run_duration_s = 0.0


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

    # Which side the oncoming traffic — and therefore the glare — is on. 
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
                       "(%s) — the ego may brake for it", actor.id, exc)

    spawn_dist = ego_tf.location.distance(spawn_loc)
    logger.info("[OBST] %s %s spawned at t=%.1f s, %.1f m ahead of the ego "
                "[anchor: %s], on the '%s' edge at (%.1f, %.1f); crossing %.2f m "
                "into the lane at %.1f m/s (%.2f s)",
                OBSTACLE_KIND, actor.id, OBSTACLE_SPAWN_AT_S, spawn_dist, anchor,
                OBSTACLE_LATERAL_EDGE, spawn_loc.x, spawn_loc.y, travel,
                OBSTACLE_SPEED, duration)
    return actor, control, duration




# =============================================================================
# DeepSight's cameras, live glare, and the stand-in agent
# =============================================================================

OUT_DIR = args.out or os.path.join('exports_closed_loop', 'severity_%g' % args.severity)
os.makedirs(OUT_DIR, exist_ok=True)
CLEAN_DIR = os.path.join(OUT_DIR, 'clean', 'CAM_FRONT')
os.makedirs(CLEAN_DIR, exist_ok=True)

# The agent: DeepSight's input path with glare added in tick(). With
# --front-only the three rear views are not rendered and their slots in the
# model input are black.
GlareDeepSightInputAgent = make_glare_agent_class()
agent_cameras = GLARE_CAMERAS if args.front_only else SURROUND_CAMERAS
agent = GlareDeepSightInputAgent(OUT_DIR, cameras=agent_cameras,
                                 model_view_every=args.model_view_every)


class SensorBuffer:
    """Holds the latest (frame, BGRA image) per camera and waits for a full set.

    Plays the leaderboard's role: before the agent runs, wait until every camera
    has delivered its image for this frame. Images are copied out of CARLA's
    buffer, as the leaderboard does.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._data = {}

    def put(self, sensor_id, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)).copy()
        with self._cond:
            self._data.setdefault(sensor_id, {})[image.frame] = arr
            buf = self._data[sensor_id]
            while len(buf) > 5:
                del buf[min(buf)]
            self._cond.notify_all()

    def wait_for(self, frame, sensor_ids, timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                if all(frame in self._data.get(s, {}) for s in sensor_ids):
                    return {s: self._data[s][frame] for s in sensor_ids}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)


sensor_buffer = SensorBuffer()
blueprints = world.get_blueprint_library()

# Spawn the agent's cameras exactly as the leaderboard would: only size and FOV
# are set, so exposure, bloom and lens flare are CARLA's defaults. (The
# open-loop script's manual exposure and extra bloom are deliberately NOT used:
# DeepSight would not see them.)
agent_sensors = []
for spec in agent.sensors():
    bp = blueprints.find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(spec['width']))
    bp.set_attribute('image_size_y', str(spec['height']))
    bp.set_attribute('fov', str(spec['fov']))
    cam = world.spawn_actor(bp, spec_transform(spec), attach_to=ego_vehicle)
    cam.listen(lambda image, sid=spec['id']: sensor_buffer.put(sid, image))
    agent_sensors.append(cam)
    logger.info("Spawned %s: %dx%d, FOV %g", spec['id'], spec['width'],
                spec['height'], spec['fov'])
camera_ids = [s['id'] for s in agent.sensors()]

# Optional 3rd-person context video, written as it goes (not held in memory).
CHASE_W, CHASE_H = 1280, 720
chase_camera = None
chase_writer = None
if args.chase_cam:
    bp = blueprints.find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(CHASE_W))
    bp.set_attribute('image_size_y', str(CHASE_H))
    bp.set_attribute('fov', '90')
    chase_camera = world.spawn_actor(
        bp, carla.Transform(carla.Location(x=-6.0, z=4.0), carla.Rotation(pitch=-15.0)),
        attach_to=ego_vehicle)
    chase_camera.listen(lambda image: sensor_buffer.put('CHASE', image))
    chase_writer = cv2.VideoWriter(os.path.join(OUT_DIR, 'chase_cam.mp4'),
                                   cv2.VideoWriter_fourcc(*'mp4v'), FPS, (CHASE_W, CHASE_H))

# The glare: seg + depth cameras matched to each front view, attached to the ego.
injector = GlareInjector(world, ego_vehicle, agent.sensors(),
                         severity=args.severity, strict=args.strict)
agent.use_glare_injector(injector)

# --- NPC behaviour and lights (unchanged from the open-loop script) ----------
for npc in npc_list:
    npc.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.update_vehicle_lights(npc, False)  # TM won't manage lights
    if npc.id in opposite_npc_ids:
        npc.set_light_state(HEADLIGHTS_ON)    # oncoming: the glare source
    else:
        npc.set_light_state(LIGHTS_OFF)       # same-direction NPCs stay dark
logger.info("Enabled autopilot for %d NPC vehicles", len(npc_list))

for npc in lead_npcs:
    try:
        traffic_manager.vehicle_percentage_speed_difference(npc, -12.0)  # ~12% faster
    except (AttributeError, RuntimeError):
        pass  # non-essential tuning; TM API varies by version

# The ego drives on Traffic Manager: the agent reproduces DeepSight's input but
# does not drive, so every severity sees the same trajectory for the same seed.
ego_vehicle.set_autopilot(True, traffic_manager.get_port())
traffic_manager.update_vehicle_lights(ego_vehicle, False)
ego_vehicle.set_light_state(EGO_LIGHTS)
try:
    route_locations = [wp.transform.location for wp in ego_route[::10]]  # every ~20 m
    traffic_manager.set_path(ego_vehicle, route_locations)
    logger.info("Ego route path set (%d waypoints toward spawn %d)",
                len(route_locations), DEST_SPAWN_IDX)
except Exception as e:
    logger.warning("traffic_manager.set_path unavailable (%s) — using plain autopilot", e)

TOTAL_TICKS = int(DURATION_S / FIXED_DELTA)
logger.info("Running for %.0f s (%d ticks), glare severity %g, output %s",
            DURATION_S, TOTAL_TICKS, args.severity, OUT_DIR)

world.set_weather(weather)
applied = world.get_weather()
logger.info("Confirmed weather: sun_altitude=%.1f, fog_density=%.1f, precip=%.1f",
            applied.sun_altitude_angle, applied.fog_density, applied.precipitation)
for npc in opposite_npcs:
    if npc.is_alive:
        npc.set_light_state(HEADLIGHTS_ON)

OBSTACLE_SPAWN_TICK = int(OBSTACLE_SPAWN_AT_S / FIXED_DELTA)
OBSTACLE_RETRY_TICKS = int(OBSTACLE_SPAWN_RETRY_S / FIXED_DELTA)


def ego_forward_speed():
    """Speed along the ego's heading, m/s: what the leaderboard's SPEED reports."""
    v = ego_vehicle.get_velocity()
    f = ego_vehicle.get_transform().get_forward_vector()
    return v.x * f.x + v.y * f.y + v.z * f.z


# --- Run --------------------------------------------------------------------
steps_skipped = 0
step_ms = []
run_wall_start = time.time()

try:
    for tick in range(TOTAL_TICKS):
        frame = world.tick()

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
                        obstacle.apply_control(obstacle_control)
                elif obs_elapsed <= obstacle_run_duration_s + 0.1:
                    if obstacle_control is not None:
                        obstacle.apply_control(carla.WalkerControl(speed=0.0))
                    elif OBSTACLE_HOLD_IN_LANE:
                        obstacle.disable_constant_velocity()
            except RuntimeError:
                pass

        if tick % 20 == 0:
            for npc in opposite_npcs:
                if npc.is_alive:
                    npc.set_light_state(HEADLIGHTS_ON)
            ego_vehicle.set_light_state(EGO_LIGHTS)

        # --- What the leaderboard does each tick: collect, then call the agent ---
        wanted = camera_ids + (['CHASE'] if chase_camera is not None else [])
        images = sensor_buffer.wait_for(frame, wanted)
        if images is None:
            steps_skipped += 1
            logger.warning("Tick %d: camera images for frame %d never arrived; "
                           "agent step skipped", tick, frame)
            continue

        input_data = {cid: (frame, images[cid]) for cid in camera_ids}
        input_data['SPEED'] = (frame, {'speed': ego_forward_speed()})
        timestamp = tick * FIXED_DELTA

        t0 = time.perf_counter()
        agent.run_step(input_data, timestamp)
        step_ms.append((time.perf_counter() - t0) * 1000.0)

        # The same CAM_FRONT frame WITHOUT glare, encoded the same way, for checking.
        cv2.imwrite(os.path.join(CLEAN_DIR, '%05d.jpg' % agent.step),
                    images['CAM_FRONT'][:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if chase_writer is not None:
            chase_writer.write(np.ascontiguousarray(images['CHASE'][:, :, :3]))

        if tick % 100 == 0:
            logger.info("%.1f s elapsed (%d/%d ticks), agent step %.0f ms",
                        tick * FIXED_DELTA, tick, TOTAL_TICKS,
                        float(np.mean(step_ms[-100:])) if step_ms else 0.0)
finally:
    run_wall_s = time.time() - run_wall_start
    logger.info("Finished run in %.0f s wall time — stopping sensors", run_wall_s)
    for s in agent_sensors + ([chase_camera] if chase_camera else []):
        try:
            s.stop()
        except RuntimeError:
            pass
    if chase_writer is not None:
        chase_writer.release()
    agent.destroy()        # writes glare_log.jsonl and removes the seg/depth cameras

if obstacle is None:
    logger.warning("[OBST] Obstacle never spawned between %.1f s and %.1f s. "
                   "Try a different OBSTACLE_AHEAD_M (now %.0f m).",
                   OBSTACLE_SPAWN_AT_S, OBSTACLE_SPAWN_AT_S + OBSTACLE_SPAWN_RETRY_S,
                   OBSTACLE_AHEAD_M)
else:
    logger.info("[OBST] Spawned at %.1f s (tick %d); in the ego's lane from %.1f s",
                obstacle_start_tick * FIXED_DELTA, obstacle_start_tick,
                obstacle_start_tick * FIXED_DELTA + obstacle_run_duration_s)


# =============================================================================
# Checks: did the glare reach the files the model reads?
# =============================================================================

n_steps = agent.step + 1
glare_log = injector.log
front_by_frame = {e['frame']: e for e in glare_log if e['camera'] == 'CAM_FRONT'}
step_frames = agent.frames                       # step -> CARLA frame

glared_steps, unchanged_ok, unchanged_bad, glared_same = [], 0, 0, 0
for step, frame in enumerate(step_frames):
    entry = front_by_frame.get(frame)
    lamps = entry['glaring_headlights'] if entry else 0
    with open(agent.frame_path('CAM_FRONT', step), 'rb') as fh:
        glared_bytes = fh.read()
    with open(os.path.join(CLEAN_DIR, '%05d.jpg' % step), 'rb') as fh:
        clean_bytes = fh.read()
    if lamps > 0:
        glared_steps.append(step)
        if glared_bytes == clean_bytes:
            glared_same += 1
    elif glared_bytes == clean_bytes:
        unchanged_ok += 1
    else:
        unchanged_bad += 1

# History: at step s, the four history slots must be CAM_FRONT from s-20..s-5,
# i.e. the same (glared) files written as current frames earlier.
history_bad = 0
history_checked = 0
for step in range(max(HISTORY_STEPS), n_steps, 10):
    with open(os.path.join(OUT_DIR, 'meta', '%04d.json' % step), encoding='utf-8') as fh:
        meta = json.load(fh)
    expected = [os.path.relpath(agent.frame_path('CAM_FRONT', step - k), OUT_DIR)
                for k in HISTORY_STEPS]
    history_checked += 1
    if [s['path'] for s in meta['slots'][:4]] != expected:
        history_bad += 1

glare_ms = [e['glare_ms'] for e in glare_log if 'glare_ms' in e]
wait_ms = [e['wait_ms'] for e in glare_log if 'wait_ms' in e]
glare_on = args.severity > 0

checks = [
    ("agent ran every tick", steps_skipped == 0,
     "%d steps, %d skipped" % (n_steps, steps_skipped)),
    ("no frame reached the agent without glare (seg/depth always arrived)",
     injector.missing_aux_frames == 0, "%d missing" % injector.missing_aux_frames),
    ("history slots point to the earlier CAM_FRONT files",
     history_bad == 0, "%d of %d sampled steps wrong" % (history_bad, history_checked)),
]
if glare_on:
    checks += [
        ("glare appeared in the model's CAM_FRONT input", len(glared_steps) > 0,
         "%d of %d steps had glaring headlights" % (len(glared_steps), n_steps)),
        ("every step with glaring headlights changed the saved file", glared_same == 0,
         "%d identical to clean" % glared_same),
        ("steps without glaring headlights are byte-identical to clean",
         unchanged_bad == 0, "%d identical, %d differ" % (unchanged_ok, unchanged_bad)),
    ]
else:
    checks += [
        ("severity 0: every saved CAM_FRONT file is byte-identical to clean",
         unchanged_ok == n_steps, "%d of %d identical" % (unchanged_ok, n_steps)),
    ]

logger.info("=" * 70)
all_pass = True
for name, ok, detail in checks:
    all_pass &= bool(ok)
    logger.info("%s  %s%s", "PASS" if ok else "FAIL", name, ("  (%s)" % detail) if detail else "")
logger.info("Agent step: mean %.0f ms.  Glare per camera: mean %.0f ms, plus %.0f ms "
            "waiting for seg/depth.",
            float(np.mean(step_ms)) if step_ms else 0.0,
            float(np.mean(glare_ms)) if glare_ms else 0.0,
            float(np.mean(wait_ms)) if wait_ms else 0.0)
logger.info("%s", "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
logger.info("=" * 70)

summary = {
    'severity': args.severity, 'seed': args.seed, 'duration_s': DURATION_S,
    'cameras': camera_ids, 'steps': n_steps, 'steps_skipped': steps_skipped,
    'steps_with_glare': len(glared_steps),
    'first_glare_step': glared_steps[0] if glared_steps else None,
    'missing_aux_frames': injector.missing_aux_frames,
    'mean_agent_step_ms': float(np.mean(step_ms)) if step_ms else None,
    'mean_glare_ms_per_camera': float(np.mean(glare_ms)) if glare_ms else None,
    'run_wall_s': run_wall_s,
    'checks': {name: bool(ok) for name, ok, _ in checks},
    'all_pass': bool(all_pass),
}
with open(os.path.join(OUT_DIR, 'run_summary.json'), 'w', encoding='utf-8') as fh:
    json.dump(summary, fh, indent=1)

# Side-by-side video of CAM_FRONT exactly as stored for the model: decoded from
# the quality-20 JPEGs, clean on the left, glared on the right.
compare_path = os.path.join(OUT_DIR, 'compare_CAM_FRONT.mp4')
writer = cv2.VideoWriter(compare_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (1600, 450))
for step in range(n_steps):
    clean = cv2.imread(os.path.join(CLEAN_DIR, '%05d.jpg' % step))
    glared = cv2.imread(agent.frame_path('CAM_FRONT', step))
    if clean is None or glared is None:
        continue
    left = cv2.resize(clean, (800, 450))
    right = cv2.resize(glared, (800, 450))
    cv2.putText(left, "CLEAN (as saved)", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "MODEL INPUT, severity %g" % args.severity, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    writer.write(np.hstack([left, right]))
writer.release()
logger.info("Wrote %s", compare_path)
if glared_steps:
    logger.info("First glared step: %d. Look at model_view/step_%05d.jpg and "
                "camera/CAM_FRONT/%05d.jpg", glared_steps[0],
                (glared_steps[0] // max(1, args.model_view_every)) * max(1, args.model_view_every),
                glared_steps[0])

# --- Cleanup ----------------------------------------------------------------
logger.info("Destroying actors")
for actor in agent_sensors + [chase_camera, ego_vehicle, obstacle] + npc_list:
    if actor is None:
        continue
    try:
        if actor.is_alive:
            actor.destroy()
    except RuntimeError:
        pass

try:
    world.apply_settings(original_settings)
    traffic_manager.set_synchronous_mode(False)
    world.tick()
    time.sleep(0.5)
except RuntimeError:
    pass

logger.info("Done")
