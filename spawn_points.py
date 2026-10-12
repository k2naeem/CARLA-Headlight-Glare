"""Visualise a CARLA map's spawn points in the simulator.

Draws each spawn point's index, position and forward direction as debug markers
visible in the spectator view, and prints the coordinates to stdout. Use this to
pick the EGO_SPAWN_IDX / DEST_SPAWN_IDX values in Carla_script.py.

Run: python3 spawn_points.py   (requires a running CARLA server)
"""
import logging
import time

import carla

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MARKER_LIFETIME_S = 120.0

# Towns shipped with a standard CARLA install. TOWN_INDEX selects one.
town_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05',
             'Town06', 'Town07', 'Town08', 'Town09', 'Town10',
             'Town11', 'Town12']
TOWN_INDEX = 6


logger.info("Connecting to CARLA at localhost:2000")
client = carla.Client('localhost', 2000)
client.set_timeout(60.0)

logger.info("Loading world: %s", town_list[TOWN_INDEX])
client.load_world(town_list[TOWN_INDEX])

world = client.get_world()
logger.info("World loaded")

spawn_points = world.get_map().get_spawn_points()
logger.info("Found %d spawn points", len(spawn_points))

debug = world.debug
for i, sp in enumerate(spawn_points):
    debug.draw_string(
        sp.location + carla.Location(z=1.0), str(i),
        draw_shadow=False, color=carla.Color(0, 255, 0),
        life_time=MARKER_LIFETIME_S,
    )
    debug.draw_point(
        sp.location + carla.Location(z=0.5), size=0.1,
        color=carla.Color(255, 0, 0), life_time=MARKER_LIFETIME_S,
    )
    forward = sp.get_forward_vector()
    debug.draw_arrow(
        sp.location + carla.Location(z=0.5),
        sp.location + carla.Location(x=forward.x * 2, y=forward.y * 2, z=0.5),
        thickness=0.1, arrow_size=0.2,
        color=carla.Color(0, 0, 255), life_time=MARKER_LIFETIME_S,
    )
    logger.info("Spawn %3d: x=%8.2f, y=%8.2f, z=%6.2f",
                i, sp.location.x, sp.location.y, sp.location.z)

logger.info("Drew %d markers - check the CARLA spectator view", len(spawn_points))
logger.info("Markers visible for %.0f seconds. Ctrl+C to exit.", MARKER_LIFETIME_S)

try:
    time.sleep(MARKER_LIFETIME_S)
except KeyboardInterrupt:
    pass
