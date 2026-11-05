"""Write route files for the pop-out-from-glare scenarios (glare_popout.py).

Run on the Ubuntu machine, from the conda env with the CARLA 0.9.15 Python API:

    python make_routes.py                       # default towns, 2 spots each
    python make_routes.py --towns Town01,Town03 --count 3

No CARLA server is needed: each town's OpenDRIVE file is read straight from
$CARLA_ROOT. A spot is a straight stretch of two-way road, one lane each way,
with no junction for the whole scenario. Each spot gets one route per
variant (VARIANTS), at the same place, so they compare like for like. Routes
go to routes/glare_popout/<id>_<variant>_<town>.xml plus routes/index.json,
which bench2drive_scenarios/run.py reads:

    python run.py run glare_popout --glare 1.0 --headlights scene

Every scenario parameter is written into the route file with its default, so
a route can be tuned by editing it (see glare_popout.py for what each means).
These routes are ours, not Bench2Drive's, so they can go in git.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = os.path.join(HERE, 'routes')
FAMILY = 'glare_popout'

PEDESTRIAN = {'distance': 90, 'car_ahead': 35, 'car_speed': 9, 'release_dist': 30, 'behind': 0.5,
              'ped_offset': 'auto', 'adversary': 'walker', 'speed': 2.5, 'arrive_before': 0.5, 'hold': 3,
              'beams': 'low'}

# Variant name: (scenario type, first route id, parameters written into its route files).
# Ids start at 91000 so they never clash with Bench2Drive's.
VARIANTS = {
    # Street lights off: lit only by headlights, the hard case.
    'PedestrianDark': ('GlarePedestrianPopOut', 91000, dict(PEDESTRIAN, street_lights='off')),
    'Overtake': ('GlareOncomingOvertake', 92000, {
        'distance': 60, 'lead_ahead': 50, 'lead_speed': 5, 'catch_gap': 25, 'pull_out_gap': 5,
        'swing_dist': 75, 'pass_speed': 15, 'return_gap': 3, 'street_lights': 'on', 'beams': 'low'}),
    # Bench2Drive's street lights: the pedestrian is lit from above, the easier case.
    'PedestrianLit': ('GlarePedestrianPopOut', 93000, dict(PEDESTRIAN, street_lights='on')),
    # A second lit car 30 m behind, timed so its glare covers the step-out itself.
    'PedestrianTwoCarsDark': ('GlarePedestrianPopOut', 94000,
                              dict(PEDESTRIAN, second_gap=30, second_dist='auto', street_lights='off')),
}
LEAD_IN_M = 30.0      # route before the trigger point, so the ego is up to speed
BEFORE_M = LEAD_IN_M + max(p['distance'] for _, _, p in VARIANTS.values())   # route start to M
AFTER_M = 90.0        # M to the route end; also room for the oncoming cars to wait (the
                      # second car of PedestrianTwoCarsDark starts ~70-85 m past M)
MAX_TURN_DEG = 8.0    # how much the road may bend over the whole stretch
MIN_SPACING_M = 250.0 # between two spots in one town

# Clear night: no sun, no fog or rain to soften the glare.
NIGHT = {'cloudiness': '10.0', 'fog_density': '0.0', 'precipitation': '0.0',
         'precipitation_deposits': '0.0', 'wetness': '0.0', 'sun_altitude_angle': '-90.0',
         'sun_azimuth_angle': '0.0', 'wind_intensity': '10.0'}


def load_map(carla, carla_root, town):
    pattern = os.path.join(carla_root, 'CarlaUE4', 'Content', 'Carla', 'Maps', '**', 'OpenDrive',
                           town + '.xodr')
    found = glob.glob(pattern, recursive=True)
    if not found:
        sys.exit('No OpenDRIVE file for %s under %s' % (town, carla_root))
    with open(found[0], encoding='utf-8') as fh:
        return carla.Map(town, fh.read())


def oncoming_if_two_way(carla, wp):
    """wp's oncoming lane if wp is the only lane its way, beside one oncoming lane; else None."""
    if wp.lane_type != carla.LaneType.Driving or wp.is_junction:
        return None
    right = wp.get_right_lane()
    if right is not None and right.lane_type == carla.LaneType.Driving:
        return None
    left = wp.get_left_lane()
    if left is None or left.lane_type != carla.LaneType.Driving or left.lane_id * wp.lane_id > 0:
        return None
    far = left.get_right_lane()
    if far is not None and far.lane_type == carla.LaneType.Driving:
        return None
    return left


def walk(carla, wp, length, forward, step=2.0):
    """The waypoint `length` m along wp's lane if the road stays straight and two-way; else None."""
    yaw = wp.transform.rotation.yaw
    travelled = 0.0
    while travelled < length:
        found = wp.next(step) if forward else wp.previous(step)
        if len(found) != 1:
            return None                      # the lane forks or ends
        wp = found[0]
        travelled += step
        turn = (wp.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0
        if abs(turn) > MAX_TURN_DEG or oncoming_if_two_way(carla, wp) is None:
            return None
    return wp


def find_spots(carla, town_map, count):
    """Up to `count` meet points M, at least MIN_SPACING_M apart, sidewalks first."""
    candidates = []
    for wp in town_map.generate_waypoints(5.0):
        onc = oncoming_if_two_way(carla, wp)
        if onc is None:
            continue
        start, end = walk(carla, wp, BEFORE_M, False), walk(carla, wp, AFTER_M, True)
        if start is None or end is None:
            continue
        far = onc.get_right_lane()
        sidewalk = far is not None and far.lane_type == carla.LaneType.Sidewalk
        candidates.append((not sidewalk, round(wp.transform.location.x), round(wp.transform.location.y),
                           wp, start, end, sidewalk))
    candidates.sort(key=lambda c: c[:3])

    spots = []
    for *_, wp, start, end, sidewalk in candidates:
        loc = wp.transform.location
        if all(loc.distance(s['meet'].transform.location) >= MIN_SPACING_M for s in spots):
            spots.append({'meet': wp, 'start': start, 'end': end, 'sidewalk': sidewalk})
        if len(spots) == count:
            break
    return spots


def position(parent, tag, wp, yaw=False):
    loc = wp.transform.location
    attrs = {'x': '%.1f' % loc.x, 'y': '%.1f' % loc.y, 'z': '%.1f' % loc.z}
    if yaw:
        attrs['yaw'] = '%.1f' % wp.transform.rotation.yaw
    return ET.SubElement(parent, tag, attrs)


def route_xml(route_id, town, scenario, params, spot):
    trigger = spot['meet'].previous(float(params['distance']))[0]
    routes = ET.Element('routes')
    route = ET.SubElement(routes, 'route', {'id': str(route_id), 'town': town})
    waypoints = ET.SubElement(route, 'waypoints')
    for key in ('start', 'meet', 'end'):
        position(waypoints, 'position', spot[key])
    scenarios = ET.SubElement(route, 'scenarios')
    node = ET.SubElement(scenarios, 'scenario', {'name': '%s_1' % scenario, 'type': scenario})
    position(node, 'trigger_point', trigger, yaw=True)
    for name, value in params.items():
        ET.SubElement(node, name, {'value': str(value)})
    weathers = ET.SubElement(route, 'weathers')
    for percent in ('0', '100'):
        ET.SubElement(weathers, 'weather', dict(NIGHT, route_percentage=percent))
    if hasattr(ET, 'indent'):         # Python 3.9+; older ones write it on one line
        ET.indent(routes)
    return ET.ElementTree(routes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--carla', help='CARLA 0.9.15 folder (default $CARLA_ROOT)')
    ap.add_argument('--towns', default='Town01,Town02,Town03,Town05,Town10HD',
                    help='comma-separated towns (default %(default)s)')
    ap.add_argument('--count', type=int, default=2, help='spots per town (default 2)')
    args = ap.parse_args()

    carla_root = os.path.abspath(os.path.expanduser(args.carla or os.environ.get('CARLA_ROOT') or ''))
    if not os.path.isdir(carla_root):
        sys.exit('Set --carla or CARLA_ROOT to your CARLA 0.9.15 folder.')
    sys.path.insert(0, os.path.join(carla_root, 'PythonAPI', 'carla'))
    import carla

    out = os.path.join(ROUTES, FAMILY)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    saved, k = [], 0
    for town in [t.strip() for t in args.towns.split(',') if t.strip()]:
        spots = find_spots(carla, load_map(carla, carla_root, town), args.count)
        print('%-9s %d spot(s)' % (town, len(spots)))
        for spot in spots:
            k += 1
            meet = spot['meet'].transform.location
            for variant, (scenario, first_id, params) in VARIANTS.items():
                route_id = first_id + k
                name = '%d_%s_%s' % (route_id, variant, town)
                path = os.path.join(out, name + '.xml')
                route_xml(route_id, town, scenario, params, spot).write(
                    path, encoding='utf-8', xml_declaration=True)
                saved.append({'id': str(route_id), 'name': name, 'family': FAMILY, 'scenario': scenario,
                              'variant': variant, 'town': town, 'file': os.path.relpath(path, ROUTES),
                              'meet': [round(meet.x, 1), round(meet.y, 1)], 'sidewalk': spot['sidewalk']})
                print('   %s  M at (%.0f, %.0f)%s' % (name, meet.x, meet.y,
                                                       '' if spot['sidewalk'] else '  no sidewalk'))

    with open(os.path.join(ROUTES, 'index.json'), 'w', encoding='utf-8') as fh:
        json.dump({'source': 'custom_scenarios/make_routes.py', 'scenarios': saved}, fh, indent=1)
    if not saved:
        sys.exit('No straight two-way stretch of %.0f m found; try other --towns.' % (BEFORE_M + AFTER_M))
    print('Saved %d routes to %s' % (len(saved), out))


if __name__ == '__main__':
    main()
