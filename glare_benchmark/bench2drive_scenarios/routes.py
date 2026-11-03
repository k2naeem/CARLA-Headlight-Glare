"""Which Bench2Drive routes the glare benchmark runs, grouped into families.

Bench2Drive's licence (CC BY-NC-ND 4.0) does not allow sharing modified
copies of its route files, so nothing here rewrites bench2drive220.xml. This
script reads your own copy, writes a small manifest of route ids
(night_routes.json), and hands run_family.sh the ids to pass to the
leaderboard's --routes-subset option.

    python3 routes.py build <DeepSight>/bench2drive/leaderboard/data/bench2drive220.xml
    python3 routes.py list                  # families and their routes
    python3 routes.py ids overtaking        # -> 1825,3410,...  (for --routes-subset)
    python3 routes.py ids smoke             # route 3514, the one night route in Dev10
    python3 routes.py ids 3410,3436         # explicit ids, checked against the manifest

A route is "night" when any of its weather keyframes has the sun below the
horizon. Each Bench2Drive route has exactly one scenario, and its family comes
from that scenario's type (FAMILIES below).
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, 'night_routes.json')

# Route 3514 (ParkingExit, Town13) is the only night route in Bench2Drive's
# Dev10 set, so it is the first thing to run on a new machine.
SMOKE_ROUTE = '3514'

# Scenario type -> family. Families 1-5 are the ones the benchmark reports;
# "other_night" runs in a full night sweep but is not a headline family.
FAMILIES = {
    # 1. Overtaking into oncoming traffic: the oncoming stream is the glare.
    'overtaking': [
        'AccidentTwoWays', 'ConstructionObstacleTwoWays', 'HazardAtSideLaneTwoWays',
        'ParkedObstacleTwoWays', 'VehicleOpensDoorTwoWays',
    ],
    # 2. Left turns and right of way: yield to a car its own headlights wash out.
    'junction_left_turns': [
        'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionLeftTurnEnterFlow',
        'SignalizedJunctionLeftTurn', 'SignalizedJunctionLeftTurnEnterFlow',
        'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight',
    ],
    # 3. Other junctions and turns with oncoming traffic.
    'other_junctions': [
        'NonSignalizedJunctionRightTurn', 'SignalizedJunctionRightTurn',
        'VanillaNonSignalizedTurn', 'VanillaNonSignalizedTurnEncounterStopsign',
        'VanillaSignalizedTurnEncounterGreenLight', 'VanillaSignalizedTurnEncounterRedLight',
        'VehicleTurningRoute', 'BlockedIntersection', 'InvadingTurn', 'T_Junction',
    ],
    # 4. Crossings: valid only on steps where the glare log shows glare while
    #    the cyclist or pedestrian is in view.
    'crossings': [
        'CrossingBicycleFlow', 'DynamicObjectCrossing', 'ParkingCrossingPedestrian',
        'PedestrianCrossing', 'VehicleTurningRoutePedestrian',
    ],
    # 5. Controls: little or no oncoming glare expected (divided highways,
    #    merges, an emergency vehicle from behind). Glare level should not
    #    change these results.
    'controls': [
        'HighwayCutIn', 'HighwayExit', 'MergerIntoSlowTraffic', 'MergerIntoSlowTrafficV2',
        'EnterActorFlow', 'YieldToEmergencyVehicle',
    ],
}
HEADLINE_FAMILIES = list(FAMILIES)
FAMILY_OF = {scenario: fam for fam, types in FAMILIES.items() for scenario in types}


def build(xml_path, out_path=MANIFEST):
    tree = ET.parse(xml_path)
    routes = []
    for route in tree.iter('route'):
        altitudes = [float(w.get('sun_altitude_angle')) for w in route.iter('weather')
                     if w.get('sun_altitude_angle') is not None]
        scenarios = [s.get('type') for s in route.iter('scenario')]
        scenario = scenarios[0] if scenarios else None
        night = bool(altitudes) and min(altitudes) < 0.0
        if night:
            family = FAMILY_OF.get(scenario, 'other_night')
        else:
            family = 'day'
        routes.append({'id': route.get('id'), 'town': route.get('town'),
                       'scenario': scenario, 'night': night,
                       'min_sun_altitude': min(altitudes) if altitudes else None,
                       'family': family})
    manifest = {
        'source': os.path.basename(xml_path),
        'smoke_route': SMOKE_ROUTE,
        'families': {fam: [r['id'] for r in routes if r['family'] == fam]
                     for fam in HEADLINE_FAMILIES + ['other_night', 'day']},
        'routes': routes,
    }
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=1)
    night = [r for r in routes if r['night']]
    print('%d routes, %d at night -> %s' % (len(routes), len(night), out_path))
    unknown = sorted({r['scenario'] for r in night if r['family'] == 'other_night'})
    print('night scenario types in other_night: %s' % ', '.join(unknown))
    return manifest


def load():
    if not os.path.exists(MANIFEST):
        sys.exit('No %s yet. Run: python3 routes.py build <bench2drive220.xml>' % MANIFEST)
    with open(MANIFEST, encoding='utf-8') as fh:
        return json.load(fh)


def ids_for(target, manifest):
    """Route ids for a family name, 'smoke', 'night', or a comma-separated id list."""
    fams = manifest['families']
    if target == 'smoke':
        return [manifest['smoke_route']]
    if target == 'night':
        return [r['id'] for r in manifest['routes'] if r['night']]
    if target in fams:
        return fams[target]
    known = {r['id'] for r in manifest['routes']}
    ids = [t.strip() for t in target.split(',') if t.strip()]
    bad = [i for i in ids if i not in known]
    if not ids or bad:
        sys.exit('Unknown family or route id(s): %s. Families: %s, smoke, night'
                 % (', '.join(bad) or target, ', '.join(fams)))
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='write night_routes.json from bench2drive220.xml')
    b.add_argument('xml')
    sub.add_parser('list', help='print the families and their routes')
    i = sub.add_parser('ids', help='comma-separated ids for --routes-subset')
    i.add_argument('target', help='a family, smoke, night, or ids like 3410,3436')
    args = ap.parse_args()

    if args.cmd == 'build':
        build(args.xml)
        return
    manifest = load()
    if args.cmd == 'ids':
        print(','.join(sorted(ids_for(args.target, manifest), key=int)))
        return
    by_id = {r['id']: r for r in manifest['routes']}
    for fam, ids in manifest['families'].items():
        if fam == 'day':
            print('\n%-20s %d routes (not run by default)' % (fam, len(ids)))
            continue
        print('\n%-20s %d routes' % (fam, len(ids)))
        for rid in sorted(ids, key=int):
            r = by_id[rid]
            print('  %-6s %-7s %s' % (rid, r['town'], r['scenario']))
    print('\nsmoke route: %s' % manifest['smoke_route'])


if __name__ == '__main__':
    main()
