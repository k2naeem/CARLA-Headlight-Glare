"""Run Bench2Drive night scenarios with headlight glare on the cameras.

One script for everything. Run it from the conda env that has the CARLA
0.9.15 Python API, on the Ubuntu GPU machine:

    python run.py extract                      # save the glare scenarios into scenarios/
    python run.py list                         # what was saved
    python run.py check                        # is the machine ready?
    python run.py run overtaking --glare 1.0   # run a family, one route at a time
    python run.py run 3410 --glare 1.0         # or one route (or 3410,3436, or all)
    python run.py summary ~/glare_runs         # glare report for every run in a folder

Our own pop-out-from-glare scenarios (../custom_scenarios) run the same way
once ../custom_scenarios/make_routes.py has written their routes:

    python run.py run glare_popout --glare 1.0 --headlights scene

Paths come from options or environment variables:
    --b2d    / B2D_DIR     the Bench2Drive folder: DeepSight's repo/bench2drive
                           (its leaderboard runs at 10 Hz, as DeepSight expects)
    --carla  / CARLA_ROOT  CARLA 0.9.15 with AdditionalMaps (Town12/13)
    --out    / GLARE_OUT   where runs are saved (default ~/glare_runs)

Every route runs as its own leaderboard call and is saved to its own folder,
<out>/<id>_<scenario>_<town>_glare<level>_<date-time>/, so a crash stops only
that route and runs never overwrite each other. The car is driven by the
leaderboard's NpcAgent (--driver npc); glare_agent.py records the cameras and,
with --headlights low|high (default low), switches on nearby traffic's
headlights, which Bench2Drive mostly leaves off at night.

Bench2Drive's license (CC BY-NC-ND 4.0) does not allow sharing its route files,
so scenarios/ is kept out of git: run `extract` on each machine.
"""
import argparse
import collections
import copy
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCENARIOS = os.path.join(HERE, 'scenarios')
INDEX = os.path.join(SCENARIOS, 'index.json')
CUSTOM = os.path.join(os.path.dirname(HERE), 'custom_scenarios')
CUSTOM_ROUTES = os.path.join(CUSTOM, 'routes')       # written by custom_scenarios/make_routes.py
CUSTOM_MODULES = ('glare_popout.py',)                # our scenario classes

# The Bench2Drive scenario types worth testing under oncoming glare, by family.
# A night route whose scenario type is not listed here is not saved.
FAMILIES = {
    # Passing a blockage into an oncoming stream whose headlights are the glare.
    'overtaking': ['AccidentTwoWays', 'ConstructionObstacleTwoWays', 'HazardAtSideLaneTwoWays',
                   'ParkedObstacleTwoWays', 'VehicleOpensDoorTwoWays'],
    # Yielding at left turns to a car its own headlights wash out.
    'junction_left_turns': ['NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionLeftTurnEnterFlow',
                            'SignalizedJunctionLeftTurn', 'SignalizedJunctionLeftTurnEnterFlow',
                            'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight'],
    # Other junctions and turns with oncoming traffic.
    'other_junctions': ['NonSignalizedJunctionRightTurn', 'SignalizedJunctionRightTurn',
                        'VanillaNonSignalizedTurn', 'VanillaNonSignalizedTurnEncounterStopsign',
                        'VanillaSignalizedTurnEncounterGreenLight',
                        'VanillaSignalizedTurnEncounterRedLight', 'VehicleTurningRoute',
                        'BlockedIntersection', 'InvadingTurn', 'T_Junction'],
    # Cyclists and pedestrians crossing; glare overlaps them only by chance.
    'crossings': ['CrossingBicycleFlow', 'DynamicObjectCrossing', 'ParkingCrossingPedestrian',
                  'PedestrianCrossing', 'VehicleTurningRoutePedestrian'],
    # Little oncoming glare expected: glare level should change nothing here.
    'controls': ['HighwayCutIn', 'HighwayExit', 'MergerIntoSlowTraffic', 'MergerIntoSlowTrafficV2',
                 'EnterActorFlow', 'YieldToEmergencyVehicle'],
}
FAMILY_OF = {scenario: fam for fam, types in FAMILIES.items() for scenario in types}

# Leaderboard agent per driver. DeepSight goes here later (glare_qwen_agent.py
# at the repo root already wraps it); its output layout differs, see README.
DRIVERS = {'npc': os.path.join(HERE, 'glare_agent.py')}

GLARE_CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')


# --- paths --------------------------------------------------------------------

def b2d_dir(args):
    path = args.b2d or os.environ.get('B2D_DIR')
    if not path:
        sys.exit('Set --b2d or B2D_DIR to the Bench2Drive folder (DeepSight repo/bench2drive).')
    return os.path.abspath(os.path.expanduser(path))


def carla_dir(args):
    path = args.carla or os.environ.get('CARLA_ROOT')
    if not path:
        sys.exit('Set --carla or CARLA_ROOT to your CARLA 0.9.15 folder.')
    return os.path.abspath(os.path.expanduser(path))


def out_dir(args):
    return os.path.abspath(os.path.expanduser(args.out or os.environ.get('GLARE_OUT') or '~/glare_runs'))


def load_index():
    """Every saved scenario: Bench2Drive's (run.py extract) and ours (make_routes.py).
    Each entry's 'file' is an absolute path; 'custom' marks ours."""
    scenarios = []
    for folder in (SCENARIOS, CUSTOM_ROUTES):
        path = os.path.join(folder, 'index.json')
        if os.path.exists(path):
            with open(path, encoding='utf-8') as fh:
                for s in json.load(fh)['scenarios']:
                    scenarios.append(dict(s, file=os.path.join(folder, s['file']),
                                          custom=folder == CUSTOM_ROUTES))
    if not scenarios:
        sys.exit('No saved scenarios yet. Run: python run.py extract '
                 '(and python ../custom_scenarios/make_routes.py for ours)')
    return scenarios


# --- extract --------------------------------------------------------------------

def cmd_extract(args):
    """Save each glare-relevant night route as its own route file."""
    source = os.path.join(b2d_dir(args), 'leaderboard', 'data', 'bench2drive220.xml')
    if not os.path.exists(source):
        sys.exit('Not found: %s' % source)
    if os.path.isdir(SCENARIOS):
        shutil.rmtree(SCENARIOS)

    saved = []
    for route in ET.parse(source).iter('route'):
        altitudes = [float(w.get('sun_altitude_angle')) for w in route.iter('weather')
                     if w.get('sun_altitude_angle') is not None]
        scenario = next((s.get('type') for s in route.iter('scenario')), None)
        family = FAMILY_OF.get(scenario)
        if not altitudes or min(altitudes) >= 0.0 or family is None:
            continue                       # daytime, or not a glare scenario
        name = '%s_%s_%s' % (route.get('id'), scenario, route.get('town'))
        path = os.path.join(SCENARIOS, family, name + '.xml')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        root = ET.Element('routes')
        root.append(copy.deepcopy(route))
        ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
        saved.append({'id': route.get('id'), 'name': name, 'family': family,
                      'scenario': scenario, 'town': route.get('town'),
                      'file': os.path.relpath(path, SCENARIOS)})

    with open(INDEX, 'w', encoding='utf-8') as fh:
        json.dump({'source': source, 'scenarios': saved}, fh, indent=1)
    counts = collections.Counter(s['family'] for s in saved)
    print('Saved %d scenarios to %s' % (len(saved), SCENARIOS))
    for fam in FAMILIES:
        print('  %-20s %d' % (fam, counts[fam]))


def families(scenarios):
    return list(dict.fromkeys(s['family'] for s in scenarios))


def cmd_list(args):
    scenarios = load_index()
    for fam in families(scenarios):
        rows = [s for s in scenarios if s['family'] == fam]
        print('\n%s (%d)' % (fam, len(rows)))
        for s in rows:
            print('  %s' % s['name'])


def select(target, scenarios):
    """Scenarios for 'all', a family, or comma-separated route ids."""
    if target == 'all':
        return scenarios
    if target in families(scenarios):
        return [s for s in scenarios if s['family'] == target]
    wanted = [t.strip() for t in target.split(',') if t.strip()]
    by_id = {s['id']: s for s in scenarios}
    missing = [w for w in wanted if w not in by_id]
    if missing or not wanted:
        sys.exit('Unknown: %s. Use all, a family (%s) or saved route ids (python run.py list).'
                 % (', '.join(missing) or target, ', '.join(families(scenarios))))
    return [by_id[w] for w in wanted]


# --- check ---------------------------------------------------------------------

def cmd_check(args):
    fails = []

    def report(ok, text, warn=False):
        print('  %s  %s' % ('ok  ' if ok else ('WARN' if warn else 'FAIL'), text))
        if not ok and not warn:
            fails.append(text)

    b2d, carla = b2d_dir(args), carla_dir(args)
    evaluator = os.path.join(b2d, 'leaderboard', 'leaderboard', 'leaderboard_evaluator.py')
    report(os.path.exists(evaluator), 'Bench2Drive leaderboard in %s' % b2d)
    report(os.path.exists(os.path.join(b2d, 'leaderboard', 'leaderboard', 'autoagents', 'npc_agent.py')),
           'NpcAgent found')
    report(os.path.exists(INDEX), 'scenarios saved (python run.py extract)')
    report(os.path.exists(os.path.join(CUSTOM_ROUTES, 'index.json')),
           'pop-out-from-glare routes (python ../custom_scenarios/make_routes.py)', warn=True)
    if os.path.exists(evaluator):
        with open(evaluator, encoding='utf-8') as fh:
            fixed_gpu = 'graphicsadapter={int(4)}' in fh.read()
        report(not fixed_gpu, 'CARLA GPU adapter: the evaluator hard-codes 4; '
               'on fewer GPUs change int(4) on its line 208', warn=True)

    report(os.path.exists(os.path.join(carla, 'CarlaUE4.sh')), 'CarlaUE4.sh in %s' % carla)
    maps = os.path.join(carla, 'CarlaUE4', 'Content', 'Carla', 'Maps')
    for town in ('Town12', 'Town13'):
        present = os.path.isdir(maps) and any(n.startswith(town) for n in os.listdir(maps))
        report(present, '%s installed (AdditionalMaps_0.9.15)' % town)

    for f in ('glare_injector.py', 'glare_mask_closed_loop.py', 'deepsight_input_agent.py'):
        report(os.path.exists(os.path.join(REPO, f)), '%s in %s' % (f, REPO))

    sys.path[:0] = [os.path.join(carla, 'PythonAPI', 'carla')]
    for name in ('carla', 'cv2', 'numpy', 'PIL', 'networkx', 'shapely'):
        try:
            __import__(name)
            report(True, 'import %s' % name)
        except Exception as exc:
            report(False, 'import %s (%s)' % (name, exc.__class__.__name__))
    report(shutil.which('nvidia-smi') is not None, 'NVIDIA driver (nvidia-smi)')

    print('\nReady. Next: python run.py run all --glare 1.0' if not fails
          else '\n%d check(s) failed.' % len(fails))
    return 1 if fails else 0


# --- run -----------------------------------------------------------------------

def scenario_overlay(b2d):
    """A scenario_runner root whose srunner/scenarios/ is Bench2Drive's plus ours.

    The leaderboard finds scenario classes by importing every
    $SCENARIO_RUNNER_ROOT/srunner/scenarios/*.py, so linking ours in beside
    Bench2Drive's lets a route use them without editing Bench2Drive.
    """
    root = tempfile.mkdtemp(prefix='glare_srunner_')
    folder = os.path.join(root, 'srunner', 'scenarios')
    os.makedirs(folder)
    source = os.path.join(b2d, 'scenario_runner', 'srunner', 'scenarios')
    for name in os.listdir(source):
        if name.endswith('.py'):
            os.symlink(os.path.join(source, name), os.path.join(folder, name))
    for name in CUSTOM_MODULES:
        os.symlink(os.path.join(CUSTOM, name), os.path.join(folder, name))
    return root


def leaderboard_env(args, b2d, carla, run_dir, route_file, scenario_root):
    env = dict(os.environ)
    paths = [os.path.join(carla, 'PythonAPI'), os.path.join(carla, 'PythonAPI', 'carla'),
             b2d, os.path.join(b2d, 'leaderboard'), os.path.join(b2d, 'leaderboard', 'team_code'),
             os.path.join(b2d, 'scenario_runner'), REPO, HERE]
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [env.get('PYTHONPATH')] + paths))
    env.update({
        'CARLA_ROOT': carla,
        'SCENARIO_RUNNER_ROOT': scenario_root,
        'LEADERBOARD_ROOT': os.path.join(b2d, 'leaderboard'),
        'CHALLENGE_TRACK_CODENAME': 'SENSORS',
        'IS_BENCH2DRIVE': 'True',
        'ROUTES': route_file,
        'SAVE_PATH': run_dir,
        'GLARE_OUTPUT': run_dir,
        'GLARE_SEVERITY': str(args.glare),
        'GLARE_HEADLIGHTS': args.headlights,
        'GLARE_DEBUG_EVERY': str(args.debug_every),
        'GLARE_STRICT': '0' if args.no_strict else '1',
        'CUDA_VISIBLE_DEVICES': str(args.gpu),
    })
    return env


def run_one(args, b2d, carla, scenario, out_root):
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(out_root, '%s_glare%g_%s_%s' % (scenario['name'], args.glare,
                                                           args.headlights, stamp))
    os.makedirs(run_dir)
    route_file = scenario['file']
    overlay = scenario_overlay(b2d) if scenario.get('custom') else None
    scenario_root = overlay or os.path.join(b2d, 'scenario_runner')
    cmd = [sys.executable, os.path.join('leaderboard', 'leaderboard', 'leaderboard_evaluator.py'),
           '--routes=%s' % route_file, '--repetitions=1', '--track=SENSORS',
           '--checkpoint=%s' % os.path.join(run_dir, 'results.json'),
           '--agent=%s' % DRIVERS[args.driver], '--agent-config=%s' % args.driver,
           '--debug=0', '--resume=True', '--port=%d' % args.port,
           '--traffic-manager-port=%d' % args.tm_port, '--gpu-rank=%d' % args.gpu]

    print('\n=== %s' % os.path.basename(run_dir), flush=True)
    started = datetime.datetime.now().isoformat(timespec='seconds')
    with open(os.path.join(run_dir, 'run.log'), 'w', encoding='utf-8') as log:
        try:
            proc = subprocess.Popen(cmd, cwd=b2d,
                                    env=leaderboard_env(args, b2d, carla, run_dir, route_file, scenario_root),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:          # show it live and keep it in run.log
                sys.stdout.write(line)
                log.write(line)
            status = proc.wait()
        finally:
            if overlay:
                shutil.rmtree(overlay, ignore_errors=True)
    with open(os.path.join(run_dir, 'run_info.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict(scenario, glare_level=args.glare, headlights=args.headlights,
                       driver=args.driver, started=started,
                       finished=datetime.datetime.now().isoformat(timespec='seconds'),
                       exit_status=status), fh, indent=1)

    # CARLA processes outlive a crashed run; Bench2Drive ships a cleaner.
    cleaner = os.path.join(b2d, 'tools', 'clean_carla.sh')
    if not args.no_clean and os.path.exists(cleaner):
        subprocess.run(['bash', cleaner], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return status


def cmd_run(args):
    b2d, carla, out_root = b2d_dir(args), carla_dir(args), out_dir(args)
    if not os.path.exists(os.path.join(b2d, 'leaderboard', 'leaderboard', 'leaderboard_evaluator.py')):
        sys.exit('No Bench2Drive leaderboard in %s (python run.py check)' % b2d)
    scenarios = select(args.target, load_index())
    os.makedirs(out_root, exist_ok=True)
    print('%d scenario(s), glare %g, headlights %s, driver %s -> %s'
          % (len(scenarios), args.glare, args.headlights, args.driver, out_root))

    failed = [s['name'] for s in scenarios if run_one(args, b2d, carla, s, out_root) != 0]
    print()
    summarize(out_root)
    if failed:
        print('Leaderboard exited with an error on: %s (see run.log)' % ', '.join(failed))
    return 1 if failed else 0


# --- summary -------------------------------------------------------------------

def summarize_run(run_dir, root):
    """Glare counts and the byte checks of carla_script_closed_loop.py for one run."""
    row = {'run': os.path.relpath(run_dir, root)}
    info = os.path.join(run_dir, 'run_info.json')
    row['exit_status'] = json.load(open(info, encoding='utf-8')).get('exit_status') if os.path.exists(info) else None

    log = []
    log_path = os.path.join(run_dir, 'glare_log.jsonl')
    if os.path.exists(log_path):
        with open(log_path, encoding='utf-8') as fh:
            log = [json.loads(line) for line in fh if line.strip()]
    frame_of = {}
    meta_dir = os.path.join(run_dir, 'meta')
    for name in os.listdir(meta_dir):
        with open(os.path.join(meta_dir, name), encoding='utf-8') as fh:
            meta = json.load(fh)
        frame_of[meta['step']] = meta['frame']
    row['steps'] = len(frame_of)
    for cam in GLARE_CAMERAS:
        row[cam] = sum(1 for e in log if e['camera'] == cam and e['glaring_headlights'] > 0)
    row['missing'] = sum(1 for e in log if e.get('aux_missing'))

    # A front frame with glaring headlights must differ from the clean copy;
    # one without must be byte-identical.
    front = {e['frame']: e['glaring_headlights'] for e in log if e['camera'] == 'CAM_FRONT'}
    row['glare_unchanged'] = row['clean_changed'] = 0
    for step, frame in frame_of.items():
        a = os.path.join(run_dir, 'camera', 'CAM_FRONT', '%05d.jpg' % step)
        b = os.path.join(run_dir, 'clean', 'CAM_FRONT', '%05d.jpg' % step)
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        with open(a, 'rb') as fa, open(b, 'rb') as fb:
            same = fa.read() == fb.read()
        if front.get(frame, 0) > 0 and same:
            row['glare_unchanged'] += 1
        elif front.get(frame, 0) == 0 and not same:
            row['clean_changed'] += 1
    row['pass'] = (row['steps'] > 0 and row['missing'] == 0
                   and row['glare_unchanged'] == 0 and row['clean_changed'] == 0)
    return row


def summarize(root):
    runs = sorted(d for d, subdirs, _ in os.walk(root) if 'meta' in subdirs)
    if not runs:
        print('No runs under %s' % root)
        return 1
    rows = [summarize_run(d, root) for d in runs]
    width = max(len(r['run']) for r in rows)
    print('%-*s %6s %6s %6s %6s %5s  %s' % (width, 'run', 'steps', 'front', 'left', 'right', 'miss', 'checks'))
    for r in rows:
        notes = []
        if r['CAM_FRONT'] + r['CAM_FRONT_LEFT'] + r['CAM_FRONT_RIGHT'] == 0:
            notes.append('no glare')
        if r['exit_status']:
            notes.append('exit %s, see run.log' % r['exit_status'])
        print('%-*s %6d %6d %6d %6d %5d  %s%s' % (
            width, r['run'], r['steps'], r['CAM_FRONT'], r['CAM_FRONT_LEFT'], r['CAM_FRONT_RIGHT'],
            r['missing'], 'PASS' if r['pass'] else 'FAIL', ('  (%s)' % ', '.join(notes)) if notes else ''))
    print('front/left/right = steps with glaring headlights on that camera')
    with open(os.path.join(root, 'summary.csv'), 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(r['pass'] for r in rows) else 1


def cmd_summary(args):
    return summarize(os.path.abspath(os.path.expanduser(args.folder)))


# --- main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument('--b2d', help='Bench2Drive folder (default $B2D_DIR)')
    paths.add_argument('--carla', help='CARLA 0.9.15 folder (default $CARLA_ROOT)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('extract', parents=[paths], help='save the glare scenarios into scenarios/')
    sub.add_parser('list', help='list the saved scenarios')
    sub.add_parser('check', parents=[paths], help='check that this machine can run them')
    r = sub.add_parser('run', parents=[paths], help='run scenarios, one leaderboard call each')
    r.add_argument('target', help='all, a family (glare_popout for ours), or route ids like 3410,3436')
    r.add_argument('--glare', type=float, default=1.0, help='glare level, 0 to 2 (default 1.0)')
    r.add_argument('--out', help='output folder (default $GLARE_OUT or ~/glare_runs)')
    r.add_argument('--headlights', choices=['low', 'high', 'scene'], default='low',
                   help="force low or high beams on nearby traffic (default low); "
                        "'scene' keeps Bench2Drive's lighting, where most cars are dark")
    r.add_argument('--debug-every', type=int, default=10,
                   help='save debug/ frames with each lit car and its glare verdict every N steps (0 = off)')
    r.add_argument('--driver', choices=sorted(DRIVERS), default='npc')
    r.add_argument('--port', type=int, default=30000)
    r.add_argument('--tm-port', type=int, default=50000)
    r.add_argument('--gpu', type=int, default=0)
    r.add_argument('--no-strict', action='store_true',
                   help='pass a frame through without glare instead of stopping when seg/depth is late')
    r.add_argument('--no-clean', action='store_true', help="don't kill CARLA processes between routes")
    s = sub.add_parser('summary', help='glare report for every run under a folder')
    s.add_argument('folder')

    args = ap.parse_args()
    handler = {'extract': cmd_extract, 'list': cmd_list, 'check': cmd_check,
               'run': cmd_run, 'summary': cmd_summary}[args.cmd]
    sys.exit(handler(args) or 0)


if __name__ == '__main__':
    main()
