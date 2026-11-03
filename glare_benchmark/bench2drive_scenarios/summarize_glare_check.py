"""Per-route report for a run_glare_check.sh output folder.

    python3 summarize_glare_check.py ~/glare_check_runs/smoke/severity_1.0

For each route under <folder>/routes/ it reads glare_log.jsonl and the saved
frames, and checks what carla_script_closed_loop.py checks:

  * no frame reached the agent without its seg/depth (missing = 0)
  * every CAM_FRONT step with glaring headlights changed the saved file
  * every CAM_FRONT step without them is byte-identical to the clean copy

It also reports how much glare each route had, so routes with none can be set
aside. Writes summary.csv next to the routes and exits 1 if any check failed.
"""
import collections
import csv
import glob
import json
import os
import sys

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')


def summarize_route(route_dir):
    row = {'route': os.path.basename(route_dir)}
    log_path = os.path.join(route_dir, 'glare_log.jsonl')
    log = []
    if os.path.exists(log_path):
        with open(log_path, encoding='utf-8') as fh:
            log = [json.loads(line) for line in fh if line.strip()]

    # step -> CARLA frame, from the stand-in agent's meta files
    frame_of = {}
    for path in glob.glob(os.path.join(route_dir, 'meta', '*.json')):
        with open(path, encoding='utf-8') as fh:
            meta = json.load(fh)
        frame_of[meta['step']] = meta['frame']
    row['steps'] = len(frame_of)

    verdicts = collections.Counter()
    for cam in CAMERAS:
        entries = [e for e in log if e['camera'] == cam]
        row[cam + '_glared'] = sum(1 for e in entries if e['glaring_headlights'] > 0)
        row[cam + '_max_lamps'] = max((e['glaring_headlights'] for e in entries), default=0)
        for e in entries:
            verdicts.update(e.get('verdicts', {}))
    row['missing'] = sum(1 for e in log if e.get('aux_missing'))
    row['warmup'] = sum(1 for e in log if e.get('warmup'))
    row['detected'] = verdicts.get('detected', 0)
    row['not_detected'] = verdicts.get('not_detected', 0)
    row['facing_away'] = verdicts.get('facing_away', 0)
    row['too_far'] = verdicts.get('too_far', 0)

    # Byte checks on CAM_FRONT, as in the CARLA test script.
    front = {e['frame']: e for e in log if e['camera'] == 'CAM_FRONT'}
    glared_same = unglared_diff = 0
    for step, frame in frame_of.items():
        glared = os.path.join(route_dir, 'camera', 'CAM_FRONT', '%05d.jpg' % step)
        clean = os.path.join(route_dir, 'clean', 'CAM_FRONT', '%05d.jpg' % step)
        if not (os.path.exists(glared) and os.path.exists(clean)):
            continue
        with open(glared, 'rb') as a, open(clean, 'rb') as b:
            same = a.read() == b.read()
        lamps = front[frame]['glaring_headlights'] if frame in front else 0
        if lamps > 0 and same:
            glared_same += 1
        elif lamps == 0 and not same:
            unglared_diff += 1
    row['glared_but_unchanged'] = glared_same
    row['unglared_but_changed'] = unglared_diff

    row['pass'] = (row['steps'] > 0 and row['missing'] == 0
                   and glared_same == 0 and unglared_diff == 0)
    row['has_glare'] = row['CAM_FRONT_glared'] + row['CAM_FRONT_LEFT_glared'] + row['CAM_FRONT_RIGHT_glared'] > 0
    return row


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    route_dirs = sorted(d for d in glob.glob(os.path.join(root, 'routes', '*')) if os.path.isdir(d))
    if not route_dirs:
        sys.exit('No routes under %s/routes/' % root)
    rows = [summarize_route(d) for d in route_dirs]

    print('%-58s %6s %8s %8s %8s %5s %7s  %s' % ('route', 'steps', 'front', 'left', 'right',
                                                 'miss', 'warmup', 'checks'))
    for r in rows:
        print('%-58s %6d %8d %8d %8d %5d %7d  %s%s' % (
            r['route'][:58], r['steps'], r['CAM_FRONT_glared'], r['CAM_FRONT_LEFT_glared'],
            r['CAM_FRONT_RIGHT_glared'], r['missing'], r['warmup'],
            'PASS' if r['pass'] else 'FAIL', '' if r['has_glare'] else '  (no glare on this route)'))
    print('front/left/right = steps with glaring headlights on that camera')

    out = os.path.join(root, 'summary.csv')
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote %s' % out)
    sys.exit(0 if all(r['pass'] for r in rows) else 1)


if __name__ == '__main__':
    main()
