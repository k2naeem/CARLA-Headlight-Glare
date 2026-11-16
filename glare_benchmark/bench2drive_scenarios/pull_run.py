"""Gather the important parts of DeepSight runs into one small folder to copy off the machine.

Run it on the machine that ran DeepSight, then copy one folder to your laptop:

    python pull_run.py ~/glare_runs/94001_*_deepsight_*        # one or more run folders
    scp -r <user>@<machine>:glare_pulls ~/Downloads/            # on the laptop

For each run folder it writes glare_pulls/<run name>/ with:
    summary.txt            driving score, infractions, how many frames were glared, the
                           scenario moments, and DeepSight's brake/throttle and reason
                           around each
    reasoning.txt          every step: game time, controls, glaring headlights, speed of
                           the plan, and DeepSight's one-line reason (its <think> text)
    CAM_FRONT.mp4         what DeepSight saw from the front camera, at 10 fps
    key/                   CAM_FRONT at 1 s, 0.5 s before and at each scenario moment,
                           each with a .txt of DeepSight's reasoning, controls, plan and
                           how many headlights glared on that frame
    results.json, run_info.json, scenario_events.jsonl, glare_log.jsonl   copied as is

A glare run and its glare-0 run give key frames with the same names, so they can be
compared side by side. Running it again on a run replaces that run's folder.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

SIM_HZ = 10                    # DeepSight's leaderboard runs CARLA at 10 Hz: one frame per 0.1 s
OFFSETS_S = (-1.0, -0.5, 0.0)  # key frames relative to each scenario moment
COPIED = ('results.json', 'run_info.json', 'scenario_events.jsonl')


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def results_lines(path):
    """Driving score and infractions from Bench2Drive's results.json."""
    if not os.path.exists(path):
        return ['no results.json: the run did not finish']
    records = json.load(open(path, encoding='utf-8')).get('_checkpoint', {}).get('records', [])
    if not records:
        return ['results.json has no route record: the run did not finish']
    rec = records[0]
    scores = rec.get('scores', {})
    lines = ['status %s' % rec.get('status'),
             'driving score %.1f  (route completion %.1f x penalty %.2f)'
             % (scores.get('score_composed', 0), scores.get('score_route', 0), scores.get('score_penalty', 0))]
    hits = {k: v for k, v in rec.get('infractions', {}).items() if v}
    lines += ['%s: %s' % (k, '; '.join(map(str, v))) for k, v in hits.items()] or ['no infractions']
    return lines


def reason(meta):
    """DeepSight's one-line reason: the text inside <think>...</think>."""
    text = str(meta.get('output', ''))
    found = re.search(r'<think>(.*?)</think>', text, re.S)
    return ' '.join((found.group(1) if found else text.split('\n')[0]).split())


def glare_by_frame(log):
    """CAM_FRONT glaring-headlight count per CARLA frame."""
    return {e['frame']: e.get('glaring_headlights', 0) for e in log if e.get('camera') == 'CAM_FRONT'}


def write_video(cam_dir, path):
    try:
        import cv2
    except ImportError:
        return 'no video: OpenCV is missing'
    frames = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
    if not frames:
        return 'no video: no CAM_FRONT frames'
    h, w = cv2.imread(frames[0]).shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), SIM_HZ, (w, h))
    for f in frames:
        out.write(cv2.imread(f))
    out.release()
    return 'CAM_FRONT.mp4: %d frames' % len(frames)


def write_reasoning(metas, events, glare, path):
    """One line per step, with each scenario moment marked where it happens."""
    timed = [e for e in events if 'time_s' in e and 'frame' in e]
    anchor = timed[0] if timed else None      # turns game time into a CARLA frame for the glare log
    marks = list(timed)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('step  time_s  brake throttle steer  glare  plan_m  reason\n')
        fh.write('(glare: glaring headlights on CAM_FRONT, - if not logged; '
                 'plan_m: how far the planned path reaches, short = slowing or stopping)\n\n')
        for m in metas:
            while marks and marks[0]['time_s'] <= m['timestamp'] + 0.05:
                fh.write('---- %s at %.1f s ----\n' % (marks[0]['event'], marks[0]['time_s']))
                marks.pop(0)
            n = None
            if anchor:
                n = glare.get(anchor['frame'] + round((m['timestamp'] - anchor['time_s']) * SIM_HZ))
            plan = m.get('planning_trajectory') or []
            reach = max((abs(p[1]) for p in plan), default=0.0) if plan and len(plan[0]) > 1 else 0.0
            fh.write('%4d  %6.1f  %5.2f %8.2f %5.2f  %5s  %6.1f  %s\n'
                     % (m['step'], m['timestamp'], m.get('brake', 0), m.get('throttle', 0), m.get('steer', 0),
                        '-' if n is None else n, reach, reason(m)))


def pull(run_dir, dest_root):
    run_dir = os.path.abspath(run_dir.rstrip('/'))
    ds_dirs = [d for d in glob.glob(os.path.join(run_dir, 'Scenarios', '*')) if os.path.isdir(os.path.join(d, 'meta'))]
    if not ds_dirs:
        print('skip %s: no Scenarios/<route>/meta folder (not a DeepSight run?)' % run_dir)
        return
    ds = ds_dirs[0]
    dest = os.path.join(dest_root, os.path.basename(run_dir))
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.join(dest, 'key'))
    for name in COPIED:
        if os.path.exists(os.path.join(run_dir, name)):
            shutil.copy(os.path.join(run_dir, name), dest)
    if os.path.exists(os.path.join(ds, 'glare_log.jsonl')):
        shutil.copy(os.path.join(ds, 'glare_log.jsonl'), dest)

    events = read_jsonl(os.path.join(run_dir, 'scenario_events.jsonl'))
    log = read_jsonl(os.path.join(ds, 'glare_log.jsonl'))
    glare = glare_by_frame(log)
    metas = []
    for f in sorted(glob.glob(os.path.join(ds, 'meta', '*.json'))):
        m = json.load(open(f, encoding='utf-8'))
        m['step'] = int(os.path.splitext(os.path.basename(f))[0])
        metas.append(m)

    info = {}
    if os.path.exists(os.path.join(run_dir, 'run_info.json')):
        info = json.load(open(os.path.join(run_dir, 'run_info.json'), encoding='utf-8'))
    summary = ['run %s' % os.path.basename(run_dir),
               'glare %s, size %s, driver %s' % (info.get('glare_level'), info.get('glare_size'), info.get('driver')), '']
    summary += results_lines(os.path.join(run_dir, 'results.json')) + ['']
    glared = sum(1 for n in glare.values() if n > 0)
    summary.append('CAM_FRONT frames with glare: %d of %d' % (glared, len(glare)) if log
                   else 'no glare_log.jsonl: glare was off (glare 0) or the run stopped early')
    summary.append('')

    for e in events:
        if not metas or 'time_s' not in e:
            continue
        summary.append('%s at %.1f s  %s' % (e['event'], e['time_s'],
                                            {k: v for k, v in e.items() if k not in ('event', 'frame', 'time_s')}))
        for off in OFFSETS_S:
            m = min(metas, key=lambda m: abs(m['timestamp'] - (e['time_s'] + off)))
            frame = e['frame'] + round((m['timestamp'] - e['time_s']) * SIM_HZ)
            n = glare.get(frame)
            name = '%s_%+.1fs' % (e['event'], off)
            img = os.path.join(ds, 'camera', 'CAM_FRONT', '%05d.jpg' % m['step'])
            if os.path.exists(img):
                shutil.copy(img, os.path.join(dest, 'key', name + '.jpg'))
            controls = 'brake %.2f  throttle %.2f  steer %.2f' % (m.get('brake', 0), m.get('throttle', 0), m.get('steer', 0))
            glare_txt = 'glaring headlights on CAM_FRONT: %s' % ('no log' if n is None else n)
            summary.append('   %+.1f s  step %4d  %s  %s\n            %s' % (off, m['step'], controls, glare_txt, reason(m)))
            with open(os.path.join(dest, 'key', name + '.txt'), 'w', encoding='utf-8') as fh:
                fh.write('%s, %+.1f s: game time %.1f s, step %d, CARLA frame %d\n'
                         % (e['event'], off, m['timestamp'], m['step'], frame))
                fh.write('%s\n%s\n' % (controls, glare_txt))
                fh.write('planned trajectory: %s\n\n' % m.get('planning_trajectory'))
                fh.write(str(m.get('output', '')))

    write_reasoning(metas, events, glare, os.path.join(dest, 'reasoning.txt'))
    summary += ['', 'reasoning.txt: DeepSight\'s reason at all %d steps' % len(metas), write_video(os.path.join(ds, 'camera', 'CAM_FRONT'), os.path.join(dest, 'CAM_FRONT.mp4'))]
    with open(os.path.join(dest, 'summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(summary) + '\n')
    print('\n'.join(summary))
    print('-> %s\n' % dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run folders (the ones holding results.json and Scenarios/)')
    ap.add_argument('--dest', default='~/glare_pulls', help='where to write (default %(default)s)')
    args = ap.parse_args()
    dest_root = os.path.expanduser(args.dest)
    for run in args.runs:
        if not os.path.isdir(run):
            sys.exit('not a folder: %s' % run)
        pull(run, dest_root)


if __name__ == '__main__':
    main()
