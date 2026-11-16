"""Gather the important parts of DeepSight runs into one small folder to copy off the machine.

Run it on the machine that ran DeepSight, then copy one folder to your laptop:

    python pull_run.py ~/glare_runs/94001_*_deepsight_*        # one or more run folders
    scp -r <user>@<machine>:glare_pulls ~/Downloads/            # on the laptop

For each run folder it writes glare_pulls/<run name>/ with:
    summary.txt            driving score, route completion, any crashes, and DeepSight's
                           reasoning over the run: one line each time its reason (the
                           <think> text) or its braking changes, with the steps and times
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
    lines = ['Driving score     %.1f' % scores.get('score_composed', 0),
             'Route completion  %.1f%%' % scores.get('score_route', 0)]
    # min_speed_infractions are Bench2Drive's notes on speed versus traffic, not crashes.
    hits = {k: v for k, v in rec.get('infractions', {}).items() if v and k != 'min_speed_infractions'}
    lines += ['%s: %s' % (k, '; '.join(map(str, v))) for k, v in hits.items()] or ['Infractions       none']
    return lines


def reason(meta):
    """DeepSight's one-line reason: the text inside <think>...</think>."""
    text = str(meta.get('output', ''))
    found = re.search(r'<think>(.*?)</think>', text, re.S)
    return ' '.join((found.group(1) if found else text.split('\n')[0]).split())


# DeepSight writes its reason in Chinese from a small set of phrases; these give the gist in English.
GLOSS = [('行人', 'pedestrian'), ('横穿', 'crossing'), ('对向来车', 'oncoming car'), ('道路狭窄', 'narrow road'),
         ('会车', 'passing risk'), ('STOP', 'STOP sign'), ('交叉车流', 'cross traffic'), ('前方车辆', 'car ahead'),
         ('追尾', 'rear-end risk'), ('能见度低', 'low visibility'), ('昏暗', 'dark'), ('视野受限', 'limited view'),
         ('碰撞', 'collision risk'), ('红灯', 'red light'), ('施工', 'construction'), ('障碍', 'obstacle'),
         ('停车等待', 'STOP AND WAIT'), ('保持当前车道', 'keep lane'), ('减速', 'slow down'), ('加速', 'speed up'),
         ('变道', 'change lane'), ('绕行', 'go around')]


def gloss(text):
    """English keywords for a reason, in the order they appear, e.g. 'Hard: pedestrian, crossing, slow down'."""
    found = sorted((text.find(zh), en) for zh, en in GLOSS if zh in text)
    level = text.split('.')[0]
    return '%s: %s' % (level, ', '.join(en for _, en in found)) if found else level


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


def timeline(metas):
    """DeepSight's reason over the run: one line each time its reason or braking changes."""
    lines = ['DeepSight\'s reasoning (1 step = 0.1 s; a new line whenever the reason or the braking changes)',
             '']
    rows = []
    for m in metas:
        key = (reason(m), 'BRAKE' if m.get('brake', 0) >= 0.5 else 'drive')
        if rows and rows[-1][0] == key:
            rows[-1][2] = m
        else:
            rows.append([key, m, m])
    for (text, action), first, last in rows:
        lines.append('%5.1f-%5.1f s  steps %3d-%3d  %-5s  %s'
                     % (first['timestamp'], last['timestamp'], first['step'], last['step'], action, gloss(text)))
        if ':' in gloss(text):          # show DeepSight's own words under the keywords
            lines.append('%37s%s' % ('', text))
    return lines


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

    for e in events:
        if not metas or 'time_s' not in e:
            continue
        for off in OFFSETS_S:
            m = min(metas, key=lambda m: abs(m['timestamp'] - (e['time_s'] + off)))
            frame = e['frame'] + round((m['timestamp'] - e['time_s']) * SIM_HZ)
            n = glare.get(frame)
            name = '%s_%+.1fs' % (e['event'], off)
            img = os.path.join(ds, 'camera', 'CAM_FRONT', '%05d.jpg' % m['step'])
            if os.path.exists(img):
                shutil.copy(img, os.path.join(dest, 'key', name + '.jpg'))
            with open(os.path.join(dest, 'key', name + '.txt'), 'w', encoding='utf-8') as fh:
                fh.write('%s, %+.1f s: game time %.1f s, step %d, CARLA frame %d\n'
                         % (e['event'], off, m['timestamp'], m['step'], frame))
                fh.write('brake %.2f  throttle %.2f  steer %.2f\n' % (m.get('brake', 0), m.get('throttle', 0), m.get('steer', 0)))
                fh.write('glaring headlights on CAM_FRONT: %s\n' % ('no log' if n is None else n))
                fh.write('planned trajectory: %s\n\n' % m.get('planning_trajectory'))
                fh.write(str(m.get('output', '')))
    write_video(os.path.join(ds, 'camera', 'CAM_FRONT'), os.path.join(dest, 'CAM_FRONT.mp4'))

    summary = [os.path.basename(run_dir), ''] + results_lines(os.path.join(run_dir, 'results.json')) + ['']
    summary += timeline(metas)
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
