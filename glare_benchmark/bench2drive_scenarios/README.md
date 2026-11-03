# Bench2Drive scenarios

Bench2Drive night routes, run with headlight glare and no new scenario code.
53 of the 220 routes in `bench2drive220.xml` are at night.

Two ways to run them:

1. **Glare check, no model** (`run_glare_check.sh`). The leaderboard's own
   `NpcAgent` drives the route; `glare_check_agent.py` passes DeepSight's six
   cameras through your glare and saves them exactly as DeepSight would read
   them, plus a clean copy. Use this to see that the glare works on real
   Bench2Drive scenes before running the model.
2. **DeepSight with glare** (`run_family.sh`, `run_sweep.sh`). Runs the real
   model through `glare_qwen_agent.py` and gets Bench2Drive scores.

| File | What it does |
| --- | --- |
| `routes.py` | Groups the night routes into families and prints route ids for the leaderboard |
| `night_routes.json` | The families and route ids, built from `bench2drive220.xml` |
| `check_setup.sh` | Checks the machine: DeepSight's leaderboard, CARLA and its Town12/13 maps, Python packages, GPU |
| `glare_check_agent.py` | Leaderboard agent for the glare check: NpcAgent driving + DeepSight's input path with glare |
| `run_glare_check.sh` | Glare check on one family, the smoke route or listed ids, at one glare level |
| `summarize_glare_check.py` | Per-route report of a glare check: glare per camera, missing frames, byte checks |
| `run_family.sh` | DeepSight with glare on one family at one glare level |
| `run_sweep.sh` | DeepSight with glare on several families and levels, with repeats |
| `leaderboard_env.sh` | Leaderboard settings shared by the run scripts |

## Families

| Family | Routes | What it tests |
| --- | --- | --- |
| `overtaking` | 6 | Passing a blockage into an oncoming stream whose headlights are the glare |
| `junction_left_turns` | 6 | Yielding to oncoming cars at left turns; the car to yield to is the glare source |
| `other_junctions` | 12 | Right turns and signalized or stop-sign turns with oncoming traffic |
| `crossings` | 8 | Cyclists and pedestrians crossing. Counts only on steps where `glare_log.jsonl` shows glare while they are in view |
| `controls` | 7 | Highways, merges, an emergency vehicle from behind: glare should change little |
| `other_night` | 14 | The remaining night routes (same-direction obstacles, cut-ins, interurban flows); run with `night` |

`python3 routes.py list` prints every route with its town and scenario.
The night routes span Town03, 04, 05, 06, 10HD, 12, 13 and 15; most are in
Town12 and Town13.

Bench2Drive's license (CC BY-NC-ND 4.0) does not allow sharing modified route
files, so nothing here rewrites them: `night_routes.json` holds only route ids,
and the runs pass them to the leaderboard's `--routes-subset` option against
your own copy of `bench2drive220.xml`.

## Set up the Ubuntu GPU machine (once)

1. CARLA 0.9.15 and its extra maps (Town12/13 come with AdditionalMaps):
   ```bash
   wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
   ```
   ```bash
   wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
   ```
   Extract the first, put the second in its `Import/` folder and run
   `ImportAssets.sh` there.
2. A Python environment with the CARLA 0.9.15 Python API, `opencv-python`,
   `numpy`, `Pillow`, `networkx` and `shapely`. For DeepSight runs, also its
   own packages (Qwen2.5-VL, `qwen_vl_utils`) and a checkpoint.
3. This repo on branch `bench2drive-glare-agent`, and DeepSight's repo for its
   leaderboard (it runs at 10 Hz, like DeepSight expects).
4. Point the scripts at them and check:
   ```bash
   export DEEPSIGHT_DIR=~/DeepSight CARLA_ROOT=~/CARLA_0.9.15
   ```
   ```bash
   bash check_setup.sh --clone
   ```
   `--clone` clones DeepSight into `$DEEPSIGHT_DIR` if it is not there.
   `MODEL_PATH` is needed only for DeepSight runs.

**GPU number.** DeepSight's `leaderboard_evaluator.py` starts CARLA with a
fixed `-graphicsadapter=4` (line 208). On a machine with fewer GPUs, change
`int(4)` there to your GPU's adapter number; Bench2Drive's README explains the
numbering. `check_setup.sh` warns while it is still 4.

**Over SSH.** Run inside `tmux` or `screen` so a dropped connection does not
stop a route. The leaderboard starts CARLA off-screen, so no display is needed.

## Glare check (no model)

```bash
OUT_ROOT=/data/glare_check bash run_glare_check.sh smoke 1.0
```

```bash
python3 summarize_glare_check.py /data/glare_check/smoke/severity_1.0
```

Then a family, e.g. `bash run_glare_check.sh overtaking 1.0`. Without
`OUT_ROOT`, output goes to `~/glare_check_runs/`.

Each route lands in `<OUT_ROOT>/<family>/severity_<level>/routes/<route>/`:
`camera/` (what DeepSight would read), `clean/CAM_FRONT/`, `meta/`,
`model_view/`, `glare_log.jsonl`. The summary marks a route PASS when no frame
missed its segmentation and depth, every front frame with glaring headlights
changed, and every front frame without them is byte-identical to the clean
copy. It also flags routes with no glare at all.

The driver obeys lights and stops for cars but ignores pedestrians and cannot
overtake, so it may wait behind a blocked lane until the route times out. The
scores mean nothing here; the images are the point.

## DeepSight with glare

Needs `MODEL_PATH` as well.

```bash
OUT_ROOT=/data/glare_runs bash run_family.sh smoke 1.0
```

```bash
LEVELS="0 1.0 2.0" REPEATS=3 bash run_sweep.sh
```

Output goes to `<OUT_ROOT>/<family>/severity_<level>/` (default
`~/glare_benchmark_runs/`; sweeps add `rep_<n>/`):

- `results.json`: Bench2Drive's scores per route (driving score, success, infractions)
- `Scenarios/<route>/`: DeepSight's own output per route: the camera JPEGs it
  read, `meta/<step>.json` with its prompt and answer, and `glare_log.jsonl`

Compare each glare level against the same family at level 0. The leaderboard
starts CARLA itself; `run_sweep.sh` calls DeepSight's `tools/clean_carla.sh`
between runs, since a crashed run leaves CARLA running.
