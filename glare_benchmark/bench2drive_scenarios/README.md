# Bench2Drive scenarios

The Bench2Drive night scenarios worth testing under oncoming headlight glare,
saved one file per route, plus one script that runs them with your glare on
DeepSight's cameras. For now the leaderboard's NpcAgent drives; no model is
loaded.

| File | |
| --- | --- |
| `run.py` | Everything: `extract`, `list`, `check`, `run`, `summary` |
| `glare_agent.py` | The leaderboard agent: NpcAgent drives, the cameras record with glare |
| `scenarios/` | Made by `run.py extract`: one route file per scenario, by family (kept out of git) |

## Setup on the Ubuntu machine (once)

You need CARLA 0.9.15 with AdditionalMaps (Town12/13), and a Bench2Drive
leaderboard. Use the `bench2drive/` folder of DeepSight's repo: its leaderboard
runs at 10 Hz, as DeepSight will later. Only the code is used, not the model.

```bash
git clone --depth 1 https://github.com/hotdogcheesewhite/DeepSight ~/DeepSight
```

```bash
echo 'export B2D_DIR=$HOME/DeepSight/bench2drive CARLA_ROOT=$HOME/CARLA_0.9.15 GLARE_OUT=$HOME/glare_runs' >> ~/.bashrc
```

Open a new terminal (or `source ~/.bashrc`), activate your CARLA conda env,
then from this folder:

```bash
python run.py extract
```

```bash
python run.py check
```

`check` lists anything missing. Two things it often flags:
- the Python packages CARLA's driver needs: `pip install networkx shapely`
- DeepSight's leaderboard starts CARLA on GPU adapter 4. On a machine with
  fewer GPUs, change `int(4)` to `int(0)` on line 208 of
  `$B2D_DIR/leaderboard/leaderboard/leaderboard_evaluator.py`.

## Run

```bash
python run.py run overtaking --glare 1.0
```

The target is a family, `all`, or route ids like `3410` or `3410,3436`
(`python run.py list` shows them). Options: `--out` (default `$GLARE_OUT`),
`--glare` (0 to 2), `--headlights`, `--gpu`, `--port`, `--no-strict`.

**Debug view.** Every 10 steps (`--debug-every`, 0 = off) each run saves
`debug/step_<step>.jpg`: the front frame with each lit car drawn where the
glare code places it, labelled with its id, verdict and distance (green =
detected, red = rejected). `glare_debug.jsonl` holds the numbers, including the
camera position the glare code uses next to where the camera should be.

**Only driving cars glare.** A car counts as driving once it has moved faster
than 1 m/s during the run. A car waiting at a light that drove there still
glares, a car pulling out of a parking spot starts glaring when it moves, and
a parked car never glares, even if Bench2Drive has its lights on.

**Headlights.** Bench2Drive's night traffic mostly drives with its headlights
off, so on its own there is little to glare. `--headlights low` (the default)
switches on low beams for every driving car within 90 m of the ego, each step,
and stops CARLA's Traffic Manager from switching them back off; `high` adds
high beams; `scene` keeps Bench2Drive's lighting. Each run records the mode in
its folder name, and `headlights.json` counts the driving, parked and lit
cars. Bench2Drive itself turns scenario cars' lights off beyond about 50 m of
the ego, so those light up only when closer; background traffic stays lit to
90 m. Keep Bench2Drive's own files unedited.

| Family | Scenarios | Why |
| --- | --- | --- |
| `overtaking` | 6 | Passing a blockage into oncoming traffic whose headlights are the glare |
| `junction_left_turns` | 6 | Yielding to a car its own headlights wash out |
| `other_junctions` | 12 | Other turns with oncoming traffic |
| `crossings` | 8 | Cyclists and pedestrians; glare overlaps them only by chance |
| `controls` | 7 | Highways and merges: little oncoming glare, results should not change |

Each route runs on its own (a crash stops only that route) into its own
folder, so runs never overwrite each other:

```
~/glare_runs/
├── 3410_AccidentTwoWays_Town13_glare1_low_20261103-143530/
│   ├── compare_CAM_FRONT.mp4   clean | glared front camera, the whole route
│   ├── model_view/             the 10 inputs as DeepSight would see them
│   ├── camera/  clean/  meta/  glare_log.jsonl  headlights.json  run.log  run_info.json
└── summary.csv
```

When the routes finish, `run.py` prints one line per run: steps with glaring
headlights on each front camera, missing frames, and PASS/FAIL. A run passes
when every frame got its segmentation and depth, every front frame with
glaring headlights changed, and every front frame without them is identical to
the clean copy. `python run.py summary <folder>` prints it again for any
folder, including the copy on your Mac.

NpcAgent obeys lights and stops for cars, but ignores pedestrians and cannot
overtake, so on blocked-lane routes it waits until the route times out. It
still sees the oncoming traffic while it waits.

Run inside `tmux` over SSH so a dropped connection does not stop a route.

## Our pop-out-from-glare scenarios

`run.py` also runs the scenarios in `../custom_scenarios` (family
`glare_popout`), once `python ../custom_scenarios/make_routes.py` has written
their routes. See that README.

```bash
python run.py run glare_popout --glare 1.0 --headlights scene
```

## Copy the videos to your Mac

On the Mac; copies each run's video and the summary, keeping folder names:

```bash
rsync -avm --include='*/' --include='compare_CAM_FRONT.mp4' --include='summary.csv' --exclude='*' <user>@<ubuntu-host>:glare_runs/ ~/glare_runs/
```

## Adding DeepSight later

`run.py` picks the agent from `DRIVERS`. DeepSight's agent with glare already
exists (`glare_qwen_agent.py` at the repo root). Adding it means a `deepsight`
entry pointing there, passing the checkpoint as the agent config, and reading
its output, which DeepSight writes under `Scenarios/<route>/` in the run folder.
