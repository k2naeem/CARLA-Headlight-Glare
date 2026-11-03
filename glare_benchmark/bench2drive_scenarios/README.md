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
`--glare` (0 to 2), `--gpu`, `--port`, `--no-strict`.

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
├── 3410_AccidentTwoWays_Town13_glare1_20261103-143530/
│   ├── compare_CAM_FRONT.mp4   clean | glared front camera, the whole route
│   ├── model_view/             the 10 inputs as DeepSight would see them
│   ├── camera/  clean/  meta/  glare_log.jsonl  run.log  run_info.json
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
