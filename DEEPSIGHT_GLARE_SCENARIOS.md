# Running DeepSight in the glare scenarios

This guide runs [DeepSight](https://github.com/hotdogcheesewhite/DeepSight)
through our night scenarios with live headlight glare on its cameras. Two
scripts do the work: `glare_benchmark/custom_scenarios/make_routes.py` writes
the routes, and `glare_benchmark/bench2drive_scenarios/run.py` runs them.

In each scenario, the hazard appears inside an oncoming car's glare:
- a pedestrian stepping out from behind the car, or
- a car or motorcycle pulling out of its glare into the ego's lane.

The glare is added to DeepSight's camera images inside the agent's `tick()`,
before DeepSight saves and reads them. So its current views and its history
frames both see the glare. Nothing in DeepSight or Bench2Drive is edited.

DeepSight is by its authors:
[hotdogcheesewhite/DeepSight](https://github.com/hotdogcheesewhite/DeepSight),
Apache License 2.0. This repo doesn't include it; you clone it yourself
below. Please cite their paper if you use it.

## What you need

- **Machine:** Ubuntu with an NVIDIA GPU (CARLA needs both).
- **CARLA 0.9.15:** our scenarios use Town01, 02, 03, 05 and 10HD. Bench2Drive's
  own routes also need AdditionalMaps (Town12/13).
- **DeepSight:**
  - the DeepSight repo, with the Python packages its README lists (torch,
    transformers, qwen_vl_utils, ...);
  - a DeepSight model checkpoint folder (what DeepSight's `TEAM_CONFIG` points to).
- **This repo:** run all the commands below from it.

Use one conda env that has both CARLA's Python API and DeepSight's packages.

## Setup (once)

**1. Clone both repos.**

```bash
git clone https://github.com/k2naeem/CARLA-Headlight-Glare.git ~/CARLA-Headlight-Glare
```

```bash
git clone --depth 1 https://github.com/hotdogcheesewhite/DeepSight ~/DeepSight
```

**2. Set the paths.** Change these to where things are on your machine:

```bash
echo 'export B2D_DIR=$HOME/DeepSight/bench2drive CARLA_ROOT=$HOME/CARLA_0.9.15 GLARE_OUT=$HOME/glare_runs DEEPSIGHT_CKPT=$HOME/ckpt/checkpoint-20000' >> ~/.bashrc
```

Then open a new terminal and activate the conda env.

**3. Install the two extra packages and check the machine.** DeepSight's env
already has numpy and OpenCV. Don't install `requirements.txt` into it, because
its pinned numpy could clash with DeepSight's torch.

```bash
pip install networkx shapely
```

```bash
cd ~/CARLA-Headlight-Glare/glare_benchmark/bench2drive_scenarios && python run.py check
```

`check` prints FAIL or WARN for anything missing. One warning matters on most
machines: DeepSight's leaderboard starts CARLA on GPU 4. With fewer GPUs, change
`int(4)` to `int(0)` on line 208 of
`$B2D_DIR/leaderboard/leaderboard/leaderboard_evaluator.py`.

You don't need to copy `glare_qwen_agent.py` into DeepSight; `run.py` loads
it from this repo.

**4. Write the scenario routes.** No CARLA server is needed for this.

```bash
cd ../custom_scenarios && python make_routes.py
```

It picks two straight two-way roads per town. Each road gets every scenario
below, so the ids run x001 to x010.

## Run DeepSight

From `glare_benchmark/bench2drive_scenarios`:

```bash
python run.py run glare_popout --driver deepsight --glare 1.0 --glare-size 2
```

That runs every scenario with glare. To get the clean baseline, run the same
routes with no glare:

```bash
python run.py run glare_popout --driver deepsight --glare 0
```

To run one scenario, give its id (`python run.py list` shows them all):

```bash
python run.py run 94001 --driver deepsight --glare 1.0 --glare-size 2
```

Each route runs as its own leaderboard call, so a crash stops only that
route. Run inside `tmux` over SSH, so a dropped connection doesn't stop the run.

| Option | Default | |
| --- | --- | --- |
| `--checkpoint` | `$DEEPSIGHT_CKPT` | DeepSight model folder |
| `--glare` | 1.0 | Glare brightness, 0 (off) to 2 |
| `--glare-size` | 1 | 2 makes the glare twice as large |
| `--gpu` | 0 | GPU for CARLA and DeepSight |
| `--port`, `--tm-port` | 30000, 50000 | Change these to run two jobs at once |
| `--out` | `$GLARE_OUT` | Where run folders go |

## The scenarios

| Ids | Name | What happens |
| --- | --- | --- |
| 91001+ | PedestrianDark | A pedestrian appears behind a lit oncoming car, then steps into the ego's lane 0.5 s before impact. Street lights off. |
| 93001+ | PedestrianLit | The same under street lights (the easier case) |
| 94001+ | PedestrianTwoCarsDark | The same plus a second lit car just behind the pedestrian, so its glare covers the step-out |
| 92001+ | Overtake | A car hidden in a slow lit car's glare pulls out to overtake, into the ego's lane, then pulls back in |
| 95001+ | OvertakeHeadOn | It pulls out at 55 m and stays in the ego's lane: a head-on course with no bail-out |
| 96001+ | OvertakeHeadOnMoto | The same with a motorcycle (one headlamp) |
| 97001+ | OvertakeHeadOnDark | OvertakeHeadOn with the street lights off |
| 98001+ | OvertakeHeadOnMotoDark | OvertakeHeadOnMoto with the street lights off |

Every route also runs Bench2Drive's background traffic. You can change any
scenario by editing parameters in its route XML (speeds, distances, beams,
lights). [custom_scenarios/README.md](glare_benchmark/custom_scenarios/README.md)
lists them.

## Running your own scenarios

`run.py` runs your own scenarios with the same glare and DeepSight setup. Today
that takes a few manual steps, because `run.py` only reads two route lists:
- `bench2drive_scenarios/scenarios/index.json`, written by `run.py extract`;
- `custom_scenarios/routes/index.json`, written by `make_routes.py`.

It also only adds one scenario file of ours, `glare_popout.py`, to
Bench2Drive's scenarios.

**If your route uses Bench2Drive's own scenario types** (for example
`PedestrianCrossing` or `AccidentTwoWays`):

1. Put your route XML in its own folder under
   `glare_benchmark/custom_scenarios/routes/`, for example
   `routes/mine/90001_MyRoute_Town05.xml`.
2. Add an entry for it to the `scenarios` list in
   `custom_scenarios/routes/index.json`:

   ```json
   {"id": "90001", "name": "90001_MyRoute_Town05", "family": "mine",
    "scenario": "PedestrianCrossing", "town": "Town05", "file": "mine/90001_MyRoute_Town05.xml"}
   ```

   `file` is relative to `routes/`. `id` and `family` are what you type after
   `run.py run`. Pick ids no other route uses.
3. Run it like any other route:

   ```bash
   python run.py run 90001 --driver deepsight --glare 1.0 --glare-size 2
   ```

**If you wrote your own scenario class** (a `.py` file, like our
`glare_popout.py`), also do this:

4. Put the file in `glare_benchmark/custom_scenarios/`.
5. Add its file name to `CUSTOM_MODULES` near the top of `run.py`:
   `CUSTOM_MODULES = ('glare_popout.py', 'my_scenario.py')`.

Routes in `custom_scenarios/routes/index.json` are run with a temporary copy of
Bench2Drive's scenario folder that also links in every file in
`CUSTOM_MODULES`. That way the leaderboard finds your class without any file in
Bench2Drive being edited.

**Things to watch:**
- **`make_routes.py` rewrites `custom_scenarios/routes/index.json`.** Running
  it again removes your entries, so add them back afterwards. It only deletes
  `routes/glare_popout/`, so your XML files in their own folder are kept.
- **Only vehicles with their headlights on glare.** With `--driver deepsight`,
  each scene keeps its own lighting, and Bench2Drive leaves most background
  cars' headlights off at night. Your scenario needs to turn on the headlights
  of the cars that should glare. `glare_popout.py` does this for its cars with
  `set_light_state`.
- **Run it at night.** Set a negative `sun_altitude_angle` in the route's
  weather; [make_routes.py](glare_benchmark/custom_scenarios/make_routes.py)'s
  `NIGHT` settings are a good start.
- **Use towns you have installed.** Town12 and Town13 need CARLA's
  AdditionalMaps.

Everything else works as it does for our scenarios: your own DeepSight weights
(`--checkpoint`) or code (`B2D_DIR`), the glare options, the run folders, and
`pull_run.py`.

## What a run gives you

```
~/glare_runs/
└── 94001_PedestrianTwoCarsDark_Town03_glare1x2_scene_deepsight_<time>/
    ├── results.json        Bench2Drive's result: driving score, collisions, route completion
    ├── run.log             the leaderboard's output
    ├── run_info.json       scenario, glare level, driver, exit status
    └── Scenarios/<route>_<time>/
        ├── ...             DeepSight's own output (its glared camera frames, predictions)
        └── glare_log.jsonl per camera frame: how many headlights glared
```

To see what glare does to DeepSight, compare `results.json` and DeepSight's
output between the `glare1x2` run and the `glare0` run of the same route.

**Copying a run to your laptop.** `pull_run.py` gathers the important parts of
one or more runs into `~/glare_pulls/<run>/`. That folder holds:
- `summary.txt`: the score, infractions, glared-frame count, and the controls at each scenario moment.
- `CAM_FRONT.mp4`
- `key/`: front-camera frames from 1 s before, 0.5 s before and at each moment, each with
  DeepSight's reasoning and the glaring-headlight count.

Run it on the GPU machine:

```bash
python pull_run.py ~/glare_runs/94001_*_deepsight_*
```

Then copy the folder from the laptop:

```bash
scp -r <user>@<machine>:glare_pulls ~/Downloads/
```

**Labels.** The per-frame labels come only from runs with the default driver,
NpcAgent. These include the clean copy of every frame, the clean-vs-glared
comparison video, and how much glare hides the pedestrian or overtaker in
each frame (`road_user_glare.jsonl`, `hazard_glare.jsonl`). The routes are
the same, so run them once without `--driver deepsight` to get them:

```bash
python run.py run glare_popout --glare 1.0 --glare-size 2 --headlights scene
```

`python run.py summary ~/glare_runs` reports those NpcAgent runs.

## How the glare is decided

A vehicle glares in a camera frame only when all of these hold:
- its low or high beams are on;
- it is within 70 m;
- it faces the camera;
- the camera can see it (segmentation and depth agree on at least 150 pixels);
- the lamp itself isn't hidden behind something.

Cars and trucks glare from two lamps, motorcycles from one, bicycles never.
The glare is added to DeepSight's three front cameras: CAM_FRONT, CAM_FRONT_LEFT
and CAM_FRONT_RIGHT.

Our scenario cars keep their headlights on the whole time. They start glaring
once they are within 70 m. With DeepSight, the background traffic keeps
Bench2Drive's own lighting, and `run.py` sets `--headlights scene`.

## Troubleshooting

- **The run stops at the first step with "DeepSight needs these environment
  variables set".** You ran DeepSight's own script. Use `run.py`, which sets
  `SAVE_PATH`, `IS_BENCH2DRIVE` and `ROUTES`.
- **"No module named team_code".** `B2D_DIR` must be DeepSight's `bench2drive`
  folder.
- **CARLA doesn't start or uses the wrong GPU.** See the `int(4)` fix in Setup.
- **The run waits a long time, then stops with a seg/depth error.** The glare
  needs a segmentation and a depth frame for every camera frame. `--no-strict`
  passes a late frame through without glare instead of stopping.
- **Running DeepSight's own `run_evaluation.sh` instead.** That works for
  Bench2Drive's routes, with
  `TEAM_AGENT=<this repo>/glare_qwen_agent.py`, and with this repo on
  `PYTHONPATH`. Our scenarios also need the scenario class linked into
  `SCENARIO_RUNNER_ROOT`. `run.py` does that for you, so use it for our
  scenarios.

## Files

| File | |
| --- | --- |
| `glare_qwen_agent.py` | DeepSight's agent with glare added in `tick()` |
| `glare_injector.py`, `glare_mask_closed_loop.py` | The glare: which lamps glare, and the rendering |
| `deepsight_input_agent.py` | DeepSight's camera rig and 10 Hz timing |
| `glare_benchmark/bench2drive_scenarios/run.py` | Runs routes with either driver; `check`, `list`, `summary` |
| `glare_benchmark/bench2drive_scenarios/glare_agent.py` | The NpcAgent driver that records clean frames and labels |
| `glare_benchmark/custom_scenarios/glare_popout.py` | The scenarios |
| `glare_benchmark/custom_scenarios/make_routes.py` | Writes their routes |
