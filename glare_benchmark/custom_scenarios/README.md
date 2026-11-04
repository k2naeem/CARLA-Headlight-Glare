# Custom scenarios

Scenarios Bench2Drive cannot guarantee, so we build them. Bench2Drive's
pop-out scenarios (DynamicObjectCrossing, ParkingCrossingPedestrian,
VehicleOpensDoor) never place a lit oncoming car where the hazard appears, and
its oncoming-traffic scenarios have no pop-out. These do both, on purpose,
every run.

| File | |
| --- | --- |
| `glare_popout.py` | The scenarios, as Bench2Drive leaderboard scenario classes |
| `make_routes.py` | Finds straight two-way roads in CARLA's maps and writes a route file per scenario |
| `routes/` | Made by `make_routes.py`: `glare_popout/<id>_<scenario>_<town>.xml` and `index.json` |

## The scenarios

Both run at night on a straight road with one lane each way. The glare
source is a car the scenario drives itself, headlights on, so the timing is
the same every run, whatever speed the ego drives.

**`GlarePedestrianPopOut`** — a pedestrian steps out from behind oncoming
headlights. The pedestrian waits on the centre line at the meet point M. A lit
oncoming car is timed so that, when the ego is 30 m from M, it is 4 m in
front of the pedestrian, about 25 m from the ego. Seen from CAM_FRONT, the
pedestrian stands right behind its headlamps. The pedestrian then walks into
the ego's lane, stands there 3 s, and walks off the road. This is the
geometry `carla_script_closed_loop.py` stages for its cyclist, made into a
route DeepSight's leaderboard can run.

**`GlareOncomingOvertake`** — an oncoming car pulls out of another car's glare,
into the ego's lane. A slow lit car (5 m/s) comes toward the ego, with a
second lit car catching up behind it, hidden in its glare. When the ego is
75 m away, the second car pulls out to overtake, head-on in the ego's lane, at
up to 15 m/s, and pulls back in once past. An ego that brakes keeps about
20 m clear; one at 8 m/s that doesn't react misses it by about 2 m.

Variants are route-file parameters, no code changes needed:

| Parameter | Scenario | Default | Other values |
| --- | --- | --- | --- |
| `adversary` | pedestrian | `walker` | `bicycle` (rides straight across) |
| `start` | pedestrian | `line` (centre line) | `curb` (far sidewalk, crosses both lanes) |
| `ped_start`, `gap` | pedestrian | 30 m, 4 m | earlier or later pop-out; car nearer or farther |
| `speed`, `hold` | pedestrian | 1.4 m/s, 3 s | |
| `lead_speed`, `pass_speed` | overtake | 5, 15 m/s | |
| `swing_dist`, `pull_out_gap`, `catch_gap`, `return_gap` | overtake | 75, 5, 25, 3 m | |
| `beams` | both | `low` | `high` |

The full list, with what each means, is in each class's docstring in
`glare_popout.py`.

## Run (on the Ubuntu machine)

Once, to write the routes (reads the maps' OpenDRIVE files, no CARLA server):

```bash
python make_routes.py
```

It prints each spot it picked, two per town in Town01, 02, 03, 05 and 10HD.
Each spot gets both scenarios. Then, from `../bench2drive_scenarios`:

```bash
python run.py run glare_popout --glare 1.0 --headlights scene
```

```bash
python run.py run glare_popout --glare 0 --headlights scene
```

The second run is the clean baseline: same routes, no glare. `--headlights
scene` keeps the background traffic as Bench2Drive lights it (mostly dark), so
the scenario's cars are the only glare. One route is a family member like any
other: `python run.py run 91001`.

Each run folder is laid out like a Bench2Drive run, plus
`scenario_events.jsonl`: one line per moment (`oncoming_meet`, `popout_start`,
`popout_in_lane`, `popout_clear`, `overtake_swing`, `overtake_return`) with its
CARLA frame, which matches the `frame` in `meta/<step>.json`. That is where
the hazard is in the video, and the frames to label for a dataset.

NpcAgent ignores pedestrians, so it drives into the pedestrian in every run.
This is expected: for now the runs are about the images. Brakes and
collisions mean something once DeepSight drives.

## How it plugs into Bench2Drive

The leaderboard loads scenario classes from every
`$SCENARIO_RUNNER_ROOT/srunner/scenarios/*.py`. For a route of ours, `run.py`
builds a temporary folder with links to Bench2Drive's scenario files plus
`glare_popout.py`, and points `SCENARIO_RUNNER_ROOT` at it. Bench2Drive's
files stay unedited. DeepSight's agent will run these routes the same way.

## Branching off

A new scenario is a new class in `glare_popout.py` built from the same
pieces:
- `_GlareScenario`: the meet point M on a two-way road, and road position `s`
- `OncomingCar`: a lit car, optionally timed to be at a place when the ego is
- `LogEvent`: a frame-stamped moment for the dataset

Then add the class and its default parameters to `SCENARIOS` in
`make_routes.py`.

## Still to build

| # | Scenario | Why Bench2Drive can't | What to build | Start from |
| --- | --- | --- | --- | --- |
| 7 | Sudden high beams on a bend | Traffic Manager picks the beams, usually low; a scenario cannot request high beams | A route where one car appears around a bend at 30–50 m with `beams=high` | Town07 rural bends (`spawn_points.py` to pick them), rural Town12 |
| 8 | Dark-road versions | Bench2Drive turns street lights on within 100 m of the ego at night | A street-light switch, applied to the Bench2Drive families and these | The pole-disabling code in `carla_script_closed_loop.py` |
| 9 | Controlled oncoming stack | Traffic density varies from run to run | Exactly 1, 2 or 4 lit oncoming cars at fixed spacing (`OncomingCar` for each) | The oncoming batches in `carla_script_closed_loop.py` |
| 10 | Sun glare at dawn or dusk | The glare code only uses headlights as light sources | A sun light source in the glare code, from the weather's sun angles; future work | — |
