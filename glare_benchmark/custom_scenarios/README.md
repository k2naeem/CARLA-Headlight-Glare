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

**`GlarePedestrianPopOut`** — a pedestrian steps out from behind an oncoming
car, inside its glare. Nobody stands in the road beforehand: the pedestrian
stays hidden until the lit oncoming car is 30 m from the ego. Then they appear
just behind its rear bumper, right beside its flank (between the car and the
centre line), inside its glare. They wait in the dark, then hurry into the
ego's lane at 2.5 m/s, timed from the ego's speed so they stop right in front
of it 0.5 s before it reaches them: too late to stop for a driver who only
sees them then. They stand there 3 s and walk off the road. Because the
release follows the car, and the step-out follows the ego, it plays out the
same way at any ego speed. The clean frame shows the pedestrian throughout.

How long the glare hides them is limited by the glare itself. Measured with
`glare()`, a pedestrian loses most of their contrast only within about 40 px
of a headlamp, and is nearly back to normal by 80 px. With one car, the
pedestrian starts at the edge of that zone (about 44 px) and is out of it
within half a second: the car's approach sweeps its lamps outward while the
pedestrian walks inward. By the time they step out, 8–14 m ahead of the ego,
the car is beside the ego and its lamps are at the image edge. So the glare
hides them appearing, and darkness (`PedestrianDark`) hides them stepping
out. Every frame is labelled (below), so the dataset records exactly which
frames hide them.

**Two cars (`PedestrianTwoCarsDark`).** A second lit car starts 24 m behind
the first. Once the pedestrian is in place, its speed is set every tick so
that its inner headlamp lines up with them a quarter of the way across
(`line_up` 0.25). That keeps it just a few metres behind the pedestrian as
they step out: about 6 m at an ego speed of 8 m/s, 3 m at 4 m/s, never under
2 m. Its glare covers the wait and the step-out itself. In the offline sim, at
`--glare-size 2` and ego speeds of 3–12 m/s, the pedestrian is inside that
glare for 1–8 s while waiting and 0.4–0.6 s of their 1.1 s crossing, and
still stops 0.4 s before impact. A lower `line_up` brings the car closer
still. For a moment mid-crossing they stand in front of the inner lamp. The
injector then skips that lamp's glare (a covered lamp does not glare), and
the outer lamp still glares.

**Bigger glare.** `python run.py run ... --glare-size 2` makes every part of
the glare twice as large: the zone where a pedestrian keeps under half their
contrast grows from about 37 px to 80 px at 25 m, and from 50 to 108 px at
15 m. At 3 the glare starts dimming the ego's lane centre itself. Runs are
named `..._glare1x2_...`. It applies to any run, Bench2Drive's too; the
default is 1.

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
| `release_dist` | pedestrian | 30 m | the car's distance to the ego when the pedestrian appears |
| `behind`, `ped_offset` | pedestrian | 0.5 m, `auto` (0.15 m clear of the car's side) | where they appear |
| `speed`, `arrive_before` | pedestrian | 2.5 m/s, 0.5 s | how fast they cross; how long before impact they stop in the ego's lane |
| `hold`, `car_speed` | pedestrian | 3 s, 9 m/s | |
| `second_gap`, `second_dist`, `line_up` | pedestrian | 0 (none), `auto`, 0.25 | a second lit car this far behind the first; its distance to the ego at the step-out; how far across the pedestrian is when its lamp lines up with them (lower = closer behind them) |
| `street_lights` | both | `on` | `off`: no street lights along the stretch, lit only by headlights |
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
Each spot gets three routes:

| Ids | Variant | |
| --- | --- | --- |
| 91001+ | `PedestrianDark` | the pedestrian with street lights off, lit only by headlights (the hard case) |
| 92001+ | `Overtake` | the oncoming overtake |
| 93001+ | `PedestrianLit` | the pedestrian under Bench2Drive's street lights (the easier case) |
| 94001+ | `PedestrianTwoCarsDark` | street lights off, plus a second lit car whose glare covers the step-out |

Then, from `../bench2drive_scenarios`:

```bash
python run.py run glare_popout --glare 1.0 --glare-size 2 --headlights scene
```

```bash
python run.py run glare_popout --glare 0 --headlights scene
```

The second run is the clean baseline: same routes, no glare. `--headlights
scene` keeps the background traffic as Bench2Drive lights it (mostly dark), so
the scenario's cars are the only glare. One route is a family member like any
other: `python run.py run 91001`.

Each run folder is laid out like a Bench2Drive run, plus two label files:
- `scenario_events.jsonl`: one line per moment (`street_lights_off`,
  `popout_start`, `second_car_at_mark`, `popout_walk`, `popout_in_lane`, `popout_clear`, `overtake_swing`,
  `overtake_return`) with its CARLA frame, which matches the `frame` in
  `meta/<step>.json`. That is where the hazard is in the video.
- `road_user_glare.jsonl`: for every step with a pedestrian or cyclist in
  CAM_FRONT, how much the glare washes them out. The segmentation camera gives
  their exact pixels. For each pixel, `cover` is how far the glare pushed it
  toward white (0 = untouched, 1 = white), measured on the saved JPEGs
  DeepSight reads. The line records the pixel count, the mean cover, the share
  of pixels at least half way to white (`washed`), and `hidden` (half or more
  washed). It needs glare on, since the segmentation camera only runs then.

`python run.py summary` adds a `hidden` column: steps with a hidden
pedestrian or cyclist out of steps with one in view.

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
| 8 | Dark-road Bench2Drive routes | Bench2Drive turns street lights on near the ego at night | Our scenarios have the switch (`street_lights="off"`); the Bench2Drive families still need it, e.g. from the agent | `StreetLightsOff` in `glare_popout.py` |
| 10 | Sun glare at dawn or dusk | The glare code only uses headlights as light sources | A sun light source in the glare code, from the weather's sun angles; future work | — |
